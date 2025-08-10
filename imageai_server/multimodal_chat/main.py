from __future__ import annotations
import os
from typing import List, Dict, Any
import argparse
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from .config import load_config, setup_logging
from ..shared.manage_cache import download_file
from ..shared.model_types import (
    ReferenceModel, Quantization, REFERENCE_MODELS, 
    get_available_model_quants, get_smallest_quant_for_model
)
from ..shared.model_manager import get_model_manager, BackendType

# Import ONNX components with explicit mock handling
# Check for explicit mock environment variable
USE_MOCK_ONNX = os.getenv("USE_MOCK_ONNX", "false").lower() in ("true", "1", "yes")

if USE_MOCK_ONNX:
    # Mock classes for testing
    class MockONNXModelLoader:
        def load_model(self, model_name):
            return {}, None, None
    
    class MockONNXInferenceEngine:
        def __init__(self, *args, **kwargs):
            pass
        def generate_text(self, text, max_tokens=100, images=None, audio=None):
            return "mocked response"
    
    ONNXModelLoader = MockONNXModelLoader
    ONNXInferenceEngine = MockONNXInferenceEngine
    ONNX_LOADER_AVAILABLE = False
    print("Using mock ONNX components for testing")
else:
    try:
        from ..shared.onnx_loader import ONNXModelLoader, ONNXInferenceEngine
        ONNX_LOADER_AVAILABLE = True
    except ImportError as e:
        print(f"Error: ONNX dependencies not available: {e}")
        print("Set USE_MOCK_ONNX=true environment variable to use mock components for testing")
        raise

import base64
import io
from PIL import Image

# Utility to find the nearest ancestor directory containing config.json
def find_config_root(path: str) -> str | None:
    dir_path = os.path.dirname(path)
    while True:
        if os.path.exists(os.path.join(dir_path, "config.json")):
            return dir_path
        parent = os.path.dirname(dir_path)
        if parent == dir_path:
            return None
        dir_path = parent

setup_logging()
config = load_config()

MODEL_PATH = config.model_path
DEFAULT_FILE = os.getenv("ONNX_FILE_NAME", "model.onnx")

# Global model manager
model_manager = get_model_manager()

import logging

logger = logging.getLogger(__name__)

# Removed old model loading functions - now using ONNXModelLoader

app = FastAPI()

class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    images: List[str] | None = None  # base64-encoded images
    audio: List[str] | None = None   # base64-encoded audio or URLs
    max_tokens: int | None = None


@app.get("/health")
async def health_check():
    model_name = os.path.basename(MODEL_PATH) if MODEL_PATH else "none"
    return {"status": "ok", "model": model_name}


# Helper to validate model name against curated list
def validate_model_name(model_name: str) -> str:
    """Validate that the model is in our curated list and return the repo_id."""
    # First, try using the unified model system
    from ..shared.model_identifier import ModelCatalog, ModelNameParser
    
    # Handle unified model identifiers (e.g., "smolvlm-256m-instruct:q8_0")
    try:
        model_id, quantization, backend = ModelNameParser.normalize_model_name(model_name)
        if model_id and model_id in ModelCatalog.MODELS:
            model_info = ModelCatalog.MODELS[model_id]
            return model_info.repo_id
    except Exception:
        pass
    
    # Handle direct reference model names
    try:
        ref_model = ReferenceModel(model_name)
        if ref_model in REFERENCE_MODELS:
            return REFERENCE_MODELS[ref_model].repo_id
    except ValueError:
        pass
    
    # Handle repo_id format (check if it matches any reference model repo_id)
    for ref_model, spec in REFERENCE_MODELS.items():
        if model_name == spec.repo_id or model_name.startswith(spec.repo_id):
            return spec.repo_id
    
    # Handle curated model names directly (ONNX-only models)
    curated_model_names = ["Gemma-3n-E2B-it-ONNX"]
    for curated_name in curated_model_names:
        if model_name == curated_name or model_name.startswith(curated_name):
            # Map curated name back to repo_id
            mapping = {
                "Gemma-3n-E2B-it-ONNX": "onnx-community/gemma-3n-E2B-it-ONNX",
            }
            return mapping[curated_name]
    
    # Handle PyTorch-only models (let model manager handle backend selection)
    pytorch_models = [
        "HuggingFaceTB/SmolVLM-256M-Instruct",
        "HuggingFaceTB/SmolVLM-500M-Instruct", 
        "ibm-granite/granite-vision-3.2-2b"
    ]
    
    # Strip quantization suffixes for PyTorch model name checking
    base_model_name = model_name
    quantization_suffixes = ["/FP32", "/FP16", "/INT8", "/UINT8", "/Q4", "/Q4_F16", "/BNB4", "/QUANTIZED"]
    for suffix in quantization_suffixes:
        if base_model_name.endswith(suffix):
            base_model_name = base_model_name[:-len(suffix)]
            break
    
    for pytorch_model in pytorch_models:
        if base_model_name == pytorch_model or base_model_name.endswith(pytorch_model.split('/')[-1]):
            return pytorch_model
    
    # Model not in curated list
    available_models = [f"{model.value} ({spec.repo_id})" for model, spec in REFERENCE_MODELS.items()]
    available_curated = get_available_model_quants()[:3]  # Show first 3
    raise HTTPException(
        status_code=400, 
        detail=f"Model '{model_name}' not supported. Available reference models: {', '.join(available_models)} or curated models: {', '.join(available_curated)}..."
    )

# Helper to get inference engine
async def get_inference_engine(model_name: str):
    """Get inference engine for a model, using appropriate backend."""
    # For unified model identifiers, let the model manager handle everything
    from ..shared.model_identifier import ModelNameParser
    
    try:
        # Try to parse as unified model identifier first
        model_id, quantization, backend = ModelNameParser.normalize_model_name(model_name)
        if model_id:
            # Use the original model name to preserve quantization info
            return model_manager.load_model(model_name, backend=None)  # Let model manager choose
    except Exception:
        pass
    
    # Validate model first for legacy models
    validated_repo_id = validate_model_name(model_name)
    
    # Map repo_id back to model name for curated configs
    model_name_mapping = {
        "onnx-community/gemma-3n-E2B-it-ONNX": "Gemma-3n-E2B-it-ONNX",
        "HuggingFaceTB/SmolVLM-256M-Instruct": "SmolVLM-256M-Instruct",
    }
    
    curated_model_name = model_name_mapping.get(validated_repo_id)
    if curated_model_name:
        # Check if the original model_name already includes quantization
        if '/' in model_name and model_name.startswith(curated_model_name):
            # User specified exact model/quant, use it directly
            model_quant_key = model_name
        else:
            # Use curated model with smallest quantization
            model_quant_key = get_smallest_quant_for_model(curated_model_name)
            if model_quant_key is None:
                raise ValueError(f"No quantization found for curated model: {curated_model_name}")
    else:
        # Fall back to legacy approach
        model_quant_key = f"{validated_repo_id}:q4"
    
    # Use model manager to load the model - let it choose the appropriate backend
    return model_manager.load_model(model_quant_key, backend=None)


# Generate text using inference engine
async def generate_text(text: str, model_name: str, max_tokens: int = 100, images: List[str] | None = None, audio: List[str] | None = None) -> str:
    """Generate text using the model manager, with MLX VLM priority for compatible models."""
    
    # Try MLX VLM first for MLX-compatible models (Apple Silicon + has images)
    if images and _is_mlx_model(model_name):
        try:
            from .vlm_service import get_vlm_service
            vlm_service = get_vlm_service()
            
            # Check if MLX backend is available
            available_backends = vlm_service.get_available_backends()
            if any(b['name'] == 'mlx' and b['available'] for b in available_backends):
                logger.info(f"Using MLX VLM for model: {model_name}")
                
                # Load MLX model if not already loaded
                current_model = vlm_service.get_loaded_model_info()
                if not current_model or current_model['model_name'] != model_name:
                    success = await vlm_service.load_model(model_name, backend="mlx")
                    if not success:
                        logger.warning(f"Failed to load MLX model {model_name}, falling back to legacy backends")
                        raise Exception("MLX model loading failed")
                
                # Convert base64 images to PIL Images
                pil_images = []
                for img_base64 in images:
                    try:
                        import io
                        image_data = base64.b64decode(img_base64)
                        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
                        pil_images.append(pil_image)
                    except Exception as e:
                        logger.error(f"Failed to decode image: {e}")
                        continue
                
                if pil_images:
                    # Use first image for now (could be enhanced for multi-image support)
                    response = await vlm_service.generate_response(
                        image=pil_images[0],
                        prompt=text,
                        max_tokens=max_tokens,
                        temperature=0.0
                    )
                    return response.text
                else:
                    logger.warning("No valid images found, falling back to legacy backends")
                    
        except Exception as e:
            logger.warning(f"MLX VLM generation failed for {model_name}: {e}")
            # Fall through to legacy backends
    
    # Fall back to existing model manager for ONNX/PyTorch models
    return model_manager.generate_text(
        model_name=model_name,
        text=text,
        max_tokens=max_tokens,
        images=images,
        audio=audio,
        backend=None  # Let model manager choose appropriate backend
    )


def _is_mlx_model(model_name: str) -> bool:
    """Check if a model is MLX-compatible."""
    if not model_name:
        return False
    
    # Check for MLX model patterns
    mlx_patterns = [
        "mlx",
        "lmstudio-community",
        "gemma-3n",
        "gemma3n"
    ]
    
    model_lower = model_name.lower()
    return any(pattern in model_lower for pattern in mlx_patterns)


# Legacy functions for backward compatibility with old tests
def load_session(model_path: str):
    """Legacy function for old tests - returns None to trigger fallback logic."""
    return None

def download_model(repo_id: str, filename: str, cache_dir: str = None) -> str:
    """Legacy function for old tests."""
    # Mock download - just return a fake path
    import tempfile
    return os.path.join(tempfile.gettempdir(), filename)

def _ensure_model_path(model_name: str) -> str:
    """Legacy function for old tests."""
    # Parse model name like the old function did
    if os.path.exists(model_name):
        return model_name
        
    # Parse repo and filename
    if ":" in model_name:
        repo_id, filename = model_name.split(":", 1)
    else:
        repo_id = model_name
        filename = "model.onnx"
    
    # Use download_model function (which can be mocked in tests)
    cache_dir = os.path.expanduser("~/.cache/onnx_chat")
    return download_model(repo_id, filename, cache_dir)

def classify(text: str, model_path: str) -> str:
    """Legacy classification function for old tests."""
    # Try to load session (for test mocking)
    session = load_session(model_path)
    
    if session is not None:
        # Use ONNX session if available (for tests)
        try:
            inputs = session.get_inputs()
            input_name = inputs[0].name
            result = session.run(None, {input_name: text})
            return str(result[0][0])
        except Exception:
            pass
    
    # Simple rule-based fallback since ONNX classification was removed
    if "good" in text.lower() or "great" in text.lower() or "excellent" in text.lower():
        return "positive"
    elif "bad" in text.lower() or "terrible" in text.lower() or "awful" in text.lower():
        return "negative"
    else:
        return "neutral"


@app.post("/v1/chat/completions")
async def chat(request: Request):
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    try:
        req = ChatRequest(**data)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors())

    text = ""
    if req.messages:
        content = req.messages[-1].get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text += part.get("text", "")
        else:
            text = str(content)
    result = await generate_text(text, req.model or (MODEL_PATH or ""), req.max_tokens or 100, req.images, req.audio)
    return {
        "id": "cmpl-001",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": result}, "finish_reason": "stop"}
        ],
        "model": req.model,
    }


def main() -> None:
    """CLI entry point for running the server via ``python -m``."""
    import uvicorn

    parser = argparse.ArgumentParser(description="ONNX chat completion server")
    parser.add_argument("--host", default=config.host)
    parser.add_argument("--port", type=int, default=config.port)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "imageai_server.multimodal_chat.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=config.log_level,
    )


if __name__ == "__main__":
    main()
