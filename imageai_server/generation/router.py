import time, io, base64
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from diffusers import StableDiffusionXLPipeline, AutoPipelineForText2Image, FluxPipeline, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel, GGUFQuantizationConfig, FluxTransformer2DModel
import numpy as np
from ..shared.diffusion_model_loader import diffusion_loader
# Legacy imports removed - using model_config now
from .model_config import model_config

try:
    import onnxruntime as ort
    try:
        from optimum.onnxruntime import ORTStableDiffusionPipeline, ORTStableDiffusionXLPipeline
        OnnxStableDiffusionPipeline = ORTStableDiffusionPipeline
        OnnxStableDiffusionXLPipeline = ORTStableDiffusionXLPipeline
    except ImportError:
        from diffusers import OnnxStableDiffusionPipeline
        OnnxStableDiffusionXLPipeline = None
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    OnnxStableDiffusionPipeline = None
    OnnxStableDiffusionXLPipeline = None
    ONNX_AVAILABLE = False

# Helper to choose device (CUDA or MPS)
def choose_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    raise HTTPException(
        status_code=503,
        detail="No supported GPU backend available; image generation requires CUDA or MPS."
    )

# Helper to choose ONNX providers
def choose_onnx_providers():
    providers = []
    if torch.cuda.is_available():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers

router = APIRouter()

# override auto-registration prefix so decorator path stands alone
router_prefix = ""
router_tag = "generation"

# Lazy-loaded pipelines
_sdxl = None
_flux1 = None
_qwen_image = None
_sdxl_turbo = None
_sd15_onnx = None
_sdxl_onnx = None
_sdxl_turbo_onnx = None

# GGUF pipeline cache
_gguf_pipelines = {}

def get_sdxl():
    global _sdxl
    if _sdxl is None:
        device = choose_device()
        dtype = torch.float16
        variant = "fp16"
        _sdxl = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=dtype,
            variant=variant,
            use_safetensors=True
        ).to(device)
    return _sdxl

def get_flux1():
    global _flux1
    if _flux1 is None:
        device = choose_device()
        dtype = torch.float16
        _flux1 = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=dtype
        )
        if device != "cpu":
            _flux1.to(device)
        _flux1.enable_model_cpu_offload()
    return _flux1

def get_qwen_image():
    global _qwen_image
    if _qwen_image is None:
        device = choose_device()
        dtype = torch.float16
        _qwen_image = DiffusionPipeline.from_pretrained(
            "Qwen/Qwen-Image",
            torch_dtype=dtype
        ).to(device)
    return _qwen_image

def get_sdxl_turbo():
    global _sdxl_turbo
    if _sdxl_turbo is None:
        device = choose_device()
        _sdxl_turbo = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to(device)
        _sdxl_turbo.enable_attention_slicing()
    return _sdxl_turbo

def get_sd15_onnx_fp16():
    global _sd15_onnx
    if _sd15_onnx is None:
        if not ONNX_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="ONNX is not available. Install onnxruntime to use ONNX models."
            )
        # ORTStableDiffusionPipeline expects a single provider string
        provider = "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
        _sd15_onnx = OnnxStableDiffusionPipeline.from_pretrained(
            "Mitchins/sd15-onnx-fp16",
            provider=provider
        )
    return _sd15_onnx

def get_sdxl_onnx_fp16():
    global _sdxl_onnx
    if _sdxl_onnx is None:
        if not ONNX_AVAILABLE or OnnxStableDiffusionXLPipeline is None:
            raise HTTPException(
                status_code=503,
                detail="ONNX SDXL is not available. Install optimum[onnxruntime] to use ONNX SDXL models."
            )
        provider = "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
        
        # Try local working model first, fallback to HuggingFace
        import os
        from pathlib import Path
        local_model_path = Path(__file__).parent.parent.parent / "onnx_models_fixed" / "sdxl-onnx-fp16-fixed"
        if local_model_path.exists():
            print(f"🔧 Using local SDXL ONNX model: {local_model_path}")
            _sdxl_onnx = OnnxStableDiffusionXLPipeline.from_pretrained(
                str(local_model_path),
                provider=provider
            )
        else:
            print("⚠️  Local model not found, trying HuggingFace (may have type mismatch issues)")
            _sdxl_onnx = OnnxStableDiffusionXLPipeline.from_pretrained(
                "Mitchins/sdxl-onnx-fp16",
                provider=provider
            )
    return _sdxl_onnx

def get_sdxl_turbo_onnx_fp16():
    global _sdxl_turbo_onnx
    if _sdxl_turbo_onnx is None:
        if not ONNX_AVAILABLE or OnnxStableDiffusionXLPipeline is None:
            raise HTTPException(
                status_code=503,
                detail="ONNX SDXL is not available. Install optimum[onnxruntime] to use ONNX SDXL models."
            )
        provider = "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
        _sdxl_turbo_onnx = OnnxStableDiffusionXLPipeline.from_pretrained(
            "Mitchins/sdxl-turbo-onnx-fp16",
            provider=provider
        )
    return _sdxl_turbo_onnx

# GGUF loaders using native diffusers support with from_single_file
def get_sd15_gguf_pipeline(model_repo: str, gguf_filename: str, cache_key: str):
    """Load SD1.5 GGUF model using from_single_file approach."""
    global _gguf_pipelines
    
    if cache_key not in _gguf_pipelines:
        device = choose_device()
        
        # Load quantized UNet using from_single_file
        # Use complete HuggingFace URL format
        gguf_url = f"https://huggingface.co/{model_repo}/blob/main/{gguf_filename}"
        try:
            quantized_unet = UNet2DConditionModel.from_single_file(
                gguf_url,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16),
                torch_dtype=torch.float16
            )
            
            # Create pipeline with quantized UNet and regular components
            _gguf_pipelines[cache_key] = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                unet=quantized_unet,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            )
            
            if device != "cpu":
                _gguf_pipelines[cache_key].to(device)
                
        except Exception as e:
            print(f"❌ Failed to load GGUF model {cache_key}: {e}")
            # Don't fallback for explicitly requested quantizations - raise the error
            raise HTTPException(
                status_code=503,
                detail=f"Failed to load GGUF model {cache_key}: {str(e)}. GGUF support requires diffusers>=0.24.0 and gguf package."
            )
            
    return _gguf_pipelines[cache_key]

def get_sd15_q8():
    return get_sd15_gguf_pipeline(
        "second-state/stable-diffusion-v1-5-GGUF",
        "stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf",
        "sd15_q8"
    )

def get_sd15_q4():
    return get_sd15_gguf_pipeline(
        "second-state/stable-diffusion-v1-5-GGUF", 
        "stable-diffusion-v1-5-pruned-emaonly-Q4_1.gguf",
        "sd15_q4"
    )

def get_sdxl_gguf_pipeline(model_repo: str, gguf_filename: str, cache_key: str):
    """Load SDXL GGUF model using from_single_file approach."""
    global _gguf_pipelines
    
    if cache_key not in _gguf_pipelines:
        device = choose_device()
        
        # Load quantized UNet using from_single_file for SDXL
        gguf_url = f"https://huggingface.co/{model_repo}/blob/main/{gguf_filename}"
        try:
            quantized_unet = UNet2DConditionModel.from_single_file(
                gguf_url,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16),
                torch_dtype=torch.float16
            )
            
            # Create SDXL pipeline with quantized UNet and regular components
            _gguf_pipelines[cache_key] = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                unet=quantized_unet,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            )
            
            if device != "cpu":
                _gguf_pipelines[cache_key].to(device)
                
        except Exception as e:
            print(f"❌ Failed to load SDXL GGUF model {cache_key}: {e}")
            # Don't fallback for explicitly requested quantizations - raise the error
            raise HTTPException(
                status_code=503,
                detail=f"Failed to load SDXL GGUF model {cache_key}: {str(e)}. GGUF support requires diffusers>=0.24.0 and gguf package."
            )
            
    return _gguf_pipelines[cache_key]

def get_sdxl_q8():
    return get_sdxl_gguf_pipeline(
        "gpustack/stable-diffusion-xl-base-1.0-GGUF",
        "stable-diffusion-xl-base-1.0-Q8_0.gguf", 
        "sdxl_q8"
    )

def get_sdxl_q4():
    return get_sdxl_gguf_pipeline(
        "gpustack/stable-diffusion-xl-base-1.0-GGUF",
        "stable-diffusion-xl-base-1.0-Q4_1.gguf",
        "sdxl_q4"
    )

def get_flux1_gguf_pipeline(model_repo: str, gguf_filename: str, cache_key: str):
    """Load FLUX GGUF model using from_single_file approach."""
    global _gguf_pipelines
    
    if cache_key not in _gguf_pipelines:
        device = choose_device()
        
        # Load quantized Transformer using from_single_file for FLUX
        gguf_url = f"https://huggingface.co/{model_repo}/blob/main/{gguf_filename}"
        try:
            quantized_transformer = FluxTransformer2DModel.from_single_file(
                gguf_url,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                torch_dtype=torch.bfloat16
            )
            
            # Create FLUX pipeline with quantized transformer and regular components
            _gguf_pipelines[cache_key] = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-schnell",
                transformer=quantized_transformer,
                torch_dtype=torch.bfloat16
            )
            
            if device != "cpu":
                _gguf_pipelines[cache_key].to(device)
            _gguf_pipelines[cache_key].enable_model_cpu_offload()
                
        except Exception as e:
            print(f"❌ Failed to load FLUX GGUF model {cache_key}: {e}")
            # Don't fallback for explicitly requested quantizations - raise the error
            raise HTTPException(
                status_code=503,
                detail=f"Failed to load FLUX GGUF model {cache_key}: {str(e)}. GGUF support requires diffusers>=0.24.0 and gguf package."
            )
            
    return _gguf_pipelines[cache_key]

def get_flux1_q8():
    return get_flux1_gguf_pipeline(
        "city96/FLUX.1-schnell-gguf",
        "flux1-schnell-Q8_0.gguf",
        "flux1_q8"
    )

def get_flux1_q4():
    return get_flux1_gguf_pipeline(
        "city96/FLUX.1-schnell-gguf", 
        "flux1-schnell-Q4_1.gguf",
        "flux1_q4"
    )

# Request / response schemas
class ImageGenRequest(BaseModel):
    prompt: str
    model: Optional[str] = "flux1-schnell"
    n: Optional[int] = 1
    width: Optional[int]
    height: Optional[int]
    negative_prompt: Optional[str] = ""
    working_set: Optional[str] = None  # For diffusion models with multiple working sets

class ImageData(BaseModel):
    b64_json: str

class ImageGenResponse(BaseModel):
    created: int
    data: List[ImageData]

# Helper to encode PIL→base64
def _pil_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# Adapter functions
def _gen_sdxl(pipe, prompt, **_):
    return [pipe(prompt=prompt).images[0]]

def _gen_sdxl_turbo(pipe, prompt, width, height, negative_prompt, n, **_):
    imgs = []
    for _ in range(n):
        imgs.append(pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
        ).images[0])
    return imgs

def _gen_flux1(pipe, prompt, **_):
    return [pipe(
        prompt,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256
    ).images[0]]

def _gen_qwen_image(pipe, prompt, width, height, negative_prompt, **_):
    return [pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width, height=height,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        generator=torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(time.time()))
    ).images[0]]

def _gen_sd15_onnx(pipe, prompt, width, height, negative_prompt, n, **_):
    imgs = []
    for _ in range(n):
        imgs.append(pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            width=width,
            height=height,
            num_inference_steps=20,
            guidance_scale=7.5
        ).images[0])
    return imgs

def _gen_sdxl_onnx(pipe, prompt, width, height, negative_prompt, n, **_):
    imgs = []
    for _ in range(n):
        imgs.append(pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            width=width,
            height=height,
            num_inference_steps=25,
            guidance_scale=8.0
        ).images[0])
    return imgs

# GGUF adapter functions
def _gen_sd15_gguf(pipe, prompt, width, height, negative_prompt, n, **_):
    imgs = []
    for _ in range(n):
        imgs.append(pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            width=width,
            height=height,
            num_inference_steps=20,
            guidance_scale=7.5
        ).images[0])
    return imgs

def _gen_sdxl_gguf(pipe, prompt, width, height, negative_prompt, n, **_):
    imgs = []
    for _ in range(n):
        imgs.append(pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            width=width,
            height=height,
            num_inference_steps=25,
            guidance_scale=8.0
        ).images[0])
    return imgs

def _gen_flux1_gguf(pipe, prompt, **_):
    return [pipe(
        prompt,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256
    ).images[0]]

# Model metadata for frontend features
MODEL_METADATA = {
    "sdxl": {
        "engine": "pytorch",
        "supports_negative_prompt": True,
        "max_resolution": 1024,
        "default_resolution": 1024,
        "min_resolution": 512,
        "display_name": "Stable Diffusion XL",
        "description": "Full precision SDXL model",
        "memory_requirement": "~8GB VRAM",
        "quantization": "FP16"
    },
    "flux1-schnell": {
        "engine": "pytorch", 
        "supports_negative_prompt": False,
        "max_resolution": 1024,
        "default_resolution": 1024,
        "min_resolution": 256
    },
    "qwen-image": {
        "engine": "pytorch",
        "supports_negative_prompt": True,
        "max_resolution": 1024,
        "default_resolution": 928,
        "min_resolution": 256
    },
    "sdxl-turbo": {
        "engine": "pytorch",
        "supports_negative_prompt": True,
        "max_resolution": 1024,
        "default_resolution": 1024,
        "min_resolution": 512,
        "display_name": "Stable Diffusion XL Turbo",
        "description": "Fast SDXL variant, fewer steps needed",
        "memory_requirement": "~8GB VRAM",
        "quantization": "FP16"
    },
    "sd15-onnx": {
        "engine": "onnx",
        "supports_negative_prompt": True,
        "max_resolution": 768,
        "default_resolution": 512,
        "min_resolution": 256,
        "display_name": "Stable Diffusion 1.5 (ONNX FP16)",
        "description": "ONNX optimized model with 50% memory reduction",
        "memory_requirement": "~2GB",
        "quantization": "FP16"
    },
    "sdxl-onnx": {
        "engine": "onnx",
        "supports_negative_prompt": True,
        "max_resolution": 1536,
        "default_resolution": 1024,
        "min_resolution": 512,
        "display_name": "Stable Diffusion XL (ONNX)",
        "description": "ONNX optimized SDXL for CPU/GPU inference",
        "memory_requirement": "~8GB",
        "quantization": "FP32"
    },
    "sdxl-turbo-onnx": {
        "engine": "onnx", 
        "supports_negative_prompt": True,
        "max_resolution": 1024,
        "default_resolution": 1024,
        "min_resolution": 512,
        "display_name": "SDXL Turbo (ONNX FP16)",
        "description": "ONNX optimized SDXL Turbo, fast generation",
        "memory_requirement": "~5GB",
        "quantization": "FP16"
    }
}

# Registry mapping model keys to (getter, adapter)
PIPE_REGISTRY = {
    "sdxl": (get_sdxl, _gen_sdxl),
    "flux1-schnell": (get_flux1, _gen_flux1),
    "qwen-image": (get_qwen_image, _gen_qwen_image),
    "sdxl-turbo": (get_sdxl_turbo, _gen_sdxl_turbo),
    # ONNX models - FP16 only
    **({'sd15-onnx': (get_sd15_onnx_fp16, _gen_sd15_onnx)} if ONNX_AVAILABLE else {}),
    **({'sdxl-onnx': (get_sdxl_onnx_fp16, _gen_sdxl_onnx)} if ONNX_AVAILABLE and OnnxStableDiffusionXLPipeline else {}),
    **({'sdxl-turbo-onnx': (get_sdxl_turbo_onnx_fp16, _gen_sdxl_onnx)} if ONNX_AVAILABLE and OnnxStableDiffusionXLPipeline else {}),
    # GGUF models
    "sd15-q8": (get_sd15_q8, _gen_sd15_gguf),
    "sd15-q4": (get_sd15_q4, _gen_sd15_gguf),
    "sdxl-q8": (get_sdxl_q8, _gen_sdxl_gguf),
    "sdxl-q4": (get_sdxl_q4, _gen_sdxl_gguf), 
    "flux1-q8": (get_flux1_q8, _gen_flux1_gguf),
    "flux1-q4": (get_flux1_q4, _gen_flux1_gguf)
}

@router.get("/v1/models/generation")
def get_generation_models():
    """Get available generation models and their metadata."""
    # Get models from both the new diffusion loader and legacy registry
    available_models = {}
    
    # Add models from new diffusion system
    diffusion_models = diffusion_loader.get_available_models()
    available_models.update(diffusion_models)
    
    # Add legacy models (keeping backward compatibility)
    for model_key in PIPE_REGISTRY.keys():
        if model_key in MODEL_METADATA and model_key not in available_models:
            available_models[model_key] = MODEL_METADATA[model_key]
    
    # Add models from config system
    config_models = model_config.get_legacy_model_metadata()
    available_models.update(config_models)
    
    return {"models": available_models}

@router.get("/v1/models/generation/families")
def get_generation_model_families():
    """Get models organized by family with quantization variants.""" 
    # Legacy endpoint - now uses config system
    return {"families": {}}

@router.get("/v1/models/generation/ui-config")
def get_ui_config():
    """Get model configuration organized for UI display."""
    return model_config.get_ui_structure()

@router.get("/v1/models/generation/loaded")
def get_loaded_models():
    """Get information about currently loaded models."""
    return {
        "loaded_pipelines": diffusion_loader.get_loaded_pipelines(),
        "legacy_models": list(PIPE_REGISTRY.keys())
    }

@router.post("/v1/models/generation/unload")
def unload_pipeline(request: dict):
    """Unload a specific pipeline to free memory."""
    model_id = request.get("model_id")
    working_set_name = request.get("working_set_name")
    
    if not model_id:
        raise HTTPException(400, "model_id is required")
    
    try:
        diffusion_loader.unload_pipeline(model_id, working_set_name)
        return {"message": f"Successfully unloaded pipeline {model_id}"}
    except Exception as e:
        raise HTTPException(500, f"Failed to unload pipeline: {str(e)}")

@router.post("/v1/models/generation/clear-cache")
def clear_cache():
    """Clear all loaded pipelines to free memory."""
    try:
        diffusion_loader.clear_cache()
        return {"message": "Successfully cleared all pipeline cache"}
    except Exception as e:
        raise HTTPException(500, f"Failed to clear cache: {str(e)}")

@router.post("/v1/images/generations", response_model=ImageGenResponse)
def generate(req: ImageGenRequest):
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"🎨 Generation request received: model={req.model}, working_set={req.working_set}")
    original_model = req.model.lower()
    
    # Try to resolve model using config system first
    resolved = model_config.resolve_model_id(original_model)
    if resolved:
        model, backend, quantization = resolved
        logger.info(f"🔧 Resolved {original_model} to model={model}, backend={backend}, quantization={quantization}")
        
        # Look up corresponding legacy ID from the mapping
        for legacy_id, mapping in model_config.id_mapping.items():
            if mapping == [model, backend, quantization]:
                key = legacy_id
                logger.info(f"🔄 Mapped to legacy ID: {key}")
                break
        else:
            # No legacy mapping found, use resolved model name as key
            key = model
            logger.info(f"🔄 Using resolved model name: {key}")
    else:
        key = original_model
        logger.info(f"❌ Could not resolve {original_model}, using as-is")
    
    # Try new diffusion system first
    try:
        return _generate_with_diffusion_system(req, key)
    except Exception as e:
        # Fall back to legacy system
        pass
    
    # Legacy system
    getter, adapter = PIPE_REGISTRY.get(key, (None, None))
    if not getter:
        available_models = list(PIPE_REGISTRY.keys()) + list(diffusion_loader.get_available_models().keys())
        config_models = list(model_config.get_legacy_model_metadata().keys())
        all_available = sorted(set(available_models + config_models))
        raise HTTPException(400, f"model must be one of: {', '.join(all_available)}")

    # Get model metadata for validation
    metadata = MODEL_METADATA.get(key, {})
    default_res = metadata.get("default_resolution", 512)
    max_res = metadata.get("max_resolution", 1024)
    min_res = metadata.get("min_resolution", 256)
    
    width = req.width or default_res
    height = req.height or default_res
    
    # Validate resolution limits
    if width > max_res or height > max_res:
        raise HTTPException(400, f"Resolution too high for {req.model}. Max: {max_res}x{max_res}")
    if width < min_res or height < min_res:
        raise HTTPException(400, f"Resolution too low for {req.model}. Min: {min_res}x{min_res}")

    # Choose device based on engine type
    if metadata.get("engine") == "pytorch":
        device = choose_device()  # This will raise HTTPException if no GPU
    # ONNX models can run on CPU, so no device check needed

    pipe = getter()
    pil_imgs = adapter(
        pipe,
        req.prompt,
        width=width,
        height=height,
        negative_prompt=req.negative_prompt,
        n=req.n
    )

    return ImageGenResponse(
        created=int(time.time()),
        data=[ImageData(b64_json=_pil_to_b64(img)) for img in pil_imgs]
    )

def _generate_with_diffusion_system(req: ImageGenRequest, model_key: str) -> ImageGenResponse:
    """Generate using the new diffusion model system."""
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"🔧 Using diffusion system for {model_key}")
    
    # Check if model exists in diffusion system
    diffusion_models = diffusion_loader.get_available_models()
    if model_key not in diffusion_models:
        raise ValueError(f"Model {model_key} not found in diffusion system")
    
    model_metadata = diffusion_models[model_key]
    
    # Validate resolution
    width = req.width or model_metadata["default_resolution"]
    height = req.height or model_metadata["default_resolution"]
    
    if width > model_metadata["max_resolution"] or height > model_metadata["max_resolution"]:
        raise HTTPException(400, f"Resolution too high for {req.model}. Max: {model_metadata['max_resolution']}x{model_metadata['max_resolution']}")
    if width < model_metadata["min_resolution"] or height < model_metadata["min_resolution"]:
        raise HTTPException(400, f"Resolution too low for {req.model}. Min: {model_metadata['min_resolution']}x{model_metadata['min_resolution']}")
    
    # Load pipeline with optional working set
    try:
        logger.info(f"📥 Loading pipeline {model_key} with working_set={req.working_set}")
        pipeline, metadata = diffusion_loader.load_pipeline(model_key, req.working_set)
        logger.info(f"✅ Pipeline loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load pipeline: {str(e)}")
        raise HTTPException(500, f"Failed to load model {model_key}: {str(e)}")
    
    # Generate images
    pil_imgs = []
    settings = metadata.get("settings", {})
    
    try:
        for _ in range(req.n or 1):
            # Prepare generation parameters
            gen_kwargs = {
                "prompt": req.prompt,
                "width": width,
                "height": height,
                "num_inference_steps": settings.get("num_inference_steps", 20),
                "guidance_scale": settings.get("guidance_scale", 7.5),
            }
            
            # Add negative prompt if supported
            if model_metadata.get("supports_negative_prompt", True) and req.negative_prompt:
                gen_kwargs["negative_prompt"] = req.negative_prompt
            
            # Generate image
            result = pipeline(**gen_kwargs)
            pil_imgs.extend(result.images)
    
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {str(e)}")
    
    return ImageGenResponse(
        created=int(time.time()),
        data=[ImageData(b64_json=_pil_to_b64(img)) for img in pil_imgs[:req.n or 1]]
    )