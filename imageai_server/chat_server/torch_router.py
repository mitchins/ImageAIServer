"""PyTorch Chat Router - Optional endpoint for PyTorch models."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any, Optional
import logging

from ..shared.model_manager import get_model_manager, BackendType
from ..shared.torch_loader import TORCH_AVAILABLE

logger = logging.getLogger(__name__)
router = APIRouter()


class TorchChatRequest(BaseModel):
    """Request model for PyTorch chat endpoint."""
    model: str
    messages: List[Dict[str, Any]]
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 0.9
    backend: Optional[str] = "pytorch"  # Can override to use ONNX if available




@router.get("/models")
async def list_models():
    """List available PyTorch models."""
    if not TORCH_AVAILABLE:
        return {
            "pytorch_available": False,
            "models": [],
            "message": "PyTorch backend not available. Install torch and transformers to enable."
        }
    
    model_manager = get_model_manager()
    all_models = model_manager.list_available_models()
    
    return {
        "pytorch_available": True,
        "pytorch_models": all_models.get(BackendType.PYTORCH, []),
        "onnx_models_count": len(all_models.get(BackendType.ONNX, [])),
        "message": "Use /v1/chat/completions/torch endpoint with any PyTorch model"
    }


@router.post("/completions")
async def torch_chat_completions(request: Request):
    """PyTorch-optimized chat completions endpoint."""
    if not TORCH_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="PyTorch backend not available. Install torch and transformers to enable."
        )
    
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    try:
        req = TorchChatRequest(**data)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors())
    
    # Extract text from messages
    text = ""
    if req.messages:
        content = req.messages[-1].get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text += part.get("text", "")
        else:
            text = str(content)
    
    try:
        # Use model manager with backend preference
        model_manager = get_model_manager()
        backend = BackendType(req.backend) if req.backend else BackendType.PYTORCH
        
        result = model_manager.generate_text(
            model_name=req.model,
            text=text,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            backend=backend
        )
        
        return {
            "id": f"torch-{hash(text) % 10000}",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result},
                    "finish_reason": "stop"
                }
            ],
            "model": req.model,
            "backend": backend.value,
        }
    except Exception as e:
        logger.error(f"Error generating text with PyTorch: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.get("/models/available")
async def list_torch_models():
    """OpenAI-compatible models endpoint for PyTorch models."""
    model_manager = get_model_manager()
    all_models = model_manager.list_available_models()
    pytorch_models = all_models.get(BackendType.PYTORCH, [])
    
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "owned_by": "pytorch",
                "permission": []
            }
            for model_id in pytorch_models
        ]
    }