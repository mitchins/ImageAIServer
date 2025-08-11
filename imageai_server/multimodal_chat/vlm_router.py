"""
VLM Router - Handles HTTP routing only
Business logic is in VLMService
"""

import logging
import base64
import io
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from PIL import Image

from .vlm_service import get_vlm_service, VLMStrategyError

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    images: Optional[List[str]] = None  # base64-encoded images
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class ModelLoadRequest(BaseModel):
    model: str
    backend: Optional[str] = None


def decode_base64_image(base64_str: str) -> Image.Image:
    """Decode base64 image string to PIL Image"""
    try:
        # Remove data URL prefix if present
        if base64_str.startswith('data:image'):
            base64_str = base64_str.split(',', 1)[1]
        
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")


def extract_prompt_from_messages(messages: List[Dict[str, Any]]) -> str:
    """Extract text prompt from messages"""
    if not messages:
        return "what"  # Default neutral prompt
    
    # Get last message content
    last_message = messages[-1]
    content = last_message.get("content", "")
    
    if isinstance(content, list):
        # Handle multimodal content format
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        return " ".join(text_parts) or "what"
    else:
        return str(content) or "what"


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        vlm_service = get_vlm_service()
        health_info = await vlm_service.health_check()
        return {"status": "ok", "vlm": health_info}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "error", "error": str(e)}


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Main VLM chat endpoint"""
    try:
        # Parse request
        data = await request.json()
        req = ChatRequest(**data)
        
        vlm_service = get_vlm_service()
        
        # Extract prompt
        prompt = extract_prompt_from_messages(req.messages)
        
        # Get image (required for VLM)
        if not req.images:
            raise HTTPException(status_code=400, detail="Image required for VLM inference")
        
        image = decode_base64_image(req.images[0])
        
        # Generate response
        response = await vlm_service.generate_response(
            image=image,
            prompt=prompt,
            max_tokens=req.max_tokens or 100,
            temperature=req.temperature or 0.0
        )
        
        # Return OpenAI-compatible response
        return {
            "id": "chatcmpl-vlm",
            "object": "chat.completion",
            "model": response.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response.text.split()),
                "total_tokens": len(prompt.split()) + len(response.text.split())
            },
            "metadata": {
                "backend": response.backend,
                "inference_time": response.inference_time
            }
        }
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors())
    except VLMStrategyError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/load_model")
async def load_model(request: ModelLoadRequest):
    """Load a VLM model"""
    try:
        vlm_service = get_vlm_service()
        
        success = await vlm_service.load_model(
            model_name=request.model,
            backend=request.backend
        )
        
        if success:
            model_info = vlm_service.get_loaded_model_info()
            return {
                "status": "success",
                "message": f"Model {request.model} loaded successfully",
                "model_info": model_info
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to load model")
            
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unload_model")
async def unload_model():
    """Unload current VLM model"""
    try:
        vlm_service = get_vlm_service()
        vlm_service.unload_model()
        
        return {
            "status": "success",
            "message": "Model unloaded successfully"
        }
        
    except Exception as e:
        logger.error(f"Model unloading failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_models():
    """List available models and backends"""
    try:
        vlm_service = get_vlm_service()
        
        # Get MLX models (other strategies can be added here)
        mlx_strategy = vlm_service.get_strategy("mlx")
        models = []
        
        if mlx_strategy:
            supported_models = mlx_strategy.get_supported_models()
            for key, info in supported_models.items():
                models.append({
                    "id": key,
                    "object": "model",
                    "model_id": info["model_id"],
                    "backend": "mlx",
                    "size": info["size"],
                    "description": info["description"],
                    "tested": info.get("tested", False)
                })
        
        return {
            "object": "list",
            "data": models,
            "available_backends": vlm_service.get_available_backends()
        }
        
    except Exception as e:
        logger.error(f"List models failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_status():
    """Get current VLM service status"""
    try:
        vlm_service = get_vlm_service()
        
        return {
            "loaded_model": vlm_service.get_loaded_model_info(),
            "available_backends": vlm_service.get_available_backends(),
            "service_health": await vlm_service.health_check()
        }
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))