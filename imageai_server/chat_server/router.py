"""Chat server router - provides OpenAI-compatible chat endpoints."""

from fastapi import APIRouter, Request, HTTPException
from typing import Dict, Any

# Module-level variables for auto-registration
router_prefix = "/v1" 
router_tag = "chat"

router = APIRouter()

# Import multimodal chat handling
try:
    from ..multimodal_chat.main import chat as multimodal_chat_handler
    
    @router.post("/chat/completions")
    async def chat_completions(request: Request):
        """OpenAI-compatible chat completions endpoint."""
        # Delegate to multimodal chat handler
        return await multimodal_chat_handler(request)
        
except ImportError as e:
    # If multimodal chat not available, provide error endpoint
    @router.post("/chat/completions")
    async def chat_completions(request: Request):
        raise HTTPException(
            status_code=503,
            detail="Chat service not available - multimodal chat module not found"
        )

@router.get("/models")
async def list_models():
    """List available chat-compatible models in OpenAI-compatible format."""
    try:
        from ..shared.manage_cache import list_cached_entries
        from ..shared.model_types import ModelType
        from fastapi import HTTPException
        
        cached_entries = list_cached_entries()
        
        # Filter for ONNX models that are LLM-capable and create OpenAI-compatible response
        models = []
        chat_compatible_types = ModelType.chat_compatible_types()
        
        for entry in cached_entries:
            if (entry["path"].endswith(".onnx") and 
                entry["kind"] in [t.value for t in chat_compatible_types]):
                
                # Always create full model ID with repo and complete file path
                # This gives users the exact string they need for API calls
                model_id = f"{entry['repo']}/{entry['path']}"
                
                models.append({
                    "id": model_id,
                    "object": "model",
                    "created": int(entry["last_used"]),
                    "owned_by": entry["repo"].split("/")[0] if "/" in entry["repo"] else "huggingface",
                })
        
        return {
            "object": "list",
            "data": models
        }
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))

# Include PyTorch-specific routes if available (without pytorch prefix)
try:
    from .torch_router import router as torch_router
    router.include_router(torch_router, tags=["chat"])
except ImportError:
    # PyTorch routes not available
    pass