"""Chat server router that combines ONNX and PyTorch endpoints."""

from fastapi import APIRouter
from ..multimodal_chat.main import app as multimodal_app

# Create main router
router = APIRouter()

# Include multimodal chat routes (preserves all middleware, mounts, etc.)
router.mount("/multimodal", multimodal_app)

# Include PyTorch routes if available
try:
    from .torch_router import router as torch_router
    router.include_router(torch_router, tags=["pytorch"])
except ImportError:
    # PyTorch routes not available, continue with multimodal only
    pass