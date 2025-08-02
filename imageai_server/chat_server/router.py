"""Chat server router that combines ONNX and PyTorch endpoints."""

from fastapi import APIRouter
from onnx_chat.main import app as onnx_app

# Create main router
router = APIRouter()

# Include ONNX chat routes
for route in onnx_app.routes:
    if hasattr(route, 'endpoint'):
        router.add_api_route(
            route.path,
            route.endpoint,
            methods=route.methods,
            **route.kwargs
        )

# Optionally include PyTorch routes if available
try:
    from .torch_router import router as torch_router
    router.include_router(torch_router, tags=["pytorch"])
except ImportError:
    # PyTorch routes not available, continue with ONNX only
    pass