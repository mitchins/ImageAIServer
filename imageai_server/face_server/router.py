"""Face server router that mounts the face API application."""

from fastapi import APIRouter
from ..face_api.main import app as face_app

# Create main router and mount the face API
router = APIRouter()
router.mount("/api", face_app)
