"""Face server router - exposes face API endpoints under /api prefix."""

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
import asyncio
import logging

# Import the face processing components
from ..face_api.main import model_loader, get_embedding, DETECTOR_MODEL, EMBEDDER_MODEL_PATH, EMBEDDER_FILE

router = APIRouter()

# Module-level variables for auto-registration
router_prefix = "/api/face"
router_tag = "face"

logger = logging.getLogger(__name__)

@router.post("/compare")
async def compare_faces(
    image_a: UploadFile = File(...),
    image_b: UploadFile = File(...),
    face_type: str = Form("photo")
):
    """Compare two face images for similarity."""
    logger.info("Received compare_faces request")
    try:
        # Validate file uploads
        if not image_a or not image_b:
            logger.error("Missing image files in request")
            return JSONResponse(
                status_code=400,
                content={"error": "Both image_a and image_b are required"}
            )
        
        data_a = await image_a.read()
        data_b = await image_b.read()
        
        if not data_a or not data_b:
            logger.error(f"Empty image data: image_a={len(data_a) if data_a else 0} bytes, image_b={len(data_b) if data_b else 0} bytes")
            return JSONResponse(
                status_code=400,
                content={"error": "Both images must contain data"}
            )
            
        logger.info(f"Processing images: a={len(data_a)} bytes, b={len(data_b)} bytes, face_type={face_type}")
        # Get embeddings for both images (not async)
        logger.debug("Getting embedding for image A")
        emb_a = get_embedding(data_a, face_type)
        logger.debug(f"Image A embedding: {'success' if emb_a is not None else 'failed'}")
        
        logger.debug("Getting embedding for image B")
        emb_b = get_embedding(data_b, face_type)
        logger.debug(f"Image B embedding: {'success' if emb_b is not None else 'failed'}")
        
        if emb_a is None or emb_b is None:
            missing = []
            if emb_a is None:
                missing.append("image_a")
            if emb_b is None:
                missing.append("image_b")
            error_msg = f"Face not detected in: {', '.join(missing)}"
            logger.warning(error_msg)
            return JSONResponse(
                status_code=400,
                content={"error": error_msg}
            )
        
        # Calculate cosine similarity
        import numpy as np
        similarity = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b)))
        
        return {
            "confidence": similarity,
            "model": f"{DETECTOR_MODEL}",
            "processing_time_ms": 0  # Would need to track timing
        }
        
    except Exception as e:
        logger.error(f"Error in compare_faces: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "model": DETECTOR_MODEL}

@router.get("/models/info")
async def models_info():
    """Get information about loaded face models."""
    return {
        "detector_model": f"{DETECTOR_MODEL}",
        "embedder_model": f"{EMBEDDER_MODEL_PATH}/{EMBEDDER_FILE}",
        "detector_loaded": model_loader._detector is not None,
        "embedder_loaded": model_loader._embedder is not None
    }