import time, io, base64
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from diffusers import StableDiffusionXLPipeline, AutoPipelineForText2Image, FluxPipeline, DiffusionPipeline
import numpy as np
from ..shared.diffusion_model_loader import diffusion_loader

try:
    import onnxruntime as ort
    from diffusers import OnnxStableDiffusionPipeline
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    OnnxStableDiffusionPipeline = None
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

# Module-level variables for auto-registration - OpenAI-compatible under /v1
router_prefix = "/v1"
router_tag = "generation"

# Lazy-loaded pipelines
_sdxl = None
_flux1 = None
_qwen_image = None
_sdxl_turbo = None
_sd15_onnx = None

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

def get_sd15_onnx():
    global _sd15_onnx
    if _sd15_onnx is None:
        if not ONNX_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="ONNX is not available. Install onnxruntime to use ONNX models."
            )
        providers = choose_onnx_providers()
        _sd15_onnx = OnnxStableDiffusionPipeline.from_pretrained(
            "tlwu/stable-diffusion-v1-5-onnxruntime",
            provider=providers
        )
    return _sd15_onnx

# Request / response schemas
class ImageGenRequest(BaseModel):
    prompt: str
    model: Optional[str] = "flux1-schnell"
    n: Optional[int] = 1
    width: Optional[int]
    height: Optional[int]
    negative_prompt: Optional[str] = ""

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

# Model metadata for frontend features
MODEL_METADATA = {
    "sdxl": {
        "engine": "pytorch",
        "supports_negative_prompt": True,
        "max_resolution": 1024,
        "default_resolution": 1024,
        "min_resolution": 512
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
        "min_resolution": 512
    },
    "sd15-onnx": {
        "engine": "onnx",
        "supports_negative_prompt": True,
        "max_resolution": 768,
        "default_resolution": 512,
        "min_resolution": 256
    }
}

# Registry mapping model keys to (getter, adapter)
PIPE_REGISTRY = {
    "sdxl": (get_sdxl, _gen_sdxl),
    "flux1-schnell": (get_flux1, _gen_flux1),
    "qwen-image": (get_qwen_image, _gen_qwen_image),
    "sdxl-turbo": (get_sdxl_turbo, _gen_sdxl_turbo),
    **({'sd15-onnx': (get_sd15_onnx, _gen_sd15_onnx)} if ONNX_AVAILABLE else {})
}

@router.get("/models/generation")
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
    
    return {"models": available_models}

@router.get("/models/generation/loaded")
def get_loaded_models():
    """Get information about currently loaded models."""
    return {
        "loaded_pipelines": diffusion_loader.get_loaded_pipelines(),
        "legacy_models": list(PIPE_REGISTRY.keys())
    }

@router.post("/models/generation/unload")
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

@router.post("/models/generation/clear-cache")
def clear_cache():
    """Clear all loaded pipelines to free memory."""
    try:
        diffusion_loader.clear_cache()
        return {"message": "Successfully cleared all pipeline cache"}
    except Exception as e:
        raise HTTPException(500, f"Failed to clear cache: {str(e)}")

@router.post("/images/generations", response_model=ImageGenResponse)
def generate(req: ImageGenRequest):
    key = req.model.lower()
    
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
        raise HTTPException(400, f"model must be one of: {', '.join(available_models)}")

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
    
    # Load pipeline
    try:
        pipeline, metadata = diffusion_loader.load_pipeline(model_key)
    except Exception as e:
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