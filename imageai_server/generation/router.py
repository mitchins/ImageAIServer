import time, io, base64
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from diffusers import StableDiffusionXLPipeline, AutoPipelineForText2Image, FluxPipeline, DiffusionPipeline

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

router = APIRouter()

# override auto-registration prefix so decorator path stands alone
router_prefix = ""
router_tag = "generation"

# Lazy-loaded pipelines
_sdxl = None
_flux1 = None
_qwen_image = None
_sdxl_turbo = None

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

# Registry mapping model keys to (getter, adapter)
PIPE_REGISTRY = {
    "sdxl": (get_sdxl, _gen_sdxl),
    "flux1-schnell": (get_flux1, _gen_flux1),
    "qwen-image": (get_qwen_image, _gen_qwen_image),
    "sdxl-turbo": (get_sdxl_turbo, _gen_sdxl_turbo),
}

@router.post("/v1/images/generations", response_model=ImageGenResponse)
def generate(req: ImageGenRequest):
    device = choose_device()

    key = req.model.lower()
    getter, adapter = PIPE_REGISTRY.get(key, (None, None))
    if not getter:
        raise HTTPException(400, "model must be one of: sdxl, flux1-schnell, qwen-image, sdxl-turbo")

    width, height = req.width or 928, req.height or 928
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