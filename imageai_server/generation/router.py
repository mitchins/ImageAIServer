import time, io, base64
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from diffusers import StableDiffusionXLPipeline, AutoPipelineForText2Image, FluxPipeline, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel, FluxTransformer2DModel
import numpy as np
from ..shared.diffusion_model_loader import diffusion_loader
# Legacy imports removed - using model_config now
from .model_config import model_config
from .runtime_manager import runtime_manager

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

# CoreML Support (Apple Silicon only)
try:
    import coremltools
    from python_coreml_stable_diffusion.pipeline import get_coreml_pipe
    from diffusers import StableDiffusionPipeline
    import platform
    # Only enable on macOS ARM64 (Apple Silicon)
    COREML_AVAILABLE = (platform.system() == "Darwin" and platform.machine() == "arm64")
    if COREML_AVAILABLE:
        print("🍎 CoreML available on Apple Silicon")
    else:
        print("⚠️  CoreML tools installed but not on Apple Silicon")
except ImportError:
    get_coreml_pipe = None
    COREML_AVAILABLE = False

# Helper to choose device (CUDA or MPS)
def choose_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"

# Helper for multi-GPU device assignment
def assign_pipeline_devices(pipeline, fallback_device="cuda"):
    """Assign pipeline components to different GPUs based on environment variables"""
    import os
    
    # Get device assignments from environment variables
    clip_device = f"cuda:{os.getenv('CLIP_GPU_DEVICE', '0')}"
    t5_device = f"cuda:{os.getenv('T5_GPU_DEVICE', '0')}"
    diffuser_device = f"cuda:{os.getenv('DIFFUSER_GPU_DEVICE', '0')}"
    vae_device = f"cuda:{os.getenv('VAE_GPU_DEVICE', '0')}"
    
    try:
        # Move components to assigned devices
        if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
            pipeline.text_encoder.to(clip_device)
        if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
            pipeline.text_encoder_2.to(t5_device)
        if hasattr(pipeline, 'transformer') and pipeline.transformer is not None:
            pipeline.transformer.to(diffuser_device)
        if hasattr(pipeline, 'vae') and pipeline.vae is not None:
            pipeline.vae.to(vae_device)
            
        print(f"🔧 Multi-GPU Assignment: CLIP→{clip_device}, T5→{t5_device}, Diffuser→{diffuser_device}, VAE→{vae_device}")
        
    except Exception as e:
        print(f"⚠️  Multi-GPU assignment failed, falling back to single device: {e}")
        pipeline.to(fallback_device)

# Helper to choose ONNX providers
def choose_onnx_providers():
    providers = []
    if torch.cuda.is_available():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers

# Enhanced ONNX provider selection using runtime_manager
def get_optimal_onnx_config():
    """Get optimal ONNX providers and configuration using runtime_manager"""
    optimal_runtime = runtime_manager.get_optimal_runtime()
    
    # Check if optimal runtime is ONNX-based
    if optimal_runtime.startswith("onnx-"):
        config = runtime_manager.get_runtime_config(optimal_runtime)
        return config["providers"]
    
    # Fallback to legacy provider selection
    return choose_onnx_providers()

router = APIRouter()

# override auto-registration prefix so decorator path stands alone
router_prefix = ""
router_tag = "generation"

# Lazy-loaded pipelines
_sdxl = None
_flux1 = None
_flux1_dev_pytorch = None
_flux1_dev_tensorrt = None
_flux1_dev_fp8 = None
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
            # Use multi-GPU assignment instead of single device
            assign_pipeline_devices(_flux1, fallback_device=device)
        _flux1.enable_model_cpu_offload()
    return _flux1

def get_flux1_dev_pytorch():
    """Get Flux1.dev PyTorch pipeline"""
    global _flux1_dev_pytorch
    if _flux1_dev_pytorch is None:
        device = choose_device()
        dtype = torch.float16
        _flux1_dev_pytorch = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=dtype
        )
        if device != "cpu":
            assign_pipeline_devices(_flux1_dev_pytorch, fallback_device=device)
        _flux1_dev_pytorch.enable_model_cpu_offload()
    return _flux1_dev_pytorch

def get_flux1_dev_tensorrt():
    """Get Flux1.dev TensorRT pipeline - uses TensorRT-RTX implementation"""
    global _flux1_dev_tensorrt
    if _flux1_dev_tensorrt is None:
        import sys
        import os
        
        # Add the TensorRT demo path
        tensorrt_path = os.path.join(os.getcwd(), "nvidia-demos/TensorRT-RTX/demo")
        if tensorrt_path not in sys.path:
            sys.path.insert(0, tensorrt_path)
        
        # Clean import - no more path gymnastics!
        from flux.pipelines.flux_pipeline import FluxPipeline as TensorRTFluxPipeline
        
        # Initialize TensorRT Flux pipeline with low VRAM mode
        # Force device to cuda:0 for TensorRT engine compatibility
        _flux1_dev_tensorrt = TensorRTFluxPipeline(
            cache_dir="./demo_cache",
            device="cuda:0",  # Force primary GPU for TensorRT engines
            low_vram=True,  # Enable for memory efficiency
            verbose=True
        )
        
        print("✅ TensorRT-RTX Flux1.dev pipeline loaded successfully")
    
    return _flux1_dev_tensorrt

def get_flux1_dev_fp8():
    """Get Flux1.dev with FP8 quantization - uses TensorRT-RTX implementation"""
    global _flux1_dev_fp8
    if _flux1_dev_fp8 is None:
        import sys
        import os
        
        # Add the TensorRT demo path
        tensorrt_path = os.path.join(os.getcwd(), "nvidia-demos/TensorRT-RTX/demo")
        if tensorrt_path not in sys.path:
            sys.path.insert(0, tensorrt_path)
        
        # Clean import
        from flux.pipelines.flux_pipeline import FluxPipeline as TensorRTFluxPipeline
        
        # Initialize TensorRT Flux pipeline with low VRAM mode
        _flux1_dev_fp8 = TensorRTFluxPipeline(
            cache_dir="./demo_cache",
            device="cuda:0",  # Force primary GPU for TensorRT engines
            low_vram=True,
            verbose=True
        )
        
        print("✅ TensorRT-RTX Flux1.dev FP8 pipeline loaded successfully")
    
    return _flux1_dev_fp8

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

def get_sd15_onnx_fp32():
    global _sd15_onnx
    if _sd15_onnx is None:
        if not ONNX_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="ONNX is not available. Install onnxruntime to use ONNX models."
            )
        provider = "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
        _sd15_onnx = OnnxStableDiffusionPipeline.from_pretrained(
            "imgailab/sd15-onnx-cpu",
            provider=provider
        )
    return _sd15_onnx

def get_sdxl_onnx_fp32():
    global _sdxl_onnx
    if _sdxl_onnx is None:
        if not ONNX_AVAILABLE or OnnxStableDiffusionXLPipeline is None:
            raise HTTPException(
                status_code=503,
                detail="ONNX SDXL is not available. Install optimum[onnxruntime] to use ONNX SDXL models."
            )
        provider = "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
        _sdxl_onnx = OnnxStableDiffusionXLPipeline.from_pretrained(
            "imgailab/sdxl-onnx-cpu",
            provider=provider
        )
    return _sdxl_onnx

def get_sdxl_turbo_onnx_fp32():
    global _sdxl_turbo_onnx
    if _sdxl_turbo_onnx is None:
        if not ONNX_AVAILABLE or OnnxStableDiffusionXLPipeline is None:
            raise HTTPException(
                status_code=503,
                detail="ONNX SDXL is not available. Install optimum[onnxruntime] to use ONNX SDXL models."
            )
        provider = "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
        model_path = "imgailab/sdxl-turbo-onnx-cpu"
        print(f"🔧 [DEBUG] Loading SDXL-Turbo ONNX from: {model_path}")
        
        _sdxl_turbo_onnx = OnnxStableDiffusionXLPipeline.from_pretrained(
            model_path,
            provider=provider
        )
        print(f"✅ [DEBUG] SDXL-Turbo ONNX loaded successfully")
    return _sdxl_turbo_onnx

# TensorRT-Optimized Model Loaders (with dynamic engines support)
_sd15_tensorrt = None
_sdxl_tensorrt = None 
_sdxl_turbo_tensorrt = None

def get_sd15_tensorrt():
    """Load SD1.5 with TensorRT optimization and dynamic engines"""
    global _sd15_tensorrt
    if _sd15_tensorrt is None:
        if not ONNX_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="ONNX is not available. Install onnxruntime to use TensorRT models."
            )
        
        # Use runtime_manager for optimal TensorRT configuration
        providers = get_optimal_onnx_config()
        print(f"🚀 [DEBUG] Loading SD1.5 TensorRT with providers: {providers}")
        
        _sd15_tensorrt = OnnxStableDiffusionPipeline.from_pretrained(
            "imgailab/sd15-onnx-cpu",
            provider=providers
        )
        print(f"✅ [DEBUG] SD1.5 TensorRT loaded successfully")
    return _sd15_tensorrt

def get_sdxl_tensorrt():
    """Load SDXL with TensorRT optimization and dynamic engines"""
    global _sdxl_tensorrt
    if _sdxl_tensorrt is None:
        if not ONNX_AVAILABLE or OnnxStableDiffusionXLPipeline is None:
            raise HTTPException(
                status_code=503,
                detail="ONNX SDXL is not available. Install optimum[onnxruntime] to use TensorRT SDXL models."
            )
            
        # Use runtime_manager for optimal TensorRT configuration
        providers = get_optimal_onnx_config()
        print(f"🚀 [DEBUG] Loading SDXL TensorRT with providers: {providers}")
        
        _sdxl_tensorrt = OnnxStableDiffusionXLPipeline.from_pretrained(
            "imgailab/sdxl-onnx-cpu",
            provider=providers
        )
        print(f"✅ [DEBUG] SDXL TensorRT loaded successfully")
    return _sdxl_tensorrt

def get_sdxl_turbo_tensorrt():
    """Load SDXL-Turbo with TensorRT optimization and dynamic engines"""
    global _sdxl_turbo_tensorrt
    if _sdxl_turbo_tensorrt is None:
        if not ONNX_AVAILABLE or OnnxStableDiffusionXLPipeline is None:
            raise HTTPException(
                status_code=503,
                detail="ONNX SDXL is not available. Install optimum[onnxruntime] to use TensorRT SDXL models."
            )
            
        # Use runtime_manager for optimal TensorRT configuration
        providers = get_optimal_onnx_config()
        model_path = "imgailab/sdxl-turbo-onnx-cpu"
        print(f"🚀 [DEBUG] Loading SDXL-Turbo TensorRT from: {model_path}")
        print(f"🚀 [DEBUG] Using providers: {providers}")
        
        _sdxl_turbo_tensorrt = OnnxStableDiffusionXLPipeline.from_pretrained(
            model_path,
            provider=providers
        )
        print(f"✅ [DEBUG] SDXL-Turbo TensorRT loaded successfully")
    return _sdxl_turbo_tensorrt

# CoreML Model Loaders (Apple Silicon only)
_sd15_coreml = None
_sdxl_coreml = None

def get_sd15_coreml():
    """Load SD1.5 with Apple CoreML optimization (Apple Silicon only)"""
    global _sd15_coreml
    if _sd15_coreml is None:
        if not COREML_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="CoreML is not available. Requires Apple Silicon (M1/M2/M3/M4) and CoreML tools."
            )
        
        print(f"🍎 [DEBUG] Loading SD1.5 CoreML...")
        
        # Download Apple's pre-converted CoreML model
        from huggingface_hub import snapshot_download
        from pathlib import Path
        import os
        
        model_id = "apple/coreml-stable-diffusion-v1-5"
        variant = "original/packages"
        
        # Create local cache directory
        cache_dir = Path("./coreml_models")
        cache_dir.mkdir(exist_ok=True)
        
        local_model_path = cache_dir / "sd15_coreml"
        
        if not local_model_path.exists():
            print(f"📥 [DEBUG] Downloading CoreML SD1.5 from {model_id}...")
            snapshot_download(
                model_id,
                allow_patterns=f"{variant}/*",
                local_dir=local_model_path,
                cache_dir=None  # Don't use default cache, use our local path
            )
            print(f"✅ [DEBUG] CoreML SD1.5 downloaded to {local_model_path}")
        
        # Create a base PyTorch pipeline for conversion
        base_pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            use_safetensors=True
        )
        
        # Convert to CoreML using Apple's official tools with local path
        _sd15_coreml = get_coreml_pipe(
            pytorch_pipe=base_pipeline,
            mlpackages_dir=str(local_model_path / variant),
            model_version="runwayml/stable-diffusion-v1-5",
            compute_unit="CPU_AND_GPU"  # Use both CPU and GPU on Apple Silicon
        )
        print(f"✅ [DEBUG] SD1.5 CoreML loaded successfully")
    return _sd15_coreml

def get_sdxl_coreml():
    """Load SDXL with Apple CoreML optimization (Apple Silicon only)"""
    global _sdxl_coreml
    if _sdxl_coreml is None:
        if not COREML_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="CoreML is not available. Requires Apple Silicon (M1/M2/M3/M4) and CoreML tools."
            )
        
        print(f"🍎 [DEBUG] Loading SDXL CoreML...")
        
        # Download Apple's pre-converted CoreML model
        from huggingface_hub import snapshot_download
        from pathlib import Path
        import os
        
        model_id = "apple/coreml-stable-diffusion-xl-base"
        variant = "packages"  # SDXL uses 'packages' directly, not 'original/packages'
        
        # Create local cache directory
        cache_dir = Path("./coreml_models")
        cache_dir.mkdir(exist_ok=True)
        
        local_model_path = cache_dir / "sdxl_coreml"
        
        if not local_model_path.exists():
            print(f"📥 [DEBUG] Downloading CoreML SDXL from {model_id}...")
            snapshot_download(
                model_id,
                allow_patterns=f"{variant}/*",
                local_dir=local_model_path,
                cache_dir=None  # Don't use default cache, use our local path
            )
            print(f"✅ [DEBUG] CoreML SDXL downloaded to {local_model_path}")
        
        # Create a base PyTorch pipeline for conversion  
        base_pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            use_safetensors=True
        )
        
        # Convert to CoreML using Apple's official tools with local path
        _sdxl_coreml = get_coreml_pipe(
            pytorch_pipe=base_pipeline,
            mlpackages_dir=str(local_model_path / variant),
            model_version="stabilityai/stable-diffusion-xl-base-1.0",
            compute_unit="ALL"  # Use all compute units on Apple Silicon
        )
        print(f"✅ [DEBUG] SDXL CoreML loaded successfully")
    return _sdxl_coreml

# INT8 Model Loaders
_sd15_int8 = None
_sdxl_int8 = None
_sdxl_turbo_int8 = None
_flux1_int8 = None

def get_sd15_pytorch_int8():
    import logging
    logger = logging.getLogger(__name__)
    global _sd15_int8
    if _sd15_int8 is None:
        device = choose_device()
        model_path = "imgailab/sd15-torch-int8"
        logger.info(f"🔧 Loading SD1.5 INT8 model from: {model_path}")
        logger.info(f"   Device: {device}")
        logger.info(f"   Format: PyTorch (.bin files, not safetensors)")
        logger.info(f"   Expected: TorchAO INT8 quantized weights")
        # Force print for debugging
        print(f"🔧 [DEBUG] Loading SD1.5 INT8 model from: {model_path}")
        print(f"   Device: {device}")
        print(f"   Format: PyTorch (.bin files, not safetensors)")
        
        _sd15_int8 = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=False  # INT8 models use PyTorch format
        ).to(device)
        
        # Verify model loaded correctly
        logger.info(f"✅ SD1.5 INT8 model loaded successfully")
        logger.info(f"   UNet device: {next(_sd15_int8.unet.parameters()).device}")
        logger.info(f"   UNet dtype: {next(_sd15_int8.unet.parameters()).dtype}")
        
        # Check if quantization info exists
        try:
            import json
            from transformers.utils import cached_file
            quant_file = cached_file(model_path, "quantization_info.json")
            if quant_file:
                with open(quant_file, 'r') as f:
                    quant_info = json.load(f)
                logger.info(f"   Quantization method: {quant_info.get('method', 'unknown')}")
                logger.info(f"   Original model: {quant_info.get('original_model', 'unknown')}")
        except Exception as e:
            logger.warning(f"   Could not load quantization info: {e}")
            
    return _sd15_int8

def get_sdxl_pytorch_int8():
    import logging
    logger = logging.getLogger(__name__)
    global _sdxl_int8
    if _sdxl_int8 is None:
        device = choose_device()
        model_path = "imgailab/sdxl-torch-int8"
        logger.info(f"🔧 Loading SDXL INT8 model from: {model_path}")
        logger.info(f"   Device: {device}")
        logger.info(f"   Format: PyTorch (.bin files, not safetensors)")
        logger.info(f"   Expected: TorchAO INT8 quantized weights")
        
        _sdxl_int8 = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=False  # INT8 models use PyTorch format
        ).to(device)
        
        # Verify model loaded correctly
        logger.info(f"✅ SDXL INT8 model loaded successfully")
        logger.info(f"   UNet device: {next(_sdxl_int8.unet.parameters()).device}")
        logger.info(f"   UNet dtype: {next(_sdxl_int8.unet.parameters()).dtype}")
        
        # Check if quantization info exists
        try:
            import json
            from transformers.utils import cached_file
            quant_file = cached_file(model_path, "quantization_info.json")
            if quant_file:
                with open(quant_file, 'r') as f:
                    quant_info = json.load(f)
                logger.info(f"   Quantization method: {quant_info.get('method', 'unknown')}")
                logger.info(f"   Original model: {quant_info.get('original_model', 'unknown')}")
        except Exception as e:
            logger.warning(f"   Could not load quantization info: {e}")
            
    return _sdxl_int8

def get_sdxl_turbo_pytorch_int8():
    import logging
    logger = logging.getLogger(__name__)
    global _sdxl_turbo_int8
    if _sdxl_turbo_int8 is None:
        device = choose_device()
        model_path = "imgailab/sdxl-turbo-torch-int8"
        logger.info(f"🔧 Loading SDXL-Turbo INT8 model from: {model_path}")
        logger.info(f"   Device: {device}")
        logger.info(f"   Format: PyTorch (.bin files, not safetensors)")
        logger.info(f"   Expected: TorchAO INT8 quantized weights")
        
        _sdxl_turbo_int8 = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=False  # INT8 models use PyTorch format
        ).to(device)
        
        # Verify model loaded correctly
        logger.info(f"✅ SDXL-Turbo INT8 model loaded successfully")
        logger.info(f"   UNet device: {next(_sdxl_turbo_int8.unet.parameters()).device}")
        logger.info(f"   UNet dtype: {next(_sdxl_turbo_int8.unet.parameters()).dtype}")
        
        # Check if quantization info exists
        try:
            import json
            from transformers.utils import cached_file
            quant_file = cached_file(model_path, "quantization_info.json")
            if quant_file:
                with open(quant_file, 'r') as f:
                    quant_info = json.load(f)
                logger.info(f"   Quantization method: {quant_info.get('method', 'unknown')}")
                logger.info(f"   Original model: {quant_info.get('original_model', 'unknown')}")
        except Exception as e:
            logger.warning(f"   Could not load quantization info: {e}")
            
    return _sdxl_turbo_int8

def get_flux1_pytorch_int8():
    import logging
    logger = logging.getLogger(__name__)
    global _flux1_int8
    if _flux1_int8 is None:
        device = choose_device()
        model_path = "imgailab/flux1-torch-int8"
        logger.info(f"🔧 Loading FLUX.1 INT8 model from: {model_path}")
        logger.info(f"   Device: {device}")
        logger.info(f"   Format: PyTorch (.bin files, not safetensors)")
        logger.info(f"   Expected: TorchAO INT8 quantized weights")
        
        _flux1_int8 = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=False  # INT8 models use PyTorch format
        )
        
        if device != "cpu":
            _flux1_int8.to(device)
        _flux1_int8.enable_model_cpu_offload()
        
        # Verify model loaded correctly (FLUX uses transformer, not unet)
        logger.info(f"✅ FLUX.1 INT8 model loaded successfully")
        logger.info(f"   Transformer device: {next(_flux1_int8.transformer.parameters()).device}")
        logger.info(f"   Transformer dtype: {next(_flux1_int8.transformer.parameters()).dtype}")
        
        # Check if quantization info exists
        try:
            import json
            from transformers.utils import cached_file
            quant_file = cached_file(model_path, "quantization_info.json")
            if quant_file:
                with open(quant_file, 'r') as f:
                    quant_info = json.load(f)
                logger.info(f"   Quantization method: {quant_info.get('method', 'unknown')}")
                logger.info(f"   Original model: {quant_info.get('original_model', 'unknown')}")
        except Exception as e:
            logger.warning(f"   Could not load quantization info: {e}")
            
    return _flux1_int8

# GGUF support removed due to shape mismatch issues
# Will be replaced with SafeTensors quantized models in the future

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
    generation_time_seconds: Optional[float] = None

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
        # SDXL-Turbo specific parameters - guidance_scale=0.0 and 1 step
        imgs.append(pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=1,   # SDXL-Turbo works best with 1 step
            guidance_scale=0.0,      # Model trained without guidance
            generator=torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
        ).images[0])
    return imgs

def _gen_flux1(pipe, prompt, **_):
    # Optimized settings for Flux.1-schnell (fast variant)
    return [pipe(
        prompt,
        guidance_scale=0.0,  # Schnell works best with no guidance
        num_inference_steps=4,  # Schnell is designed for 4 steps
        max_sequence_length=256
    ).images[0]]

def _gen_flux1_dev_pytorch(pipe, prompt, **_):
    # Optimized settings for Flux.1-dev (quality variant)
    return [pipe(
        prompt,
        guidance_scale=3.5,  # Dev benefits from guidance
        num_inference_steps=28,  # Dev needs more steps for quality
        max_sequence_length=256
    ).images[0]]

def _gen_flux1_tensorrt(pipe, prompt, width=1024, height=1024, **_):
    """TensorRT-RTX Flux adapter - uses different interface"""
    import tempfile
    import os
    from PIL import Image
    
    # TensorRT pipeline needs engines loaded first (engines should exist in demo_cache)
    if not hasattr(pipe, '_engines_loaded'):
        # Force cleanup first to ensure clean state
        pipe.cleanup()
        
        pipe.load_engines(
            transformer_precision="bf16",
            opt_batch_size=1,
            opt_height=height,
            opt_width=width,
            shape_mode="static",
        )
        pipe.load_resources(
            batch_size=1,
            height=height,
            width=width,
        )
        pipe._engines_loaded = True
    
    # Create temp directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # TensorRT-RTX pipeline uses infer() method and saves to file
        pipe.infer(
            prompt=prompt,
            save_path=temp_dir,
            height=height,
            width=width,
            seed=None,
            batch_size=1,
            num_inference_steps=4,
            guidance_scale=3.5,
        )
        
        # Load the generated image from file
        output_files = [f for f in os.listdir(temp_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if output_files:
            image_path = os.path.join(temp_dir, output_files[0])
            return [Image.open(image_path)]
        else:
            raise RuntimeError("TensorRT pipeline did not generate any output images")

def _gen_flux1_tensorrt_fp8(pipe, prompt, width=1024, height=1024, **_):
    """TensorRT-RTX Flux adapter with FP8 precision"""
    import tempfile
    import os
    from PIL import Image
    
    # TensorRT pipeline needs engines loaded first (FP8 engines need to be built)
    if not hasattr(pipe, '_engines_loaded'):
        # Force cleanup first to ensure clean state
        pipe.cleanup()
        
        pipe.load_engines(
            transformer_precision="fp8",  # Use FP8 for this variant
            opt_batch_size=1,
            opt_height=height,
            opt_width=width,
            shape_mode="static",
        )
        pipe.load_resources(
            batch_size=1,
            height=height,
            width=width,
        )
        pipe._engines_loaded = True
    
    # Create temp directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # TensorRT-RTX pipeline uses infer() method and saves to file
        pipe.infer(
            prompt=prompt,
            save_path=temp_dir,
            height=height,
            width=width,
            seed=None,
            batch_size=1,
            num_inference_steps=4,
            guidance_scale=3.5,
        )
        
        # Load the generated image from file
        output_files = [f for f in os.listdir(temp_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if output_files:
            image_path = os.path.join(temp_dir, output_files[0])
            return [Image.open(image_path)]
        else:
            raise RuntimeError("TensorRT pipeline did not generate any output images")

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

def _gen_sdxl_turbo_onnx(pipe, prompt, width, height, negative_prompt, n, **_):
    imgs = []
    for _ in range(n):
        # SDXL-Turbo ONNX needs the same 1-step, no-guidance parameters
        imgs.append(pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            width=width,
            height=height,
            num_inference_steps=1,   # SDXL-Turbo works best with 1 step
            guidance_scale=0.0       # Model trained without guidance
        ).images[0])
    return imgs

# INT8 adapter functions
def _gen_sd15_int8(pipe, prompt, width, height, negative_prompt, n, **_):
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

def _gen_sdxl_int8(pipe, prompt, width, height, negative_prompt, n, **_):
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

def _gen_sdxl_turbo_int8(pipe, prompt, width, height, negative_prompt, n, **_):
    imgs = []
    for _ in range(n):
        # SDXL-Turbo INT8 uses the same parameters as FP16 - 1 step, no guidance
        imgs.append(pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=1,   # SDXL-Turbo works best with 1 step
            guidance_scale=0.0,      # Model trained without guidance
            generator=torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
        ).images[0])
    return imgs

def _gen_flux1_int8(pipe, prompt, **_):
    return [pipe(
        prompt,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256
    ).images[0]]

# CoreML adapter functions
def _gen_sd15_coreml(pipe, prompt, width, height, negative_prompt, n, **_):
    """Generate images using Apple CoreML pipeline"""
    imgs = []
    for _ in range(n):
        # Apple CoreML pipeline uses similar interface to diffusers
        imgs.append(pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            width=width,
            height=height,
            num_inference_steps=20,
            guidance_scale=7.5
        ).images[0])
    return imgs

def _gen_sdxl_coreml(pipe, prompt, width, height, negative_prompt, n, **_):
    """Generate images using Apple CoreML SDXL pipeline - optimized for sharpness"""
    imgs = []
    
    # Import scheduler for better quality
    from diffusers import DPMSolverMultistepScheduler

    # Debug: verify SDXL VAE scaling factor (should be ~0.13025 for SDXL)
    try:
        sf = getattr(pipe.vae.config, "scaling_factor", None)
        if sf is not None:
            print(f"🍎 [DEBUG] SDXL VAE scaling_factor={sf} (expected ≈ 0.13025)")
    except Exception:
        print("🍎 [DEBUG] Could not access SDXL VAE scaling factor, assuming default")
    
    for _ in range(n):
        # SDXL CoreML Optimization for Sharpness (based on expert advice):
        # 1. Use DPM-Solver++ (Karras) scheduler instead of Euler for sharper results
        # 2. Use 24-30 steps for optimal quality with DPM-Solver++
        # 3. CFG 6.5-7.0 range (we're using 7.0)
        # 4. Generate at native 1024x1024 when possible
        # 5. Apple's models should already use full SDXL VAE (not TAESD)
        
        try:
            # Set up DPM-Solver++ (Karras) scheduler for sharper results
            original_scheduler = pipe.scheduler
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                use_karras_sigmas=True,  # Karras noise schedule for better quality
                algorithm_type="dpmsolver++",  # DPM-Solver++ variant
            )
            
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                width=width,
                height=height,
                num_inference_steps=24,  # Optimal for DPM-Solver++ (Karras): 24-30 steps
                guidance_scale=6.8,      # Sweet spot in 6.5-7.0 range for sharpness
                generator=torch.Generator(device="cpu").manual_seed(42)  # Consistent results
            )
            
            # Restore original scheduler
            pipe.scheduler = original_scheduler
            
            imgs.append(result.images[0])
            
        except Exception as e:
            # Fallback to original parameters if DPM-Solver++ fails
            print(f"🍎 [DEBUG] DPM-Solver++ failed, falling back to default: {e}")
            imgs.append(pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                width=width,
                height=height,
                num_inference_steps=20,
                guidance_scale=7.0
            ).images[0])
    
    return imgs


# GGUF adapter functions removed - replaced by SafeTensors quantized models

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
        "default_resolution": 512,  # SDXL-Turbo works best at 512x512
        "min_resolution": 512,
        "display_name": "Stable Diffusion XL Turbo",
        "description": "Fast SDXL variant, 1-step inference",
        "memory_requirement": "~8GB VRAM",
        "quantization": "FP16"
    },
    "sd15-onnx": {
        "engine": "onnx",
        "supports_negative_prompt": True,
        "max_resolution": 768,
        "default_resolution": 512,
        "min_resolution": 256,
        "display_name": "Stable Diffusion 1.5 (ONNX FP32)",
        "description": "ONNX CPU/GPU optimized, reliable controlled repo",
        "memory_requirement": "~4GB",
        "quantization": "FP32"
    },
    "sdxl-onnx": {
        "engine": "onnx",
        "supports_negative_prompt": True,
        "max_resolution": 1536,
        "default_resolution": 1024,
        "min_resolution": 512,
        "display_name": "Stable Diffusion XL (ONNX FP32)",
        "description": "ONNX CPU/GPU optimized, reliable controlled repo",
        "memory_requirement": "~12GB",
        "quantization": "FP32"
    },
    "sdxl-turbo-onnx": {
        "engine": "onnx",
        "supports_negative_prompt": True,
        "max_resolution": 1024,
        "default_resolution": 512,  # SDXL-Turbo works best at 512x512
        "min_resolution": 512,
        "display_name": "Stable Diffusion XL Turbo (ONNX FP32)",
        "description": "Fast ONNX generation, 1-step inference",
        "memory_requirement": "~12GB",
        "quantization": "FP32"
    },
    # INT8 Models
    "sd15-pytorch:int8": {
        "engine": "pytorch",
        "supports_negative_prompt": True,
        "max_resolution": 768,
        "default_resolution": 512,
        "min_resolution": 256,
        "display_name": "Stable Diffusion 1.5 (PyTorch INT8)",
        "description": "TorchAO quantized, ~50% memory reduction",
        "memory_requirement": "~1GB VRAM",
        "quantization": "INT8"
    },
    "sdxl-pytorch:int8": {
        "engine": "pytorch",
        "supports_negative_prompt": True,
        "max_resolution": 1536,
        "default_resolution": 1024,
        "min_resolution": 512,
        "display_name": "Stable Diffusion XL (PyTorch INT8)",
        "description": "TorchAO quantized, ~50% memory reduction",
        "memory_requirement": "~3.5GB VRAM",
        "quantization": "INT8"
    },
    "sdxl-turbo-pytorch:int8": {
        "engine": "pytorch",
        "supports_negative_prompt": True,
        "max_resolution": 1024,
        "default_resolution": 512,  # SDXL-Turbo works best at 512x512
        "min_resolution": 512,
        "display_name": "Stable Diffusion XL Turbo (PyTorch INT8)",
        "description": "TorchAO quantized, 1-step inference with 50% memory reduction",
        "memory_requirement": "~3.5GB VRAM",
        "quantization": "INT8"
    },
    "flux1-pytorch:int8": {
        "engine": "pytorch",
        "supports_negative_prompt": False,
        "max_resolution": 1024,
        "default_resolution": 1024,
        "min_resolution": 256,
        "display_name": "FLUX.1 Schnell (PyTorch INT8)",
        "description": "TorchAO quantized, ~50% memory reduction for complex prompts",
        "memory_requirement": "~12GB VRAM",
        "quantization": "INT8"
    },
    # CoreML Models
    "sd15-coreml": {
        "engine": "coreml",
        "supports_negative_prompt": True,
        "max_resolution": 768,
        "default_resolution": 512,
        "min_resolution": 256,
        "display_name": "Stable Diffusion 1.5 (Apple CoreML FP16)",
        "description": "Apple Silicon optimized, Neural Engine acceleration",
        "memory_requirement": "~2GB Unified Memory",
        "quantization": "FP16"
    },
    "sdxl-coreml": {
        "engine": "coreml",
        "supports_negative_prompt": True,
        "max_resolution": 1536,
        "default_resolution": 1024,  # SDXL trained at 1024 - use native resolution for sharpness
        "min_resolution": 512,
        "display_name": "Stable Diffusion XL (Apple CoreML FP16) - Optimized",
        "description": "Apple Silicon optimized with DPM-Solver++ for enhanced sharpness",
        "memory_requirement": "~6GB Unified Memory",
        "quantization": "FP16"
    },
}

# Registry mapping model keys to (getter, adapter)
PIPE_REGISTRY = {
    "sdxl": (get_sdxl, _gen_sdxl),
    "flux1-schnell": (get_flux1, _gen_flux1),
    "flux1-dev": (get_flux1_dev_tensorrt, _gen_flux1_tensorrt),
    "flux1-dev-fp8": (get_flux1_dev_fp8, _gen_flux1_tensorrt_fp8),
    "flux1-dev-pytorch": (get_flux1_dev_pytorch, _gen_flux1_dev_pytorch),
    "qwen-image": (get_qwen_image, _gen_qwen_image),
    "sdxl-turbo": (get_sdxl_turbo, _gen_sdxl_turbo),
    # ONNX models - FP32 controlled repos
    **({'sd15-onnx': (get_sd15_onnx_fp32, _gen_sd15_onnx)} if ONNX_AVAILABLE else {}),
    **({'sdxl-onnx': (get_sdxl_onnx_fp32, _gen_sdxl_onnx)} if ONNX_AVAILABLE and OnnxStableDiffusionXLPipeline else {}),
    **({'sdxl-turbo-onnx': (get_sdxl_turbo_onnx_fp32, _gen_sdxl_turbo_onnx)} if ONNX_AVAILABLE and OnnxStableDiffusionXLPipeline else {}),
    # TensorRT models - Same ONNX repos but with TensorRT optimization
    **({'sd15-tensorrt': (get_sd15_tensorrt, _gen_sd15_onnx)} if ONNX_AVAILABLE else {}),
    **({'sdxl-tensorrt': (get_sdxl_tensorrt, _gen_sdxl_onnx)} if ONNX_AVAILABLE and OnnxStableDiffusionXLPipeline else {}),
    **({'sdxl-turbo-tensorrt': (get_sdxl_turbo_tensorrt, _gen_sdxl_turbo_onnx)} if ONNX_AVAILABLE and OnnxStableDiffusionXLPipeline else {}),
    # CoreML models - Apple Silicon optimized 
    **({'sd15-coreml': (get_sd15_coreml, _gen_sd15_coreml)} if COREML_AVAILABLE else {}),
    **({'sdxl-coreml': (get_sdxl_coreml, _gen_sdxl_coreml)} if COREML_AVAILABLE else {}),
    # INT8 models
    "sd15-pytorch:int8": (get_sd15_pytorch_int8, _gen_sd15_int8),
    "sdxl-pytorch:int8": (get_sdxl_pytorch_int8, _gen_sdxl_int8),
    "sdxl-turbo-pytorch:int8": (get_sdxl_turbo_pytorch_int8, _gen_sdxl_turbo_int8),
    "flux1-pytorch:int8": (get_flux1_pytorch_int8, _gen_flux1_int8)
}

# Log INT8 model registration on import (this will show at startup)
import logging
_startup_logger = logging.getLogger(__name__)
_int8_models = [k for k in PIPE_REGISTRY.keys() if "int8" in k]
if _int8_models:
    print(f"🔧 INT8 Models Registered: {', '.join(_int8_models)}")
    _startup_logger.info(f"INT8 Models available: {', '.join(_int8_models)}")

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

@router.get("/v1/models/generation/runtime-status")
def get_runtime_status():
    """Get current runtime status and model compatibility information."""
    return model_config.get_runtime_status()

@router.get("/v1/models/generation/gpu-status")
def get_gpu_status():
    """Get GPU device assignment status and memory information."""
    import os
    
    gpu_status = {
        "device_assignment": {
            "CLIP_GPU_DEVICE": os.getenv("CLIP_GPU_DEVICE", "0"),
            "T5_GPU_DEVICE": os.getenv("T5_GPU_DEVICE", "0"),
            "DIFFUSER_GPU_DEVICE": os.getenv("DIFFUSER_GPU_DEVICE", "0"),
            "VAE_GPU_DEVICE": os.getenv("VAE_GPU_DEVICE", "0")
        },
        "assigned_devices": {
            "clip_text_encoder": f"cuda:{os.getenv('CLIP_GPU_DEVICE', '0')}",
            "t5_text_encoder": f"cuda:{os.getenv('T5_GPU_DEVICE', '0')}",
            "transformer": f"cuda:{os.getenv('DIFFUSER_GPU_DEVICE', '0')}",
            "vae_decoder": f"cuda:{os.getenv('VAE_GPU_DEVICE', '0')}"
        },
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    # Add memory info for each GPU if CUDA is available
    if torch.cuda.is_available():
        gpu_status["gpu_memory"] = {}
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_device(i)
                total_memory = torch.cuda.get_device_properties(i).total_memory
                allocated_memory = torch.cuda.memory_allocated(i)
                free_memory = total_memory - allocated_memory
                
                gpu_status["gpu_memory"][f"cuda:{i}"] = {
                    "total_gb": round(total_memory / 1024**3, 2),
                    "allocated_gb": round(allocated_memory / 1024**3, 2),
                    "free_gb": round(free_memory / 1024**3, 2),
                    "device_name": torch.cuda.get_device_name(i)
                }
            except Exception as e:
                gpu_status["gpu_memory"][f"cuda:{i}"] = {"error": str(e)}
    
    return gpu_status

@router.get("/v1/models/generation/optimal")
def get_optimal_models(memory_efficient: bool = False):
    """Get optimal model recommendations based on detected runtime."""
    return {
        "optimal_models": model_config.get_optimal_models(prefer_memory_efficient=memory_efficient),
        "runtime_status": model_config.get_runtime_status()
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
    start_time = time.time()
    logger.info(f"🎨 Generation request received: model={req.model}, working_set={req.working_set}")
    original_model = req.model.lower()
    
    # Log detailed model resolution process
    logger.info(f"🔍 Resolving model ID: '{original_model}'")
    # Force print for debugging regardless of log level
    print(f"🔍 [DEBUG] Resolving model ID: '{original_model}'")
    
    # Try to resolve model using config system first
    resolved = model_config.resolve_model_id(original_model)
    if resolved:
        model, backend, quantization = resolved
        logger.info(f"✅ Resolved '{original_model}' to:")
        logger.info(f"   Model: {model}")
        logger.info(f"   Backend: {backend}")
        logger.info(f"   Quantization: {quantization}")
        # Force print for debugging
        print(f"✅ [DEBUG] Resolved '{original_model}' to: {model}/{backend}/{quantization}")
        
        # Look up corresponding router ID from the mapping
        for router_id, mapping in model_config.id_mapping.items():
            if mapping == [model, backend, quantization]:
                key = router_id
                logger.info(f"🔄 Found router mapping: '{original_model}' -> '{key}'")
                print(f"🔄 [DEBUG] Found router mapping: '{original_model}' -> '{key}'")
                break
        else:
            # No router mapping found, use resolved model name as key
            key = model
            logger.info(f"🔄 No router mapping found, using resolved model name: '{key}'")
    else:
        key = original_model
        logger.info(f"❌ Could not resolve '{original_model}' in model config, using as-is: '{key}'")
    
    # Try new diffusion system first
    try:
        return _generate_with_diffusion_system(req, key, start_time)
    except Exception as e:
        # Fall back to legacy system
        pass
    
    # Router system
    logger.info(f"🔧 Attempting to load model using router system with key: '{key}'")
    getter, adapter = PIPE_REGISTRY.get(key, (None, None))
    if not getter:
        available_models = list(PIPE_REGISTRY.keys()) + list(diffusion_loader.get_available_models().keys())
        config_models = list(model_config.get_legacy_model_metadata().keys())
        all_available = sorted(set(available_models + config_models))
        logger.error(f"❌ Model '{key}' not found in PIPE_REGISTRY")
        logger.info(f"   Available in PIPE_REGISTRY: {list(PIPE_REGISTRY.keys())}")
        logger.info(f"   Available in diffusion_loader: {list(diffusion_loader.get_available_models().keys())}")
        raise HTTPException(400, f"model must be one of: {', '.join(all_available)}")
    
    logger.info(f"✅ Found model loader for '{key}': {getter.__name__}")
    logger.info(f"   Adapter function: {adapter.__name__}")

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

    # Load the pipeline (this will trigger our detailed logging)
    logger.info(f"📥 Calling model loader: {getter.__name__}()")
    print(f"📥 [DEBUG] Calling model loader: {getter.__name__}()")
    pipe = getter()
    logger.info(f"✅ Pipeline loaded successfully")
    print(f"✅ [DEBUG] Pipeline loaded successfully")
    
    # Generate images
    logger.info(f"🎨 Starting generation with adapter: {adapter.__name__}")
    logger.info(f"   Prompt: '{req.prompt[:50]}{'...' if len(req.prompt) > 50 else ''}'")
    logger.info(f"   Resolution: {width}x{height}")
    logger.info(f"   Count: {req.n}")
    
    pil_imgs = adapter(
        pipe,
        req.prompt,
        width=width,
        height=height,
        negative_prompt=req.negative_prompt,
        n=req.n
    )

    generation_time = time.time() - start_time
    return ImageGenResponse(
        created=int(time.time()),
        data=[ImageData(b64_json=_pil_to_b64(img)) for img in pil_imgs],
        generation_time_seconds=generation_time
    )

def _generate_with_diffusion_system(req: ImageGenRequest, model_key: str, start_time: float) -> ImageGenResponse:
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
    
    generation_time = time.time() - start_time
    return ImageGenResponse(
        created=int(time.time()),
        data=[ImageData(b64_json=_pil_to_b64(img)) for img in pil_imgs[:req.n or 1]],
        generation_time_seconds=generation_time
    )