"""
TensorRT-RTX Backend for Diffusion Models

This backend wraps the NVIDIA TensorRT-RTX pipeline demos to provide
high-performance inference for diffusion models like SDXL and Flux.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass

import torch
import numpy as np
from PIL import Image

from .model_backend import ModelBackend, BackendConfig
from .diffusion_model_registry import Backend, Quantization, WorkingSet
from .engine_manager import get_engine_manager, EngineManager

def _setup_tensorrt_rtx_environment():
    """Setup TensorRT-RTX environment variables."""
    trt_rtx_lib = "/data/nvidia/TensorRT-RTX-1.0.0.21/targets/x86_64-linux-gnu/lib"
    if os.path.exists(trt_rtx_lib):
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if trt_rtx_lib not in current_ld_path:
            os.environ["LD_LIBRARY_PATH"] = f"{trt_rtx_lib}:{current_ld_path}"
    os.environ["POLYGRAPHY_USE_TENSORRT_RTX"] = "1"

logger = logging.getLogger(__name__)

@dataclass
class TensorRTRTXConfig(BackendConfig):
    """Configuration for TensorRT-RTX backend."""
    backend_type: str = "tensorrt_rtx"
    cache_dir: str = "./tensorrt_cache"
    hf_token: Optional[str] = None
    low_vram: bool = False
    verbose: bool = True
    enable_runtime_cache: bool = True
    
    # Engine management
    hf_repo: str = "your-org/tensorrt-engines"
    auto_download_engines: bool = True
    preferred_quantization: str = "bf16"


class TensorRTRTXBackend(ModelBackend):
    """TensorRT-RTX backend for diffusion models."""
    
    def __init__(self, config: TensorRTRTXConfig):
        self.config = config
        self._pipeline = None
        self._current_model = None
        self._current_working_set = None
        
        # Setup TensorRT-RTX environment first
        _setup_tensorrt_rtx_environment()
        
        # Initialize engine manager
        self.engine_manager = get_engine_manager(
            cache_dir=f"{config.cache_dir}/engines",
            hf_repo=config.hf_repo,
            hf_token=config.hf_token
        )
        
        # Disable ONNX Runtime to prevent conflicts with native TensorRT
        os.environ["OMP_NUM_THREADS"] = "1"  # Prevent threading conflicts
    
    def is_available(self) -> bool:
        """Check if TensorRT-RTX backend is available."""
        try:
            # Check if CUDA is available
            if not torch.cuda.is_available():
                logger.warning("CUDA not available")
                return False
                
            # Check if TensorRT-RTX is available (for .plan engine loading)
            import tensorrt_rtx as trt
            # Handle TensorRT-RTX API
            try:
                version = getattr(trt, '__version__', 'unknown')
                logger.debug(f"TensorRT-RTX version: {version}")
            except:
                logger.debug("TensorRT-RTX version detection failed, but TensorRT-RTX is available")
            
            # Check compute capability for TensorRT engines
            compute_capability = torch.cuda.get_device_capability()
            if compute_capability < (8, 0):  # Minimum Ampere for our engines
                logger.warning(f"TensorRT-RTX requires compute capability >= 8.0, found {compute_capability}")
                return False
                
            # Check if GPU architecture is supported
            arch = self._get_gpu_architecture()
            if not arch:
                logger.warning("GPU architecture not supported for TensorRT-RTX engines")
                return False
            
            # Check if we can get engines for this architecture
            try:
                # Try to get HuggingFace repo name (this validates architecture)
                repo_name = self._get_hf_repo_name("sdxl", "bf16")
                logger.debug(f"TensorRT-RTX engines available from: {repo_name}")
                return True
            except Exception as e:
                logger.warning(f"TensorRT-RTX engines not available: {e}")
                
            # Fallback: check if our demo components are available
            try:
                from ..tensorrt.nvidia_sdxl_pipeline import NVIDIASDXLPipeline
                logger.debug("NVIDIA TensorRT-RTX SDXL pipeline available as fallback")
                return True
            except ImportError:
                logger.warning("NVIDIA TensorRT-RTX demo components not available")
                return False
            
        except ImportError as e:
            logger.warning(f"TensorRT dependencies not available: {e}")
            return False
        except Exception as e:
            logger.warning(f"Error checking TensorRT-RTX availability: {e}")
            return False
    
    def _get_gpu_architecture(self) -> str:
        """Get GPU architecture string with blackwell support and future compatibility."""
        major, minor = torch.cuda.get_device_capability()
        
        if major == 8:
            if minor >= 9:  # Ada Lovelace
                return "ada"
            else:  # Ampere
                return "ampere"
        elif major >= 9:  # Blackwell and future
            return "blackwell"
        else:  # Turing and older - not supported
            return None  # Will cause engines to be unavailable
    
    def _get_supported_quantizations(self) -> List[str]:
        """Get supported quantizations for current GPU architecture."""
        arch = self._get_gpu_architecture()
        if arch == "ampere":
            return ["bf16"]
        elif arch == "ada":
            return ["bf16", "fp8"]
        elif arch == "blackwell":
            return ["bf16", "fp8", "fp4"]
        else:
            return []  # Turing and older not supported
    
    def _get_hf_repo_name(self, model_id: str, quantization: str = "bf16") -> str:
        """Get HuggingFace repository name for model and current architecture."""
        arch = self._get_gpu_architecture()
        if not arch:
            raise RuntimeError("GPU architecture not supported for TensorRT-RTX engines")
        return f"imgailab/{model_id}-{quantization}-{arch}"
    
    def _download_engine_repo(self, model_id: str, quantization: str = "bf16") -> Path:
        """Download and cache TensorRT-RTX engines from HuggingFace."""
        repo_name = self._get_hf_repo_name(model_id, quantization)
        cache_dir = Path(self.config.cache_dir) / "hf_engines" / repo_name.replace("/", "_")
        
        # Check if already cached
        if cache_dir.exists() and list(cache_dir.glob("*.plan")):
            logger.debug(f"Using cached engines: {cache_dir}")
            return cache_dir
        
        logger.info(f"Downloading TensorRT-RTX engines from {repo_name}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=repo_name,
                local_dir=cache_dir,
                token=self.config.hf_token or os.environ.get("HF_TOKEN"),
                local_dir_use_symlinks=False  # Copy files instead of symlinks
            )
            
            # Verify engines were downloaded
            engine_files = list(cache_dir.glob("*.plan"))
            if not engine_files:
                raise RuntimeError(f"No engine files found in {repo_name}")
            
            logger.info(f"Downloaded {len(engine_files)} engine files to {cache_dir}")
            return cache_dir
            
        except Exception as e:
            logger.error(f"Failed to download engines from {repo_name}: {e}")
            # Clean up failed download
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir)
            raise RuntimeError(f"Engine download failed: {e}")
    
    def get_supported_models(self) -> List[str]:
        """Return list of model identifiers supported by this backend."""
        return ["sdxl", "flux1-dev", "flux1-schnell"]
    
    def supports_quantization(self, quant_type: str) -> bool:
        """Check if backend supports specific quantization type."""
        supported_quants = {"bf16", "fp8", "fp4"}
        return quant_type.lower() in supported_quants
    
    def _get_pipeline_class(self, model_id: str, architecture: str):
        """Get the appropriate pipeline class for the model architecture."""
        if architecture == "sdxl":
            try:
                # We'll need to create an SDXL wrapper based on the existing demo
                # For now, let's assume we have access to the TensorRT demo
                from nvidia.demo.Diffusion.demo_txt2img_xl import StableDiffusionXLPipeline
                return StableDiffusionXLPipeline
            except ImportError:
                raise RuntimeError("SDXL TensorRT-RTX pipeline not available")
        
        elif architecture == "flux":
            try:
                from flux1.dev.pipelines.flux_pipeline import FluxPipeline
                return FluxPipeline
            except ImportError:
                raise RuntimeError("Flux TensorRT-RTX pipeline not available")
        
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
    
    def load_pipeline(self, model_id: str, working_set: WorkingSet) -> Tuple[Any, Dict[str, Any]]:
        """Load a TensorRT-RTX pipeline for the specified model and working set."""
        logger.info(f"Loading TensorRT-RTX pipeline for {model_id} with working set {working_set.name}")
        
        # Clean up existing pipeline if different model
        if self._pipeline and (self._current_model != model_id or self._current_working_set != working_set.name):
            logger.info("Cleaning up previous pipeline")
            self._cleanup_pipeline()
        
        if self._pipeline:
            logger.info("Using existing pipeline")
            return self._pipeline, self._get_pipeline_metadata(working_set)
        
        # Create new pipeline based on architecture
        if model_id == "sdxl":
            pipeline = self._create_sdxl_pipeline(working_set)
        elif model_id.startswith("flux1"):
            pipeline = self._create_flux_pipeline(model_id, working_set)
        else:
            raise ValueError(f"Unsupported model: {model_id}")
        
        self._pipeline = pipeline
        self._current_model = model_id
        self._current_working_set = working_set.name
        
        return pipeline, self._get_pipeline_metadata(working_set)
    
    def _create_sdxl_pipeline(self, working_set: WorkingSet):
        """Create SDXL TensorRT pipeline using our HuggingFace engines."""
        from ..tensorrt.nvidia_sdxl_pipeline import NVIDIASDXLPipeline
        
        # Get quantization from working set
        from .diffusion_model_registry import Component
        quantization = working_set.components[Component.UNET].quantization.value
        
        # Validate quantization is supported for current architecture
        supported_quants = self._get_supported_quantizations()
        if quantization not in supported_quants:
            raise RuntimeError(f"Quantization {quantization} not supported on {self._get_gpu_architecture()}. Supported: {supported_quants}")
        
        engine_base_path = None
        
        logger.info(f"🔍 SDXL Engine path selection for {quantization} on {self._get_gpu_architecture()}:")
        
        # 1. Try HuggingFace engines (PRIMARY - our optimized engines)
        try:
            hf_engine_path = self._download_engine_repo("sdxl", quantization)
            if hf_engine_path.exists() and list(hf_engine_path.glob("*.plan")):
                engine_base_path = str(hf_engine_path)
                logger.info(f"✓ Using HuggingFace engines: {engine_base_path}")
        except Exception as e:
            logger.warning(f"HuggingFace engine download failed: {e}")
        
        # 2. Fallback to local development engines  
        if not engine_base_path:
            local_cache_dir = Path(f"{self.config.cache_dir}/engines")
            local_identifier = f"sdxl-{quantization}-{self._get_gpu_architecture()}"
            local_engine_path = local_cache_dir / local_identifier
            
            if local_engine_path.exists() and list(local_engine_path.glob("*.plan")):
                engine_base_path = str(local_engine_path)
                logger.info(f"✓ Using local engines: {engine_base_path}")
        
        # 3. Final fallback to NVIDIA demo engines (may not match quantization)
        if not engine_base_path:
            nvidia_engine_path = Path("/data/nvidia/sdxl_tensorrt_rtx/engines_xl_b1_1024")
            if nvidia_engine_path.exists() and list(nvidia_engine_path.glob("*.plan")):
                engine_base_path = str(nvidia_engine_path)
                logger.warning(f"⚠️  Using NVIDIA demo engines (may not match {quantization}): {engine_base_path}")
        
        if not engine_base_path:
            raise RuntimeError(f"No SDXL engines found for {quantization} on {self._get_gpu_architecture()}. Available repos: {self._get_hf_repo_name('sdxl', quantization)}")
        
        # Get settings from working set
        guidance_scale = working_set.optimal_settings.get("guidance_scale", 8.0)
        num_inference_steps = working_set.optimal_settings.get("num_inference_steps", 30)
        
        # Create pipeline using NVIDIA's working implementation
        pipeline = NVIDIASDXLPipeline(
            device=self.config.device,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            verbose=self.config.verbose,
            hf_token=self.config.hf_token
        )
        
        # Auto-load engines if available
        if engine_base_path:
            try:
                # Extract directories from engine path
                engine_path = Path(engine_base_path)
                
                # Use different framework paths for NVIDIA vs local engines
                if "nvidia" in str(engine_path):
                    # NVIDIA demo structure
                    framework_dir = str(engine_path.parent / "pytorch_model")
                    onnx_dir = str(engine_path.parent / "onnx")
                else:
                    # Local cache structure  
                    framework_dir = str(engine_path.parent / "framework")
                    onnx_dir = str(engine_path.parent / "onnx")
                
                pipeline.load_engines(
                    engine_dir=str(engine_path),
                    framework_model_dir=framework_dir,
                    onnx_dir=onnx_dir,
                    build_engines=False  # Don't build, use existing engines
                )
                
                # Activate engines
                pipeline.activate_engines()
                
                logger.info(f"✓ Auto-loaded NVIDIA engines successfully")
                
            except Exception as e:
                logger.warning(f"Failed to auto-load engines: {e}")
                logger.info("Pipeline created but engines need to be loaded manually")
        
        return pipeline
    
    def _create_flux_pipeline(self, model_id: str, working_set: WorkingSet):
        """Create Flux TensorRT-RTX pipeline using our HuggingFace engines."""
        
        # Get quantization from working set  
        from .diffusion_model_registry import Component
        quantization = working_set.components[Component.UNET].quantization.value
        
        # Validate quantization is supported for current architecture
        supported_quants = self._get_supported_quantizations()
        if quantization not in supported_quants:
            raise RuntimeError(f"Quantization {quantization} not supported on {self._get_gpu_architecture()}. Supported: {supported_quants}")
        
        logger.info(f"🔍 Flux Engine path selection for {model_id} {quantization} on {self._get_gpu_architecture()}:")
        
        # 1. Try HuggingFace engines (PRIMARY)
        engine_base_path = None
        try:
            hf_engine_path = self._download_engine_repo(model_id, quantization)
            if hf_engine_path.exists() and list(hf_engine_path.glob("*.plan")):
                engine_base_path = str(hf_engine_path)
                logger.info(f"✓ Using HuggingFace engines: {engine_base_path}")
        except Exception as e:
            logger.warning(f"HuggingFace engine download failed: {e}")
        
        # 2. Fallback to local development engines
        if not engine_base_path:
            local_cache_dir = Path(f"{self.config.cache_dir}/engines")
            local_identifier = f"{model_id}-{quantization}-{self._get_gpu_architecture()}"
            local_engine_path = local_cache_dir / local_identifier
            
            if local_engine_path.exists() and list(local_engine_path.glob("*.plan")):
                engine_base_path = str(local_engine_path)
                logger.info(f"✓ Using local engines: {engine_base_path}")
        
        if not engine_base_path:
            raise RuntimeError(f"No {model_id} engines found for {quantization} on {self._get_gpu_architecture()}. Available repos: {self._get_hf_repo_name(model_id, quantization)}")
        
        # Import the Flux pipeline from our TensorRT-RTX demo
        sys.path.append('/data/imagai/nvidia-demos/TensorRT-RTX/demo/flux1.dev')
        from pipelines.flux_pipeline import FluxPipeline
        
        # Create pipeline with configuration
        pipeline = FluxPipeline(
            cache_dir=str(engine_base_path),  # Point directly to engine directory
            device=self.config.device or "cuda",
            verbose=self.config.verbose,
            hf_token=self.config.hf_token or os.environ.get("HF_TOKEN"),
            low_vram=self.config.low_vram,
            guidance_scale=working_set.optimal_settings.get("guidance_scale", 3.5),
            num_inference_steps=working_set.optimal_settings.get("num_inference_steps", 50)
        )
        
        # Load pre-built engines
        logger.info(f"Loading pre-built {model_id} engines...")
        pipeline.load_engines(
            transformer_precision=quantization,
            opt_batch_size=1,
            opt_height=1024,
            opt_width=1024,
            shape_mode="static"
        )
        
        return pipeline
    
    def _get_pipeline_metadata(self, working_set: WorkingSet) -> Dict[str, Any]:
        """Get metadata for the loaded pipeline."""
        return {
            "backend": Backend.TENSORRT_RTX.value,
            "working_set": working_set.name,
            "settings": working_set.optimal_settings,
            "constraints": working_set.constraints,
            "quantization": list({comp.quantization.value for comp in working_set.components.values()}),
            "device": self.config.device
        }
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        batch_size: int = 1,
        **kwargs
    ) -> List[Image.Image]:
        """Generate images using the loaded TensorRT pipeline."""
        if not self._pipeline:
            raise RuntimeError("No pipeline loaded. Call load_pipeline first.")
        
        logger.info(f"Generating image with TensorRT: '{prompt}' at {width}x{height}")
        
        # Check if this is a Flux-style pipeline (TensorRT-RTX)
        if hasattr(self._pipeline, 'load_engines') and hasattr(self._pipeline, 'engines'):
            # Flux-style pipeline (TensorRT-RTX)
            precision = "bf16"  # Default, can be overridden
            if hasattr(self._pipeline, 'precision_config'):
                transformer_precision = list(self._pipeline.precision_config.values())[0] if self._pipeline.precision_config else precision
            else:
                transformer_precision = precision
                
            # Load engines if not already loaded
            if not self._pipeline.engines:
                logger.info("Loading TensorRT engines...")
                self._pipeline.load_engines(
                    transformer_precision=transformer_precision,
                    opt_batch_size=batch_size,
                    opt_height=height,
                    opt_width=width
                )
            
            # Load resources
            self._pipeline.load_resources(
                batch_size=batch_size,
                height=height,
                width=width
            )
            
            # Run inference
            images_np, _ = self._pipeline.infer(
                prompt=prompt,
                batch_size=batch_size,
                height=height,
                width=width,
                seed=seed,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            
            # Convert numpy arrays to PIL Images
            images = []
            for img_np in images_np:
                images.append(Image.fromarray(img_np))
            
            return images
        
        # Check if this is the NVIDIA SDXL pipeline
        elif hasattr(self._pipeline, 'load_engines'):
            # NVIDIA SDXL pipeline
            
            # Engines should already be loaded and activated during creation
            # Just run inference directly
            images, inference_time = self._pipeline.infer(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                batch_size=batch_size,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed
            )
            
            return images
        
        else:
            raise NotImplementedError(f"Pipeline type not supported: {type(self._pipeline)}")
    
    def _cleanup_pipeline(self):
        """Clean up the current pipeline."""
        if self._pipeline:
            if hasattr(self._pipeline, 'cleanup'):
                self._pipeline.cleanup()
            del self._pipeline
            self._pipeline = None
            self._current_model = None
            self._current_working_set = None
            
            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def __del__(self):
        """Cleanup on destruction."""
        self._cleanup_pipeline()


class TensorRTRTXDiffusionLoader:
    """Loader specifically for TensorRT-RTX diffusion models."""
    
    def __init__(self, config: TensorRTRTXConfig):
        self.config = config
        self.backend = TensorRTRTXBackend(config)
        self._loaded_pipelines: Dict[str, Any] = {}
    
    def is_available(self) -> bool:
        """Check if TensorRT-RTX is available."""
        return self.backend.is_available()
    
    def load_model_pipeline(self, model_id: str, working_set: WorkingSet):
        """Load a model pipeline with TensorRT-RTX backend."""
        cache_key = f"{model_id}:{working_set.name}"
        
        if cache_key in self._loaded_pipelines:
            return self._loaded_pipelines[cache_key]
        
        pipeline, metadata = self.backend.load_pipeline(model_id, working_set)
        self._loaded_pipelines[cache_key] = (pipeline, metadata)
        
        return pipeline, metadata
    
    def generate_image(self, pipeline, **kwargs) -> List[Image.Image]:
        """Generate images using the TensorRT-RTX backend."""
        return self.backend.generate_image(**kwargs)
    
    def cleanup(self):
        """Clean up all loaded pipelines."""
        for cache_key in list(self._loaded_pipelines.keys()):
            del self._loaded_pipelines[cache_key]
        self._loaded_pipelines.clear()
        self.backend._cleanup_pipeline()