"""
Diffusion Model Loader - Handles loading and caching of diffusion model components.

This system provides:
- Component-aware loading with validation
- Intelligent caching and memory management
- Backend abstraction (PyTorch vs ONNX)
- Working set validation and fallbacks
"""

import logging
from typing import Dict, Optional, Any, Tuple, Union
from pathlib import Path
import torch
from .diffusion_model_registry import (
    diffusion_registry, ModelDefinition, WorkingSet, Component, Backend, Quantization
)

# Lazy import ONNX dependencies only when needed
ort = None
ORTStableDiffusionPipeline = None
ONNX_AVAILABLE = False

def _ensure_onnx_imports():
    """Lazy import ONNX dependencies only when ONNX backend is used."""
    global ort, ORTStableDiffusionPipeline, ONNX_AVAILABLE
    
    if ONNX_AVAILABLE:
        return  # Already imported
    
    try:
        import onnxruntime as ort_module
        from optimum.onnxruntime import ORTStableDiffusionPipeline as ORT_Pipeline
        
        ort = ort_module
        ORTStableDiffusionPipeline = ORT_Pipeline
        ONNX_AVAILABLE = True
    except ImportError as e:
        raise RuntimeError(f"ONNX Runtime not available: {e}")

try:
    from .tensorrt_rtx_backend import TensorRTRTXDiffusionLoader, TensorRTRTXConfig
    TENSORRT_RTX_AVAILABLE = True
except ImportError:
    TensorRTRTXDiffusionLoader = None
    TensorRTRTXConfig = None
    TENSORRT_RTX_AVAILABLE = False

try:
    from diffusers import (
        StableDiffusionPipeline, StableDiffusionXLPipeline,
        AutoPipelineForText2Image,
        DiffusionPipeline
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class DiffusionModelLoader:
    """Loads and manages diffusion model components with intelligent caching."""
    
    def __init__(self):
        self._loaded_pipelines: Dict[str, Any] = {}
        self._component_cache: Dict[str, Any] = {}
        self._registry = diffusion_registry
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get all available models with their metadata for API consumption."""
        result = {}
        
        for model_id, model_def in self._registry.get_available_models().items():
            # Get the best working set for this environment
            optimal_ws = self._registry.suggest_optimal_working_set(
                model_id, 
                prefer_gpu=torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()),
                prefer_quality=True
            )
            
            if optimal_ws:
                result[model_id] = {
                    "display_name": model_def.display_name,
                    "engine": optimal_ws.components[Component.UNET].backend.value,
                    "supports_negative_prompt": model_def.supports_negative_prompt,
                    "max_resolution": model_def.max_resolution,
                    "default_resolution": model_def.default_resolution,
                    "min_resolution": model_def.min_resolution,
                    "working_set": optimal_ws.name,
                    "description": optimal_ws.description,
                    "architecture": model_def.architecture
                }
        
        return result
    
    def _choose_device(self) -> str:
        """Choose the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _choose_onnx_providers(self) -> list:
        """Choose ONNX execution providers."""
        from .provider_utils import get_standard_providers
        return get_standard_providers(use_tensorrt=False)
    
    def _validate_environment(self, working_set: WorkingSet) -> Tuple[bool, str]:
        """Validate if the current environment can run the working set."""
        # Check if backend is available
        backend = next(iter({comp.backend for comp in working_set.components.values()}))
        
        if backend == Backend.ONNX:
            try:
                _ensure_onnx_imports()
            except RuntimeError as e:
                return False, str(e)
        
        if backend == Backend.PYTORCH and not DIFFUSERS_AVAILABLE:
            return False, "Diffusers library not available"
        
        if backend == Backend.TENSORRT_RTX:
            if not TENSORRT_RTX_AVAILABLE:
                return False, "TensorRT-RTX backend not available"
            
            # Create a test loader to check availability
            try:
                config = TensorRTRTXConfig(device=self._choose_device())
                test_loader = TensorRTRTXDiffusionLoader(config)
                if not test_loader.is_available():
                    return False, "TensorRT-RTX environment requirements not met"
            except Exception as e:
                return False, f"TensorRT-RTX backend initialization failed: {e}"
        
        # Check GPU requirements
        requires_gpu = working_set.constraints.get("requires_gpu", False)
        has_gpu = torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
        
        if requires_gpu and not has_gpu:
            return False, "GPU required but not available"
        
        return True, "OK"
    
    def load_pipeline(self, model_id: str, working_set_name: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        """Load a complete diffusion pipeline."""
        cache_key = f"{model_id}:{working_set_name or 'default'}"
        
        # Return cached pipeline if available
        if cache_key in self._loaded_pipelines:
            logger.info(f"Using cached pipeline for {cache_key}")
            return self._loaded_pipelines[cache_key]
        
        # Get model definition
        model_def = self._registry.get_model(model_id)
        if not model_def:
            raise ValueError(f"Unknown model: {model_id}")
        
        # Get working set
        if working_set_name:
            working_set = self._registry.get_working_set(model_id, working_set_name)
        else:
            working_set = self._registry.suggest_optimal_working_set(model_id)
        
        if not working_set:
            raise ValueError(f"No suitable working set found for {model_id}")
        
        # Validate environment
        can_run, reason = self._validate_environment(working_set)
        if not can_run:
            raise RuntimeError(f"Cannot run {model_id} with {working_set.name}: {reason}")
        
        # Validate working set
        is_valid, issues = self._registry.validate_working_set(working_set)
        if not is_valid:
            raise ValueError(f"Invalid working set {working_set.name}: {'; '.join(issues)}")
        
        logger.info(f"Loading pipeline {model_id} with working set {working_set.name}")
        
        # Load based on backend
        backend = next(iter({comp.backend for comp in working_set.components.values()}))
        
        if backend == Backend.PYTORCH:
            pipeline = self._load_pytorch_pipeline(model_def, working_set)
        elif backend == Backend.ONNX:
            pipeline = self._load_onnx_pipeline(model_def, working_set)
        elif backend == Backend.TENSORRT_RTX:
            pipeline = self._load_tensorrt_rtx_pipeline(model_def, working_set)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        
        # Cache the pipeline
        metadata = {
            "model_id": model_id,
            "working_set": working_set.name,
            "backend": backend.value,
            "settings": working_set.optimal_settings,
            "constraints": working_set.constraints
        }
        
        self._loaded_pipelines[cache_key] = (pipeline, metadata)
        logger.info(f"Successfully loaded and cached pipeline {cache_key}")
        
        return pipeline, metadata
    
    def _load_pytorch_pipeline(self, model_def: ModelDefinition, working_set: WorkingSet) -> Any:
        """Load a PyTorch-based pipeline."""
        device = self._choose_device()
        torch_dtype = torch.float16 if Quantization.FP16 in {comp.quantization for comp in working_set.components.values()} else torch.float32
        
        # Use the base repo for standard pipelines
        base_repo = model_def.base_repo
        
        # Select appropriate pipeline class based on architecture
        if model_def.architecture == "sd15":
            pipeline_class = StableDiffusionPipeline
        elif model_def.architecture == "sdxl":
            pipeline_class = StableDiffusionXLPipeline
        else:
            # Fallback to auto pipeline
            pipeline_class = AutoPipelineForText2Image
        
        # Load pipeline
        pipeline = pipeline_class.from_pretrained(
            base_repo,
            torch_dtype=torch_dtype,
            variant="fp16" if torch_dtype == torch.float16 else None,
            use_safetensors=True,
            safety_checker=None,  # Disable safety checker for speed
            requires_safety_checker=False
        )
        
        # Move to device
        if device != "cpu":
            pipeline = pipeline.to(device)
            
        # Apply optimizations
        if hasattr(pipeline, 'enable_attention_slicing'):
            pipeline.enable_attention_slicing()
        
        if hasattr(pipeline, 'enable_model_cpu_offload') and device != "cpu":
            pipeline.enable_model_cpu_offload()
        
        return pipeline
    
    def _load_onnx_pipeline(self, model_def: ModelDefinition, working_set: WorkingSet) -> Any:
        """Load an ONNX-based pipeline."""
        _ensure_onnx_imports()  # Lazy import ONNX dependencies
        
        providers = self._choose_onnx_providers()
        
        # Get the base repo from the UNet component (assumes all components from same repo)
        unet_spec = working_set.components[Component.UNET]
        base_repo = unet_spec.repo_id
        
        # Load ONNX pipeline using Optimum
        pipeline = ORTStableDiffusionPipeline.from_pretrained(
            base_repo,
            provider=providers[0] if providers else "CPUExecutionProvider"  # Use first provider
        )
        
        return pipeline
    
    def _load_tensorrt_rtx_pipeline(self, model_def: ModelDefinition, working_set: WorkingSet) -> Any:
        """Load a TensorRT-RTX-based pipeline."""
        if not TENSORRT_RTX_AVAILABLE:
            raise RuntimeError("TensorRT-RTX backend not available")
        
        # Ensure clean TensorRT-RTX environment (no ONNX Runtime interference)
        import os
        old_providers = os.environ.get("ORT_PROVIDERS", None)
        os.environ["ORT_PROVIDERS"] = "CPUExecutionProvider"  # Force ONNX Runtime to CPU only
        
        try:
            device = self._choose_device()
            
            # Create TensorRT-RTX configuration
            config = TensorRTRTXConfig(
                device=device,
                cache_dir="./tensorrt_cache",  # Use same cache dir as setup script
                verbose=True,  # Can be made configurable
                enable_runtime_cache=True
            )
            
            # Create loader and load the pipeline
            loader = TensorRTRTXDiffusionLoader(config)
            pipeline, metadata = loader.load_model_pipeline(model_def.model_id, working_set)
            
            # Store the loader for later use
            pipeline._tensorrt_loader = loader
            
            return pipeline
            
        finally:
            # Restore original provider settings
            if old_providers:
                os.environ["ORT_PROVIDERS"] = old_providers
            elif "ORT_PROVIDERS" in os.environ:
                del os.environ["ORT_PROVIDERS"]
    
    def unload_pipeline(self, model_id: str, working_set_name: Optional[str] = None):
        """Unload a pipeline to free memory."""
        cache_key = f"{model_id}:{working_set_name or 'default'}"
        
        if cache_key in self._loaded_pipelines:
            pipeline, _ = self._loaded_pipelines[cache_key]
            
            # Handle TensorRT-RTX pipeline cleanup
            if hasattr(pipeline, '_tensorrt_loader'):
                logger.info(f"Cleaning up TensorRT-RTX pipeline {cache_key}")
                pipeline._tensorrt_loader.cleanup()
            elif hasattr(pipeline, 'to'):
                # Move PyTorch/ONNX pipeline to CPU to free GPU memory
                pipeline.to('cpu')
            
            del self._loaded_pipelines[cache_key]
            logger.info(f"Unloaded pipeline {cache_key}")
            
            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_loaded_pipelines(self) -> Dict[str, Dict[str, Any]]:
        """Get information about currently loaded pipelines."""
        result = {}
        for cache_key, (pipeline, metadata) in self._loaded_pipelines.items():
            result[cache_key] = {
                "metadata": metadata,
                "memory_usage": self._estimate_memory_usage(pipeline)
            }
        return result
    
    def _estimate_memory_usage(self, pipeline) -> str:
        """Estimate memory usage of a pipeline."""
        try:
            if hasattr(pipeline, 'hf_device_map'):
                return "GPU (device mapped)"
            elif hasattr(pipeline, 'device') and 'cuda' in str(pipeline.device):
                return "GPU"
            else:
                return "CPU"
        except:
            return "Unknown"
    
    def clear_cache(self):
        """Clear all cached pipelines."""
        for cache_key in list(self._loaded_pipelines.keys()):
            self.unload_pipeline(*cache_key.split(':', 1))
        
        logger.info("Cleared all cached pipelines")

# Global loader instance
diffusion_loader = DiffusionModelLoader()