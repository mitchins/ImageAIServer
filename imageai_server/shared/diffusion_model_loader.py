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

try:
    import onnxruntime as ort
    from optimum.onnxruntime import ORTStableDiffusionPipeline
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    ORTStableDiffusionPipeline = None
    ONNX_AVAILABLE = False

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
        providers = []
        if torch.cuda.is_available():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers
    
    def _validate_environment(self, working_set: WorkingSet) -> Tuple[bool, str]:
        """Validate if the current environment can run the working set."""
        # Check if backend is available
        backend = next(iter({comp.backend for comp in working_set.components.values()}))
        
        if backend == Backend.ONNX and not ONNX_AVAILABLE:
            return False, "ONNX runtime not available"
        
        if backend == Backend.PYTORCH and not DIFFUSERS_AVAILABLE:
            return False, "Diffusers library not available"
        
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
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX runtime not available")
        
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
    
    def unload_pipeline(self, model_id: str, working_set_name: Optional[str] = None):
        """Unload a pipeline to free memory."""
        cache_key = f"{model_id}:{working_set_name or 'default'}"
        
        if cache_key in self._loaded_pipelines:
            pipeline, _ = self._loaded_pipelines[cache_key]
            
            # Move to CPU to free GPU memory
            if hasattr(pipeline, 'to'):
                pipeline.to('cpu')
            
            del self._loaded_pipelines[cache_key]
            logger.info(f"Unloaded pipeline {cache_key}")
            
            # Force garbage collection
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