"""
Diffusion Model Registry - Manages complex diffusion model configurations and component compatibility.

This system handles:
- Multi-component models (UNet, VAE, Text Encoder, Scheduler)
- Quantization compatibility (FP16, INT8, FP32)
- Backend differences (PyTorch vs ONNX)
- Working sets of compatible components
- Configuration validation and suggestions
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class Backend(Enum):
    PYTORCH = "pytorch"
    ONNX = "onnx"

class Quantization(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    BF16 = "bf16"

class Component(Enum):
    UNET = "unet"
    VAE = "vae"
    TEXT_ENCODER = "text_encoder"
    SCHEDULER = "scheduler"
    TOKENIZER = "tokenizer"

@dataclass
class ComponentSpec:
    """Specification for a single model component."""
    repo_id: str
    filename: Optional[str] = None
    subfolder: Optional[str] = None
    quantization: Quantization = Quantization.FP16
    backend: Backend = Backend.PYTORCH
    required: bool = True

@dataclass
class WorkingSet:
    """A validated combination of components that work together."""
    name: str
    components: Dict[Component, ComponentSpec]
    optimal_settings: Dict[str, Union[int, float, str]] = field(default_factory=dict)
    constraints: Dict[str, Union[int, bool]] = field(default_factory=dict)
    description: str = ""

@dataclass
class ModelDefinition:
    """Complete definition of a diffusion model with all variants and working sets."""
    model_id: str
    display_name: str
    base_repo: str
    architecture: str  # "sd15", "sdxl", "flux", etc.
    working_sets: List[WorkingSet]
    default_working_set: str
    supports_negative_prompt: bool = True
    max_resolution: int = 1024
    default_resolution: int = 512
    min_resolution: int = 256

class DiffusionModelRegistry:
    """Central registry for managing diffusion model configurations."""
    
    def __init__(self):
        self.models: Dict[str, ModelDefinition] = {}
        self._load_builtin_models()
    
    def _load_builtin_models(self):
        """Load built-in model definitions."""
        
        # Stable Diffusion 1.5 (Multiple backends and quantizations)
        sd15_pytorch_fp16 = WorkingSet(
            name="pytorch_fp16",
            components={
                Component.UNET: ComponentSpec("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="unet", quantization=Quantization.FP16),
                Component.VAE: ComponentSpec("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="vae", quantization=Quantization.FP16),
                Component.TEXT_ENCODER: ComponentSpec("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="text_encoder", quantization=Quantization.FP16),
                Component.SCHEDULER: ComponentSpec("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="scheduler"),
                Component.TOKENIZER: ComponentSpec("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="tokenizer"),
            },
            optimal_settings={
                "num_inference_steps": 20,
                "guidance_scale": 7.5,
                "scheduler": "DPMSolverMultistepScheduler"
            },
            constraints={
                "max_resolution": 768,
                "default_resolution": 512,
                "requires_gpu": True
            },
            description="Standard PyTorch FP16 - Good quality, GPU required"
        )
        
        sd15_onnx_fp16 = WorkingSet(
            name="onnx_fp16",
            components={
                Component.UNET: ComponentSpec("Mitchins/sd15-onnx-fp16", filename="unet/model.onnx", quantization=Quantization.FP16, backend=Backend.ONNX),
                Component.VAE: ComponentSpec("Mitchins/sd15-onnx-fp16", filename="vae_decoder/model.onnx", quantization=Quantization.FP16, backend=Backend.ONNX),
                Component.TEXT_ENCODER: ComponentSpec("Mitchins/sd15-onnx-fp16", filename="text_encoder/model.onnx", quantization=Quantization.FP16, backend=Backend.ONNX),
                Component.SCHEDULER: ComponentSpec("Mitchins/sd15-onnx-fp16", subfolder="scheduler", backend=Backend.ONNX),
                Component.TOKENIZER: ComponentSpec("Mitchins/sd15-onnx-fp16", subfolder="tokenizer", backend=Backend.ONNX),
            },
            optimal_settings={
                "num_inference_steps": 20,
                "guidance_scale": 7.5,
                "provider": "CPUExecutionProvider"
            },
            constraints={
                "max_resolution": 768,
                "default_resolution": 512,
                "requires_gpu": False
            },
            description="ONNX FP16 CPU - Best quality, Raspberry Pi compatible"
        )
        
        sd15_onnx_int8 = WorkingSet(
            name="onnx_int8",
            components={
                Component.UNET: ComponentSpec("Mitchins/sd15-onnx-int8", filename="unet/model.onnx", quantization=Quantization.INT8, backend=Backend.ONNX),
                Component.VAE: ComponentSpec("Mitchins/sd15-onnx-int8", filename="vae_decoder/model.onnx", quantization=Quantization.INT8, backend=Backend.ONNX),
                Component.TEXT_ENCODER: ComponentSpec("Mitchins/sd15-onnx-int8", filename="text_encoder/model.onnx", quantization=Quantization.INT8, backend=Backend.ONNX),
                Component.SCHEDULER: ComponentSpec("Mitchins/sd15-onnx-int8", subfolder="scheduler", backend=Backend.ONNX),
                Component.TOKENIZER: ComponentSpec("Mitchins/sd15-onnx-int8", subfolder="tokenizer", backend=Backend.ONNX),
            },
            optimal_settings={
                "num_inference_steps": 25,  # Slightly more steps for INT8
                "guidance_scale": 8.0,      # Higher guidance for INT8
                "provider": "CPUExecutionProvider"
            },
            constraints={
                "max_resolution": 768,
                "default_resolution": 512,
                "requires_gpu": False
            },
            description="ONNX INT8 CPU - Fastest inference, optimized for speed"
        )
        
        self.models["sd15"] = ModelDefinition(
            model_id="sd15",
            display_name="Stable Diffusion 1.5",
            base_repo="stable-diffusion-v1-5/stable-diffusion-v1-5",
            architecture="sd15",
            working_sets=[sd15_pytorch_fp16, sd15_onnx_fp16, sd15_onnx_int8],
            default_working_set="onnx_fp16",  # ONNX FP16 as default for best quality/performance balance
            supports_negative_prompt=True,
            max_resolution=768,
            default_resolution=512,
            min_resolution=256
        )
        
        # SDXL (PyTorch only for now)
        sdxl_pytorch_fp16 = WorkingSet(
            name="pytorch_fp16",
            components={
                Component.UNET: ComponentSpec("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet", quantization=Quantization.FP16),
                Component.VAE: ComponentSpec("stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae", quantization=Quantization.FP16),
                Component.TEXT_ENCODER: ComponentSpec("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder", quantization=Quantization.FP16),
            },
            optimal_settings={
                "num_inference_steps": 30,
                "guidance_scale": 8.0
            },
            constraints={
                "max_resolution": 1536,
                "default_resolution": 1024,
                "requires_gpu": True
            },
            description="SDXL PyTorch FP16 - High quality, GPU required"
        )
        
        self.models["sdxl"] = ModelDefinition(
            model_id="sdxl",
            display_name="Stable Diffusion XL",
            base_repo="stabilityai/stable-diffusion-xl-base-1.0",
            architecture="sdxl",
            working_sets=[sdxl_pytorch_fp16],
            default_working_set="pytorch_fp16",
            supports_negative_prompt=True,
            max_resolution=1536,
            default_resolution=1024,
            min_resolution=512
        )
    
    def get_model(self, model_id: str) -> Optional[ModelDefinition]:
        """Get a model definition by ID."""
        return self.models.get(model_id)
    
    def get_available_models(self) -> Dict[str, ModelDefinition]:
        """Get all available model definitions."""
        return self.models.copy()
    
    def get_working_sets(self, model_id: str) -> List[WorkingSet]:
        """Get all working sets for a model."""
        model = self.get_model(model_id)
        return model.working_sets if model else []
    
    def get_working_set(self, model_id: str, working_set_name: str) -> Optional[WorkingSet]:
        """Get a specific working set for a model."""
        model = self.get_model(model_id)
        if not model:
            return None
        
        for ws in model.working_sets:
            if ws.name == working_set_name:
                return ws
        return None
    
    def validate_working_set(self, working_set: WorkingSet) -> Tuple[bool, List[str]]:
        """Validate if a working set is internally consistent."""
        issues = []
        
        # Check quantization compatibility
        quantizations = {comp.quantization for comp in working_set.components.values()}
        backends = {comp.backend for comp in working_set.components.values()}
        
        # Mixed backends are generally not allowed
        if len(backends) > 1:
            issues.append(f"Mixed backends not supported: {list(backends)}")
        
        # Some quantization combinations are problematic
        if Quantization.INT8 in quantizations and Quantization.FP32 in quantizations:
            issues.append("INT8 and FP32 quantizations should not be mixed")
        
        # Check required components
        required_components = {comp for comp, spec in working_set.components.items() if spec.required}
        if Component.UNET not in required_components:
            issues.append("UNet component is required")
        
        return len(issues) == 0, issues
    
    def suggest_optimal_working_set(self, model_id: str, prefer_gpu: bool = True, prefer_quality: bool = True) -> Optional[WorkingSet]:
        """Suggest the best working set based on preferences."""
        model = self.get_model(model_id)
        if not model:
            return None
        
        # Score working sets based on preferences
        scored_sets = []
        for ws in model.working_sets:
            score = 0
            
            # GPU preference
            requires_gpu = ws.constraints.get("requires_gpu", False)
            if prefer_gpu and requires_gpu:
                score += 10
            elif not prefer_gpu and not requires_gpu:
                score += 10
            
            # Quality preference (FP16 > INT8 > FP32 for diffusion)
            quantizations = {comp.quantization for comp in ws.components.values()}
            if prefer_quality:
                if Quantization.FP16 in quantizations:
                    score += 8
                elif Quantization.FP32 in quantizations:
                    score += 6
                elif Quantization.INT8 in quantizations:
                    score += 4
            else:  # Prefer speed
                if Quantization.INT8 in quantizations:
                    score += 8
                elif Quantization.FP16 in quantizations:
                    score += 6
                elif Quantization.FP32 in quantizations:
                    score += 2
            
            # Backend preference (ONNX is more portable)
            backends = {comp.backend for comp in ws.components.values()}
            if Backend.ONNX in backends:
                score += 5
            
            scored_sets.append((score, ws))
        
        # Return highest scored working set
        if scored_sets:
            return max(scored_sets, key=lambda x: x[0])[1]
        
        return None

# Global registry instance
diffusion_registry = DiffusionModelRegistry()