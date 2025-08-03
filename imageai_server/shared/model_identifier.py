"""Unified model identifier system for consistent model representation across backends."""

from typing import Dict, List, Optional, Tuple, NamedTuple
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


class ModelBackendType(str, Enum):
    """Available model backends."""
    ONNX = "onnx"
    PYTORCH = "pytorch"
    AUTO = "auto"  # Let system choose best backend


class QuantizationType(str, Enum):
    """Unified quantization types across backends."""
    # Precision types (common to both)
    FP32 = "fp32"
    FP16 = "fp16" 
    BF16 = "bf16"
    
    # Integer quantization
    INT8 = "int8"
    INT4 = "int4"
    UINT8 = "uint8"
    
    # Advanced quantization
    Q4 = "q4"
    Q4_F16 = "q4_f16"
    Q4_MIXED = "q4_mixed"
    BNB4 = "bnb4"
    QUANTIZED = "quantized"  # Generic quantized
    
    # GGUF quantization levels (for PyTorch via llama.cpp)
    Q8_0 = "q8_0"
    Q6_K = "q6_k"
    Q6_K_L = "q6_k_l"
    Q5_K_S = "q5_k_s"
    Q5_K_M = "q5_k_m"
    Q5_K_L = "q5_k_l"
    Q4_K_S = "q4_k_s"
    Q4_K_M = "q4_k_m"
    Q4_K_L = "q4_k_l"
    Q4_0 = "q4_0"
    Q4_1 = "q4_1"
    Q3_K_S = "q3_k_s"
    Q3_K_M = "q3_k_m"
    Q3_K_L = "q3_k_l"
    Q2_K = "q2_k"
    
    # Advanced GGUF quants
    IQ4_XS = "iq4_xs"
    IQ4_NL = "iq4_nl"
    IQ3_M = "iq3_m"
    IQ3_XS = "iq3_xs"
    IQ3_XXS = "iq3_xxs"
    IQ2_M = "iq2_m"
    
    # Special
    AUTO = "auto"  # Let backend choose best available


class ModelIdentifier(NamedTuple):
    """Unified model identifier structure."""
    family: str          # e.g., "smolvlm", "gemma3n", "granite-vision"
    size: str           # e.g., "256m", "2b", "7b"
    variant: str        # e.g., "instruct", "chat", "base"
    repo_id: str        # Hugging Face repo ID
    quantization: QuantizationType = QuantizationType.AUTO
    backend: ModelBackendType = ModelBackendType.AUTO


class ModelCatalog:
    """Centralized catalog of all supported models with unified identifiers."""
    
    # GGUF quantization configurations - similar to ONNX MODEL_QUANT_CONFIGS
    GGUF_QUANTIZATIONS = {
        # SmolVLM-256M GGUF variants 
        "ggml-org/SmolVLM-256M-Instruct-GGUF": [
            QuantizationType.Q8_0,   # 175MB - highest quality
            QuantizationType.FP16,   # 328MB - full precision
        ],
        
        # SmolVLM-500M GGUF variants
        "ggml-org/SmolVLM-500M-Instruct-GGUF": [
            QuantizationType.Q8_0,   # ~400MB estimated
            QuantizationType.FP16,   # ~750MB estimated  
        ],
        
        # Granite Vision 3.2-2B GGUF variants (bartowski)
        "bartowski/ibm-granite_granite-vision-3.2-2b-GGUF": [
            # Recommended quantizations in order of quality/size
            QuantizationType.Q6_K,      # 2.08GB - excellent quality
            QuantizationType.Q5_K_M,    # 1.80GB - very good quality
            QuantizationType.Q4_K_M,    # 1.55GB - good quality
            QuantizationType.IQ4_XS,    # 1.38GB - compact high quality
            QuantizationType.Q4_0,      # 1.46GB - standard 4-bit
            QuantizationType.Q3_K_M,    # 1.25GB - aggressive but usable
            QuantizationType.Q2_K,      # 0.98GB - very aggressive
        ],
        
        # Pixtral 12B GGUF variants (bartowski)
        "bartowski/mistral-community_pixtral-12b-GGUF": [
            QuantizationType.Q6_K,      # ~10GB - excellent quality
            QuantizationType.Q5_K_M,    # ~8.5GB - very good quality
            QuantizationType.Q4_K_M,    # ~7GB - good balance
            QuantizationType.Q4_0,      # ~6.5GB - standard 4-bit
            QuantizationType.Q3_K_M,    # ~5.5GB - aggressive
        ],
        
        # Gemma-3 27B GGUF variants (bartowski)
        "bartowski/google_gemma-3-27b-it-GGUF": [
            QuantizationType.Q6_K,      # ~22GB - excellent quality
            QuantizationType.Q5_K_M,    # ~19GB - very good quality
            QuantizationType.Q4_K_M,    # ~16GB - good balance
            QuantizationType.Q4_0,      # ~15GB - standard 4-bit
            QuantizationType.Q3_K_M,    # ~13GB - aggressive
        ],
    }
    
    MODELS = {
        # SmolVLM family - GGUF quantized preferred (ultra-lightweight)
        "smolvlm-256m-instruct": ModelIdentifier(
            family="smolvlm",
            size="256m", 
            variant="instruct",
            repo_id="ggml-org/SmolVLM-256M-Instruct-GGUF",
            backend=ModelBackendType.PYTORCH
        ),
        "smolvlm-500m-instruct": ModelIdentifier(
            family="smolvlm",
            size="500m",
            variant="instruct", 
            repo_id="ggml-org/SmolVLM-500M-Instruct-GGUF",
            backend=ModelBackendType.PYTORCH
        ),
        
        # Granite Vision family - GGUF quantized preferred
        "granite-vision-3.2-2b": ModelIdentifier(
            family="granite-vision",
            size="2b",
            variant="3.2",
            repo_id="bartowski/ibm-granite_granite-vision-3.2-2b-GGUF", 
            backend=ModelBackendType.PYTORCH
        ),
        
        # Gemma-3n family - ONNX optimized
        "gemma3n-e2b-it": ModelIdentifier(
            family="gemma3n",
            size="2b",
            variant="e2b-it",
            repo_id="onnx-community/gemma-3n-E2B-it-ONNX",
            backend=ModelBackendType.ONNX
        ),
        
        # Qwen2.5-VL family - AWQ quantized preferred
        "qwen2.5-vl-7b-awq": ModelIdentifier(
            family="qwen2.5-vl",
            size="7b",
            variant="awq",
            repo_id="Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
            backend=ModelBackendType.PYTORCH
        ),
        
        # Pixtral family - High quality 12B
        "pixtral-12b": ModelIdentifier(
            family="pixtral",
            size="12b",
            variant="instruct",
            repo_id="bartowski/mistral-community_pixtral-12b-GGUF",
            backend=ModelBackendType.PYTORCH
        ),
        
        # Gemma-3 family - Large 27B option
        "gemma3-27b-it": ModelIdentifier(
            family="gemma3",
            size="27b",
            variant="it",
            repo_id="bartowski/google_gemma-3-27b-it-GGUF",
            backend=ModelBackendType.PYTORCH
        ),
    }

    @classmethod
    def get_model(cls, identifier: str) -> Optional[ModelIdentifier]:
        """Get model by unified identifier."""
        return cls.MODELS.get(identifier.lower())
    
    @classmethod
    def list_models(cls, backend: Optional[ModelBackendType] = None) -> List[str]:
        """List all model identifiers, optionally filtered by backend."""
        if backend is None:
            return list(cls.MODELS.keys())
        return [
            model_id for model_id, model in cls.MODELS.items() 
            if model.backend == backend or model.backend == ModelBackendType.AUTO
        ]
    
    @classmethod
    def get_pytorch_models(cls) -> List[str]:
        """Get all PyTorch-compatible models."""
        return cls.list_models(ModelBackendType.PYTORCH)
    
    @classmethod
    def get_onnx_models(cls) -> List[str]:
        """Get all ONNX-compatible models."""
        return cls.list_models(ModelBackendType.ONNX)
    
    @classmethod
    def get_available_quantizations(cls, repo_id: str) -> List[QuantizationType]:
        """Get available quantizations for a GGUF repo."""
        return cls.GGUF_QUANTIZATIONS.get(repo_id, [])
    
    @classmethod
    def get_recommended_quantization(cls, repo_id: str) -> Optional[QuantizationType]:
        """Get the recommended quantization for a model (first in list = best quality/size tradeoff)."""
        quantizations = cls.get_available_quantizations(repo_id)
        return quantizations[0] if quantizations else None
    
    @classmethod
    def format_gguf_filename(cls, repo_id: str, quantization: QuantizationType) -> str:
        """Format GGUF filename based on actual repository naming conventions."""
        # Map quantization to actual filename formats used in repos
        quant_mapping = {
            # SmolVLM uses different format for FP16
            QuantizationType.FP16: "f16",
            QuantizationType.Q8_0: "Q8_0",
            # Standard formats for other models
            QuantizationType.Q6_K: "Q6_K",
            QuantizationType.Q5_K_M: "Q5_K_M", 
            QuantizationType.Q4_K_M: "Q4_K_M",
            QuantizationType.Q4_0: "Q4_0",
            QuantizationType.IQ4_XS: "IQ4_XS",
            QuantizationType.Q3_K_M: "Q3_K_M",
            QuantizationType.Q2_K: "Q2_K",
        }
        
        quant_suffix = quant_mapping.get(quantization, quantization.value.upper())
        
        # Use actual naming conventions from the repositories
        if "SmolVLM-256M" in repo_id:
            return f"SmolVLM-256M-Instruct-{quant_suffix}.gguf"
        elif "SmolVLM-500M" in repo_id:
            return f"SmolVLM-500M-Instruct-{quant_suffix}.gguf"
        elif "granite-vision-3.2-2b" in repo_id:
            # Bartowski uses the full repo name format
            return f"ibm-granite_granite-vision-3.2-2b-{quant_suffix}.gguf"
        else:
            # Fallback: use repo name
            model_name = repo_id.split('/')[-1].replace('-GGUF', '')
            return f"{model_name}-{quant_suffix}.gguf"


class ModelNameParser:
    """Unified parser for different model name formats."""
    
    @staticmethod
    def parse_legacy_format(name: str) -> Tuple[str, Optional[QuantizationType]]:
        """Parse legacy format like 'SmolVLM-256M-Instruct/UINT8' or 'model:q8_0'."""
        quantization = None
        base_name = name
        
        # Handle colon format (unified): model:quantization
        if ":" in name:
            base_name, quant_str = name.rsplit(":", 1)
            try:
                quantization = QuantizationType(quant_str.lower())
            except ValueError:
                quantization = None
        # Handle slash format (legacy): model/QUANTIZATION  
        elif "/" in name:
            base_name, quant_str = name.rsplit("/", 1)
            try:
                quantization = QuantizationType(quant_str.lower())
            except ValueError:
                quantization = None
        
        return base_name, quantization
    
    @staticmethod
    def parse_repo_id(repo_id: str) -> Optional[str]:
        """Map HF repo ID to unified identifier."""
        repo_to_identifier = {
            # Original PyTorch models
            "HuggingFaceTB/SmolVLM-256M-Instruct": "smolvlm-256m-instruct",
            "HuggingFaceTB/SmolVLM-500M-Instruct": "smolvlm-500m-instruct", 
            "ibm-granite/granite-vision-3.2-2b": "granite-vision-3.2-2b",
            
            # GGUF quantized models (preferred)
            "ggml-org/SmolVLM-256M-Instruct-GGUF": "smolvlm-256m-instruct",
            "ggml-org/SmolVLM-500M-Instruct-GGUF": "smolvlm-500m-instruct",
            "bartowski/ibm-granite_granite-vision-3.2-2b-GGUF": "granite-vision-3.2-2b",
            
            # ONNX models
            "onnx-community/gemma-3n-E2B-it-ONNX": "gemma3n-e2b-it",
            "onnx-community/Qwen2-VL-2B-Instruct-ONNX": "qwen2-vl-2b-instruct",
            "microsoft/Phi-3.5-vision-instruct": "phi3.5-vision-instruct",
        }
        return repo_to_identifier.get(repo_id)
    
    @staticmethod
    def normalize_model_name(name: str) -> Tuple[str, Optional[QuantizationType], ModelBackendType]:
        """Normalize any model name format to unified identifier."""
        # Try parsing as legacy format first
        base_name, quantization = ModelNameParser.parse_legacy_format(name)
        
        # Look up in catalog by various methods
        model_id = None
        
        # 1. Direct lookup in catalog
        model_id = base_name.lower()
        if model_id in ModelCatalog.MODELS:
            model = ModelCatalog.MODELS[model_id]
            return model_id, quantization, model.backend
        
        # 2. Try as repo ID
        model_id = ModelNameParser.parse_repo_id(base_name)
        if model_id and model_id in ModelCatalog.MODELS:
            model = ModelCatalog.MODELS[model_id]
            return model_id, quantization, model.backend
        
        # 3. Fuzzy matching for legacy names
        legacy_mappings = {
            "SmolVLM-256M-Instruct": "smolvlm-256m-instruct",
            "SmolVLM-500M-Instruct": "smolvlm-500m-instruct",
            "Gemma-3n-E2B-it-ONNX": "gemma3n-e2b-it",
            "granite-vision-3.2-2b": "granite-vision-3.2-2b",
        }
        
        model_id = legacy_mappings.get(base_name)
        if model_id:
            model = ModelCatalog.MODELS[model_id]
            return model_id, quantization, model.backend
        
        # 4. If not found, return None to indicate unrecognized model
        return None, quantization, ModelBackendType.PYTORCH


class UnifiedModelManager:
    """High-level interface for model management across backends."""
    
    @staticmethod
    def resolve_model(name: str, preferred_backend: Optional[ModelBackendType] = None) -> Tuple[str, ModelBackendType, Optional[QuantizationType]]:
        """Resolve any model name to normalized form with backend selection."""
        model_id, quantization, suggested_backend = ModelNameParser.normalize_model_name(name)
        
        # Use preferred backend if specified, otherwise use suggested
        final_backend = preferred_backend if preferred_backend and preferred_backend != ModelBackendType.AUTO else suggested_backend
        
        # Get the repo_id for loading
        if model_id:
            model = ModelCatalog.get_model(model_id)
            if model:
                repo_id = model.repo_id
                
                # Special handling for GGUF models when using PyTorch backend
                # Use original model repo instead of GGUF repo for transformers compatibility
                if (final_backend == ModelBackendType.PYTORCH and 
                    repo_id in ModelCatalog.GGUF_QUANTIZATIONS):
                    
                    # Map GGUF repos to their original counterparts
                    gguf_to_original = {
                        "ggml-org/SmolVLM-256M-Instruct-GGUF": "HuggingFaceTB/SmolVLM-256M-Instruct",
                        "ggml-org/SmolVLM-500M-Instruct-GGUF": "HuggingFaceTB/SmolVLM-500M-Instruct",
                        "bartowski/ibm-granite_granite-vision-3.2-2b-GGUF": "ibm-granite/granite-vision-3.2-2b",
                        "bartowski/mistral-community_pixtral-12b-GGUF": "mistralai/pixtral-12b",
                        "bartowski/google_gemma-3-27b-it-GGUF": "google/gemma-3-27b-it",
                    }
                    
                    original_repo = gguf_to_original.get(repo_id)
                    if original_repo:
                        logger.info(f"Using original repo {original_repo} instead of GGUF repo {repo_id} for PyTorch backend")
                        repo_id = original_repo
                
                # If quantization specified, validate it's available for this model
                if quantization and model.repo_id in ModelCatalog.GGUF_QUANTIZATIONS:
                    available_quants = ModelCatalog.get_available_quantizations(model.repo_id)
                    if quantization not in available_quants:
                        # Fall back to recommended quantization
                        quantization = ModelCatalog.get_recommended_quantization(model.repo_id)
                        if quantization:
                            logger.info(f"Quantization not available, using recommended: {quantization}")
                
                # If no quantization specified but GGUF model, use recommended
                elif not quantization and model.repo_id in ModelCatalog.GGUF_QUANTIZATIONS:
                    quantization = ModelCatalog.get_recommended_quantization(model.repo_id)
                    
            else:
                # Fallback to using the model_id as repo_id
                repo_id = model_id
        else:
            # Parse as legacy format and extract base name
            base_name, _ = ModelNameParser.parse_legacy_format(name)
            repo_id = base_name
        
        return repo_id, final_backend, quantization
    
    @staticmethod
    def get_available_models() -> Dict[str, Dict]:
        """Get all available models organized by backend."""
        result = {
            "onnx": [],
            "pytorch": [],
            "auto": []
        }
        
        for model_id, model in ModelCatalog.MODELS.items():
            model_info = {
                "id": model_id,
                "family": model.family,
                "size": model.size,
                "variant": model.variant,
                "repo_id": model.repo_id,
                "description": f"{model.family.title()} {model.size.upper()} {model.variant}"
            }
            
            if model.backend == ModelBackendType.ONNX:
                result["onnx"].append(model_info)
            elif model.backend == ModelBackendType.PYTORCH:
                result["pytorch"].append(model_info)
            else:
                result["auto"].append(model_info)
        
        return result