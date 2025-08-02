"""ONNX backend implementation for consistency with PyTorch backend."""

import logging
from typing import List

from .model_backend import ModelBackend, BackendConfig

# Import ONNX availability with error handling
try:
    from .onnx_loader import ONNX_AVAILABLE
    from .model_types import get_available_model_quants
except ImportError:
    ONNX_AVAILABLE = False
    def get_available_model_quants():
        return []

logger = logging.getLogger(__name__)


class ONNXBackend(ModelBackend):
    """ONNX backend implementation."""
    
    def __init__(self, config: BackendConfig):
        super().__init__(config)
    
    def is_available(self) -> bool:
        """Check if ONNX backend is available."""
        return ONNX_AVAILABLE
    
    def get_supported_models(self) -> List[str]:
        """Return list of models supported by ONNX backend."""
        if not ONNX_AVAILABLE:
            return []
        
        # Return all curated ONNX model/quant combinations
        return get_available_model_quants()
    
    def supports_quantization(self, quant_type: str) -> bool:
        """Check if ONNX backend supports specific quantization type."""
        if not ONNX_AVAILABLE:
            return False
        
        # ONNX supports these quantization types
        supported_quants = {
            "q4": True,
            "q4_f16": True,
            "q4_mixed": True,
            "fp16": True,
            "fp32": True,
            "int8": True,
            "uint8": True,
            "bnb4": True,
            "quantized": True,
        }
        return supported_quants.get(quant_type.lower(), False)


def create_onnx_backend(config: BackendConfig = None) -> ONNXBackend:
    """Create ONNX backend."""
    if config is None:
        config = BackendConfig(backend_type="onnx")
    
    return ONNXBackend(config)