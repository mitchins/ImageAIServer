"""
ONNX Runtime Provider Utilities

Centralizes provider configuration to avoid format inconsistencies.
"""

import torch
from typing import List, Union, Tuple, Dict, Any

def normalize_providers(providers: List[Union[str, Tuple[str, Dict[str, Any]]]]) -> List[str]:
    """
    Normalize provider format to avoid ONNX Runtime validation errors.
    
    Args:
        providers: Mixed list of provider names and (name, options) tuples
        
    Returns:
        Consistent list of provider names (strings only) - ONNX Runtime handles options separately
    """
    normalized = []
    
    for provider in providers:
        if isinstance(provider, str):
            # Keep string providers as-is
            normalized.append(provider)
        else:
            # Extract provider name from tuple, ignore options
            provider_name = provider[0] if isinstance(provider, tuple) else str(provider)
            normalized.append(provider_name)
    
    return normalized

def get_standard_providers(use_tensorrt: bool = False, use_cuda: bool = None) -> List[str]:
    """
    Get standard provider list for ONNX Runtime.
    
    Args:
        use_tensorrt: Whether to include TensorRT provider
        use_cuda: Whether to include CUDA provider (auto-detect if None)
        
    Returns:
        Consistent list of provider names (strings only)
    """
    providers = []
    
    if use_cuda is None:
        use_cuda = torch.cuda.is_available()
    
    # Add TensorRT if requested and CUDA is available
    # Note: TensorRT options should be handled via provider_options parameter separately
    if use_tensorrt and use_cuda:
        providers.append("TensorrtExecutionProvider")
    
    # Add CUDA if available
    if use_cuda:
        providers.append("CUDAExecutionProvider")
    
    # Always add CPU as fallback
    providers.append("CPUExecutionProvider")
    
    return providers

def get_tensorrt_rtx_providers() -> List[str]:
    """
    Get providers for TensorRT-RTX backend (no ONNX Runtime needed).
    
    TensorRT-RTX uses native TensorRT engines, not ONNX Runtime,
    so this returns empty list to avoid provider conflicts.
    """
    return []