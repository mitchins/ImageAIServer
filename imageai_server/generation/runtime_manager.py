"""
Runtime Manager - Detects and manages available execution providers
Simple, robust, failure-avoidant design
"""

import torch
import platform
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

@dataclass
class RuntimeInfo:
    """Information about an available runtime"""
    name: str
    display_name: str
    available: bool
    performance_tier: int  # 1=fastest, 4=slowest
    memory_efficient: bool
    hardware_required: Optional[str] = None
    notes: str = ""

class RuntimeManager:
    """Detects and manages available execution runtimes"""
    
    def __init__(self):
        self._runtimes = {}
        self._detected = False
        
    def detect_runtimes(self) -> Dict[str, RuntimeInfo]:
        """Detect all available runtimes once, cache results"""
        if self._detected:
            return self._runtimes
            
        self._runtimes = {
            # Base ONNX - Always available
            "onnx-cpu": RuntimeInfo(
                name="onnx-cpu",
                display_name="ONNX CPU",
                available=self._check_onnx_cpu(),
                performance_tier=4,
                memory_efficient=True,
                notes="Universal compatibility, CPU-only"
            ),
            
            # ONNX CUDA - NVIDIA GPUs
            "onnx-cuda": RuntimeInfo(
                name="onnx-cuda", 
                display_name="ONNX CUDA",
                available=self._check_onnx_cuda(),
                performance_tier=2,
                memory_efficient=False,
                hardware_required="NVIDIA GPU",
                notes="NVIDIA GPU acceleration"
            ),
            
            # ONNX TensorRT - High-end NVIDIA
            "onnx-tensorrt": RuntimeInfo(
                name="onnx-tensorrt",
                display_name="ONNX TensorRT", 
                available=self._check_onnx_tensorrt(),
                performance_tier=1,
                memory_efficient=True,
                hardware_required="NVIDIA RTX/A/H Series",
                notes="Maximum NVIDIA performance, 2-3x speedup"
            ),
            
            # ONNX OpenVINO - Intel optimization
            "onnx-openvino": RuntimeInfo(
                name="onnx-openvino",
                display_name="ONNX OpenVINO",
                available=self._check_onnx_openvino(), 
                performance_tier=2,
                memory_efficient=True,
                hardware_required="Intel CPU/iGPU",
                notes="Intel hardware optimization"
            ),
            
            # CoreML - Apple Silicon (direct, not ONNX)
            "coreml": RuntimeInfo(
                name="coreml",
                display_name="CoreML",
                available=self._check_coreml(),
                performance_tier=1,
                memory_efficient=True,
                hardware_required="Apple Silicon",
                notes="Native Apple Silicon, Neural Engine"
            ),
            
            # PyTorch - Fallback
            "pytorch-cpu": RuntimeInfo(
                name="pytorch-cpu",
                display_name="PyTorch CPU",
                available=True,  # Always available
                performance_tier=4,
                memory_efficient=False,
                notes="PyTorch CPU fallback"
            ),
            
            "pytorch-cuda": RuntimeInfo(
                name="pytorch-cuda",
                display_name="PyTorch CUDA",
                available=self._check_pytorch_cuda(),
                performance_tier=3,
                memory_efficient=False,
                hardware_required="NVIDIA GPU",
                notes="PyTorch GPU acceleration"
            ),
            
            "pytorch-mps": RuntimeInfo(
                name="pytorch-mps",
                display_name="PyTorch MPS",
                available=self._check_pytorch_mps(),
                performance_tier=2, 
                memory_efficient=True,
                hardware_required="Apple Silicon",
                notes="Apple Metal Performance Shaders"
            )
        }
        
        self._detected = True
        return self._runtimes
        
    def get_optimal_runtime(self, prefer_memory_efficient=False) -> str:
        """Get the best available runtime"""
        runtimes = self.detect_runtimes()
        available = {k: v for k, v in runtimes.items() if v.available}
        
        if not available:
            return "pytorch-cpu"  # Final fallback
            
        # Sort by performance tier (lower is better), memory efficiency as tiebreaker
        best = min(available.values(), 
                  key=lambda x: (x.performance_tier, not (x.memory_efficient and prefer_memory_efficient)))
        
        return best.name
        
    def get_providers_for_runtime(self, runtime: str) -> List[str]:
        """Get ONNX providers for a runtime"""
        provider_map = {
            "onnx-tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            "onnx-cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"], 
            "onnx-openvino": ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
            "onnx-cpu": ["CPUExecutionProvider"],
        }
        return provider_map.get(runtime, ["CPUExecutionProvider"])
        
    def get_runtime_config(self, runtime: str) -> Dict:
        """Get configuration for a specific runtime"""
        configs = {
            "onnx-tensorrt": {
                "providers": [
                    ("TensorrtExecutionProvider", {
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": "./trt_cache",
                        "trt_fp16_enable": True,
                        "trt_max_workspace_size": 4 << 30,  # 4GB
                        "trt_max_partition_iterations": 3,
                        "trt_min_subgraph_size": 1
                    }),
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider"
                ]
            },
            "onnx-cuda": {
                "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]
            },
            "onnx-openvino": {
                "providers": [
                    ("OpenVINOExecutionProvider", {
                        "device_type": "AUTO",  # Auto-detect best Intel device
                        "precision": "FP16",
                        "num_threads": 0  # Use all available cores
                    }),
                    "CPUExecutionProvider"
                ]
            },
            "onnx-cpu": {
                "providers": ["CPUExecutionProvider"]
            }
        }
        return configs.get(runtime, {"providers": ["CPUExecutionProvider"]})
        
    # Detection methods - simple and robust
    def _check_onnx_cpu(self) -> bool:
        try:
            import onnxruntime as ort
            return "CPUExecutionProvider" in ort.get_available_providers()
        except ImportError:
            return False
            
    def _check_onnx_cuda(self) -> bool:
        try:
            import onnxruntime as ort
            return "CUDAExecutionProvider" in ort.get_available_providers()
        except ImportError:
            return False
            
    def _check_onnx_tensorrt(self) -> bool:
        try:
            import onnxruntime as ort
            return "TensorrtExecutionProvider" in ort.get_available_providers()
        except ImportError:
            return False
            
    def _check_onnx_openvino(self) -> bool:
        try:
            import onnxruntime as ort
            return "OpenVINOExecutionProvider" in ort.get_available_providers()
        except ImportError:
            return False
            
    def _check_coreml(self) -> bool:
        """Check if CoreML is available (macOS + diffusers CoreML support)"""
        if platform.system() != "Darwin":
            return False
        try:
            import coremltools
            from diffusers import StableDiffusionPipeline
            # Check if we're on Apple Silicon
            return platform.machine() in ["arm64", "arm"]
        except ImportError:
            return False
            
    def _check_pytorch_cuda(self) -> bool:
        try:
            return torch.cuda.is_available()
        except:
            return False
            
    def _check_pytorch_mps(self) -> bool:
        try:
            return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        except:
            return False

# Global instance
runtime_manager = RuntimeManager()