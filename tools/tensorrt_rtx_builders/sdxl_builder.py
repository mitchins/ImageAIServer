#!/usr/bin/env python3
"""
SDXL TensorRT-RTX Engine Builder

Clean, self-contained SDXL engine builder using TensorRT-RTX.
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any

from .base_builder import BaseTensorRTRTXBuilder, setup_tensorrt_rtx_env

logger = logging.getLogger(__name__)

class SDXLTensorRTRTXBuilder(BaseTensorRTRTXBuilder):
    """SDXL TensorRT-RTX engine builder."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "sdxl"
        # Use our build-only script instead of the full demo
        self.demo_script = Path(__file__).parent / "sdxl_build_only.py"
        
        if not self.demo_script.exists():
            raise FileNotFoundError(f"SDXL demo script not found: {self.demo_script}")
    
    def get_supported_quantizations(self) -> List[str]:
        """Get supported quantizations based on GPU capabilities."""
        import torch
        
        if not torch.cuda.is_available():
            return ["bf16"]  # Conservative fallback
            
        device = torch.cuda.get_device_properties(0)
        compute_major = device.major
        compute_minor = device.minor
        
        # Determine supported quantizations
        if compute_major == 8:
            if compute_minor >= 9:  # Ada Lovelace
                return ["bf16", "fp8"]
            else:  # Ampere
                return ["bf16"]
        elif compute_major >= 9:  # Blackwell
            return ["bf16", "fp8", "fp4"]
        else:
            return ["bf16"]
    
    def get_engine_files(self) -> List[str]:
        """Get list of expected SDXL engine files."""
        return [
            "clip.trt1.0.0.21.plan",
            "clip2.trt1.0.0.21.plan", 
            "unetxl.trt1.0.0.21.plan",
            "vae.trt1.0.0.21.plan"
        ]
    
    def build_engines(self, 
                     quantization: str,
                     batch_size: int = 1,
                     height: int = 1024,
                     width: int = 1024) -> bool:
        """Build SDXL engines using TensorRT-RTX demo."""
        
        logger.info(f"Building SDXL engines with {quantization} quantization")
        
        # Create subdirectories
        engines_dir = self.output_dir / "engines"
        onnx_dir = self.output_dir / "onnx"
        framework_dir = self.output_dir / "framework"
        
        for directory in [engines_dir, onnx_dir, framework_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Build command using SDXL TensorRT-RTX demo
        command = [
            sys.executable, str(self.demo_script),
            "test prompt for engine building",
            f"--batch-size={batch_size}",
            f"--height={height}",
            f"--width={width}",
            f"--engine-dir={engines_dir}",
            f"--onnx-dir={onnx_dir}",
            f"--framework-model-dir={framework_dir}",
            "--build-static-batch",
            "--num-warmup-runs=0",
            "--batch-count=1",
            "--version=xl-1.0"
        ]
        
        # Add quantization flag
        if quantization == "bf16":
            command.append("--bf16")
        elif quantization == "fp8":
            command.append("--fp8")
        elif quantization == "fp4":
            command.append("--fp4")
        else:
            raise ValueError(f"Unsupported quantization: {quantization}")
        
        # Add optional flags
        # Note: SDXL does not support --low-vram flag (only Flux and SD3.5 do)
        
        if self.verbose:
            command.append("--verbose")
            
        if self.hf_token:
            command.append(f"--hf-token={self.hf_token}")
        
        logger.info(f"Executing: {' '.join(command)}")
        
        try:
            # Run in the demo script directory for proper imports
            result = subprocess.run(
                command,
                cwd=Path("/data/nvidia/sdxl_tensorrt_rtx"),  # Run from original demo directory for imports
                check=True,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            
            logger.info("SDXL engine build completed successfully")
            logger.debug(f"Build output: {result.stdout}")
            
            # Validate engines
            if self.validate_engines():
                # Generate model card
                self.save_model_card()
                return True
            return False
            
        except subprocess.CalledProcessError as e:
            logger.error(f"SDXL engine build failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
        
        except subprocess.TimeoutExpired:
            logger.error("SDXL engine build timed out after 2 hours")
            return False
    
    def validate_engines(self) -> bool:
        """Validate SDXL engines (they're in the engines/ subdirectory)."""
        engines_dir = self.output_dir / "engines"
        if not engines_dir.exists():
            logger.error(f"Engines directory not found: {engines_dir}")
            return False
        
        expected_files = self.get_engine_files()
        
        for engine_file in expected_files:
            engine_path = engines_dir / engine_file
            if not engine_path.exists():
                logger.error(f"Missing engine file: {engine_path}")
                return False
            
            # Check file size (engines should be substantial)
            file_size_mb = engine_path.stat().st_size / (1024 * 1024)
            if file_size_mb < 10:  # Engines should be at least 10MB
                logger.error(f"Engine file too small: {engine_path} ({file_size_mb:.1f}MB)")
                return False
                
        logger.info(f"All {len(expected_files)} SDXL engine files validated successfully")
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get SDXL-specific information for README generation."""
        return {
            'license': 'openrail++',
            'base_model': 'stabilityai/stable-diffusion-xl-base-1.0',
            'model_name': 'SDXL',
            'description': 'TensorRT-RTX optimized engines for Stable Diffusion XL on NVIDIA Ampere architecture (RTX 30 series, A100, etc.) with BF16 precision.',
            'tags': [
                'stable-diffusion',
                'stable-diffusion-xl', 
                'text-to-image',
                'diffusers',
                'tensorrt',
                'tensorrt-rtx',
                'nvidia',
                'ampere',
                'bf16'
            ],
            'precision': 'bf16',
            'precision_description': '16-bit brain floating point',
            'resolution': '1024x1024',
            'engine_descriptions': {
                'clip.trt1.0.0.21.plan': 'CLIP text encoder',
                'clip2.trt1.0.0.21.plan': 'CLIP text encoder 2',
                'unetxl.trt1.0.0.21.plan': 'U-Net XL diffusion model', 
                'vae.trt1.0.0.21.plan': 'VAE decoder'
            },
            'hw_requirements': 'NVIDIA RTX 30 series (RTX 3060, 3070, 3080, 3090) or A100',
            'min_vram': 12,
            'usage_example': '''# Example usage with TensorRT-RTX backend
from imageai_server.shared.tensorrt_rtx_backend import TensorRTRTXBackend

backend = TensorRTRTXBackend()
backend.load_engines("path/to/engines")
image = backend.generate("A beautiful sunset over mountains")''',
            'performance': {
                'inference_speed': '~2-3 seconds per image (RTX 3090)',
                'memory_usage': '~6-8GB VRAM',
                'optimizations': 'Static shapes, BF16 precision, Ampere-specific kernels'
            },
            'license_text': 'This model is released under the same license as the base SDXL model (OpenRAIL++).',
            'demo_source': '[NVIDIA Diffusion Demo](https://github.com/NVIDIA/TensorRT/tree/release/10.6/demo/Diffusion)'
        }
    
    def get_total_size_gb(self) -> float:
        """Get total size of SDXL engines in GB (look in engines subdirectory)."""
        engines_dir = self.output_dir / "engines"
        total_bytes = 0
        for engine_file in self.get_engine_files():
            engine_path = engines_dir / engine_file
            if engine_path.exists():
                total_bytes += engine_path.stat().st_size
        
        return total_bytes / (1024**3)