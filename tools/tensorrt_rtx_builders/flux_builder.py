#!/usr/bin/env python3
"""
Flux TensorRT-RTX Engine Builder

Clean, self-contained Flux engine builder using TensorRT-RTX.
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any

from .base_builder import BaseTensorRTRTXBuilder, setup_tensorrt_rtx_env

logger = logging.getLogger(__name__)

class FluxTensorRTRTXBuilder(BaseTensorRTRTXBuilder):
    """Flux TensorRT-RTX engine builder."""
    
    def __init__(self, model_variant: str = "flux1-dev", **kwargs):
        super().__init__(**kwargs)
        self.model_variant = model_variant  # flux1-dev or flux1-schnell
        # Use our build-only script instead of the full demo
        self.demo_script = Path(__file__).parent / "flux_build_only.py"
        
        if not self.demo_script.exists():
            raise FileNotFoundError(f"Flux demo script not found: {self.demo_script}")
    
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
        """Get list of expected Flux engine files."""
        # These are the cache files created by the TensorRT-RTX Flux demo
        return [
            "clip.plan",
            "t5.plan", 
            "transformer.plan",
            "vae.plan"
        ]
    
    def build_engines(self, 
                     quantization: str,
                     batch_size: int = 1,
                     height: int = 1024,
                     width: int = 1024) -> bool:
        """Build Flux engines using TensorRT-RTX demo."""
        
        logger.info(f"Building {self.model_variant} engines with {quantization} quantization")
        
        # Create cache directory (Flux demo manages its own structure)
        cache_dir = self.output_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Build command using Flux TensorRT-RTX demo  
        command = [
            sys.executable, str(self.demo_script),
            "--prompt", "test prompt for engine building",
            "--batch-size", str(batch_size),
            "--height", str(height),
            "--width", str(width),
            f"--cache-dir={cache_dir}",
            "--cache-mode=full", 
            f"--precision={quantization}"
        ]
        
        # Add optional flags
        if self.low_vram:
            command.append("--low-vram")
        
        if self.verbose:
            command.append("--verbose")
            
        if self.hf_token:
            command.append(f"--hf-token={self.hf_token}")
        
        logger.info(f"Executing: {' '.join(command)}")
        
        try:
            # Run in the demo script directory for proper imports
            result = subprocess.run(
                command,
                cwd=Path("/data/imagai/nvidia-demos/TensorRT-RTX/demo/flux1.dev"),  # Run from original demo directory for imports
                check=True,
                capture_output=True,
                text=True,
                timeout=10800  # 3 hour timeout (Flux is bigger)
            )
            
            logger.info(f"{self.model_variant} engine build completed successfully")
            logger.debug(f"Build output: {result.stdout}")
            
            # Move engines from cache to expected location
            self._organize_engine_files(cache_dir)
            
            # Validate engines
            if self.validate_engines():
                # Generate model card
                self.save_model_card()
                return True
            return False
            
        except subprocess.CalledProcessError as e:
            logger.error(f"{self.model_variant} engine build failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
        
        except subprocess.TimeoutExpired:
            logger.error(f"{self.model_variant} engine build timed out after 3 hours")
            return False
    
    def _organize_engine_files(self, cache_dir: Path):
        """Organize engine files from cache structure to expected location."""
        engines_dir = self.output_dir / "engines"
        engines_dir.mkdir(parents=True, exist_ok=True)
        
        # Map actual engine files to expected names
        engine_mapping = {
            "flux_clip_text_encoder_static.engine": "clip.plan",
            "flux_t5_text_encoder_static.engine": "t5.plan", 
            "flux_transformer_static.engine": "transformer.plan",
            "flux_vae_decoder_static.engine": "vae.plan"
        }
        
        # Find and copy engine files from cache
        for actual_name, expected_name in engine_mapping.items():
            # Look for the actual engine file in cache subdirectories
            for engine_file in cache_dir.rglob(actual_name):
                target_path = engines_dir / expected_name
                if not target_path.exists():
                    # Copy instead of move to preserve cache
                    import shutil
                    shutil.copy2(engine_file, target_path)
                    logger.info(f"Copied engine: {engine_file} -> {target_path}")
                    break
    
    def validate_engines(self) -> bool:
        """Validate Flux engines (check in engines directory)."""
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
                
        logger.info(f"All {len(expected_files)} Flux engine files validated successfully")
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Flux-specific information for README generation."""
        return {
            'license': 'other',
            'base_model': f'black-forest-labs/FLUX.1-{self.model_variant.split("-")[1]}',
            'model_name': f'Flux.1-{self.model_variant.split("-")[1]}',
            'description': f'TensorRT-RTX optimized engines for Flux.1-{self.model_variant.split("-")[1]} on NVIDIA Ampere architecture (RTX 30 series, A100, etc.) with BF16 precision.',
            'tags': [
                'flux',
                f'flux1-{self.model_variant.split("-")[1]}',
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
                'clip.plan': 'CLIP text encoder',
                't5.plan': 'T5 text encoder',
                'transformer.plan': 'Flux transformer model',
                'vae.plan': 'VAE decoder'
            },
            'hw_requirements': 'NVIDIA RTX 30 series (RTX 3080, 3090) or A100',
            'min_vram': 24,
            'usage_example': '''# Example usage with TensorRT-RTX backend
from nvidia_demos.TensorRT_RTX.demo.flux1_dev.pipelines.flux_pipeline import FluxPipeline

pipeline = FluxPipeline(
    cache_dir="./cache",
    hf_token="your_hf_token"
)

# Load pre-built engines
pipeline.load_engines(
    transformer_precision="bf16",
    opt_batch_size=1,
    opt_height=1024,
    opt_width=1024
)

# Generate image
image = pipeline.infer(
    prompt="A beautiful landscape with mountains",
    height=1024,
    width=1024
)''',
            'performance': {
                'inference_speed': '~8-12 seconds per image (RTX 3090)',
                'memory_usage': '~18-20GB VRAM',
                'optimizations': 'Static shapes, BF16 precision, Ampere-specific kernels'
            },
            'license_text': f'This model follows the Flux.1-{self.model_variant.split("-")[1]} license terms. Please refer to the original model repository for licensing details.',
            'demo_source': '[NVIDIA Flux Demo](https://github.com/NVIDIA/TensorRT-RTX/)'
        }
    
    def get_total_size_gb(self) -> float:
        """Get total size of Flux engines in GB (look in engines subdirectory)."""
        engines_dir = self.output_dir / "engines"
        total_bytes = 0
        for engine_file in self.get_engine_files():
            engine_path = engines_dir / engine_file
            if engine_path.exists():
                total_bytes += engine_path.stat().st_size
        
        return total_bytes / (1024**3)