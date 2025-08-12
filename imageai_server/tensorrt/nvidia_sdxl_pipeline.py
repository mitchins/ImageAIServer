#!/usr/bin/env python3
"""
NVIDIA SDXL TensorRT-RTX Pipeline
Adapted from /data/nvidia/sdxl_tensorrt_rtx/demo_txt2img_xl.py

This integrates the working NVIDIA TensorRT-RTX demo code as a class
while maintaining our existing pipeline architecture.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image

import torch
from cuda import cudart

# Setup TensorRT-RTX environment variables
def _setup_tensorrt_rtx_environment():
    """Setup TensorRT-RTX environment variables before imports."""
    # Set TensorRT-RTX library path
    trt_rtx_lib = "/data/nvidia/TensorRT-RTX-1.0.0.21/targets/x86_64-linux-gnu/lib"
    if os.path.exists(trt_rtx_lib):
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if trt_rtx_lib not in current_ld_path:
            os.environ["LD_LIBRARY_PATH"] = f"{trt_rtx_lib}:{current_ld_path}"
    
    os.environ["POLYGRAPHY_USE_TENSORRT_RTX"] = "1"

# Setup environment first
_setup_tensorrt_rtx_environment()

# Setup paths to import NVIDIA demo modules
current_dir = Path(__file__).parent
demo_diffusion_path = current_dir / "demo_diffusion"
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import NVIDIA demo modules (after environment setup)
from demo_diffusion import dd_argparse
from demo_diffusion import pipeline as pipeline_module

logger = logging.getLogger(__name__)

class NVIDIASDXLPipeline:
    """
    SDXL Pipeline using NVIDIA's TensorRT-RTX implementation.
    
    This class wraps the working NVIDIA demo code to provide a clean
    Python API while maintaining compatibility with our backend system.
    """
    
    def __init__(self,
                 device: str = "cuda",
                 guidance_scale: float = 5.0,
                 num_inference_steps: int = 30,
                 scheduler: str = "Euler",
                 vae_scaling_factor: float = 0.13025,
                 verbose: bool = True,
                 hf_token: Optional[str] = None,
                 enable_refiner: bool = False):
        """
        Initialize the NVIDIA SDXL pipeline.
        
        Args:
            device: PyTorch device ('cuda')
            guidance_scale: Classifier-free guidance scale  
            num_inference_steps: Number of denoising steps
            scheduler: Scheduler type
            vae_scaling_factor: VAE scaling factor for SDXL
            verbose: Enable verbose logging
            hf_token: HuggingFace token
            enable_refiner: Enable SDXL refiner
        """
        self.device = device
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.scheduler = scheduler
        self.vae_scaling_factor = vae_scaling_factor
        self.verbose = verbose
        self.hf_token = hf_token
        self.enable_refiner = enable_refiner
        
        # Initialize NVIDIA pipeline components
        self.nvidia_pipeline = None
        self.engines_loaded = False
        self.engines_activated = False
        
        logger.info("Initialized NVIDIA SDXL TensorRT-RTX pipeline")
    
    
    def load_engines(self,
                    engine_dir: str,
                    framework_model_dir: str = "./pytorch_model",
                    onnx_dir: str = "./onnx",
                    build_engines: bool = False,
                    **kwargs):
        """
        Load TensorRT engines using NVIDIA's pipeline.
        
        Args:
            engine_dir: Directory containing TensorRT engine files
            framework_model_dir: Directory for PyTorch model cache
            onnx_dir: Directory for ONNX model cache
            build_engines: Whether to build engines if missing
        """
        logger.info(f"Loading NVIDIA SDXL engines from: {engine_dir}")
        
        try:
            # Create NVIDIA pipeline with our settings
            pipeline_kwargs = {
                'version': 'xl-1.0',
                'max_batch_size': 1,
                'denoising_steps': self.num_inference_steps,
                'scheduler': self.scheduler,
                'guidance_scale': self.guidance_scale,
                'device': self.device,
                'output_dir': './output',
                'hf_token': self.hf_token,
                'verbose': self.verbose,
                'nvtx_profile': False,
                'use_cuda_graph': False,
                'framework_model_dir': framework_model_dir
            }
            
            # Initialize NVIDIA SDXL pipeline
            self.nvidia_pipeline = NVIDIAStableDiffusionXLPipeline(
                vae_scaling_factor=self.vae_scaling_factor,
                enable_refiner=self.enable_refiner,
                **pipeline_kwargs
            )
            
            # Load engines with required parameters
            self.nvidia_pipeline.loadEngines(
                framework_model_dir=framework_model_dir,
                onnx_dir=onnx_dir,
                engine_dir=engine_dir,
                onnx_opset=kwargs.get('onnx_opset', 17),
                opt_batch_size=kwargs.get('opt_batch_size', 1),
                opt_image_height=kwargs.get('opt_image_height', 1024),
                opt_image_width=kwargs.get('opt_image_width', 1024),
                **{k: v for k, v in kwargs.items() if k not in ['onnx_opset', 'opt_batch_size', 'opt_image_height', 'opt_image_width']}
            )
            
            self.engines_loaded = True
            logger.info("✓ NVIDIA engines loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load NVIDIA engines: {e}")
            raise RuntimeError(f"Engine loading failed: {e}")
    
    def activate_engines(self, shared_device_memory: Optional[int] = None):
        """Activate TensorRT engines."""
        if not self.engines_loaded:
            raise RuntimeError("Engines not loaded - call load_engines first")
        
        try:
            # Get memory requirements and activate
            if shared_device_memory is None:
                max_memory = self.nvidia_pipeline.get_max_device_memory()
                _, shared_device_memory = cudart.cudaMalloc(max_memory)
                logger.info(f"Allocated {max_memory / (1024**3):.2f} GB shared device memory")
            
            self.nvidia_pipeline.activateEngines(shared_device_memory)
            self.engines_activated = True
            
            logger.info("✓ NVIDIA engines activated successfully")
            
        except Exception as e:
            logger.error(f"Failed to activate engines: {e}")
            raise RuntimeError(f"Engine activation failed: {e}")
    
    def load_resources(self, 
                      batch_size: int,
                      height: int,
                      width: int,
                      seed: Optional[int] = None):
        """Load inference resources for specific dimensions."""
        if not self.engines_activated:
            raise RuntimeError("Engines not activated - call activate_engines first")
        
        logger.info(f"Loading resources for {width}x{height}, batch_size={batch_size}")
        
        try:
            self.nvidia_pipeline.loadResources(height, width, batch_size, seed)
            logger.info("✓ Resources loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load resources: {e}")
            raise RuntimeError(f"Resource loading failed: {e}")
    
    def infer(self,
             prompt: str,
             negative_prompt: Optional[str] = None,
             height: int = 1024,
             width: int = 1024,
             batch_size: int = 1,
             num_inference_steps: Optional[int] = None,
             guidance_scale: Optional[float] = None,
             seed: Optional[int] = None,
             warmup: bool = False) -> Tuple[List[Image.Image], float]:
        """
        Run SDXL inference using NVIDIA's pipeline.
        
        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative text prompt
            height: Image height (multiple of 8)
            width: Image width (multiple of 8)
            batch_size: Number of images to generate
            num_inference_steps: Override number of denoising steps
            guidance_scale: Override guidance scale
            seed: Random seed
            warmup: Whether this is a warmup run
            
        Returns:
            Tuple of (generated_images, inference_time)
        """
        if not self.engines_activated:
            raise RuntimeError("Engines not activated")
        
        # Use instance defaults or provided values
        steps = num_inference_steps or self.num_inference_steps
        guidance = guidance_scale or self.guidance_scale
        
        # Load resources for this configuration
        self.load_resources(batch_size, height, width, seed)
        
        if not warmup:
            logger.info(f"Generating {batch_size} images at {width}x{height}")
            logger.info(f"Prompt: '{prompt}'")
        
        try:
            # Prepare prompts as lists (NVIDIA format)
            prompt_list = [prompt] if isinstance(prompt, str) else prompt
            negative_prompt_list = [negative_prompt] if negative_prompt else [""]
            
            # Run NVIDIA inference
            import time
            start_time = time.time()
            
            images_pil, inference_time = self.nvidia_pipeline.base.infer(
                prompt_list,
                negative_prompt_list, 
                height,
                width,
                warmup=warmup
            )
            
            # Convert to our expected format
            if isinstance(images_pil, np.ndarray):
                # Convert numpy arrays to PIL Images
                if images_pil.ndim == 4:  # Batch of images
                    images = []
                    for img_np in images_pil:
                        if img_np.dtype != np.uint8:
                            img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
                        if img_np.shape[0] == 3:  # CHW -> HWC
                            img_np = np.transpose(img_np, (1, 2, 0))
                        images.append(Image.fromarray(img_np))
                else:
                    # Single image
                    if images_pil.dtype != np.uint8:
                        images_pil = (np.clip(images_pil, 0, 1) * 255).astype(np.uint8)
                    if images_pil.shape[0] == 3:  # CHW -> HWC
                        images_pil = np.transpose(images_pil, (1, 2, 0))
                    images = [Image.fromarray(images_pil)]
            elif isinstance(images_pil, list):
                images = images_pil  # Already PIL Images
            else:
                raise ValueError(f"Unexpected image format: {type(images_pil)}")
            
            total_time = (time.time() - start_time) * 1000  # Convert to ms
            
            if not warmup:
                logger.info(f"✓ Generated {len(images)} images in {total_time:.1f}ms")
            
            return images, total_time
            
        except Exception as e:
            logger.error(f"NVIDIA inference failed: {e}")
            raise RuntimeError(f"Inference failed: {e}")
    
    def teardown(self):
        """Clean up resources."""
        logger.info("Cleaning up NVIDIA SDXL pipeline...")
        
        if self.nvidia_pipeline:
            self.nvidia_pipeline.teardown()
            self.nvidia_pipeline = None
        
        self.engines_loaded = False
        self.engines_activated = False
        
        logger.info("✓ NVIDIA cleanup completed")


class NVIDIAStableDiffusionXLPipeline(pipeline_module.StableDiffusionPipeline):
    """
    NVIDIA SDXL Pipeline - extracted from demo_txt2img_xl.py
    
    This is the exact NVIDIA implementation but adapted as a class
    without CLI dependencies.
    """
    
    def __init__(self, vae_scaling_factor=0.13025, enable_refiner=False, **kwargs):
        self.enable_refiner = enable_refiner
        self.nvtx_profile = kwargs.get('nvtx_profile', False)
        
        self.base = pipeline_module.StableDiffusionPipeline(
            pipeline_type=pipeline_module.PIPELINE_TYPE.XL_BASE,
            vae_scaling_factor=vae_scaling_factor,
            return_latents=self.enable_refiner,
            **kwargs,
        )
        
        if self.enable_refiner:
            self.refiner = pipeline_module.StableDiffusionPipeline(
                pipeline_type=pipeline_module.PIPELINE_TYPE.XL_REFINER,
                vae_scaling_factor=vae_scaling_factor,
                return_latents=False,
                **kwargs,
            )

    def loadEngines(self, framework_model_dir, onnx_dir, engine_dir, 
                   onnx_refiner_dir='onnx_xl_refiner', 
                   engine_refiner_dir='engine_xl_refiner', **kwargs):
        # Add default values for required parameters
        engine_kwargs = {
            'onnx_opset': kwargs.get('onnx_opset', 17),
            'opt_batch_size': kwargs.get('opt_batch_size', 1),
            'opt_image_height': kwargs.get('opt_image_height', 1024),
            'opt_image_width': kwargs.get('opt_image_width', 1024),
            **{k: v for k, v in kwargs.items() if k not in ['onnx_refiner_dir', 'engine_refiner_dir']}
        }
        
        self.base.loadEngines(engine_dir, framework_model_dir, onnx_dir, **engine_kwargs)
        if self.enable_refiner:
            self.refiner.loadEngines(engine_refiner_dir, framework_model_dir, onnx_refiner_dir, **engine_kwargs)

    def activateEngines(self, shared_device_memory=None):
        self.base.activateEngines(shared_device_memory)
        if self.enable_refiner:
            self.refiner.activateEngines(shared_device_memory)

    def loadResources(self, image_height, image_width, batch_size, seed):
        self.base.loadResources(image_height, image_width, batch_size, seed)
        if self.enable_refiner:
            # Use a different seed for refiner
            self.refiner.loadResources(image_height, image_width, batch_size, 
                                     ((seed+1) if seed is not None else None))

    def get_max_device_memory(self):
        max_device_memory = self.base.calculateMaxDeviceMemory()
        if self.enable_refiner:
            max_device_memory = max(max_device_memory, self.refiner.calculateMaxDeviceMemory())
        return max_device_memory

    def teardown(self):
        self.base.teardown()
        if self.enable_refiner:
            self.refiner.teardown()