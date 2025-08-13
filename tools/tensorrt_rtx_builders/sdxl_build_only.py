#!/usr/bin/env python3
"""
SDXL TensorRT-RTX Build-Only Script

Modified from NVIDIA's demo_txt2img_xl.py to ONLY build engines, not run inference.
"""

import sys
import argparse

# Add NVIDIA demo path
sys.path.append('/data/nvidia/sdxl_tensorrt_rtx')

from cuda import cudart
from demo_diffusion import dd_argparse
from demo_diffusion import pipeline as pipeline_module


def parseArgs():
    parser = argparse.ArgumentParser(description="SDXL Engine Build-Only", conflict_handler='resolve')
    parser = dd_argparse.add_arguments(parser)
    parser.add_argument('--version', type=str, default="xl-1.0", choices=["xl-1.0", "xl-turbo"], help="Version of Stable Diffusion XL")
    parser.add_argument('--height', type=int, default=1024, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument('--width', type=int, default=1024, help="Height of image to generate (must be multiple of 8)")
    return parser.parse_args()


class StableDiffusionXLPipeline(pipeline_module.StableDiffusionPipeline):
    def __init__(self, vae_scaling_factor=0.13025, enable_refiner=False, **kwargs):
        self.enable_refiner = enable_refiner
        self.nvtx_profile = kwargs['nvtx_profile']
        self.base = pipeline_module.StableDiffusionPipeline(
            pipeline_type=pipeline_module.PIPELINE_TYPE.XL_BASE,
            vae_scaling_factor=vae_scaling_factor,
            **kwargs
        )
    
    def loadEngines(self, framework_model_dir, onnx_dir, engine_dir, **kwargs):
        self.base.loadEngines(engine_dir, framework_model_dir, onnx_dir, **kwargs)


if __name__ == "__main__":
    print("[I] SDXL Engine Build-Only Script")
    print("[I] This script ONLY builds engines and exits")
    args = parseArgs()

    kwargs_init_pipeline, kwargs_load_engine, args_run_demo = dd_argparse.process_pipeline_args(args)

    # Initialize pipeline
    print("[I] Initializing pipeline...")
    demo = StableDiffusionXLPipeline(vae_scaling_factor=0.13025, enable_refiner=False, **kwargs_init_pipeline)

    # Load/Build TensorRT engines ONLY
    print("[I] Building TensorRT engines...")
    demo.loadEngines(
        args.framework_model_dir,
        args.onnx_dir,
        args.engine_dir,
        **kwargs_load_engine)
    
    print("[I] ✅ Engine building complete!")
    print("[I] Engines saved to:", args.engine_dir)
    print("[I] Exiting without running inference.")
    
    # Exit successfully without running inference
    sys.exit(0)