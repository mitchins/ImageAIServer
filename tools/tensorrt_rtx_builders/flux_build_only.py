#!/usr/bin/env python3
"""
Flux TensorRT-RTX Build-Only Script

Modified to ONLY build engines, not run inference.
Based on NVIDIA's TensorRT-RTX Flux demo.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Setup TensorRT-RTX environment FIRST
def setup_tensorrt_rtx_env():
    trt_rtx_lib = "/data/nvidia/TensorRT-RTX-1.0.0.21/targets/x86_64-linux-gnu/lib"
    if os.path.exists(trt_rtx_lib):
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if trt_rtx_lib not in current_ld_path:
            os.environ["LD_LIBRARY_PATH"] = f"{trt_rtx_lib}:{current_ld_path}"
    os.environ["POLYGRAPHY_USE_TENSORRT_RTX"] = "1"

setup_tensorrt_rtx_env()

# Add TensorRT-RTX demo path and our utils
sys.path.append('/data/imagai/nvidia-demos/TensorRT-RTX/demo/flux1.dev')
sys.path.append('/data/imagai/tools/tensorrt_rtx_builders')

from pipelines.flux_pipeline import FluxPipeline

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Flux Engine Build-Only")
    
    # Required arguments
    parser.add_argument("--prompt", type=str, default="test prompt", help="Test prompt for engine building")
    parser.add_argument("--hf-token", type=str, required=True, help="Hugging Face token")
    
    # Build parameters
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--precision", choices=['bf16', 'fp8', 'fp4'], default='bf16', help="Precision")
    parser.add_argument("--cache-dir", type=str, default="./cache", help="Cache directory")
    parser.add_argument("--cache-mode", choices=['full', 'lean'], default='full', help="Cache mode")
    parser.add_argument("--low-vram", action="store_true", help="Enable low VRAM mode")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    print("[I] Flux Engine Build-Only Script")
    print("[I] This script ONLY builds engines and exits")
    
    # Initialize pipeline - this will trigger engine building
    print("[I] Initializing Flux pipeline (this will build engines)...")
    pipeline = FluxPipeline(
        hf_token=args.hf_token,
        cache_dir=args.cache_dir,
        cache_mode=args.cache_mode,
        low_vram=args.low_vram,
        verbose=args.verbose
    )
    
    # Build engines - this is the key step that creates the TensorRT engines
    print("[I] Building TensorRT engines...")
    jit_times = pipeline.load_engines(
        transformer_precision=args.precision,
        opt_batch_size=args.batch_size,
        opt_height=args.height,
        opt_width=args.width,
        shape_mode="static"
    )
    
    print(f"[I] JIT compilation times:")
    for model, jit_time in jit_times.items():
        print(f"[I]   {model}: {jit_time:.2f}s")
    
    print("[I] Loading resources...")
    pipeline.load_resources(
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
    )
    
    # The pipeline initialization builds engines as needed
    # We don't need to run generate() for inference
    
    print("[I] ✅ Engine building complete!")
    print("[I] Engines cached in:", args.cache_dir)
    print("[I] Exiting without running inference.")
    
    # Exit successfully
    sys.exit(0)

if __name__ == "__main__":
    main()