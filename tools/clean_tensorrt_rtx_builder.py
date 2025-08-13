#!/usr/bin/env python3
"""
Clean TensorRT-RTX Engine Builder

Self-contained, clean engine builder using our own TensorRT-RTX builder classes.
No external dependencies on scattered demo directories.
"""

import os
import sys
import logging
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any

# Add our builders to path
sys.path.insert(0, str(Path(__file__).parent))

from tensorrt_rtx_builders.sdxl_builder import SDXLTensorRTRTXBuilder
from tensorrt_rtx_builders.flux_builder import FluxTensorRTRTXBuilder
from tensorrt_rtx_builders.base_builder import setup_tensorrt_rtx_env

# Setup environment first
setup_tensorrt_rtx_env()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_gpu_architecture():
    """Detect current GPU architecture and capabilities."""
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        device = torch.cuda.get_device_properties(0)
        compute_major = device.major
        compute_minor = device.minor
        compute_capability = f"{compute_major}.{compute_minor}"
        
        # Determine architecture
        if compute_major == 8:
            if compute_minor >= 9:
                arch = "ada"
            else:
                arch = "ampere" 
        elif compute_major >= 9:
            arch = "blackwell"
        else:
            arch = "legacy"
        
        return {
            "name": device.name,
            "compute_capability": compute_capability,
            "architecture": arch,
            "compute_major": compute_major,
            "compute_minor": compute_minor
        }
        
    except Exception as e:
        logger.warning(f"Could not detect GPU capabilities: {e}")
        return {
            "name": "Unknown GPU",
            "compute_capability": "unknown",
            "architecture": "ampere",  # Conservative fallback
            "compute_major": 8,
            "compute_minor": 6
        }

def build_all_variants(output_dir: str, 
                      low_vram: bool = False,
                      models: List[str] = None) -> Dict[str, Any]:
    """Build all TensorRT-RTX engine variants."""
    
    logger.info("🚀 Starting Clean TensorRT-RTX Engine Build")
    logger.info("=" * 80)
    
    # Detect GPU
    gpu_info = detect_gpu_architecture()
    logger.info(f"GPU: {gpu_info['name']} (Compute {gpu_info['compute_capability']})")
    logger.info(f"Architecture: {gpu_info['architecture']}")
    
    # Determine what to build
    models_to_build = models or ["sdxl", "flux1-dev", "flux1-schnell"]
    
    results = {
        "gpu_info": gpu_info,
        "builds": {},
        "summary": {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "total_size_gb": 0.0
        }
    }
    
    # Build each model variant
    for model in models_to_build:
        logger.info(f"\n📦 Building {model.upper()}")
        logger.info("-" * 40)
        
        try:
            if model == "sdxl":
                builder = SDXLTensorRTRTXBuilder(
                    output_dir=Path(output_dir) / f"{model}-bf16-{gpu_info['architecture']}",
                    low_vram=low_vram
                )
            elif model in ["flux1-dev", "flux1-schnell"]:
                builder = FluxTensorRTRTXBuilder(
                    model_variant=model,
                    output_dir=Path(output_dir) / f"{model}-bf16-{gpu_info['architecture']}",
                    low_vram=low_vram
                )
            else:
                logger.error(f"Unknown model: {model}")
                continue
            
            # Build engines (only BF16 for current GPU)
            start_time = time.time()
            success = builder.build_engines("bf16")
            build_time = time.time() - start_time
            
            if success:
                size_gb = builder.get_total_size_gb()
                logger.info(f"✅ {model} completed successfully in {build_time:.1f}s ({size_gb:.1f}GB)")
                
                results["builds"][model] = {
                    "status": "success",
                    "build_time_seconds": build_time,
                    "size_gb": size_gb,
                    "quantization": "bf16",
                    "architecture": gpu_info["architecture"]
                }
                results["summary"]["successful"] += 1
                results["summary"]["total_size_gb"] += size_gb
            else:
                logger.error(f"❌ {model} failed")
                results["builds"][model] = {
                    "status": "failed",
                    "build_time_seconds": build_time,
                    "quantization": "bf16",
                    "architecture": gpu_info["architecture"]
                }
                results["summary"]["failed"] += 1
            
            results["summary"]["total"] += 1
            
        except Exception as e:
            logger.error(f"❌ {model} failed with exception: {e}")
            results["builds"][model] = {
                "status": "error",
                "error": str(e),
                "quantization": "bf16",
                "architecture": gpu_info["architecture"]
            }
            results["summary"]["failed"] += 1
            results["summary"]["total"] += 1
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("BUILD SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total variants: {results['summary']['total']}")
    logger.info(f"Successful: {results['summary']['successful']}")
    logger.info(f"Failed: {results['summary']['failed']}")
    logger.info(f"Total size: {results['summary']['total_size_gb']:.1f} GB")
    
    for model, result in results["builds"].items():
        status = result["status"]
        if status == "success":
            logger.info(f"  ✅ {model}: {result['size_gb']:.1f}GB")
        else:
            logger.info(f"  ❌ {model}: {status}")
    
    return results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Clean TensorRT-RTX Engine Builder")
    parser.add_argument("--output-dir", default="/data/imagai/tensorrt_complete_engines",
                       help="Output directory for engines")
    parser.add_argument("--hf-token", 
                       help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--models", nargs="+", 
                       choices=["sdxl", "flux1-dev", "flux1-schnell"],
                       help="Models to build (default: all)")
    parser.add_argument("--low-vram", action="store_true",
                       help="Use low VRAM mode")
    
    args = parser.parse_args()
    
    # Build all variants
    results = build_all_variants(
        output_dir=args.output_dir,
        low_vram=args.low_vram,
        models=args.models
    )
    
    # Save results
    import json
    results_file = Path(args.output_dir) / "clean_build_report.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_file}")
    
    # Exit with appropriate code
    if results["summary"]["failed"] > 0:
        logger.error("Some builds failed")
        sys.exit(1)
    else:
        logger.info("All builds successful!")
        sys.exit(0)

if __name__ == "__main__":
    main()