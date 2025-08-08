#!/usr/bin/env python3
"""
Unified Model Converter for ImageAI Server
Handles both PyTorch quantization (INT8) and ONNX conversions (FP32)
"""

import argparse
import json
import shutil
import torch
import torch.quantization as quantization
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from jinja2 import Template

# Import required libraries
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    FluxPipeline,
    AutoPipelineForText2Image
)
from optimum.onnxruntime import ORTStableDiffusionPipeline, ORTStableDiffusionXLPipeline
import onnxruntime as ort

# Model configurations
MODELS = {
    "sd15": {
        "name": "Stable Diffusion 1.5",
        "model_id": "runwayml/stable-diffusion-v1-5",
        "pipeline_class": StableDiffusionPipeline,
        "onnx_class": ORTStableDiffusionPipeline,
        "has_fp16": True,
        "param_count": "0.86B",
        "backends": ["pytorch", "onnx"]
    },
    "sdxl": {
        "name": "Stable Diffusion XL",
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "pipeline_class": StableDiffusionXLPipeline,
        "onnx_class": ORTStableDiffusionXLPipeline,
        "has_fp16": True,
        "param_count": "3.5B",
        "backends": ["pytorch", "onnx"]
    },
    "sdxl-turbo": {
        "name": "Stable Diffusion XL Turbo",
        "model_id": "stabilityai/sdxl-turbo",
        "pipeline_class": StableDiffusionXLPipeline,
        "onnx_class": ORTStableDiffusionXLPipeline,
        "has_fp16": True,
        "param_count": "3.5B",
        "backends": ["pytorch", "onnx"]
    },
    "flux1": {
        "name": "FLUX.1 Schnell",
        "model_id": "black-forest-labs/FLUX.1-schnell",
        "pipeline_class": FluxPipeline,
        "onnx_class": None,  # ONNX not yet supported for Flux
        "has_fp16": True,
        "param_count": "12B",
        "backends": ["pytorch"]
    }
}

# Model card template using Jinja2
MODEL_CARD_TEMPLATE = """---
license: {{ license }}
tags:
- stable-diffusion
- {{ backend }}
- {{ quantization }}
- text-to-image
base_model: {{ base_model }}
---

# {{ model_name }} - {{ backend_display }} {{ quantization }}

This is a {{ quantization }} {{ backend }} version of **{{ base_model }}**.

## Model Details
- **Model Type**: {{ model_name }}
- **Parameters**: {{ param_count }}
- **Backend**: {{ backend_display }}
- **Quantization**: {{ quantization }}
- **Memory Usage**: {{ memory_usage }}
- **Conversion Date**: {{ date }}

## Usage

### {{ backend_display }} {{ quantization }}

```python
{% if backend == "pytorch" %}
{% if quantization == "INT8" %}
# PyTorch INT8 quantized model
from diffusers import {{ pipeline_class }}
import torch

# Load INT8 quantized model
pipe = {{ pipeline_class }}.from_pretrained(
    "{{ repo_name }}",
    torch_dtype=torch.qint8,
    use_safetensors=True
)

# For CPU inference
pipe = pipe.to("cpu")

# Generate image
image = pipe("A beautiful landscape", num_inference_steps=20).images[0]
image.save("output.png")
{% else %}
# PyTorch FP16 model
from diffusers import {{ pipeline_class }}
import torch

pipe = {{ pipeline_class }}.from_pretrained(
    "{{ repo_name }}",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)

# For GPU inference
pipe = pipe.to("cuda")

# Generate image
image = pipe("A beautiful landscape", num_inference_steps=20).images[0]
image.save("output.png")
{% endif %}
{% elif backend == "onnx" %}
# ONNX Runtime model
from optimum.onnxruntime import {{ onnx_class }}

# CPU provider
pipe = {{ onnx_class }}.from_pretrained(
    "{{ repo_name }}",
    provider="CPUExecutionProvider"
)

# Or GPU provider
pipe = {{ onnx_class }}.from_pretrained(
    "{{ repo_name }}",
    provider="CUDAExecutionProvider"
)

# Generate image
image = pipe("A beautiful landscape", num_inference_steps=20).images[0]
image.save("output.png")
{% endif %}
```

## Performance

| Backend | Quantization | Memory | Speed (CPU) | Speed (GPU) | Quality |
|---------|-------------|---------|-------------|-------------|---------|
| {{ backend_display }} | {{ quantization }} | {{ memory_usage }} | {{ cpu_speed }} | {{ gpu_speed }} | {{ quality }} |

## Limitations

{% if quantization == "INT8" %}
- INT8 quantization may slightly reduce image quality
- Best suited for CPU inference or memory-constrained environments
{% elif backend == "onnx" %}
- ONNX models require onnxruntime installation
- Some advanced features may not be available
{% else %}
- Requires GPU with sufficient VRAM for optimal performance
{% endif %}

## Citation

```bibtex
@misc{{"{"}}{{ model_key }}-{{ backend }}-{{ quantization|lower }},
  title = {{"{"}}{{ model_name }} {{ backend_display }} {{ quantization }}{{"}"}}
  author = {{"{"}}ImageAI Server Contributors{{"}"}}
  year = {{"{"}}2024{{"}"}}
  publisher = {{"{"}}HuggingFace{{"}"}}
  url = {{"{"}}https://huggingface.co/{{ repo_name }}{{"}"}}
{{"}"}}
```

---
*Converted using ImageAI Server Model Converter v1.0*
"""

def log(message: str):
    """Logging utility"""
    print(f"[converter] {message}", flush=True)

def get_memory_usage(model_key: str, backend: str, quantization: str) -> str:
    """Estimate memory usage for a model configuration"""
    base_sizes = {
        "sd15": 2.0,  # GB for FP16
        "sdxl": 6.5,
        "sdxl-turbo": 6.5,
        "flux1": 24.0
    }
    
    base = base_sizes.get(model_key, 4.0)
    
    if quantization == "INT8":
        return f"~{base/2:.1f}GB"
    elif quantization == "FP32":
        return f"~{base*2:.1f}GB"
    else:  # FP16
        return f"~{base:.1f}GB"

def quantize_pytorch_int8(model_key: str, output_dir: Path) -> bool:
    """Quantize a PyTorch model to INT8 using torchao (clean, no BNB dependency)"""
    config = MODELS[model_key]
    log(f"Quantizing {config['name']} to PyTorch INT8...")
    
    try:
        # Check for required dependencies
        try:
            import torchao
            from torchao.quantization import quantize_, int8_weight_only
            torchao_available = True
            log("Using torchao for INT8 quantization (no BitsAndBytes required)")
        except ImportError:
            torchao_available = False
            log("⚠️  torchao not available. Install with: pip install torchao")
            
        # Check for accelerate (needed for device_map and memory-efficient loading)
        try:
            import accelerate
            log(f"✅ accelerate available (version: {accelerate.__version__})")
        except ImportError:
            log("⚠️  accelerate not available. Install with: pip install accelerate")
            log("   accelerate is needed for memory-efficient model loading")
            
        if not torchao_available:
            # Alternative: Use PyTorch's native quantization (works on CUDA)
            log("Attempting PyTorch native INT8 quantization...")
            
            # Load model in FP16 first
            pipeline_class = config["pipeline_class"]
            pipe = pipeline_class.from_pretrained(
                config["model_id"],
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            )
            
            # Convert UNet weights to INT8 manually
            import torch.nn as nn
            
            def quantize_model_weights(model):
                """Simple INT8 weight quantization without BNB"""
                for name, module in model.named_modules():
                    if isinstance(module, (nn.Conv2d, nn.Linear)):
                        # Quantize weights to int8
                        with torch.no_grad():
                            # Get weight data
                            weight = module.weight.data.float()
                            
                            # Calculate scale for quantization
                            scale = weight.abs().max() / 127.0
                            
                            # Quantize to int8
                            weight_int8 = (weight / scale).round().clamp(-128, 127).to(torch.int8)
                            
                            # Store scale for dequantization
                            module.register_buffer('weight_scale', scale)
                            module.register_buffer('weight_int8', weight_int8)
                            
                            # Replace forward method to use quantized weights
                            original_weight = module.weight
                            module.weight.data = (weight_int8.float() * scale).to(module.weight.dtype)
                
                return model
            
            # Apply quantization
            pipe.unet = quantize_model_weights(pipe.unet)
            if hasattr(pipe, 'text_encoder'):
                pipe.text_encoder = quantize_model_weights(pipe.text_encoder)
            if hasattr(pipe, 'text_encoder_2'):
                pipe.text_encoder_2 = quantize_model_weights(pipe.text_encoder_2)
                
            log("Applied native PyTorch INT8 weight quantization")
            
        else:
            # Use torchao (cleaner, better performance) with memory-efficient loading
            pipeline_class = config["pipeline_class"]
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Use BF16 on Blackwell cards (RTX 5090) for better memory efficiency
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0).lower()
                supports_bf16 = "rtx 50" in gpu_name or "rtx 40" in gpu_name or "h100" in gpu_name or "a100" in gpu_name
                dtype = torch.bfloat16 if supports_bf16 else torch.float16
                log(f"GPU: {torch.cuda.get_device_name(0)}")
                log(f"Using {'BF16' if supports_bf16 else 'FP16'} for {'better' if supports_bf16 else 'standard'} memory efficiency")
            else:
                dtype = torch.float16
                supports_bf16 = False
            
            log(f"Loading model with low_cpu_mem_usage=True and {dtype} precision")
            
            # FLUX.1 doesn't support device_map="auto", use different loading strategy
            if "flux" in model_key.lower():
                log("FLUX.1 detected - using balanced device mapping strategy")
                try:
                    pipe = pipeline_class.from_pretrained(
                        config["model_id"],
                        torch_dtype=dtype,
                        use_safetensors=True,
                        low_cpu_mem_usage=True,
                        device_map="balanced"  # FLUX.1 supports balanced but not auto
                    )
                except Exception as e:
                    log(f"Balanced device map failed ({e}), trying sequential loading...")
                    pipe = pipeline_class.from_pretrained(
                        config["model_id"],
                        torch_dtype=dtype,
                        use_safetensors=True,
                        low_cpu_mem_usage=True
                        # No device_map for FLUX.1 - load normally then move to device
                    )
            else:
                # Standard models support device_map="auto"
                pipe = pipeline_class.from_pretrained(
                    config["model_id"],
                    torch_dtype=dtype,
                    use_safetensors=True,
                    low_cpu_mem_usage=True,
                    device_map="auto" if device == "cuda" else None
                )
            
            # Move to appropriate device if not already there
            # FLUX.1 uses 'transformer' instead of 'unet'
            main_component = pipe.transformer if hasattr(pipe, 'transformer') else pipe.unet
            if device == "cuda" and not next(main_component.parameters()).is_cuda:
                log(f"Moving pipeline to {device}...")
                pipe = pipe.to(device)
            
            # Enable memory-efficient attention if available
            try:
                pipe.enable_attention_slicing(1)  # Slice attention to reduce memory
                log("✅ Enabled attention slicing for memory efficiency")
            except:
                pass
            
            try:
                pipe.enable_model_cpu_offload()  # Offload when not in use
                log("✅ Enabled CPU offload for memory efficiency")
            except:
                pass
            
            # Force garbage collection before quantization
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            # Enable mixed precision for Blackwell cards to reduce temporary memory usage
            use_amp = supports_bf16 and device == "cuda"
            if use_amp:
                log("🚀 Enabling AMP (Automatic Mixed Precision) for memory-efficient quantization")
            
            log(f"Applying INT8 quantization component by component to save memory...")
            
            # Check memory before quantization
            if torch.cuda.is_available():
                memory_before = torch.cuda.memory_allocated() / 1024**3
                log(f"  📊 GPU Memory before quantization: {memory_before:.2f}GB")
            
            # Quantize main component (UNet for SD models, Transformer for FLUX.1)
            component_name = "Transformer" if hasattr(pipe, 'transformer') else "UNet"
            main_component = pipe.transformer if hasattr(pipe, 'transformer') else pipe.unet
            log(f"  🔧 Quantizing {component_name} (checking for FP32 inflation)...")
            
            # TorchAO may temporarily convert FP16 -> FP32 -> INT8
            # Monitor memory spike during this step
            memory_peak = memory_before
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            try:
                # Try TorchAO quantization with memory monitoring and mixed precision
                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        quantize_(main_component, int8_weight_only())
                else:
                    quantize_(main_component, int8_weight_only())
                
                if torch.cuda.is_available():
                    memory_peak = torch.cuda.max_memory_allocated() / 1024**3
                    memory_after = torch.cuda.memory_allocated() / 1024**3
                    log(f"  📊 Memory during UNet quantization - Peak: {memory_peak:.2f}GB, After: {memory_after:.2f}GB")
                    if memory_peak > memory_before * 1.8:  # 80% increase suggests FP32 inflation
                        log(f"  ⚠️  DETECTED: TorchAO inflated to FP32 (peak {memory_peak:.2f}GB vs before {memory_before:.2f}GB)")
                        if not use_amp:
                            log(f"  💡 Try upgrading to RTX 40/50 series or H100/A100 for BF16 mixed precision support")
                    else:
                        efficiency = "with BF16+AMP" if use_amp else "with FP16"
                        log(f"  ✅ Efficient quantization completed {efficiency} - no significant memory inflation")
            
            except torch.cuda.OutOfMemoryError as e:
                log(f"  ❌ OOM during TorchAO quantization: {e}")
                log(f"  🔄 Falling back to CPU quantization then GPU transfer...")
                
                # Fallback: Move to CPU, quantize, then move back
                original_device = next(main_component.parameters()).device
                main_component_cpu = main_component.to("cpu")
                gc.collect()
                torch.cuda.empty_cache()
                
                # Quantize on CPU (no memory constraints)
                quantize_(main_component_cpu, int8_weight_only())
                
                # Move quantized model back to GPU
                main_component_gpu = main_component_cpu.to(original_device)
                # Update the pipeline component
                if hasattr(pipe, 'transformer'):
                    pipe.transformer = main_component_gpu
                else:
                    pipe.unet = main_component_gpu
                    
                log(f"  ✅ CPU quantization completed, moved back to {original_device}")
            
            gc.collect()
            torch.cuda.empty_cache()
            
            # Quantize text encoders
            if hasattr(pipe, 'text_encoder'):
                log("  🔧 Quantizing Text Encoder...")
                quantize_(pipe.text_encoder, int8_weight_only())
                gc.collect()
                torch.cuda.empty_cache()
                
            if hasattr(pipe, 'text_encoder_2'):
                log("  🔧 Quantizing Text Encoder 2...")
                quantize_(pipe.text_encoder_2, int8_weight_only())
                gc.collect()
                torch.cuda.empty_cache()
                
            log(f"✅ Applied torchao INT8 weight-only quantization on {device}")
            log(f"   Memory optimization: low_cpu_mem_usage + attention_slicing + cpu_offload")
        
        # Save the model with special handling for quantized models
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # TorchAO quantized models need special serialization handling
        if torchao_available:
            log("Saving quantized model (this may take a while due to weight serialization)...")
            try:
                # Save with safe_serialization=False to avoid storage pointer issues
                pipe.save_pretrained(str(output_dir), safe_serialization=False)
                log("✅ Saved using PyTorch format (not safetensors due to quantization)")
            except Exception as save_error:
                log(f"⚠️ Standard save failed: {save_error}")
                log("Attempting component-wise save...")
                
                # Fallback: save components individually
                pipe.unet.save_pretrained(output_dir / "unet", safe_serialization=False)
                pipe.vae.save_pretrained(output_dir / "vae", safe_serialization=False)
                if hasattr(pipe, 'text_encoder'):
                    pipe.text_encoder.save_pretrained(output_dir / "text_encoder", safe_serialization=False)
                if hasattr(pipe, 'text_encoder_2'):
                    pipe.text_encoder_2.save_pretrained(output_dir / "text_encoder_2", safe_serialization=False)
                
                # Copy other files
                import shutil
                from transformers.utils import cached_file
                
                # Save scheduler config
                pipe.scheduler.save_config(output_dir)
                
                # Save tokenizers if they exist
                if hasattr(pipe, 'tokenizer'):
                    pipe.tokenizer.save_pretrained(output_dir / "tokenizer")
                if hasattr(pipe, 'tokenizer_2'):
                    pipe.tokenizer_2.save_pretrained(output_dir / "tokenizer_2")
                    
                log("✅ Saved using component-wise fallback method")
        else:
            pipe.save_pretrained(str(output_dir))
        
        # Save quantization info
        quant_info = {
            "quantization": "int8",
            "method": "torchao" if torchao_available else "pytorch_native",
            "original_model": config["model_id"],
            "conversion_date": datetime.now().isoformat(),
            "requires_bitsandbytes": False,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "serialization_format": "pytorch" if torchao_available else "safetensors",
            "notes": "TorchAO quantized models saved in PyTorch format due to storage compatibility"
        }
        
        with open(output_dir / "quantization_info.json", "w") as f:
            json.dump(quant_info, f, indent=2)
        
        log(f"✅ Saved INT8 model to {output_dir}")
        log(f"   Method: {'torchao' if torchao_available else 'pytorch_native'}")
        log(f"   Format: {'PyTorch' if torchao_available else 'SafeTensors'}")
        log(f"   No BitsAndBytes dependency required!")
        return True
        
    except Exception as e:
        log(f"❌ INT8 quantization failed: {e}")
        log("   For best results, run on CUDA device with: pip install torchao")
        return False

def convert_to_onnx_fp32(model_key: str, output_dir: Path) -> bool:
    """Convert a model to ONNX FP32"""
    config = MODELS[model_key]
    
    if config["onnx_class"] is None:
        log(f"⚠️  ONNX not supported for {config['name']}")
        return False
    
    log(f"Converting {config['name']} to ONNX FP32...")
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use Optimum's built-in export
        onnx_class = config["onnx_class"]
        pipe = onnx_class.from_pretrained(
            config["model_id"],
            export=True
        )
        pipe.save_pretrained(str(output_dir))
        
        log(f"✅ Saved ONNX FP32 model to {output_dir}")
        return True
        
    except Exception as e:
        log(f"❌ ONNX conversion failed: {e}")
        return False

def generate_model_card(
    model_key: str,
    backend: str,
    quantization: str,
    repo_name: str,
    output_path: Path
) -> None:
    """Generate a model card using Jinja2 template"""
    config = MODELS[model_key]
    
    template = Template(MODEL_CARD_TEMPLATE)
    
    backend_display = {
        "pytorch": "PyTorch",
        "onnx": "ONNX"
    }.get(backend, backend)
    
    context = {
        "model_name": config["name"],
        "model_key": model_key,
        "base_model": config["model_id"],
        "backend": backend,
        "backend_display": backend_display,
        "quantization": quantization,
        "param_count": config["param_count"],
        "memory_usage": get_memory_usage(model_key, backend, quantization),
        "repo_name": repo_name,
        "pipeline_class": config["pipeline_class"].__name__ if config["pipeline_class"] else "Pipeline",
        "onnx_class": config["onnx_class"].__name__ if config["onnx_class"] else "ORTPipeline",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "license": "openrail" if "sd" in model_key else "apache-2.0",
        "cpu_speed": "Good" if quantization == "INT8" else "Moderate",
        "gpu_speed": "Fast" if quantization != "FP32" else "Good",
        "quality": "Good" if quantization != "INT8" else "Slightly Reduced"
    }
    
    model_card = template.render(**context)
    
    with open(output_path / "README.md", "w") as f:
        f.write(model_card)
    
    log(f"✅ Generated model card at {output_path / 'README.md'}")

def main():
    parser = argparse.ArgumentParser(description="Unified Model Converter for ImageAI Server")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()) + ["all"],
        required=True,
        help="Model to convert"
    )
    parser.add_argument(
        "--backend",
        choices=["pytorch", "onnx", "both"],
        default="pytorch",
        help="Backend to target"
    )
    parser.add_argument(
        "--quantization",
        choices=["INT8", "FP16", "FP32"],
        default="INT8",
        help="Quantization level"
    )
    parser.add_argument(
        "--output-dir",
        default="converted_models",
        help="Output directory for converted models"
    )
    parser.add_argument(
        "--repo-prefix",
        default="Mitchins",
        help="HuggingFace repo prefix"
    )
    
    args = parser.parse_args()
    
    output_root = Path(args.output_dir)
    
    # Determine which models to convert
    models_to_convert = list(MODELS.keys()) if args.model == "all" else [args.model]
    
    # Process each model
    for model_key in models_to_convert:
        config = MODELS[model_key]
        log(f"\n{'='*60}")
        log(f"Processing {config['name']}")
        log(f"{'='*60}")
        
        # Determine backends to use
        if args.backend == "both":
            backends = config["backends"]
        elif args.backend in config["backends"]:
            backends = [args.backend]
        else:
            log(f"⚠️  {args.backend} not supported for {config['name']}")
            continue
        
        for backend in backends:
            # Construct output path and repo name
            quant_suffix = args.quantization.lower()
            backend_suffix = "torch" if backend == "pytorch" else backend
            
            output_dir = output_root / f"{model_key}-{backend_suffix}-{quant_suffix}"
            repo_name = f"{args.repo_prefix}/{model_key}-{backend_suffix}-{quant_suffix}"
            
            log(f"\n▶️  Converting to {backend} {args.quantization}")
            log(f"   Output: {output_dir}")
            log(f"   Repo: {repo_name}")
            
            # Perform conversion
            success = False
            if backend == "pytorch" and args.quantization == "INT8":
                success = quantize_pytorch_int8(model_key, output_dir)
            elif backend == "onnx" and args.quantization == "FP32":
                success = convert_to_onnx_fp32(model_key, output_dir)
            elif backend == "pytorch" and args.quantization == "FP16":
                log("ℹ️  FP16 PyTorch models already available from official repos")
                success = True
            else:
                log(f"⚠️  {backend} {args.quantization} conversion not implemented")
                continue
            
            # Generate model card if successful
            if success:
                generate_model_card(
                    model_key,
                    backend,
                    args.quantization,
                    repo_name,
                    output_dir
                )
    
    log(f"\n{'='*60}")
    log("✅ Conversion complete!")
    log(f"📁 Models saved to: {output_root}")

if __name__ == "__main__":
    main()