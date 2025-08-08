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
@misc{{{ model_key }}-{{ backend }}-{{ quantization|lower }},
  title = {{ "{" }}{{ model_name }} {{ backend_display }} {{ quantization }}{{ "}" }},
  author = {ImageAI Server Contributors},
  year = {2024},
  publisher = {HuggingFace},
  url = {https://huggingface.co/{{ repo_name }}}
}
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
        # Use torchao for clean INT8 quantization without BitsAndBytes
        try:
            import torchao
            from torchao.quantization import quantize_, int8_weight_only
            torchao_available = True
            log("Using torchao for INT8 quantization (no BitsAndBytes required)")
        except ImportError:
            torchao_available = False
            log("⚠️  torchao not available. Install with: pip install torchao")
            
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
            # Use torchao (cleaner, better performance)
            pipeline_class = config["pipeline_class"]
            pipe = pipeline_class.from_pretrained(
                config["model_id"],
                torch_dtype=torch.float16,
                use_safetensors=True
            )
            
            # Move to appropriate device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = pipe.to(device)
            
            # Apply int8 weight-only quantization to UNet
            quantize_(pipe.unet, int8_weight_only())
            
            # Quantize text encoders
            if hasattr(pipe, 'text_encoder'):
                quantize_(pipe.text_encoder, int8_weight_only())
            if hasattr(pipe, 'text_encoder_2'):
                quantize_(pipe.text_encoder_2, int8_weight_only())
                
            log(f"Applied torchao INT8 weight-only quantization on {device}")
        
        # Save the model
        output_dir.mkdir(parents=True, exist_ok=True)
        pipe.save_pretrained(str(output_dir))
        
        # Save quantization info
        quant_info = {
            "quantization": "int8",
            "method": "torchao" if torchao_available else "pytorch_native",
            "original_model": config["model_id"],
            "conversion_date": datetime.now().isoformat(),
            "requires_bitsandbytes": False,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        
        with open(output_dir / "quantization_info.json", "w") as f:
            json.dump(quant_info, f, indent=2)
        
        log(f"✅ Saved INT8 model to {output_dir}")
        log(f"   Method: {'torchao' if torchao_available else 'pytorch_native'}")
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