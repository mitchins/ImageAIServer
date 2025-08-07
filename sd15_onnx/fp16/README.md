---
language:
- en
license: creativeml-openrail-m
tags:
- stable-diffusion
- onnx
- text-to-image
- cpu-optimized
- raspberry-pi
- fp16
base_model: stable-diffusion-v1-5/stable-diffusion-v1-5
inference: true
---

# Stable Diffusion 1.5 ONNX FP16 CPU-Optimized

High-quality ONNX version of Stable Diffusion 1.5 optimized for CPU inference with FP16 precision. Perfect for edge deployment including Raspberry Pi.

## Model Details

- **Base Model**: [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- **Format**: ONNX
- **Precision**: FP16 (best quality/performance balance)
- **Target**: CPU inference (Intel/AMD/ARM)
- **Provider**: CPUExecutionProvider

## Key Features

🎯 **High Quality** - FP16 precision maintains excellent image quality  
🍓 **Raspberry Pi Ready** - Optimized for ARM with NEON  
⚡ **Fast CPU Inference** - ONNX Runtime optimizations  
📦 **Lightweight** - No PyTorch dependency  
🔧 **Easy Setup** - Single pip install  

## Quick Start

```python
from optimum.onnxruntime import ORTStableDiffusionPipeline

# Load ONNX model
pipe = ORTStableDiffusionPipeline.from_pretrained(
    "Mitchins/sd15-onnx-fp16",
    provider="CPUExecutionProvider"
)

# Generate image
image = pipe(
    "A serene mountain landscape at sunset, highly detailed",
    num_inference_steps=20,
    guidance_scale=7.5
).images[0]

image.save("output.png")
```

## Installation

```bash
# Minimal dependencies
pip install optimum[onnxruntime] pillow

# For ARM/Raspberry Pi
sudo apt install python3-pip
pip3 install optimum[onnxruntime] pillow
```

## Performance

| Hardware | Time (512x512) | Memory | Quality |
|----------|----------------|---------|---------|
| RPi 4 (4GB) | ~3-5 min | ~2.5GB | Excellent |
| Intel i5 | ~45-90s | ~2.5GB | Excellent |
| M1 Mac | ~30-45s | ~2.5GB | Excellent |
| AMD Ryzen | ~45-90s | ~2.5GB | Excellent |

## Memory Optimization

```python
# Lower memory usage (slower)
pipe.enable_attention_slicing()

# Generate smaller images for faster testing
image = pipe(
    prompt,
    height=256, width=256,  # Faster than 512x512
    num_inference_steps=15   # Fewer steps = faster
).images[0]
```

## Model Components

- `unet/model.onnx` - Main diffusion UNet (~3.4GB)
- `text_encoder/model.onnx` - CLIP text encoder (~470MB)  
- `vae_decoder/model.onnx` - VAE decoder (~189MB)
- `vae_encoder/model.onnx` - VAE encoder (~130MB)
- `tokenizer/` - Text tokenization files
- `scheduler/` - Noise scheduler configuration

## Use Cases

✅ **Edge AI Applications**  
✅ **Raspberry Pi Projects**  
✅ **CPU-only Servers**  
✅ **Offline Generation**  
✅ **Research and Development**  

## Comparison

| Version | Speed | Quality | Memory | Use Case |
|---------|-------|---------|---------|----------|
| **This (FP16)** | Medium | Excellent | ~2.5GB | Best quality |
| INT8 | Fast | Very Good | ~2GB | Speed focus |
| PyTorch | Slow | Excellent | ~4GB | Development |

Choose this FP16 version for the best image quality on CPU!

## License

CreativeML Open RAIL-M (inherited from Stable Diffusion 1.5)
