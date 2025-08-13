# TensorRT-RTX Tools

This directory contains tools for managing the complete TensorRT-RTX engine lifecycle for diffusion models.

## Overview

The TensorRT-RTX integration provides 3-5x faster inference compared to PyTorch for models like SDXL, Flux, and SD3.5. These tools help you:

1. **Build** optimized engines for different GPU architectures
2. **Test** engines locally before distribution  
3. **Upload** engines to HuggingFace for easy distribution
4. **Validate** downloaded engines work correctly

## Supported Configurations

### Models
- **SDXL**: Stable Diffusion XL (primary support)
- **Flux1-Dev**: FLUX.1 development model
- **Flux1-Schnell**: FLUX.1 fast model
- **SD3.5**: Stable Diffusion 3.5
- **SDVideo**: Stable Video Diffusion

### GPU Architectures & Quantizations
- **Ampere** (RTX 3090, A100): BF16
- **Ada Lovelace** (RTX 4090): FP8, BF16
- **Blackwell** (H200): FP4, FP8, BF16

This gives us **6 engine variants per model** for comprehensive GPU support.

## Tools

### 1. `build_tensorrt_engines.py` - Engine Builder

Builds TensorRT-RTX engines for multiple models and configurations.

```bash
# Build all SDXL variants for current GPU
python build_tensorrt_engines.py --models sdxl

# Build specific configuration
python build_tensorrt_engines.py --single sdxl ampere bf16

# Build for specific architecture
python build_tensorrt_engines.py --models sdxl --architecture ada

# Force rebuild existing engines
python build_tensorrt_engines.py --models sdxl --force-rebuild
```

**Output Structure:**
```
tensorrt_engines/
└── engines/
    ├── sdxl-bf16-ampere/
    │   ├── engines/
    │   │   ├── clip.plan
    │   │   ├── clip2.plan
    │   │   ├── unetxl.plan
    │   │   └── vae.plan
    │   ├── framework/
    │   └── onnx/
    └── sdxl-fp8-ada/
        └── ...
```

### 2. `upload_engines_to_hf.py` - HuggingFace Uploader

Uploads built engines to HuggingFace Hub for distribution.

```bash
# Upload all built engines
python upload_engines_to_hf.py --engines-dir ./tensorrt_engines --hf-token $HF_TOKEN

# Upload specific models
python upload_engines_to_hf.py --engines-dir ./tensorrt_engines --models sdxl --hf-token $HF_TOKEN

# Test upload process (dry run)
python upload_engines_to_hf.py --engines-dir ./tensorrt_engines --dry-run
```

**HuggingFace Structure:**
```
imgailab/tensorrt-rtx-sdxl/
├── README.md
├── sdxl-bf16-ampere/
│   ├── engines/
│   ├── config.json
│   └── README.md
├── sdxl-fp8-ada/
└── ...
```

### 3. `tensorrt_workflow.py` - Complete Workflow Manager

Orchestrates the full engine lifecycle: build → test → upload.

```bash
# Full workflow for SDXL
python tensorrt_workflow.py full-workflow --models sdxl --hf-token $HF_TOKEN

# Build only
python tensorrt_workflow.py build --models sdxl flux1-dev

# Upload existing engines
python tensorrt_workflow.py upload --models sdxl --hf-token $HF_TOKEN

# Validate downloaded engines
python tensorrt_workflow.py validate sdxl sdxl-bf16-ampere
```

### 4. Legacy Tools

- `legacy_build_sdxl.py`: Original SDXL-only builder (kept for reference)
- `model_converter.py`: General model conversion utilities

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install huggingface_hub torch

# Set environment variables
export HF_TOKEN="your_huggingface_token"
export LD_LIBRARY_PATH="/data/nvidia/TensorRT-RTX-1.0.0.21/targets/x86_64-linux-gnu/lib:$LD_LIBRARY_PATH"
export POLYGRAPHY_USE_TENSORRT_RTX=1
```

### Basic Workflow

1. **Build engines for your GPU:**
   ```bash
   cd /data/imagai/tools
   python build_tensorrt_engines.py --models sdxl
   ```

2. **Test the engines:**
   ```bash
   python tensorrt_workflow.py test --models sdxl
   ```

3. **Upload to HuggingFace (optional):**
   ```bash
   python upload_engines_to_hf.py --engines-dir ./tensorrt_engines --hf-token $HF_TOKEN
   ```

### Production Workflow

For building and distributing engines for all architectures:

```bash
# Build Ampere variants (on Ampere GPU)
python tensorrt_workflow.py full-workflow --models sdxl --architecture ampere --hf-token $HF_TOKEN

# Build Ada variants (on Ada GPU)  
python tensorrt_workflow.py full-workflow --models sdxl --architecture ada --hf-token $HF_TOKEN

# Build Blackwell variants (on Blackwell GPU)
python tensorrt_workflow.py full-workflow --models sdxl --architecture blackwell --hf-token $HF_TOKEN
```

## Integration with ImageAI Server

The ImageAI server automatically detects and uses these engines:

1. **Local engines**: Engines in `./tensorrt_engines/` are detected automatically
2. **HuggingFace engines**: Downloaded automatically when needed
3. **Model selection**: Use model IDs like `sdxl-tensorrt_rtx:bf16` in the web UI

Example API usage:
```python
# Engines are used automatically when available
curl -X POST "http://localhost:8001/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful landscape",
    "model": "sdxl-tensorrt_rtx:bf16",
    "width": 1024,
    "height": 1024
  }'
```

## Performance Expectations

| GPU | Quantization | Memory | Speed (1024x1024) |
|-----|-------------|--------|--------------------|
| RTX 3090 (Ampere) | BF16 | ~4GB | ~5.7s |
| RTX 4090 (Ada) | FP8 | ~2GB | ~3.2s |
| RTX 4090 (Ada) | BF16 | ~4GB | ~4.1s |
| H200 (Blackwell) | FP4 | ~1GB | ~1.8s |
| H200 (Blackwell) | FP8 | ~2GB | ~2.1s |

*(Benchmarks with 30 inference steps, may vary by specific hardware)*

## Troubleshooting

### Common Issues

1. **CUDA OOM**: Use lower quantization (FP8 → BF16 → FP4)
2. **Missing dependencies**: Install `tensorrt`, `cuda-python`, `huggingface_hub`
3. **Build failures**: Check GPU compute capability (8.0+ required)
4. **Upload failures**: Verify HuggingFace token permissions

### Debug Mode

Add `--verbose` to any command for detailed logging:
```bash
python build_tensorrt_engines.py --models sdxl --verbose
```

### Engine Validation

Test engines before upload:
```bash
python tensorrt_workflow.py test --models sdxl
```

## Contributing

When adding support for new models:

1. Add model configuration to `ModelConfiguration.CONFIGURATIONS`
2. Update the NVIDIA demo script mapping
3. Test with all supported quantizations
4. Update this README

## Future Improvements

- [ ] Automatic engine optimization for different batch sizes
- [ ] Multi-resolution engine variants
- [ ] Automatic model architecture detection
- [ ] Integration with model training pipelines
- [ ] Performance benchmarking automation