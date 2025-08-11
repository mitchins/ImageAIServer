# Model Conversion Guide

The `model_converter.py` script is the official tool for converting models for ImageAI Server.

## Features

- **PyTorch INT8 Quantization** using TorchAO (no BitsAndBytes required)
- **ONNX FP32 Conversion** for CPU inference
- **Automatic Model Card Generation** with proper documentation
- **Batch Conversion** support for all models
- **Clean Dependencies** - no messy third-party libraries

## Installation

### For PyTorch INT8 Quantization (CUDA Required)
```bash
pip install torchao torch diffusers transformers accelerate
```

### For ONNX Conversion
```bash
pip install optimum[onnxruntime] onnx onnxruntime
```

## Usage

### Convert Single Model to INT8
```bash
python model_converter.py --model sd15 --backend pytorch --quantization INT8
```

### Convert All Models to INT8
```bash
python model_converter.py --model all --backend pytorch --quantization INT8
```

### Convert to ONNX FP32
```bash
python model_converter.py --model sd15 --backend onnx --quantization FP32
```

### Convert Everything
```bash
# All models, both backends
python model_converter.py --model all --backend both --quantization INT8
python model_converter.py --model all --backend both --quantization FP32
```

## Supported Models

| Model Key | Model Name | PyTorch | ONNX |
|-----------|------------|---------|------|
| sd15 | Stable Diffusion 1.5 | ✅ | ✅ |
| sdxl | Stable Diffusion XL | ✅ | ✅ |
| sdxl-turbo | Stable Diffusion XL Turbo | ✅ | ✅ |
| flux1 | FLUX.1 Schnell | ✅ | ❌ |

## Supported Quantizations

| Backend | Quantizations | Best For |
|---------|---------------|----------|
| PyTorch | FP16 (official), INT8 | GPU inference |
| ONNX | FP32 | CPU inference |

## Output Structure

```
converted_models/
├── sd15-torch-int8/
│   ├── unet/
│   ├── vae/
│   ├── text_encoder/
│   ├── quantization_info.json
│   └── README.md  # Auto-generated model card
├── sd15-onnx-fp32/
│   └── ...
```

## Uploading to HuggingFace

After conversion, upload to your controlled repos:

```bash
cd converted_models/sd15-torch-int8
huggingface-cli upload imgailab/sd15-torch-int8 . --repo-type model
```

## Model Card Generation

Each converted model includes a comprehensive README.md with:
- Original upstream model reference
- Memory requirements
- Usage examples
- Performance characteristics
- Limitations and fitness for purpose

## Notes

- **INT8 Quantization** works best on CUDA devices
- **TorchAO** is used instead of BitsAndBytes for cleaner dependencies
- **ONNX models** are always FP32 for maximum compatibility
- **Model cards** use Jinja2 templates for consistency

## Troubleshooting

### INT8 Conversion Fails
- Ensure you have a CUDA device
- Install torchao: `pip install torchao`
- Run on a machine with GPU support

### ONNX Conversion Fails
- Ensure optimum is installed: `pip install optimum[onnxruntime]`
- Check disk space (ONNX models can be large)

### Out of Memory
- Use a machine with more RAM/VRAM
- Convert one model at a time
- Close other applications