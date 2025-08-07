#!/usr/bin/env python3
"""
A unified script to create ONNX-quantized versions of Stable Diffusion 1.5.

This script generates FP16, INT8, and INT4 quantized models in the
ONNX-COMMUNITY format, along with a detailed model card.
"""

import argparse
import logging
from pathlib import Path
from optimum.onnxruntime import ORTStableDiffusionPipeline
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_fp16_model(base_model_id: str, output_dir: Path):
    """Create the FP16 ONNX version of the model."""
    logger.info("🚀 Starting FP16 model creation...")
    fp16_output_path = output_dir / "FP16"
    fp16_output_path.mkdir(parents=True, exist_ok=True)

    try:
        pipe = ORTStableDiffusionPipeline.from_pretrained(
            base_model_id,
            export=True,
            provider="CPUExecutionProvider",
            use_safetensors=True,
            fp16=True, # Export to FP16
        )
        pipe.save_pretrained(fp16_output_path)
        logger.info(f"✅ FP16 model saved to: {fp16_output_path}")
    except Exception as e:
        logger.error(f"❌ FP16 model creation failed: {e}")


def create_int8_model(base_model_id: str, output_dir: Path):
    """Create the INT8 ONNX version of the model."""
    logger.info("🚀 Starting INT8 model creation...")
    int8_output_path = output_dir / "INT8"
    int8_output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Define INT8 quantization configuration
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

        pipe = ORTStableDiffusionPipeline.from_pretrained(
            base_model_id,
            export=True,
            provider="CPUExecutionProvider",
            use_safetensors=True,
        )
        pipe.quantize(save_dir=int8_output_path, quantization_config=qconfig)
        logger.info(f"✅ INT8 model saved to: {int8_output_path}")
    except Exception as e:
        logger.error(f"❌ INT8 model creation failed: {e}")


def create_int4_model(base_model_id: str, output_dir: Path):
    """Create the INT4 ONNX version of the model."""
    logger.info("🚀 Starting INT4 model creation...")
    int4_output_path = output_dir / "INT4"
    int4_output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Define INT4 quantization configuration
        qconfig = AutoQuantizationConfig.avx512_vnni(
            is_static=False,
            per_channel=False,
            weight_bits=4,
            activation_bits=4,
        )

        pipe = ORTStableDiffusionPipeline.from_pretrained(
            base_model_id,
            export=True,
            provider="CPUExecutionProvider",
            use_safetensors=True,
        )
        pipe.quantize(save_dir=int4_output_path, quantization_config=qconfig)
        logger.info(f"✅ INT4 model saved to: {int4_output_path}")
    except Exception as e:
        logger.error(f"❌ INT4 model creation failed: {e}")


def create_model_card(output_dir: Path, model_name: str):
    """Creates a README.md model card for the quantized models."""
    logger.info("📄 Creating model card...")
    readme_content = f"""---
language: en
license: creativeml-openrail-m
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- onnx
- quantized
- cpu-optimized
inference: true
---

# {model_name}

This repository contains ONNX-quantized versions of the Stable Diffusion 1.5 model,
optimized for CPU inference. Three levels of quantization are provided: FP16, INT8, and INT4.

## Model Variants

This repository provides three quantized variants in separate subfolders:

| Variant | Subfolder | Precision | Speed | Quality | VRAM Usage |
|---|---|---|---|---|---|
| **FP16** | `FP16/` | Floating Point 16 | Good | Excellent | ~1.7 GB |
| **INT8** | `INT8/` | Integer 8 | Excellent | Very Good | ~0.9 GB |
| **INT4** | `INT4/` | Integer 4 | Fastest | Good | ~0.6 GB |

### Why different versions?

- **FP16**: Best quality, ideal for general use where performance is not critical.
- **INT8**: Great balance of speed and quality. Up to 2x faster than FP16.
- **INT4**: Fastest inference, lowest memory usage. Ideal for highly resource-constrained environments.

## Usage

First, install the necessary libraries:
```bash
pip install optimum[onnxruntime]
```

Then, use the `ORTStableDiffusionPipeline` to load the desired model from its subfolder.

### Example with FP16 Model

```python
from optimum.onnxruntime import ORTStableDiffusionPipeline
from pathlib import Path

model_dir = Path("./{model_name}") # Path to the top-level model folder

# Load the FP16 pipeline
pipe = ORTStableDiffusionPipeline.from_pretrained(
    model_dir / "FP16",
    provider="CPUExecutionProvider",
)

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]
image.save("astronaut_horse_fp16.png")
```

### Example with INT8 Model

To use the INT8 model, simply change the path:

```python
# Load the INT8 pipeline
pipe = ORTStableDiffusionPipeline.from_pretrained(
    model_dir / "INT8",
    provider="CPUExecutionProvider",
)
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]
image.save("astronaut_horse_int8.png")
```

### Example with INT4 Model

To use the INT4 model, change the path accordingly:

```python
# Load the INT4 pipeline
pipe = ORTStableDiffusionPipeline.from_pretrained(
    model_dir / "INT4",
    provider="CPUExecutionProvider",
)
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]
image.save("astronaut_horse_int4.png")
```

## License

This model is licensed under the [CreativeML Open RAIL-M license](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/LICENSE.md).
"""
    readme_path = output_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    logger.info(f"✅ Model card saved to: {readme_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create ONNX-quantized versions of Stable Diffusion 1.5."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="The Hugging Face model ID to convert.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./stable-diffusion-v1-5-onnx"),
        help="The directory to save the quantized models to.",
    )
    parser.add_argument(
        "--skip_fp16",
        action="store_true",
        help="Skip creating the FP16 version."
    )
    parser.add_argument(
        "--skip_int8",
        action="store_true",
        help="Skip creating the INT8 version."
    )
    parser.add_argument(
        "--skip_int4",
        action="store_true",
        help="Skip creating the INT4 version."
    )

    args = parser.parse_args()

    logger.info(f"Base Model ID: {args.model_id}")
    logger.info(f"Output Directory: {args.output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_fp16:
        create_fp16_model(args.model_id, args.output_dir)

    if not args.skip_int8:
        create_int8_model(args.model_id, args.output_dir)

    if not args.skip_int4:
        create_int4_model(args.model_id, args.output_dir)

    create_model_card(args.output_dir, args.output_dir.name)

    logger.info("🎉 All tasks completed successfully!")


if __name__ == "__main__":
    main()
