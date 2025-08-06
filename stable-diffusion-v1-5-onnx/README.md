---
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

# stable-diffusion-v1-5-onnx

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

model_dir = Path("./stable-diffusion-v1-5-onnx") # Path to the top-level model folder

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
