# SDXL-Turbo CoreML

This repository contains CoreML models for SDXL-Turbo converted using Apple's official `python_coreml_stable_diffusion` tools.

## Models

- `Stable_Diffusion_version_stabilityai_sdxl-turbo_unet.mlpackage` - UNet model
- `Stable_Diffusion_version_stabilityai_sdxl-turbo_vae_decoder.mlpackage` - VAE decoder
- `Stable_Diffusion_version_stabilityai_sdxl-turbo_text_encoder.mlpackage` - Text encoder
- `Stable_Diffusion_version_stabilityai_sdxl-turbo_text_encoder_2.mlpackage` - Text encoder 2

## Usage

These models are optimized for Apple Silicon devices (M1/M2/M3/M4) and can be used with the python_coreml_stable_diffusion package.

## Original Model

Based on [stabilityai/sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo)

## Conversion Details

- Attention implementation: SPLIT_EINSUM_V2
- Compute unit: CPU_AND_GPU
- Optimized for Apple Silicon Neural Engine