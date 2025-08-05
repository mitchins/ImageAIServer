# Quick Start

This guide provides detailed instructions for getting started with ImageAIServer.

## Installation

### PyPI Installation (Recommended)
```bash
pip install imageaiserver
```

### Docker Installation

**Multi-tier Docker strategy available:**

```bash
# Base (ONNX only, ~2GB)
docker-compose --profile base up -d

# Full (ONNX + PyTorch, ~8GB) 
docker-compose --profile torch up -d

# GPU (ONNX + PyTorch + CUDA, requires nvidia-docker)
docker-compose --profile gpu up -d
```

**📖 Detailed Docker Guide:** [DOCKER.md](DOCKER.md)

### Development Installation
```bash
git clone https://github.com/mitchins/ImageAIServer.git
cd ImageAIServer
pip install -e .[dev]
```

## Run Individual Services
```bash
# Multimodal Chat Server (vision-language models)
python -m imageai_server.multimodal_chat.main

# Face Comparison API  
PRESET=photo uvicorn imageai_server.face_api.main:app --port 7860

# Unified Server (all services + model management)
uvicorn imageai_server.main:app --host 0.0.0.0 --port 8000
# Or using the CLI command:
imageaiserver
```

## Recent Features

### Local-Only Model Listing
- **Smart model detection** – only shows models that are actually downloaded locally
- **No more "model not downloaded" errors** – prevents selection of unavailable models
- **Dynamic model discovery** – automatically detects ONNX and PyTorch/GGUF models

### GPU Acceleration Status
- **Hardware detection** – shows CUDA, MPS (Apple Silicon), or CPU-only status
- **Real-time monitoring** – displays current device usage (e.g., "Using: mps")
- **Multi-GPU support** – shows device count for CUDA systems

### Generation Time Tracking
- **Performance monitoring** – shows inference time for each request
- **Client-side timing** – accurate to milliseconds in test interface
- **User feedback** – helps users understand model performance
