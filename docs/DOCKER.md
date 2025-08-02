# Docker Deployment Guide

ImageAIServer provides multiple Docker configurations to match different use cases and resource constraints.

## 🏗️ Multi-Tier Docker Strategy

### 1. **Base (ONNX Only)** - Lightweight
- **Size**: ~2GB
- **Backends**: ONNX only
- **Models**: Curated ONNX models (Gemma-3n, SmolVLM, etc.)
- **Use Case**: Production deployments, resource-constrained environments

### 2. **Torch (Full Featured)** - Complete
- **Size**: ~8GB  
- **Backends**: ONNX + PyTorch
- **Models**: All ONNX + PyTorch models (Llama, Mistral, SmolVLM, etc.)
- **Use Case**: Development, research, maximum model support

### 3. **GPU (Accelerated)** - High Performance
- **Size**: ~8GB + CUDA
- **Backends**: ONNX + PyTorch with GPU acceleration
- **Requirements**: NVIDIA GPU, nvidia-docker
- **Use Case**: High-throughput inference, large models

## 🚀 Quick Start

### Option 1: Base (Recommended for Production)
```bash
# Build and run lightweight ONNX-only version
docker-compose --profile base up -d

# Access at http://localhost:8000
curl http://localhost:8000/v1/backends
```

### Option 2: Full PyTorch Support
```bash
# Build and run with PyTorch support
docker-compose --profile torch up -d

# Access at http://localhost:8001
curl http://localhost:8001/v1/backends
```

### Option 3: GPU Accelerated
```bash
# Requires nvidia-docker
docker-compose --profile gpu up -d

# Access at http://localhost:8002
curl http://localhost:8002/v1/backends
```

## 🛠️ Build Options

### Build Arguments

#### TORCH_VERSION
Controls PyTorch installation variant:

```bash
# CPU only (default)
docker build -f Dockerfile.torch --build-arg TORCH_VERSION=cpu .

# CUDA 11.8
docker build -f Dockerfile.torch --build-arg TORCH_VERSION=cu118 .

# CUDA 12.1  
docker build -f Dockerfile.torch --build-arg TORCH_VERSION=cu121 .
```

### Manual Building

```bash
# Base image (ONNX only)
docker build -f Dockerfile.base -t imageaiserver:base .

# PyTorch image (full)
docker build -f Dockerfile.torch -t imageaiserver:torch .

# PyTorch with CUDA
docker build -f Dockerfile.torch --build-arg TORCH_VERSION=cu121 -t imageaiserver:gpu .
```

## 📊 Resource Requirements

| Variant | Size | RAM | CPU | GPU | Startup Time |
|---------|------|-----|-----|-----|--------------|
| Base    | 2GB  | 4GB | 2 cores | No | 30s |
| Torch   | 8GB  | 8GB | 4 cores | Optional | 60s |
| GPU     | 8GB+ | 8GB | 4 cores | Required | 60s |

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Values |
|----------|-------------|---------|--------|
| `BACKEND_TYPE` | Backend selection | `auto` | `onnx`, `pytorch`, `auto` |
| `LOG_LEVEL` | Logging level | `info` | `debug`, `info`, `warning`, `error` |
| `HF_HOME` | HuggingFace cache | `/app/.cache/huggingface` | Path |
| `CUDA_VISIBLE_DEVICES` | GPU selection | `0` | GPU indices |

### Volume Mounts

```bash
# Model cache (recommended)
-v ./models:/app/.cache/huggingface

# Logs (optional)
-v ./logs:/app/logs

# Local model cache (if pre-downloaded)
-v ~/.cache/huggingface:/app/.cache/huggingface
```

## 🚦 Health Checks

All containers include health checks:

```bash
# Check container health
docker ps

# View health check logs
docker inspect imageai-base | jq '.[0].State.Health'

# Manual health check
curl http://localhost:8000/health
curl http://localhost:8000/v1/backends
```

## 📝 Usage Examples

### 1. Development Setup
```bash
# Start full-featured version for development
docker-compose --profile torch up -d

# Test both backends
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Gemma-3n-E2B-it-ONNX/Q4_MIXED", "messages": [{"role": "user", "content": "Hello"}]}'

curl -X POST http://localhost:8001/chat-server/v1/chat/completions/torch \
  -H "Content-Type: application/json" \
  -d '{"model": "HuggingFaceTB/SmolVLM-256M-Instruct", "messages": [{"role": "user", "content": "Hello"}]}'
```

### 2. Production Deployment
```bash
# Lightweight production deployment
docker-compose --profile base up -d

# Only ONNX models available
curl http://localhost:8000/v1/backends
# Returns: {"backends": {"onnx": {"available": true}, "pytorch": {"available": false}}}
```

### 3. GPU Inference
```bash
# GPU-accelerated inference (requires nvidia-docker)
docker-compose --profile gpu up -d

# Verify GPU access
docker exec imageai-torch-gpu nvidia-smi
```

## 🔄 Profiles Usage

Docker Compose profiles allow selective service deployment:

```bash
# Base only
docker-compose --profile base up -d

# PyTorch only  
docker-compose --profile torch up -d

# GPU only
docker-compose --profile gpu up -d

# All services
docker-compose --profile all up -d

# Legacy (original Dockerfile)
docker-compose --profile legacy up -d
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Container Won't Start
```bash
# Check logs
docker-compose logs imageai-base

# Common causes:
# - Insufficient RAM (need 4GB+ for base, 8GB+ for torch)
# - Port conflicts (check ports 8000-8003)
```

#### 2. Models Not Loading
```bash
# Check model cache permissions
docker exec imageai-base ls -la /app/.cache/huggingface

# Verify internet connectivity
docker exec imageai-base curl -I https://huggingface.co

# Check available disk space
docker exec imageai-base df -h
```

#### 3. PyTorch Not Available
```bash
# Verify PyTorch installation
docker exec imageai-torch python -c "import torch; print(torch.__version__)"

# Check backend status
curl http://localhost:8001/v1/backends
```

#### 4. GPU Not Detected
```bash
# Verify nvidia-docker setup
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# Check CUDA in container
docker exec imageai-torch-gpu python -c "import torch; print(torch.cuda.is_available())"
```

### Performance Optimization

#### Memory Usage
```bash
# Monitor memory usage
docker stats

# For memory-constrained environments, use base image
docker-compose --profile base up -d
```

#### Startup Time
```bash
# Pre-build images
docker build -f Dockerfile.base -t imageaiserver:base .
docker build -f Dockerfile.torch -t imageaiserver:torch .

# Use pre-downloaded models
-v ~/.cache/huggingface:/app/.cache/huggingface
```

## 🏷️ Image Tags

| Tag | Dockerfile | Backend | Size | Use Case |
|-----|------------|---------|------|----------|
| `latest` | `Dockerfile` | ONNX | 2GB | Legacy compatibility |
| `base` | `Dockerfile.base` | ONNX | 2GB | Production |
| `torch` | `Dockerfile.torch` | ONNX+PyTorch | 8GB | Development |
| `gpu` | `Dockerfile.torch` | ONNX+PyTorch+CUDA | 8GB+ | High performance |

## 📚 Additional Resources

- **Model Management**: Access web UI at `http://localhost:8000/manage/ui/`
- **API Documentation**: `http://localhost:8000/docs`
- **Backend Status**: `http://localhost:8000/v1/backends`
- **Health Check**: `http://localhost:8000/health`

## 🔐 Security Considerations

- Containers run as non-root user
- No unnecessary ports exposed
- Model cache isolated in volumes
- Health checks prevent unhealthy deployments
- Environment variables for configuration (no hardcoded secrets)

Choose the deployment that matches your needs:
- **Resource-constrained**: Base image
- **Full features**: Torch image  
- **High performance**: GPU image