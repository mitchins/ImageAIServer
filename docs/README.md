# ImageAIServer Documentation

This directory contains all technical documentation for ImageAIServer.

## 📚 Documentation Structure

### Core Documentation
- **[../README.md](../README.md)** - Main project overview and quick start
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and design principles
- **[PYTORCH_BACKEND.md](PYTORCH_BACKEND.md)** - PyTorch backend integration guide
- **[DOCKER.md](DOCKER.md)** - Docker deployment strategies

### API Documentation
- **[API.md](API.md)** - Complete API reference
- **[ENDPOINTS.md](ENDPOINTS.md)** - Endpoint details and examples

### Development
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Development setup and guidelines
- **[TESTING.md](TESTING.md)** - Testing strategies and guidelines
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines

### Service-Specific
- **[../onnx_chat/README.md](../onnx_chat/README.md)** - ONNX chat service details
- **[../face_api/README.md](../face_api/README.md)** - Face API service details
- **[../tests/README.md](../tests/README.md)** - Testing structure and usage

## 🏗️ Architecture Overview

```
ImageAIServer/
├── Core Services
│   ├── ONNX Chat (Vision + Text)
│   ├── Face API (Detection + Comparison)  
│   └── Model Management (Web UI)
├── Optional Backends
│   ├── ONNX (Always available)
│   └── PyTorch (Optional, for more models)
└── Deployment
    ├── Base Docker (~2GB, ONNX only)
    ├── Torch Docker (~8GB, Full featured)
    └── GPU Docker (~8GB+, CUDA accelerated)
```

## 🚀 Quick Navigation

| Topic | Document | Description |
|-------|----------|-------------|
| Getting Started | [../README.md](../README.md) | Installation and basic usage |
| Architecture | [ARCHITECTURE.md](ARCHITECTURE.md) | Design principles and structure |
| PyTorch Backend | [PYTORCH_BACKEND.md](PYTORCH_BACKEND.md) | Advanced model support |
| Docker Deployment | [DOCKER.md](DOCKER.md) | Container deployment options |
| API Reference | [API.md](API.md) | Complete API documentation |
| Development Setup | [DEVELOPMENT.md](DEVELOPMENT.md) | Local development guide |
| Testing | [TESTING.md](TESTING.md) | Testing approaches and examples |