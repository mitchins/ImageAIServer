# Architecture

This document provides a detailed overview of the ImageAIServer architecture, configuration, and security features.

## Service Structure
```
ImageAIServer/
├── imageai_server/           # Main package
│   ├── main.py              # Unified server entry point
│   ├── multimodal_chat/     # Vision-language model server (renamed)
│   │   ├── main.py          # Chat server entry point  
│   │   └── README.md        # Detailed chat documentation
│   ├── face_api/            # Face comparison service
│   │   ├── main.py          # Face server entry point
│   │   ├── presets.py       # Model presets configuration
│   │   └── README.md        # Detailed face documentation
│   ├── chat_server/         # Unified chat endpoints
│   │   ├── router.py        # Combined ONNX + PyTorch routing
│   │   └── torch_router.py  # Optional PyTorch endpoints
│   ├── manage_api/          # Model management API
│   │   └── router.py        # Management endpoints
│   ├── shared/              # Shared utilities
│   │   ├── unified_model_registry.py  # Model discovery and tracking
│   │   ├── model_types.py   # Model configurations
│   │   ├── model_backend.py # Abstract backend interfaces
│   │   ├── model_manager.py # Backend selection strategy
│   │   ├── onnx_loader.py   # ONNX implementation
│   │   ├── torch_loader.py  # PyTorch implementation (optional)
│   │   └── manage_cache.py  # HuggingFace cache utilities
│   └── static/              # Web interface assets
│       └── manage/          # Model management UI
├── tests/                   # Test suite
├── docs/                    # Documentation
└── scripts/                 # Build utilities
```

## Communication Flow
```
Client Applications → HTTP API → Model Manager → Backend Selection
         ↓                ↓            ↓                ↓
    Web Interface → Management API → Model Registry → ONNX/PyTorch
```

## Backend Architecture
- **Strategy Pattern**: Model Manager selects appropriate backend
- **Abstract Interfaces**: Common API for ONNX and PyTorch
- **Lazy Loading**: PyTorch only loaded when needed
- **Fallback Support**: Automatic failover between backends

## Configuration

### Environment Variables
```bash
# ONNX Chat Server
MODEL_NAME="Qwen2-VL-2B-Instruct/Q4"    # Default model
ONNX_CHAT_HOST="0.0.0.0"                # Server host
ONNX_CHAT_PORT="8080"                   # Server port

# Face API Server  
PRESET="photo"                          # Model preset
DETECTOR_MODEL="deepghs/real_face_detection"
EMBEDDER_MODEL_PATH="openailab/onnx-arcface-resnet100-ms1m"

# Model Management
PRELOAD_MODELS="1"                      # Load models at startup
FACE_MODEL_PROVIDERS="CUDAExecutionProvider,CPUExecutionProvider"
```

### Model Storage
- **HuggingFace Cache**: `~/.cache/huggingface/hub/`
- **Auto-discovery**: Models automatically detected from cache
- **Quantization support**: Multiple variants per model family

## Privacy & Security

### Local-First Design
- **No cloud dependency** for core functionality
- **All inference runs locally** on your hardware
- **Model data stays local** with HuggingFace cache

### Security Features  
- **Input validation** on all API endpoints
- **Path traversal protection** for file operations
- **Resource limits** to prevent abuse
- **Error handling** without exposing internal details

## Performance

### Optimization Features
- **ONNX runtime** for efficient inference
- **Multiple quantization levels** (FP32 → FP16 → Q4 → INT8)
- **GPU acceleration** when available (CUDA, CoreML)
- **Model caching** for faster subsequent loads
- **Companion file management** for large models

### Resource Requirements
- **Minimum**: 8GB RAM, CPU inference
- **Recommended**: 16GB+ RAM, dedicated GPU
- **Storage**: 2-10GB per model (varies by quantization)
