# ImageAIServer <img src="imageai_server/static/icon.png" alt="ImageAIServer Icon" width="30" height="30">

ImageAIServer provides privacy-focused, offline AI inference capabilities. These servers offer local alternatives to cloud-based AI services for vision, chat, and face analysis.

## 🚀 Quick Start

### Installation

#### PyPI Installation (Recommended)
```bash
pip install imageaiserver
```

#### Docker Installation

**Multi-tier Docker strategy available:**

```bash
# Base (ONNX only, ~2GB)
docker-compose --profile base up -d

# Full (ONNX + PyTorch, ~8GB) 
docker-compose --profile torch up -d

# GPU (ONNX + PyTorch + CUDA, requires nvidia-docker)
docker-compose --profile gpu up -d
```

**📖 Detailed Docker Guide:** [docs/DOCKER.md](docs/DOCKER.md)

#### Development Installation
```bash
git clone https://github.com/mitchins/ImageAIServer.git
cd ImageAIServer
pip install -e .
```

### Run Individual Services
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

---

## ✨ Recent Features

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

---

## 🧠 Multimodal Chat Server

**Local vision-language model inference** with OpenAI-compatible API.

### Features
- **OpenAI API compatibility** – drop-in replacement for remote endpoints
- **Multimodal support** – text, vision, and multi-image analysis
- **Multiple quantizations** – FP16, Q4, INT8 for performance tuning
- **Dynamic model loading** – automatic model discovery and caching
- **Optional PyTorch backend** – support for larger models when PyTorch is installed

### Supported Models

#### ONNX Models (Always Available)
- **Qwen2-VL-2B-Instruct** – 8 quantization variants
- **Gemma-3n-E2B-it-ONNX** – 3 quantization variants  
- **Phi-3.5-vision-instruct** – 2 quantization variants
- **SmolVLM-256M-Instruct** – 8 quantization variants (ultra-lightweight)

#### PyTorch Models (Optional - requires `pip install -r requirements-torch.txt`)
- **SmolVLM Series** – 256M and 500M ultra-lightweight vision-language models

**For quantized models, use GGUF versions:**
- `ggml-org/SmolVLM-256M-Instruct-GGUF` (Q8_0: 175MB, F16: 328MB)
- `lmstudio-community/granite-vision-3.2-2b-GGUF` (various quantizations)

### Configuration
```bash
# Basic usage (ONNX models)
python -m imageai_server.multimodal_chat.main

# Custom configuration
MODEL_NAME="Qwen2-VL-2B-Instruct/Q4" python -m imageai_server.multimodal_chat.main
ONNX_CHAT_HOST=0.0.0.0 ONNX_CHAT_PORT=8080 python -m imageai_server.multimodal_chat.main

# Enable PyTorch backend (minimal - SmolVLM only)
pip install -r requirements-torch.txt

# Access PyTorch models via chat-server endpoints
# GET /chat-server/models - List available PyTorch models
# POST /chat-server/v1/chat/completions/torch - Use PyTorch models

# Test PyTorch backend with SmolVLM
python tests/manual/test_pizza_recognition.py  # Pizza recognition test
python scripts/test_backends.py                # Backend detection test
```

**📖 Detailed Documentation:** [imageai_server/multimodal_chat/README.md](imageai_server/multimodal_chat/README.md)

---

## 👤 Face Comparison API

**High-accuracy face recognition and similarity scoring** across different image styles.

### Features
- **Multi-style support** – real photos, anime, and CG characters
- **Preset configurations** – optimized for different use cases
- **ONNX-powered inference** – fast, efficient processing
- **Batch processing** – handle multiple comparisons

### Presets
- **`photo`** – Real photographs (default preset)
- **`anime`** – Anime-style characters and artwork
- **`cg`** – Computer-generated and digital art

### Supported Models
- **Face Detection**: InsightFace v1.2-v1.4 (multiple variants)
- **Face Embedding**: ArcFace ResNet100, CLIP ViT Base
- **Specialized**: Real face detection, anime face detection

### Usage
```bash
# Photo preset (default)
PRESET=photo uvicorn face_api.main:app

# Anime preset  
PRESET=anime uvicorn face_api.main:app --port 7860

# Custom configuration
DETECTOR_MODEL=deepghs/anime_face_detection \
EMBEDDER_MODEL_PATH=Xenova/clip-vit-base-patch32 \
uvicorn face_api.main:app
```

**📖 Detailed Documentation:** [imageai_server/face_api/README.md](imageai_server/face_api/README.md)

---

## 🔧 Unified Server & Model Management

**All-in-one ImageAIServer** combining chat, face, and model management APIs.

### Features
- **Unified endpoint** – single server for all ImageAI services
- **Model management UI** – web-based interface at `/manage/ui/`
- **Curated model catalog** – pre-configured ONNX models
- **Quantization control** – download specific model variants
- **Status tracking** – real-time download progress and file verification

### Model Management Features
- **13 total models** (3 chat + 10 face models)  
- **Companion file enforcement** – automatic detection of required data files
- **Quantization-level granularity** – "X/Y SIZES" status indicators
- **Organized interface** – collapsible sections by server type
- **Bulk operations** – download/clear entire model configurations

### Usage
```bash
# Start unified server
uvicorn main:app --host 0.0.0.0 --port 8000
# Or using the CLI command:
imageaiserver

# Access services:
# - Chat API: http://localhost:8000/multimodal-chat/
# - Face API: http://localhost:8000/face-server/  
# - Model Management: http://localhost:8000/manage/ui/
# - API Documentation: http://localhost:8000/docs
```

**📖 Model Management Guide:** Available through the web interface at `/manage/ui/`

---

## 🏗️ Architecture

### Service Structure
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

### Communication Flow
```
Client Applications → HTTP API → Model Manager → Backend Selection
         ↓                ↓            ↓                ↓
    Web Interface → Management API → Model Registry → ONNX/PyTorch
```

### Backend Architecture
- **Strategy Pattern**: Model Manager selects appropriate backend
- **Abstract Interfaces**: Common API for ONNX and PyTorch
- **Lazy Loading**: PyTorch only loaded when needed
- **Fallback Support**: Automatic failover between backends

---

## ⚙️ Configuration

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

---

## 🔒 Privacy & Security

### Local-First Design
- **No cloud dependency** for core functionality
- **All inference runs locally** on your hardware
- **Model data stays local** with HuggingFace cache

### Security Features  
- **Input validation** on all API endpoints
- **Path traversal protection** for file operations
- **Resource limits** to prevent abuse
- **Error handling** without exposing internal details

---

## 🚀 Performance

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

---

## 📚 Documentation

- **[docs/](docs/)** - Complete technical documentation
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System design and architecture review
- **[docs/PYTORCH_BACKEND.md](docs/PYTORCH_BACKEND.md)** - PyTorch backend integration guide  
- **[docs/DOCKER.md](docs/DOCKER.md)** - Multi-tier Docker deployment strategies
- **[tests/README.md](tests/README.md)** - Testing strategies and guidelines

---

**Ready to deploy your own AI infrastructure? Start with the unified server and scale as needed!** 🚀