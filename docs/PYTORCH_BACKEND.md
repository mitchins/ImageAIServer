# PyTorch Backend Integration

This document describes the optional PyTorch backend integration for ImageAIServer, which extends model support beyond the core ONNX models while maintaining the lightweight architecture.

## Overview

The PyTorch backend provides access to a wider range of models, including:
- **SmolVLM-256M-Instruct** with INT8 quantization
- Large language models (Llama, Mistral, Phi, Gemma)
- Vision-language models (LLaVA, BLIP2, Kosmos-2)
- Code generation models (StarCoder, CodeLlama)

## Architecture

### Design Principles
- **DRY (Don't Repeat Yourself)**: Shared interfaces and common code
- **SOLID**: Single responsibility, open/closed, interface segregation
- **Lightweight Core**: PyTorch is completely optional
- **Strategy Pattern**: Automatic backend selection based on model type

### Components

```
shared/
├── model_backend.py      # Abstract interfaces
├── model_manager.py      # Strategy pattern implementation
├── onnx_loader.py        # ONNX backend (always available)
├── torch_loader.py       # PyTorch backend (optional)
└── model_types.py        # Model configurations
```

### Backend Selection Flow

1. **Model Request** → Model Manager
2. **Check ONNX Registry** → If curated ONNX model → Use ONNX
3. **Check PyTorch Support** → If in PyTorch list → Use PyTorch
4. **Default Fallback** → ONNX preferred, then PyTorch
5. **Load Model** → Cache for future requests

## Installation

### Core (ONNX Only)
```bash
pip install -r requirements.txt
```

### With PyTorch Backend
```bash
pip install -r requirements.txt
pip install -r requirements-torch.txt
```

## Usage

### API Endpoints

#### ONNX Models (Always Available)
```bash
POST /v1/chat/completions
{
  "model": "Gemma-3n-E2B-it-ONNX/Q4_MIXED",
  "messages": [{"role": "user", "content": "Hello"}]
}
```

#### PyTorch Models (When Available)
```bash
POST /chat-server/v1/chat/completions/torch
{
  "model": "HuggingFaceTB/SmolVLM-256M-Instruct",
  "messages": [{"role": "user", "content": "Hello"}],
  "backend": "pytorch"
}
```

### Model Manager API
```python
from shared.model_manager import get_model_manager, BackendType

manager = get_model_manager()

# Auto-select backend
result = manager.generate_text(
    model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
    text="What's in this image?",
    images=["base64_image_data"]
)

# Force specific backend
result = manager.generate_text(
    model_name="HuggingFaceTB/SmolVLM-256M-Instruct", 
    text="Hello world",
    backend=BackendType.PYTORCH
)
```

## SmolVLM-256M Integration

### Features
- **Ultra-lightweight**: Only 256M parameters
- **Vision + Text**: Multimodal capabilities
- **INT8 Quantization**: ~4x memory reduction
- **CPU Optimized**: Runs efficiently on CPU

### Quantization Support
```python
from shared.model_backend import BackendConfig
from shared.torch_loader import PyTorchBackend, PyTorchModelLoader

# Configure INT8 quantization
config = BackendConfig(
    backend_type="pytorch",
    device="auto",
    precision="int8"
)

backend = PyTorchBackend(config)
loader = PyTorchModelLoader(backend)
model, tokenizer, config = loader.load_model("HuggingFaceTB/SmolVLM-256M-Instruct")
```

### Memory Usage
- **FP32**: ~1024 MB
- **INT8**: ~256 MB (4x reduction)
- **Actual GPU**: ~300-400 MB (with overhead)

## Testing

### Unit Tests (No Dependencies)
```bash
# Test backend logic without external dependencies
pytest tests/unit/test_torch_simple.py -v
```

### Integration Tests (Requires PyTorch)
```bash
# Install dependencies first
pip install -r requirements-torch.txt

# Run integration tests
pytest tests/integration/test_smolvlm_int8.py -v

# Skip slow tests (model downloads)
pytest tests/integration/ -m "not slow"
```

### Manual Testing
```bash
# Backend detection test
python test_backends.py

# Pizza recognition test (tests vision capabilities)
python test_pizza_recognition.py
```

## Pizza Recognition Test

The pizza recognition test validates that SmolVLM can identify food content without explicit hints:

### Test Design
- **Synthetic Pizza Image**: Programmatically generated pizza with pepperoni
- **Neutral Prompts**: "What is in this image?" (no food hints)
- **Success Criteria**: Model mentions pizza, food, or related terms
- **Scoring**: 10 = pizza detected, 7 = food detected, 5 = food terms, 3 = visual features

### Example Output
```
🔍 Test 1: 'What is in this image?'
   Response: I can see a circular pizza with pepperoni toppings and cheese on a crust.
   🎯 PIZZA DETECTED - Score: 10/10

✅ EXCELLENT: Model correctly identified pizza!
```

## Performance Characteristics

### Model Loading
- **First Load**: 30-60 seconds (download + quantization)
- **Subsequent Loads**: 5-10 seconds (from cache)
- **Memory Usage**: ~256 MB for INT8 SmolVLM

### Inference Speed
- **Text Generation**: ~10-20 tokens/second (CPU)
- **Vision + Text**: ~5-15 tokens/second (CPU)
- **GPU Acceleration**: 2-5x faster when available

### Storage Requirements
- **SmolVLM INT8**: ~500 MB disk space
- **Larger Models**: 1-10 GB depending on size

## Supported Models

### Vision-Language Models
- **HuggingFaceTB/SmolVLM-256M-Instruct** ← New addition
- llava-hf/llava-1.5-7b-hf
- llava-hf/llava-1.5-13b-hf
- microsoft/kosmos-2-patch14-224
- Salesforce/blip2-opt-2.7b

### Text Models
- meta-llama/Llama-2-7b-chat-hf
- meta-llama/Meta-Llama-3-8B-Instruct
- mistralai/Mistral-7B-Instruct-v0.2
- microsoft/phi-2, phi-3-mini-4k-instruct
- google/gemma-7b-it, gemma-2b-it

### Code Models
- bigcode/starcoder2-15b
- codellama/CodeLlama-7b-Instruct-hf
- WizardLM/WizardCoder-Python-34B-V1.0

## Troubleshooting

### PyTorch Not Available
```python
from shared.torch_loader import TORCH_AVAILABLE
if not TORCH_AVAILABLE:
    print("Install PyTorch: pip install -r requirements-torch.txt")
```

### Model Loading Errors
- Check internet connection for first download
- Verify sufficient disk space (~500MB per model)
- Ensure adequate RAM (~2GB for model loading)

### CUDA Issues
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
```

### Memory Issues
- Use INT8 quantization: `precision="int8"`
- Close other applications
- Consider smaller models for limited hardware

## Future Enhancements

### Planned Features
- **More Quantization Types**: AWQ, GPTQ support
- **Batch Processing**: Multiple images/prompts
- **Streaming Responses**: Real-time token generation
- **Model Caching**: Persistent model storage

### Additional Models
- **Qwen-VL Series**: When PyTorch versions available
- **InternVL**: Advanced vision understanding
- **CogVLM**: Grounding and detailed descriptions

## Contributing

When adding new PyTorch models:

1. Add to `get_supported_models()` in `torch_loader.py`
2. Add vision detection logic if needed
3. Create integration test in `tests/integration/`
4. Update documentation

Example:
```python
def get_supported_models(self) -> List[str]:
    return [
        # ... existing models ...
        "new-org/new-model-name",  # Add here
    ]
```