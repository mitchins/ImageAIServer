# ImageAIServer Tests

This directory contains both unit and integration tests for ImageAIServer.

## Test Structure

```
tests/
├── unit/                    # Unit tests (no external dependencies)
│   ├── test_torch_backend.py   # PyTorch backend unit tests
│   └── ...                     # Other unit tests
│
├── integration/             # Integration tests (require dependencies)
│   ├── test_smolvlm_int8.py    # SmolVLM INT8 quantization tests
│   └── ...                     # Other integration tests
│
└── README.md               # This file
```

## Running Tests

### Unit Tests
Unit tests use mocks and don't require external dependencies:

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run specific unit test
pytest tests/unit/test_torch_backend.py -v
```

### Integration Tests
Integration tests require dependencies and may download models:

```bash
# First, install PyTorch dependencies
pip install -r requirements-torch.txt

# Run all integration tests
pytest tests/integration/ -v

# Run specific integration test
pytest tests/integration/test_smolvlm_int8.py -v

# Skip slow tests (model downloads)
pytest tests/integration/ -v -m "not slow"

# Run face detection pipeline tests specifically
python tests/test_face_pipeline.py --verbose
```

### Face Detection Pipeline Tests
Comprehensive tests for face detection across different face types:

```bash
# Run all face detection tests (unit + integration)
python tests/test_face_pipeline.py

# Run only unit tests (ONNX model tests)
python tests/test_face_pipeline.py --unit-only

# Run only integration tests (full pipeline with real images)
python tests/test_face_pipeline.py --integration-only --verbose
```

## Test Categories

### Unit Tests (`tests/unit/`)
- **Purpose**: Test code logic without external dependencies
- **Requirements**: Basic Python packages only
- **Speed**: Fast (< 1 second per test)
- **Mocking**: Heavy use of mocks for external dependencies
- **Examples**:
  - Backend detection logic
  - Configuration handling
  - Model selection strategy
  - API compatibility
  - **Face detection ONNX models** (`test_face_onnx_models.py`)
    - Image preprocessing functions
    - CLIP embedding generation
    - Model loader functionality
    - Error handling

### Integration Tests (`tests/integration/`)
- **Purpose**: Test actual model inference end-to-end
- **Requirements**: 
  - PyTorch and transformers (`pip install -r requirements-torch.txt`)
  - Internet connection (for model downloads)
  - ~1GB disk space
  - ~2GB RAM (more for larger models)
- **Speed**: Slow (may download models on first run)
- **Examples**:
  - SmolVLM INT8 quantization
  - Vision + text generation
  - Memory efficiency validation
  - Inference speed benchmarks
  - **Face detection pipeline** (`test_face_detection_pipeline.py`)
    - Real photo face detection and comparison
    - Anime face detection and comparison
    - Cross-type embedding consistency
    - Model loading and caching

## Writing New Tests

### Unit Test Template
```python
import unittest
from unittest.mock import Mock, patch

class TestNewFeature(unittest.TestCase):
    def setUp(self):
        # Set up mocks
        self.mock_torch = Mock()
        
    def test_feature_logic(self):
        with patch.dict('sys.modules', {'torch': self.mock_torch}):
            # Test logic without real dependencies
            pass
```

### Integration Test Template
```python
import pytest

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
class TestNewIntegration:
    @pytest.mark.slow
    def test_real_inference(self):
        # Test with actual model loading
        pass
```

## CI/CD Considerations

For continuous integration:

1. **Unit tests** should run on every commit
2. **Integration tests** should run:
   - On pull requests
   - Nightly
   - Before releases

Example GitHub Actions workflow:
```yaml
- name: Run unit tests
  run: pytest tests/unit/ -v

- name: Run integration tests
  if: github.event_name == 'pull_request' || github.event_name == 'schedule'
  run: |
    pip install -r requirements-torch.txt
    pytest tests/integration/ -v
```

## Test Markers

We use pytest markers to categorize tests:

- `@pytest.mark.slow` - Tests that take > 10 seconds
- `@pytest.mark.requires_gpu` - Tests that need GPU
- `@pytest.mark.requires_internet` - Tests that download models

Run specific categories:
```bash
pytest -m "not slow"  # Skip slow tests
pytest -m "requires_gpu"  # Only GPU tests
```