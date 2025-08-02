"""Abstract base classes for model backends (ONNX, PyTorch, etc.)"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class BackendConfig:
    """Configuration for a model backend."""
    backend_type: str  # "onnx", "pytorch", etc.
    device: str = "cpu"  # "cpu", "cuda", "mps", etc.
    precision: str = "fp32"  # "fp32", "fp16", "int8", etc.
    extra_args: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_args is None:
            self.extra_args = {}


class ModelBackend(ABC):
    """Abstract base class for model backends - handles availability and capabilities."""
    
    @abstractmethod
    def __init__(self, config: BackendConfig):
        """Initialize the backend with configuration."""
        self.config = config
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available in the current environment."""
        pass
    
    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """Return list of model identifiers supported by this backend."""
        pass
    
    @abstractmethod
    def supports_quantization(self, quant_type: str) -> bool:
        """Check if backend supports specific quantization type."""
        pass


class ModelLoader(ABC):
    """Abstract base class for model loaders."""
    
    @abstractmethod
    def parse_model_name(self, name: str) -> Tuple[str, str]:
        """Parse model identifier into components."""
        pass
    
    @abstractmethod
    def download_model_files(self, repo_id: str, **kwargs) -> Dict[str, str]:
        """Download model files and return paths."""
        pass
    
    @abstractmethod
    def load_model(self, model_name: str) -> Tuple[Any, Any, Any]:
        """Load model, tokenizer, and config."""
        pass


class InferenceEngine(ABC):
    """Abstract base class for inference engines."""
    
    @abstractmethod
    def __init__(self, model: Any, tokenizer: Any, config: Any):
        """Initialize with model, tokenizer, and config."""
        self.tokenizer = tokenizer
        self.config = config
    
    @abstractmethod
    def generate_text(
        self, 
        text: str, 
        max_tokens: int = 100,
        images: Optional[List[str]] = None,
        audio: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt and optional multimodal inputs."""
        pass
    
    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to embeddings."""
        pass
    
    @abstractmethod
    def supports_modality(self, modality: str) -> bool:
        """Check if engine supports specific modality (text, vision, audio)."""
        pass