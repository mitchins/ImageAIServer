"""Model Manager - Strategy pattern for backend selection and model management."""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum

from .model_backend import ModelBackend, ModelLoader, InferenceEngine, BackendConfig
from .model_identifier import UnifiedModelManager, ModelBackendType

logger = logging.getLogger(__name__)

# Import backends with error handling
try:
    from .onnx_backend import create_onnx_backend
    from .onnx_loader import ONNXModelLoader, ONNXInferenceEngine, ONNX_AVAILABLE
    from .model_types import ONNXModelConfig, get_onnx_model_config
except ImportError as e:
    logger.warning(f"ONNX backend unavailable: {e}")
    ONNX_AVAILABLE = False
    create_onnx_backend = None
    ONNXModelLoader = None
    ONNXInferenceEngine = None
    get_onnx_model_config = lambda x: None

try:
    from .torch_loader import create_pytorch_backend, PyTorchModelLoader, PyTorchInferenceEngine, TORCH_AVAILABLE
except ImportError as e:
    logger.warning(f"PyTorch backend unavailable: {e}")
    TORCH_AVAILABLE = False
    create_pytorch_backend = lambda: None
    PyTorchModelLoader = None
    PyTorchInferenceEngine = None

try:
    from ..multimodal_chat.vlm_service import get_vlm_service
    MLX_AVAILABLE = True
except ImportError as e:
    logger.warning(f"MLX VLM backend unavailable: {e}")
    MLX_AVAILABLE = False
    get_vlm_service = None


class BackendType(str, Enum):
    """Available backend types."""
    ONNX = "onnx"
    PYTORCH = "pytorch"
    MLX = "mlx"  # Apple Silicon MLX backend
    AUTO = "auto"  # Automatically select best backend


class ModelManager:
    """Manages model loading across different backends using strategy pattern."""
    
    def __init__(self, default_backend: BackendType = BackendType.AUTO):
        self.default_backend = default_backend
        self.backends: Dict[BackendType, ModelBackend] = {}
        self.loaders: Dict[BackendType, ModelLoader] = {}
        self._initialize_backends()
        
        # Cache for loaded models
        self.loaded_models: Dict[str, Tuple[InferenceEngine, BackendType]] = {}
    
    def _initialize_backends(self):
        """Initialize available backends."""
        # Initialize ONNX backend
        if create_onnx_backend and ONNX_AVAILABLE:
            try:
                onnx_backend = create_onnx_backend()
                if onnx_backend.is_available():
                    self.backends[BackendType.ONNX] = onnx_backend
                    self.loaders[BackendType.ONNX] = ONNXModelLoader()
                    logger.info("ONNX backend initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize ONNX backend: {e}")
        
        # Initialize PyTorch backend
        if create_pytorch_backend and TORCH_AVAILABLE:
            try:
                pytorch_backend = create_pytorch_backend()
                if pytorch_backend and pytorch_backend.is_available():
                    self.backends[BackendType.PYTORCH] = pytorch_backend
                    self.loaders[BackendType.PYTORCH] = PyTorchModelLoader(pytorch_backend)
                    logger.info("PyTorch backend initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize PyTorch backend: {e}")
        
        # Initialize MLX VLM backend (Apple Silicon only)
        if get_vlm_service and MLX_AVAILABLE:
            try:
                vlm_service = get_vlm_service()
                available_backends = vlm_service.get_available_backends()
                if any(b['name'] == 'mlx' and b['available'] for b in available_backends):
                    # Create a simple adapter to match the ModelBackend interface
                    class MLXVLMAdapter:
                        def is_available(self): return True
                        def get_config(self): return {}
                    
                    class MLXVLMLoader:
                        def __init__(self, vlm_service):
                            self.vlm_service = vlm_service
                    
                    self.backends[BackendType.MLX] = MLXVLMAdapter()
                    self.loaders[BackendType.MLX] = MLXVLMLoader(vlm_service)
                    logger.info("MLX VLM backend initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize MLX VLM backend: {e}")
    
    def get_available_backends(self) -> List[BackendType]:
        """Get list of available backends."""
        return list(self.loaders.keys())
    
    def select_backend_for_model(self, model_name: str) -> BackendType:
        """Select the best backend for a given model."""
        # If default backend is specified and available, use it
        if self.default_backend != BackendType.AUTO:
            if self.default_backend in self.loaders:
                return self.default_backend
            else:
                logger.warning(f"Requested backend {self.default_backend} not available")
        
        # Check if model is in ONNX model registry (curated models)
        if BackendType.ONNX in self.loaders:
            onnx_loader = self.loaders[BackendType.ONNX]
            # Check if it's a curated ONNX model
            if '/' in model_name and hasattr(onnx_loader, 'load_curated_model'):
                from .model_types import get_curated_model_config
                if get_curated_model_config(model_name) is not None:
                    logger.info(f"Model {model_name} is a curated ONNX model")
                    return BackendType.ONNX
            
            # Check if repo_id matches ONNX configs
            repo_id = model_name.split('/')[0] if '/' in model_name else model_name
            if get_onnx_model_config(repo_id) is not None:
                logger.info(f"Model {model_name} has ONNX configuration")
                return BackendType.ONNX
        
        # Check if model is in PyTorch supported list
        if BackendType.PYTORCH in self.backends:
            pytorch_backend = self.backends[BackendType.PYTORCH]
            if model_name in pytorch_backend.get_supported_models():
                logger.info(f"Model {model_name} is in PyTorch supported list")
                return BackendType.PYTORCH
        
        # Default fallback order: ONNX first (lighter), then PyTorch
        if BackendType.ONNX in self.loaders:
            logger.info(f"Defaulting to ONNX backend for {model_name}")
            return BackendType.ONNX
        elif BackendType.PYTORCH in self.loaders:
            logger.info(f"Defaulting to PyTorch backend for {model_name}")
            return BackendType.PYTORCH
        else:
            raise RuntimeError("No backends available")
    
    def load_model(
        self, 
        model_name: str, 
        backend: Optional[BackendType] = None,
        _already_resolved: bool = False,
        _original_model_name: Optional[str] = None,
        **kwargs
    ) -> InferenceEngine:
        """Load a model using appropriate backend."""
        # Check cache first
        cache_key = f"{model_name}:{backend or 'auto'}"
        if cache_key in self.loaded_models:
            engine, used_backend = self.loaded_models[cache_key]
            logger.info(f"Using cached model {model_name} with {used_backend} backend")
            return engine
        
        # Resolve model name using unified system first (unless already resolved)
        resolved_model_name = model_name
        resolved_backend = backend
        
        if not _already_resolved:
            try:
                # Try resolving with unified model manager
                preferred_backend = None
                if backend == BackendType.ONNX:
                    preferred_backend = ModelBackendType.ONNX
                elif backend == BackendType.PYTORCH:
                    preferred_backend = ModelBackendType.PYTORCH
                
                repo_id, unified_backend, quantization = UnifiedModelManager.resolve_model(
                    model_name, preferred_backend
                )
                
                # Use resolved repo_id as the model name for loading
                resolved_model_name = repo_id
                
                # Use resolved backend if we didn't specify one
                if backend is None:
                    resolved_backend = BackendType.ONNX if unified_backend == ModelBackendType.ONNX else BackendType.PYTORCH
                    logger.info(f"Resolved '{model_name}' -> repo_id='{repo_id}', backend={resolved_backend}, quantization={quantization}")
            except Exception as e:
                logger.debug(f"Could not resolve model with unified system: {e}")
                # Fall back to original logic
        else:
            logger.debug(f"Skipping unified resolution for already resolved model: '{model_name}'")
        
        # Select backend
        if resolved_backend is None:
            resolved_backend = self.select_backend_for_model(resolved_model_name)
        
        if resolved_backend not in self.loaders:
            raise ValueError(f"Backend {resolved_backend} not available. Available: {self.get_available_backends()}")
        
        loader = self.loaders[resolved_backend]
        
        try:
            # Load model based on backend type
            if resolved_backend == BackendType.ONNX:
                # For ONNX curated models, preserve original model name with quantization
                # Check if original model name has curated config
                from .model_types import get_curated_model_config
                if (_already_resolved and _original_model_name and 
                    '/' in _original_model_name and 
                    get_curated_model_config(_original_model_name) is not None):
                    # Use original model name to preserve quantization info
                    onnx_model_name = _original_model_name
                    logger.info(f"Using original curated model name for ONNX: '{onnx_model_name}' (instead of resolved '{resolved_model_name}')")
                else:
                    # Use resolved repo_id for non-curated models
                    onnx_model_name = resolved_model_name
                
                # ONNX returns (sessions, tokenizer, config)
                sessions, tokenizer, config = loader.load_model(onnx_model_name)
                engine = ONNXInferenceEngine(sessions, tokenizer, config)
            elif resolved_backend == BackendType.PYTORCH:
                # PyTorch returns (model, tokenizer, config)
                model, tokenizer, config = loader.load_model(resolved_model_name)
                engine = PyTorchInferenceEngine(model, tokenizer, config)
            else:
                raise ValueError(f"Unknown backend type: {resolved_backend}")
            
            # Cache the loaded model
            self.loaded_models[cache_key] = (engine, resolved_backend)
            logger.info(f"Successfully loaded {model_name} with {resolved_backend} backend")
            
            return engine
            
        except Exception as e:
            logger.error(f"Failed to load {model_name} with {resolved_backend} backend: {e}")
            
            # Try fallback backend if using auto, but be smart about ONNX-only repos
            if resolved_backend == self.select_backend_for_model(resolved_model_name):
                # Don't try PyTorch fallback for ONNX-only repositories
                is_onnx_only_repo = (
                    "onnx-community" in resolved_model_name or 
                    resolved_model_name.endswith("-ONNX") or
                    "onnx" in resolved_model_name.lower()
                )
                
                if not is_onnx_only_repo:
                    other_backends = [b for b in self.get_available_backends() if b != resolved_backend]
                    if other_backends:
                        logger.info(f"Attempting fallback to {other_backends[0]} backend")
                        return self.load_model(model_name, backend=other_backends[0], **kwargs)
                else:
                    logger.warning(f"ONNX-only repository {resolved_model_name} failed to load - no fallback attempted")
            
            raise
    
    def generate_text(
        self,
        model_name: str,
        text: str,
        max_tokens: int = 100,
        images: Optional[List[str]] = None,
        audio: Optional[List[str]] = None,
        backend: Optional[BackendType] = None,
        **kwargs
    ) -> str:
        """Generate text using specified model and backend."""
        # Resolve model name using unified system
        preferred_backend = None
        if backend == BackendType.ONNX:
            preferred_backend = ModelBackendType.ONNX
        elif backend == BackendType.PYTORCH:
            preferred_backend = ModelBackendType.PYTORCH
        
        repo_id, resolved_backend, quantization = UnifiedModelManager.resolve_model(
            model_name, preferred_backend
        )
        
        # Convert back to our BackendType enum
        final_backend = BackendType.ONNX if resolved_backend == ModelBackendType.ONNX else BackendType.PYTORCH
        
        logger.info(f"Resolved '{model_name}' -> repo_id='{repo_id}', backend={final_backend}, quantization={quantization}")
        
        # Skip the unified resolution in load_model since we already resolved it here
        # Pass both resolved repo_id and original model name for curated models
        engine = self.load_model(repo_id, backend=final_backend, _already_resolved=True, _original_model_name=model_name)
        return engine.generate_text(text, max_tokens, images, audio, **kwargs)
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model."""
        backend = self.select_backend_for_model(model_name)
        
        info = {
            "model_name": model_name,
            "recommended_backend": backend.value,
            "available_backends": [b.value for b in self.get_available_backends()],
        }
        
        # Add backend-specific info
        if backend == BackendType.ONNX and '/' in model_name:
            from .model_types import get_curated_model_config
            config = get_curated_model_config(model_name)
            if config:
                info["components"] = list(config.keys())
                info["architecture"] = "multi-component"
        
        return info
    
    def list_available_models(self) -> Dict[BackendType, List[str]]:
        """List all available models by backend."""
        models = {}
        
        # ONNX models
        if BackendType.ONNX in self.loaders:
            from .model_types import get_available_model_quants
            models[BackendType.ONNX] = get_available_model_quants()
        
        # PyTorch models
        if BackendType.PYTORCH in self.backends:
            pytorch_backend = self.backends[BackendType.PYTORCH]
            models[BackendType.PYTORCH] = pytorch_backend.get_supported_models()
        
        return models


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get or create global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def load_model(
    model_name: str,
    backend: Optional[Union[str, BackendType]] = None,
    **kwargs
) -> InferenceEngine:
    """Convenience function to load a model."""
    if isinstance(backend, str):
        backend = BackendType(backend)
    
    manager = get_model_manager()
    return manager.load_model(model_name, backend=backend, **kwargs)


def generate_text(
    model_name: str,
    text: str,
    max_tokens: int = 100,
    images: Optional[List[str]] = None,
    audio: Optional[List[str]] = None,
    backend: Optional[Union[str, BackendType]] = None,
    **kwargs
) -> str:
    """Convenience function to generate text."""
    if isinstance(backend, str):
        backend = BackendType(backend)
    
    manager = get_model_manager()
    return manager.generate_text(
        model_name=model_name,
        text=text,
        max_tokens=max_tokens,
        images=images,
        audio=audio,
        backend=backend,
        **kwargs
    )