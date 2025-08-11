"""
Base VLM Strategy Interface
Following the pattern established in generation/runtime_manager.py
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from PIL import Image
import asyncio
import logging

logger = logging.getLogger(__name__)


class BaseVLMStrategy(ABC):
    """Abstract base class for VLM backend strategies"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.model_name = None
        self._loaded = False
    
    @abstractmethod
    async def load_model(self, model_name: str, **kwargs) -> bool:
        """
        Load the VLM model asynchronously
        
        Args:
            model_name: Model identifier/path
            **kwargs: Additional model-specific parameters
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def generate_response(self, 
                              image: Image.Image,
                              prompt: str,
                              max_tokens: int = 100,
                              temperature: float = 0.0,
                              **kwargs) -> str:
        """
        Generate text response from image and prompt
        
        Args:
            image: PIL Image object
            prompt: Text prompt/question
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            str: Generated text response
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this strategy can run on current system
        
        Returns:
            bool: True if dependencies and hardware are available
        """
        pass
    
    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Name of the backend (e.g., 'mlx', 'onnx', 'torch')"""
        pass
    
    @property
    def performance_tier(self) -> int:
        """Performance tier (1=fastest, 4=slowest). Override in subclasses."""
        return 4
    
    @property
    def memory_efficient(self) -> bool:
        """Whether this strategy is memory efficient. Override in subclasses."""
        return True
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._loaded
    
    def unload_model(self):
        """Unload the model to free memory"""
        self.model = None
        self.processor = None
        self.model_name = None
        self._loaded = False
        logger.info(f"{self.backend_name} model unloaded")
    
    def _format_prompt(self, prompt: str, model_name: str) -> str:
        """
        Format prompt with appropriate image tokens
        Override in subclasses for model-specific formatting
        """
        return prompt
    
    async def _run_in_executor(self, func, *args):
        """Run blocking function in thread executor"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args)


class VLMStrategyError(Exception):
    """Base exception for VLM strategy errors"""
    pass


class ModelNotLoadedError(VLMStrategyError):
    """Raised when trying to use a model that isn't loaded"""
    pass


class UnsupportedModelError(VLMStrategyError):
    """Raised when model is not supported by this strategy"""
    pass