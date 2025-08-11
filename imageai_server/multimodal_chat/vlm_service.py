"""
VLM Service Layer - Handles model inference strategies
Separate from routing concerns
"""

import logging
from typing import Optional, Dict, Any, List
from PIL import Image
from dataclasses import dataclass

from .strategies.base_strategy import BaseVLMStrategy, VLMStrategyError
from .strategies.mlx_strategy import MLXVLMStrategy

logger = logging.getLogger(__name__)


@dataclass
class VLMResponse:
    """VLM inference response"""
    text: str
    model_name: str
    backend: str
    inference_time: float
    metadata: Dict[str, Any] = None


class VLMService:
    """
    VLM Service - manages strategies and provides unified inference interface
    Similar to how diffusion_loader works in generation
    """
    
    def __init__(self):
        self._strategies: Dict[str, BaseVLMStrategy] = {}
        self._current_strategy: Optional[BaseVLMStrategy] = None
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """Initialize available VLM strategies"""
        # Register available strategies
        strategies = [
            MLXVLMStrategy(),
            # ONNXVLMStrategy(),  # TODO: Refactor existing ONNX code
            # TorchVLMStrategy(), # TODO: Add PyTorch VLM support
        ]
        
        for strategy in strategies:
            if strategy.is_available():
                self._strategies[strategy.backend_name] = strategy
                logger.info(f"✅ {strategy.backend_name.upper()} VLM strategy available")
            else:
                logger.debug(f"❌ {strategy.backend_name.upper()} VLM strategy not available")
    
    def get_optimal_strategy(self, prefer_memory_efficient: bool = True) -> Optional[BaseVLMStrategy]:
        """Get the best available VLM strategy"""
        if not self._strategies:
            return None
        
        # Sort by performance tier, then memory efficiency
        available = list(self._strategies.values())
        
        best = min(available, key=lambda s: (
            s.performance_tier,
            not (s.memory_efficient and prefer_memory_efficient)
        ))
        
        return best
    
    def get_strategy(self, backend_name: str) -> Optional[BaseVLMStrategy]:
        """Get specific strategy by backend name"""
        return self._strategies.get(backend_name)
    
    async def load_model(self, 
                        model_name: str, 
                        backend: Optional[str] = None,
                        **kwargs) -> bool:
        """
        Load a VLM model using best available or specified strategy
        
        Args:
            model_name: Model identifier
            backend: Specific backend to use (optional)
            **kwargs: Additional model parameters
            
        Returns:
            bool: True if successful
        """
        try:
            # Choose strategy
            if backend:
                strategy = self.get_strategy(backend)
                if not strategy:
                    logger.error(f"Requested backend '{backend}' not available")
                    return False
            else:
                strategy = self.get_optimal_strategy()
                if not strategy:
                    logger.error("No VLM strategies available")
                    return False
            
            # Load model
            success = await strategy.load_model(model_name, **kwargs)
            if success:
                self._current_strategy = strategy
                logger.info(f"Loaded model '{model_name}' with {strategy.backend_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            return False
    
    async def generate_response(self,
                              image: Image.Image,
                              prompt: str,
                              max_tokens: int = 100,
                              temperature: float = 0.0,
                              **kwargs) -> VLMResponse:
        """
        Generate VLM response
        
        Args:
            image: PIL Image
            prompt: Text prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            VLMResponse: Generated response with metadata
        """
        if not self._current_strategy:
            raise VLMStrategyError("No VLM model loaded")
        
        import time
        start_time = time.time()
        
        try:
            text = await self._current_strategy.generate_response(
                image=image,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            inference_time = time.time() - start_time
            
            return VLMResponse(
                text=text,
                model_name=self._current_strategy.model_name,
                backend=self._current_strategy.backend_name,
                inference_time=inference_time,
                metadata={
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "image_size": image.size
                }
            )
            
        except Exception as e:
            logger.error(f"VLM generation failed: {e}")
            raise VLMStrategyError(f"Generation failed: {e}")
    
    def get_loaded_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about currently loaded model"""
        if not self._current_strategy:
            return None
        
        return {
            "model_name": self._current_strategy.model_name,
            "backend": self._current_strategy.backend_name,
            "performance_tier": self._current_strategy.performance_tier,
            "memory_efficient": self._current_strategy.memory_efficient,
            "loaded": self._current_strategy.is_loaded
        }
    
    def get_available_backends(self) -> List[Dict[str, Any]]:
        """Get list of available backends"""
        return [
            {
                "name": strategy.backend_name,
                "performance_tier": strategy.performance_tier,
                "memory_efficient": strategy.memory_efficient,
                "available": strategy.is_available()
            }
            for strategy in self._strategies.values()
        ]
    
    def unload_model(self):
        """Unload current model"""
        if self._current_strategy:
            self._current_strategy.unload_model()
            self._current_strategy = None
            logger.info("VLM model unloaded")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for VLM service"""
        strategies_status = {}
        
        for name, strategy in self._strategies.items():
            strategies_status[name] = await strategy.health_check()
        
        return {
            "service": "vlm",
            "strategies_available": len(self._strategies),
            "current_strategy": self._current_strategy.backend_name if self._current_strategy else None,
            "loaded_model": self.get_loaded_model_info(),
            "strategies": strategies_status
        }


# Global VLM service instance
_vlm_service = None

def get_vlm_service() -> VLMService:
    """Get global VLM service instance"""
    global _vlm_service
    if _vlm_service is None:
        _vlm_service = VLMService()
    return _vlm_service