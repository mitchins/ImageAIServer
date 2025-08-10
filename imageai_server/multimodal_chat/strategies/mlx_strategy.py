"""
MLX VLM Strategy - Apple Silicon optimized Vision Language Models
Based on our successful Gemma-3N testing
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from PIL import Image
import platform

from .base_strategy import BaseVLMStrategy, ModelNotLoadedError, UnsupportedModelError

logger = logging.getLogger(__name__)


class MLXVLMStrategy(BaseVLMStrategy):
    """MLX-based VLM strategy for Apple Silicon devices"""
    
    def __init__(self):
        super().__init__()
        self._mlx_vlm = None
    
    @property
    def backend_name(self) -> str:
        return "mlx"
    
    @property
    def performance_tier(self) -> int:
        return 1  # Fastest on Apple Silicon
    
    @property
    def memory_efficient(self) -> bool:
        return True
    
    def is_available(self) -> bool:
        """Check if MLX VLM is available on this system"""
        try:
            # Must be on macOS
            if platform.system() != "Darwin":
                return False
            
            # Try importing MLX VLM
            import mlx_vlm
            return True
            
        except ImportError:
            logger.debug("MLX VLM not available: import failed")
            return False
    
    async def load_model(self, model_name: str, **kwargs) -> bool:
        """Load MLX VLM model"""
        if not self.is_available():
            logger.error("MLX VLM not available on this system")
            return False
        
        try:
            # Import here to avoid issues on non-Apple systems
            from mlx_vlm import load
            
            logger.info(f"Loading MLX VLM model: {model_name}")
            
            # Run in executor to avoid blocking
            self.model, self.processor = await self._run_in_executor(
                load, model_name
            )
            
            self.model_name = model_name
            self._loaded = True
            
            logger.info(f"✅ MLX VLM model loaded successfully: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load MLX VLM model {model_name}: {e}")
            self._loaded = False
            return False
    
    async def generate_response(self, 
                              image: Image.Image,
                              prompt: str,
                              max_tokens: int = 100,
                              temperature: float = 0.0,
                              **kwargs) -> str:
        """Generate response using MLX VLM"""
        if not self._loaded:
            raise ModelNotLoadedError("MLX VLM model not loaded")
        
        try:
            from mlx_vlm import generate
            
            # Format prompt with appropriate image tokens
            formatted_prompt = self._format_prompt(prompt, self.model_name)
            
            logger.debug(f"Generating response with prompt: {formatted_prompt[:100]}...")
            
            # Run generation in executor
            response = await self._run_in_executor(
                self._generate_sync,
                formatted_prompt,
                image,
                max_tokens,
                temperature
            )
            
            # Extract text from response
            text = response.text if hasattr(response, 'text') else str(response)
            
            logger.debug(f"Generated response: {text[:100]}...")
            return text.strip()
            
        except Exception as e:
            logger.error(f"MLX VLM generation failed: {e}")
            raise
    
    def _generate_sync(self, prompt: str, image: Image.Image, max_tokens: int, temperature: float):
        """Synchronous generation helper"""
        from mlx_vlm import generate
        
        return generate(
            model=self.model,
            processor=self.processor,
            prompt=prompt,
            image=image,
            max_tokens=max_tokens,
            temperature=temperature
        )
    
    def _format_prompt(self, prompt: str, model_name: str) -> str:
        """Format prompt with model-specific image tokens"""
        if not model_name:
            return prompt
        
        model_lower = model_name.lower()
        
        # Model-specific image token formatting
        if "gemma-3n" in model_lower or "gemma3n" in model_lower:
            return f"<start_of_image><image_soft_token><end_of_image>{prompt}"
        
        elif "llava" in model_lower:
            return f"<image>{prompt}"
        
        elif "qwen2-vl" in model_lower or "qwen2vl" in model_lower:
            # Qwen2-VL specific formatting
            return f"<|image|>{prompt}"
        
        elif "smolvlm" in model_lower:
            return f"<image>{prompt}"
        
        else:
            # Default format - most models use <image>
            logger.warning(f"Unknown model format for {model_name}, using default <image> token")
            return f"<image>{prompt}"
    
    def get_supported_models(self) -> Dict[str, Dict[str, Any]]:
        """Get list of supported MLX VLM models"""
        return {
            "gemma-3n-e2b-4bit": {
                "model_id": "lmstudio-community/gemma-3n-E2B-it-MLX-4bit",
                "size": "3B",
                "quantization": "4bit",
                "description": "Gemma 3N E2B vision model, 4-bit quantized",
                "performance_tier": 1,
                "tested": True
            },
            "gemma-3n-e2b-6bit": {
                "model_id": "lmstudio-community/gemma-3n-E2B-it-MLX-6bit",
                "size": "3B", 
                "quantization": "6bit",
                "description": "Gemma 3N E2B vision model, 6-bit quantized",
                "performance_tier": 1,
                "tested": False
            },
            "llava-interleave-qwen-0.5b": {
                "model_id": "mlx-community/llava-interleave-qwen-0.5b-bf16",
                "size": "0.5B",
                "quantization": "bf16",
                "description": "Small LLaVA model with Qwen backbone",
                "performance_tier": 2,
                "tested": False
            },
            "smolvlm-256m": {
                "model_id": "mlx-community/SmolVLM-256M-bf16",
                "size": "0.256B",
                "quantization": "bf16", 
                "description": "Very small VLM, fastest inference",
                "performance_tier": 1,
                "tested": False
            }
        }
    
    def get_model_info(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        supported = self.get_supported_models()
        return supported.get(model_key)
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for MLX VLM strategy"""
        return {
            "backend": self.backend_name,
            "available": self.is_available(),
            "loaded": self._loaded,
            "model": self.model_name,
            "platform": platform.system(),
            "architecture": platform.machine()
        }