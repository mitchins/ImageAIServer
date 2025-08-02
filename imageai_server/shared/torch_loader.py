"""PyTorch model loader with optional detection and lazy loading."""

import os
import logging
from typing import Dict, Optional, List, Any, Tuple
import numpy as np

from .model_backend import ModelBackend, ModelLoader, InferenceEngine, BackendConfig
from .model_types import ONNXModelConfig

logger = logging.getLogger(__name__)

# Check if PyTorch is available
TORCH_AVAILABLE = False
try:
    import torch
    import transformers
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, AutoProcessor,
        AutoModelForVision2Seq, AutoConfig
    )
    TORCH_AVAILABLE = True
    logger.info("PyTorch backend available")
except ImportError:
    logger.info("PyTorch backend not available - will use ONNX only")


class PyTorchBackend(ModelBackend):
    """PyTorch backend implementation."""
    
    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self.device = None
        if TORCH_AVAILABLE:
            # Set up device
            if config.device == "auto":
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                elif torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                else:
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device(config.device)
            logger.info(f"PyTorch using device: {self.device}")
    
    def is_available(self) -> bool:
        """Check if PyTorch backend is available."""
        return TORCH_AVAILABLE
    
    def get_supported_models(self) -> List[str]:
        """Return list of models supported by PyTorch backend (minimal set)."""
        return [
            # Ultra-lightweight vision-language models only
            "HuggingFaceTB/SmolVLM-256M-Instruct",  # 256M params - minimal VLM
            "HuggingFaceTB/SmolVLM-500M-Instruct",  # 500M params - still tiny
            
            # Note: For quantized versions, use GGUF instead:
            # - ggml-org/SmolVLM-256M-Instruct-GGUF (Q8_0: 175MB, F16: 328MB)
            # - lmstudio-community/granite-vision-3.2-2b-GGUF
        ]
    
    def supports_quantization(self, quant_type: str) -> bool:
        """Check if backend supports specific quantization type."""
        if not TORCH_AVAILABLE:
            return False
        
        # Minimal quantization support - native PyTorch only
        supported_quants = {
            "fp16": True,  # native torch
            "bf16": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
            # Note: For advanced quantization, use GGUF versions:
            # - ggml-org/SmolVLM-256M-Instruct-GGUF
            # - lmstudio-community/granite-vision-3.2-2b-GGUF
        }
        return supported_quants.get(quant_type.lower(), False)


class PyTorchModelLoader(ModelLoader):
    """Loader for PyTorch models."""
    
    def __init__(self, backend: PyTorchBackend):
        self.backend = backend
        self.models_cache: Dict[str, Any] = {}
        self.tokenizers_cache: Dict[str, Any] = {}
    
    def parse_model_name(self, name: str) -> Tuple[str, str]:
        """Parse model name into repo_id and revision."""
        if "@" in name:
            repo_id, revision = name.split("@", 1)
        else:
            repo_id = name
            revision = "main"
        return repo_id, revision
    
    def download_model_files(self, repo_id: str, **kwargs) -> Dict[str, str]:
        """Download model files - PyTorch uses lazy loading from HF."""
        # PyTorch/transformers handles downloading automatically
        # This method is mainly for compatibility
        return {"model": repo_id}
    
    def load_model(self, model_name: str) -> Tuple[Any, Any, Any]:
        """Load PyTorch model, tokenizer, and config."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch backend not available")
        
        if model_name in self.models_cache:
            return (
                self.models_cache[model_name],
                self.tokenizers_cache[model_name],
                None  # Config is embedded in model
            )
        
        repo_id, revision = self.parse_model_name(model_name)
        
        # Determine model loading arguments based on backend config
        model_kwargs = {
            "trust_remote_code": True,
            "revision": revision,
        }
        
        # Handle quantization - native PyTorch only (minimal approach)
        if self.backend.config.precision in ["int8", "int4"]:
            logger.warning(f"{self.backend.config.precision} quantization not supported in minimal backend. Use GGUF models for quantization. Falling back to FP16.")
            model_kwargs["torch_dtype"] = torch.float16
        elif self.backend.config.precision == "fp16":
            model_kwargs["torch_dtype"] = torch.float16
        elif self.backend.config.precision == "bf16" and self.backend.supports_quantization("bf16"):
            model_kwargs["torch_dtype"] = torch.bfloat16
        
        # Set device map for multi-GPU
        if self.backend.device.type == "cuda" and torch.cuda.device_count() > 1:
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = {"": self.backend.device}
        
        # Add any extra arguments from config
        model_kwargs.update(self.backend.config.extra_args)
        
        try:
            # Check if it's a vision-language model
            config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
            is_vision_model = (
                hasattr(config, 'model_type') and 'vision' in config.model_type.lower()
            ) or (
                hasattr(config, 'architectures') and 
                any('vision' in arch.lower() or 'vlm' in arch.lower() for arch in config.architectures)
            ) or repo_id in [
                "HuggingFaceTB/SmolVLM-256M-Instruct",
                "llava-hf/llava-1.5-7b-hf",
                "llava-hf/llava-1.5-13b-hf",
            ]
            
            # Load tokenizer/processor
            if is_vision_model:
                try:
                    # Try loading processor for vision models
                    processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
                    tokenizer = processor
                    logger.info(f"Loaded processor for vision model: {repo_id}")
                except:
                    # Fall back to tokenizer
                    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
                    logger.info(f"Loaded tokenizer for vision model: {repo_id}")
            else:
                tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
            
            # Load model
            logger.info(f"Loading PyTorch model: {repo_id} with {model_kwargs}")
            
            # Try vision model loaders first for vision models
            if is_vision_model:
                try:
                    # Check for Idefics3 models (like SmolVLM-256M)
                    if hasattr(config, 'model_type') and config.model_type == 'idefics3':
                        from transformers import Idefics3ForConditionalGeneration
                        model = Idefics3ForConditionalGeneration.from_pretrained(repo_id, **model_kwargs)
                        logger.info(f"Loaded as Idefics3 model: {repo_id}")
                    else:
                        model = AutoModelForVision2Seq.from_pretrained(repo_id, **model_kwargs)
                        logger.info(f"Loaded as Vision2Seq model: {repo_id}")
                except Exception as e:
                    logger.warning(f"Vision model loading failed: {e}")
                    # Fall back to causal LM
                    model = AutoModelForCausalLM.from_pretrained(repo_id, **model_kwargs)
                    logger.info(f"Loaded as CausalLM model: {repo_id}")
            else:
                model = AutoModelForCausalLM.from_pretrained(repo_id, **model_kwargs)
            
            # Move to device if not using device_map
            if "device_map" not in model_kwargs:
                model = model.to(self.backend.device)
            
            # Set to eval mode
            model.eval()
            
            # Cache the model
            self.models_cache[model_name] = model
            self.tokenizers_cache[model_name] = tokenizer
            
            logger.info(f"Successfully loaded PyTorch model: {repo_id}")
            return model, tokenizer, config
            
        except Exception as e:
            logger.error(f"Failed to load PyTorch model {repo_id}: {e}")
            raise


class PyTorchInferenceEngine(InferenceEngine):
    """Inference engine for PyTorch models."""
    
    def __init__(self, model: Any, tokenizer: Any, config: Any):
        super().__init__(model, tokenizer, config)
        self.model = model
        self.device = next(model.parameters()).device
    
    def generate_text(
        self, 
        text: str, 
        max_tokens: int = 100,
        images: Optional[List[str]] = None,
        audio: Optional[List[str]] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """Generate text using PyTorch model."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch backend not available")
        
        # Handle multimodal inputs if model supports them
        if images:
            # Check if we have a processor (vision models)
            if hasattr(self.tokenizer, 'image_processor') or hasattr(self.tokenizer, '__call__'):
                # Process images
                import base64
                import io
                from PIL import Image
                
                pil_images = []
                for img_base64 in images:
                    img_bytes = base64.b64decode(img_base64)
                    img = Image.open(io.BytesIO(img_bytes))
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    pil_images.append(img)
                
                # Use processor for vision models
                if hasattr(self.tokenizer, 'apply_chat_template'):
                    # SmolVLM style
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": text}
                        ]
                    }]
                    prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                    inputs = self.tokenizer(text=prompt, images=pil_images[0], return_tensors="pt")
                else:
                    # Generic vision model
                    inputs = self.tokenizer(text=text, images=pil_images[0], return_tensors="pt")
                
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            else:
                logger.warning("Vision input provided but model doesn't support vision")
                # Regular text-only tokenization
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            # Text-only input
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            generation_kwargs.update(kwargs)
            
            outputs = self.model.generate(
                **inputs,
                **generation_kwargs
            )
        
        # Decode output
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to embeddings using PyTorch model."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch backend not available")
        
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Use last hidden state of last token as embedding
            embeddings = outputs.hidden_states[-1][:, -1, :].cpu().numpy()
        
        return embeddings
    
    def supports_modality(self, modality: str) -> bool:
        """Check if model supports specific modality."""
        if modality == "text":
            return True
        elif modality == "vision":
            # Check if model has vision components
            return hasattr(self.model, "vision_tower") or hasattr(self.model, "vision_model")
        elif modality == "audio":
            # Check if model has audio components
            return hasattr(self.model, "audio_encoder") or hasattr(self.model, "audio_model")
        return False


# Factory function to create PyTorch backend
def create_pytorch_backend(config: Optional[BackendConfig] = None) -> Optional[PyTorchBackend]:
    """Create PyTorch backend if available."""
    if not TORCH_AVAILABLE:
        return None
    
    if config is None:
        config = BackendConfig(backend_type="pytorch", device="auto")
    
    return PyTorchBackend(config)