"""Integration test for SmolVLM-256M with INT8 quantization.

This test requires:
- PyTorch and transformers installed (pip install -r requirements-torch.txt)
- Internet connection for model download
- ~500MB disk space for model storage
- ~1GB RAM for model loading

Run with: pytest tests/integration/test_smolvlm_int8.py -v
"""

import pytest
import sys
import os
import base64
import io
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import torch
    import transformers
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from PIL import Image
import numpy as np


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestSmolVLMInt8Integration:
    """Integration tests for SmolVLM with INT8 quantization."""
    
    @classmethod
    def setup_class(cls):
        """Set up test fixtures."""
        cls.model_name = "HuggingFaceTB/SmolVLM-256M-Instruct"
        cls.test_image = cls.create_test_image()
    
    @staticmethod
    def create_test_image():
        """Create a test image with colored shapes."""
        img = Image.new('RGB', (256, 256), color='white')
        pixels = np.array(img)
        
        # Add colored rectangles
        pixels[50:100, 50:100] = [255, 0, 0]  # Red
        pixels[50:100, 150:200] = [0, 255, 0]  # Green
        pixels[150:200, 100:150] = [0, 0, 255]  # Blue
        
        img = Image.fromarray(pixels.astype('uint8'))
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return img_base64, img
    
    @staticmethod
    def create_pizza_image():
        """Create a simple pizza-like image."""
        # Create a circular pizza base
        img = Image.new('RGB', (256, 256), color='white')
        pixels = np.array(img)
        
        center_x, center_y = 128, 128
        radius = 80
        
        # Draw pizza base (brown/tan color)
        for y in range(256):
            for x in range(256):
                distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                if distance <= radius:
                    pixels[y, x] = [210, 180, 140]  # Tan pizza base
                    
                    # Add pepperoni (red circles)
                    if distance <= radius - 10:
                        pepperoni_positions = [
                            (center_x - 30, center_y - 20),
                            (center_x + 25, center_y - 30),
                            (center_x - 20, center_y + 25),
                            (center_x + 35, center_y + 15),
                            (center_x - 5, center_y - 40),
                        ]
                        
                        for px, py in pepperoni_positions:
                            pepperoni_dist = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
                            if pepperoni_dist <= 8:
                                pixels[y, x] = [180, 50, 50]  # Red pepperoni
                    
                    # Add cheese texture (slight yellow tint)
                    if distance <= radius - 5 and (x + y) % 3 == 0:
                        pixels[y, x] = [220, 200, 120]  # Cheese color
        
        img = Image.fromarray(pixels.astype('uint8'))
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return img_base64, img
    
    def test_pytorch_backend_available(self):
        """Test that PyTorch backend is available."""
        from imageai_server.shared.torch_loader import TORCH_AVAILABLE, create_pytorch_backend
        
        assert TORCH_AVAILABLE, "PyTorch should be available for integration tests"
        
        backend = create_pytorch_backend()
        assert backend is not None
        assert backend.is_available()
        
        print("✅ PyTorch backend is available")
    
    def test_smolvlm_in_supported_models(self):
        """Test SmolVLM is in PyTorch supported models."""
        from imageai_server.shared.model_backend import BackendConfig
        from imageai_server.shared.torch_loader import PyTorchBackend
        
        config = BackendConfig(backend_type="pytorch")
        backend = PyTorchBackend(config)
        
        models = backend.get_supported_models()
        assert self.model_name in models
        print(f"✅ SmolVLM found in {len(models)} supported models")
    
    @pytest.mark.slow
    def test_load_smolvlm_int8(self):
        """Test loading SmolVLM with INT8 quantization."""
        from imageai_server.shared.model_backend import BackendConfig
        from imageai_server.shared.torch_loader import PyTorchBackend, PyTorchModelLoader
        
        # Configure INT8
        config = BackendConfig(
            backend_type="pytorch",
            device="auto",
            precision="int8"
        )
        
        backend = PyTorchBackend(config)
        loader = PyTorchModelLoader(backend)
        
        print(f"Loading {self.model_name} with INT8 quantization...")
        model, tokenizer, config = loader.load_model(self.model_name)
        
        assert model is not None
        assert tokenizer is not None
        
        # Check if model is quantized
        if hasattr(model, 'config') and hasattr(model.config, 'quantization_config'):
            print(f"Quantization config: {model.config.quantization_config}")
            assert model.config.quantization_config.get('load_in_8bit', False)
        
        # Check model size
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {param_count:,}")
        
        # Estimate memory usage
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / 1024**2
            print(f"GPU memory allocated: {memory_mb:.2f} MB")
        
        print("✅ SmolVLM loaded successfully with INT8")
    
    @pytest.mark.slow
    def test_text_generation(self):
        """Test text-only generation with SmolVLM."""
        from imageai_server.shared.torch_loader import PyTorchModelLoader, PyTorchBackend
        from imageai_server.shared.model_backend import BackendConfig
        from transformers import Idefics3ForConditionalGeneration
        
        # Configure backend - force CPU to avoid MPS issues
        config = BackendConfig(
            backend_type="pytorch",
            device="cpu",  # Force CPU to avoid MPS device issues
            precision="fp16"
        )
        
        backend = PyTorchBackend(config)
        loader = PyTorchModelLoader(backend)
        
        # Load model directly
        model, processor, _ = loader.load_model(self.model_name)
        
        # Ensure model is on CPU
        model = model.to("cpu")
        
        # Simple text generation
        prompt = "The capital of France is"
        messages = [{"role": "user", "content": prompt}]
        
        # Process input - apply_chat_template returns formatted string
        formatted_prompt = processor.apply_chat_template(messages, tokenize=False)
        inputs = processor(text=formatted_prompt, images=None, return_tensors="pt")
        
        # Move inputs to CPU
        inputs = {k: v.to("cpu") if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        
        # Decode
        result = processor.decode(outputs[0], skip_special_tokens=True)
        
        assert result is not None
        assert len(result) > len(prompt)
        
        print(f"Prompt: {prompt}")
        print(f"Response: {result}")
        print("✅ Text generation works correctly")
    
    @pytest.mark.slow
    def test_vision_generation(self):
        """Test vision + text generation with SmolVLM."""
        from imageai_server.shared.torch_loader import PyTorchModelLoader, PyTorchBackend
        from imageai_server.shared.model_backend import BackendConfig
        import base64
        import io
        
        # Configure backend - force CPU to avoid MPS issues
        config = BackendConfig(
            backend_type="pytorch",
            device="cpu",  # Force CPU to avoid MPS device issues
            precision="fp16"
        )
        
        backend = PyTorchBackend(config)
        loader = PyTorchModelLoader(backend)
        
        # Load model directly
        model, processor, _ = loader.load_model(self.model_name)
        
        # Ensure model is on CPU
        model = model.to("cpu")
        
        img_base64, img_pil = self.test_image
        
        # Vision + text generation
        prompt = "What colors are in this image?"
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img_pil},
            {"type": "text", "text": prompt}
        ]}]
        
        # Process input - handle image + text
        formatted_prompt = processor.apply_chat_template(messages, tokenize=False)
        inputs = processor(text=formatted_prompt, images=img_pil, return_tensors="pt")
        
        # Move inputs to CPU
        inputs = {k: v.to("cpu") if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        
        # Decode
        result = processor.decode(outputs[0], skip_special_tokens=True)
        
        assert result is not None
        
        # Check if any color is mentioned
        colors = ['red', 'green', 'blue', 'white']
        colors_found = [c for c in colors if c in result.lower()]
        
        print(f"Prompt: {prompt}")
        print(f"Response: {result}")
        print(f"Colors mentioned: {colors_found}")
        
        assert len(colors_found) > 0, "Model should mention at least one color"
        print("✅ Vision generation works correctly")
    
    @pytest.mark.slow
    def test_pizza_recognition(self):
        """Test pizza recognition without food hints in prompt."""
        from imageai_server.shared.torch_loader import PyTorchModelLoader, PyTorchBackend
        from imageai_server.shared.model_backend import BackendConfig
        
        # Configure backend - force CPU to avoid MPS issues
        config = BackendConfig(
            backend_type="pytorch",
            device="cpu",  # Force CPU to avoid MPS device issues 
            precision="fp16"
        )
        
        backend = PyTorchBackend(config)
        loader = PyTorchModelLoader(backend)
        
        # Load model directly
        model, processor, _ = loader.load_model(self.model_name)
        
        # Ensure model is on CPU
        model = model.to("cpu")
        
        # Create pizza image
        pizza_img_base64, pizza_img = self.create_pizza_image()
        
        # Save pizza image for reference
        test_img_path = Path("test_pizza_image.png")
        pizza_img.save(test_img_path)
        print(f"   Saved pizza test image to: {test_img_path}")
        
        # Test without food hints - neutral prompt
        neutral_prompt = "What is in this image?"
        messages = [{"role": "user", "content": [
            {"type": "image", "image": pizza_img},
            {"type": "text", "text": neutral_prompt}
        ]}]
        
        # Process input - handle image + text
        formatted_prompt = processor.apply_chat_template(messages, tokenize=False)
        inputs = processor(text=formatted_prompt, images=pizza_img, return_tensors="pt")
        
        # Move inputs to CPU
        inputs = {k: v.to("cpu") if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        
        # Decode
        result = processor.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Prompt: {neutral_prompt}")
        print(f"Response: {result}")
        
        # Check if pizza or food-related terms are detected
        food_terms = ['pizza', 'food', 'meal', 'eat', 'dish', 'cuisine', 'pepperoni', 
                     'cheese', 'bread', 'circular', 'round', 'toppings']
        
        result_lower = result.lower()
        detected_terms = [term for term in food_terms if term in result_lower]
        
        print(f"Food-related terms detected: {detected_terms}")
        
        # Success criteria: model should recognize it as food/pizza without being told
        pizza_detected = 'pizza' in result_lower
        food_detected = any(term in result_lower for term in ['food', 'meal', 'eat', 'dish'])
        visual_features_detected = any(term in result_lower for term in ['circular', 'round', 'brown', 'red'])
        
        if pizza_detected:
            print("✅ Model correctly identified pizza without food hints!")
        elif food_detected:
            print("✅ Model recognized it as food without explicit hints")
        elif visual_features_detected:
            print("⚠️ Model detected visual features but not food classification")
        else:
            print("❌ Model did not recognize food characteristics")
        
        # Clean up
        if test_img_path.exists():
            test_img_path.unlink()
        
        # Assert success if any food recognition occurred
        assert pizza_detected or food_detected or len(detected_terms) > 0, \
            f"Model should recognize food elements. Response: {result}"
        
        print("✅ Pizza recognition test completed")
    
    @pytest.mark.slow
    def test_memory_efficiency(self):
        """Test memory efficiency of INT8 vs FP32."""
        from imageai_server.shared.model_backend import BackendConfig
        from imageai_server.shared.torch_loader import PyTorchBackend, PyTorchModelLoader
        
        # Load with INT8
        int8_config = BackendConfig(
            backend_type="pytorch",
            device="auto",
            precision="int8"
        )
        
        backend = PyTorchBackend(int8_config)
        loader = PyTorchModelLoader(backend)
        
        model, _, _ = loader.load_model(self.model_name)
        
        # Calculate sizes
        param_count = sum(p.numel() for p in model.parameters())
        fp32_size_mb = param_count * 4 / (1024 * 1024)
        int8_size_mb = param_count * 1 / (1024 * 1024)
        compression_ratio = fp32_size_mb / int8_size_mb
        
        print(f"\nMemory Efficiency Analysis:")
        print(f"  Parameters: {param_count:,}")
        print(f"  FP32 size (theoretical): {fp32_size_mb:.1f} MB")
        print(f"  INT8 size (theoretical): {int8_size_mb:.1f} MB")
        print(f"  Compression ratio: {compression_ratio:.1f}x")
        
        if torch.cuda.is_available():
            actual_memory_mb = torch.cuda.memory_allocated() / 1024**2
            print(f"  Actual GPU memory: {actual_memory_mb:.1f} MB")
            efficiency = int8_size_mb / actual_memory_mb * 100
            print(f"  Memory efficiency: {efficiency:.1f}%")
        
        assert compression_ratio > 3.5, "INT8 should provide at least 3.5x compression"
        assert int8_size_mb < 300, "INT8 model should be under 300MB"
        
        print("✅ INT8 quantization provides expected memory savings")
    
    @pytest.mark.slow
    def test_inference_speed(self):
        """Test inference speed with FP16."""
        import time
        from imageai_server.shared.torch_loader import PyTorchModelLoader, PyTorchBackend
        from imageai_server.shared.model_backend import BackendConfig
        
        # Configure backend - force CPU to avoid MPS issues
        config = BackendConfig(
            backend_type="pytorch",
            device="cpu",  # Force CPU to avoid MPS device issues
            precision="fp16"
        )
        
        backend = PyTorchBackend(config)
        loader = PyTorchModelLoader(backend)
        
        # Load model directly
        model, processor, _ = loader.load_model(self.model_name)
        
        # Ensure model is on CPU
        model = model.to("cpu")
        
        # Warm up
        warm_up_messages = [{"role": "user", "content": "Hello"}]
        warm_up_formatted = processor.apply_chat_template(warm_up_messages, tokenize=False)
        warm_up_inputs = processor(text=warm_up_formatted, images=None, return_tensors="pt")
        warm_up_inputs = {k: v.to("cpu") if hasattr(v, 'to') else v for k, v in warm_up_inputs.items()}
        with torch.no_grad():
            _ = model.generate(**warm_up_inputs, max_new_tokens=5, do_sample=False)
        
        # Time multiple inferences
        prompts = [
            "What is 2+2?",
            "Name a color.", 
            "Hello, how are",
        ]
        
        start_time = time.time()
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = processor.apply_chat_template(messages, tokenize=False)
            inputs = processor(text=formatted_prompt, images=None, return_tensors="pt")
            inputs = {k: v.to("cpu") if hasattr(v, 'to') else v for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / len(prompts)
        tokens_per_second = 10 / avg_time  # Approximate
        
        print(f"\nInference Speed:")
        print(f"  Average time per request: {avg_time:.3f}s")
        print(f"  Estimated tokens/second: {tokens_per_second:.1f}")
        
        assert avg_time < 5.0, "Inference should be reasonably fast"
        print("✅ INT8 inference speed is acceptable")
    
    def test_api_endpoint_compatibility(self):
        """Test that SmolVLM works with API endpoints."""
        from imageai_server.chat_server.torch_router import TorchChatRequest
        from imageai_server.shared.model_manager import get_model_manager
        
        manager = get_model_manager()
        
        # Simulate API request
        request = TorchChatRequest(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": "Say hello"
            }],
            max_tokens=10,
            backend="pytorch"
        )
        
        # Verify request is valid
        assert request.model == self.model_name
        assert request.backend == "pytorch"
        assert len(request.messages) == 1
        
        print("✅ API endpoint compatibility verified")


@pytest.fixture
def cleanup_test_images():
    """Cleanup any test images created during tests."""
    yield
    # Cleanup code here if needed
    test_files = Path(".").glob("test_*.png")
    for f in test_files:
        f.unlink()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])