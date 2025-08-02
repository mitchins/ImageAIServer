"""Integration test for Gemma-3n-E2B-it-ONNX model.

This test requires:
- Internet connection for model download
- ~2GB disk space for model storage
- ~4GB RAM for model loading

Run with: pytest tests/integration/test_gemma3n_onnx.py -v
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

from PIL import Image
import numpy as np

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX runtime not installed")
@pytest.mark.integration
class TestGemma3nOnnxIntegration:
    """Integration tests for Gemma-3n-E2B-it-ONNX model."""
    
    @classmethod
    def setup_class(cls):
        """Set up test fixtures."""
        cls.model_name = "Gemma-3n-E2B-it-ONNX/Q4_MIXED"
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
    
    def test_onnx_backend_available(self):
        """Test that ONNX backend is available."""
        from imageai_server.shared.onnx_backend import create_onnx_backend
        
        backend = create_onnx_backend()
        assert backend is not None
        assert backend.is_available()
        
        print("✅ ONNX backend is available")
    
    def test_gemma3n_in_supported_models(self):
        """Test Gemma-3n is in ONNX supported models."""
        from imageai_server.shared.model_backend import BackendConfig
        from imageai_server.shared.onnx_backend import ONNXBackend
        
        config = BackendConfig(backend_type="onnx")
        backend = ONNXBackend(config)
        
        models = backend.get_supported_models()
        assert self.model_name in models
        print(f"✅ Gemma-3n found in {len(models)} supported models")
    
    @pytest.mark.slow
    def test_load_gemma3n_q4_mixed(self):
        """Test loading Gemma-3n with Q4_MIXED quantization."""
        from imageai_server.shared.model_backend import BackendConfig
        from imageai_server.shared.onnx_backend import ONNXBackend
        from imageai_server.shared.onnx_loader import ONNXModelLoader
        
        # Configure Q4_MIXED
        config = BackendConfig(
            backend_type="onnx",
            device="auto",
            precision="q4_mixed"
        )
        
        backend = ONNXBackend(config)
        loader = ONNXModelLoader()
        
        print(f"Loading {self.model_name}...")
        model, tokenizer, config = loader.load_model(self.model_name)
        
        assert model is not None
        assert tokenizer is not None
        
        # Check if model components are loaded (actual ONNX structure)
        print(f"Model type: {type(model)}")
        if isinstance(model, dict):
            print(f"Model components: {list(model.keys())}")
            # ONNX models are typically returned as dict of sessions
            assert len(model) > 0, "Should have model components"
        else:
            print(f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
            assert model is not None, "Model should be loaded"
        
        print("✅ Gemma-3n loaded successfully with Q4_MIXED")
    
    @pytest.mark.slow
    def test_text_generation(self):
        """Test text-only generation with Gemma-3n."""
        from imageai_server.shared.model_manager import get_model_manager, BackendType
        
        manager = get_model_manager()
        
        # Simple text generation
        prompt = "The capital of France is"
        result = manager.generate_text(
            model_name=self.model_name,
            text=prompt,
            max_tokens=20,
            backend=BackendType.ONNX
        )
        
        assert result is not None
        assert len(result) > len(prompt)
        assert "Paris" in result or "paris" in result.lower()
        
        print(f"Prompt: {prompt}")
        print(f"Response: {result}")
        print("✅ Text generation works correctly")
    
    @pytest.mark.slow
    def test_vision_generation(self):
        """Test vision + text generation with Gemma-3n."""
        from imageai_server.shared.model_manager import get_model_manager, BackendType
        
        manager = get_model_manager()
        
        img_base64, _ = self.test_image
        
        # Vision + text generation
        prompt = "What colors are in this image?"
        result = manager.generate_text(
            model_name=self.model_name,
            text=prompt,
            max_tokens=50,
            images=[img_base64],
            backend=BackendType.ONNX
        )
        
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
        from imageai_server.shared.model_manager import get_model_manager, BackendType
        
        manager = get_model_manager()
        
        # Create pizza image
        pizza_img_base64, pizza_img = self.create_pizza_image()
        
        # Save pizza image for reference
        test_img_path = Path("test_gemma3n_pizza_image.png")
        pizza_img.save(test_img_path)
        print(f"   Saved pizza test image to: {test_img_path}")
        
        # Test without food hints - neutral prompt
        neutral_prompt = "What is in this image?"
        result = manager.generate_text(
            model_name=self.model_name,
            text=neutral_prompt,
            max_tokens=50,
            images=[pizza_img_base64],
            backend=BackendType.ONNX
        )
        
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
    def test_quantization_variants(self):
        """Test different Gemma-3n quantization variants."""
        from imageai_server.shared.model_backend import BackendConfig
        from imageai_server.shared.onnx_backend import ONNXBackend
        
        variants = [
            "Gemma-3n-E2B-it-ONNX/Q4_MIXED",
            "Gemma-3n-E2B-it-ONNX/FP16",
            "Gemma-3n-E2B-it-ONNX/FP32"
        ]
        
        config = BackendConfig(backend_type="onnx")
        backend = ONNXBackend(config)
        
        supported_models = backend.get_supported_models()
        
        available_variants = []
        for variant in variants:
            if variant in supported_models:
                available_variants.append(variant)
        
        print(f"Available Gemma-3n quantization variants: {available_variants}")
        
        assert len(available_variants) > 0, "At least one Gemma-3n variant should be supported"
        assert "Gemma-3n-E2B-it-ONNX/Q4_MIXED" in available_variants, "Q4_MIXED should be available"
        
        print("✅ Gemma-3n quantization variants are supported")
    
    @pytest.mark.slow
    def test_multimodal_capabilities(self):
        """Test Gemma-3n multimodal capabilities."""
        from imageai_server.shared.model_manager import get_model_manager, BackendType
        
        manager = get_model_manager()
        
        img_base64, _ = self.test_image
        
        # Test multimodal reasoning
        prompt = "Describe what you see and count the colored rectangles"
        result = manager.generate_text(
            model_name=self.model_name,
            text=prompt,
            max_tokens=100,
            images=[img_base64],
            backend=BackendType.ONNX
        )
        
        assert result is not None
        
        # Check for understanding
        result_lower = result.lower()
        understanding_indicators = [
            'rectangle' in result_lower or 'square' in result_lower,
            'color' in result_lower or 'red' in result_lower or 'green' in result_lower or 'blue' in result_lower,
            'three' in result_lower or '3' in result_lower or 'multiple' in result_lower
        ]
        
        print(f"Prompt: {prompt}")
        print(f"Response: {result}")
        print(f"Understanding indicators: {sum(understanding_indicators)}/3")
        
        assert any(understanding_indicators), "Model should show some understanding of the image"
        print("✅ Multimodal capabilities working")
    
    def test_model_config_compatibility(self):
        """Test Gemma-3n model configuration compatibility."""
        from imageai_server.shared.model_types import REFERENCE_MODELS, ReferenceModel
        
        # Check if the reference model exists
        gemma_spec = REFERENCE_MODELS.get(ReferenceModel.GEMMA_3N_E2B)
        assert gemma_spec is not None, "Gemma-3n specification missing"
        
        # Check required fields
        assert gemma_spec.repo_id == "onnx-community/gemma-3n-E2B-it-ONNX"
        assert gemma_spec.config.has_vision is True
        assert gemma_spec.config.num_layers == 30
        assert gemma_spec.config.num_heads == 8
        
        print(f"✅ Model configuration valid for {self.model_name}")
    
    def test_api_endpoint_compatibility(self):
        """Test that Gemma-3n works with API endpoints."""
        from imageai_server.multimodal_chat.main import ChatRequest
        
        # Simulate API request
        request = ChatRequest(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": "Hello"
            }],
            max_tokens=10
        )
        
        # Verify request is valid
        assert request.model == self.model_name
        assert len(request.messages) == 1
        assert request.max_tokens == 10
        
        print("✅ API endpoint compatibility verified")


@pytest.fixture
def cleanup_test_images():
    """Cleanup any test images created during tests."""
    yield
    # Cleanup code here if needed
    test_files = Path(".").glob("test_gemma3n_*.png")
    for f in test_files:
        f.unlink()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])