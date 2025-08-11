#!/usr/bin/env python3
"""
VLM Smoke Tests - Real Image Recognition Tests

Tests SmolVLM-256M with actual image recognition to prevent hallucination issues.
Uses pizza and anime images to ensure the models can correctly identify visual content.

Requirements:
- SmolVLM-256M model (150MB download on first run)
- Test images in tests/fixtures/
- PyTorch backend for SmolVLM
- MLX backend for Apple Silicon (optional)
"""

import pytest
import requests
import base64
import json
import sys
import os
from pathlib import Path
import subprocess
import socket
from contextlib import closing

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def find_free_port():
    """Find a free port to use for testing."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def vlm_test_server():
    """Start the test server before running VLM tests and stop it after."""
    port = find_free_port()
    base_url = f"http://localhost:{port}"
    
    # Start the server
    process = subprocess.Popen([
        sys.executable, "-c",
        f"""
import uvicorn
from imageai_server.main import app
uvicorn.run(app, host='127.0.0.1', port={port}, log_level='error')
"""
    ])
    
    # Wait for server to start
    max_retries = 30
    for _ in range(max_retries):
        try:
            response = requests.get(f"{base_url}/v1/health", timeout=1)
            if response.status_code == 200:
                break
        except:
            pass
        import time
        time.sleep(0.5)
    else:
        process.terminate()
        pytest.fail("Server failed to start within 15 seconds")
    
    yield base_url
    
    # Stop the server
    process.terminate()
    process.wait(timeout=5)


class TestVLMSmoke:
    """VLM Smoke tests with real image recognition."""
    
    @classmethod
    def setup_class(cls):
        """Set up test fixtures."""
        cls.fixtures_dir = Path(__file__).parent.parent / "fixtures"
        cls.pizza_image = cls.fixtures_dir / "real_pizza_from_wikipedia.png"
        cls.anime_image = cls.fixtures_dir / "anime-face-a.png"
        
        # Verify test images exist
        assert cls.pizza_image.exists(), f"Pizza test image not found: {cls.pizza_image}"
        assert cls.anime_image.exists(), f"Anime test image not found: {cls.anime_image}"
    
    def encode_image(self, image_path: Path) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def test_vision_models_available(self, vlm_test_server):
        """Test that vision models are available."""
        response = requests.get(f"{vlm_test_server}/v1/vision-models")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["data"]) > 0, "No vision models available"
        
        # Look for SmolVLM models
        smolvlm_models = [m for m in data["data"] if "smolvlm" in m["name"].lower() or "smol" in m["name"].lower()]
        assert len(smolvlm_models) > 0, "No SmolVLM models found"
        print(f"Found {len(smolvlm_models)} SmolVLM models")
    
    @pytest.mark.parametrize("backend", ["pytorch", "mlx"])
    def test_smolvlm_pizza_recognition(self, vlm_test_server, backend):
        """Test SmolVLM pizza recognition on different backends."""
        # Get available models
        response = requests.get(f"{vlm_test_server}/v1/vision-models")
        data = response.json()
        
        # Find appropriate SmolVLM model for backend
        target_models = []
        for model in data["data"]:
            model_name = model["name"].lower()
            model_backend = model["backend"].lower()
            
            if backend == "pytorch" and ("torch" in model_backend or model_backend == "pytorch"):
                if "smolvlm" in model_name or "smol" in model_name:
                    target_models.append(model)
            elif backend == "mlx" and "mlx" in model_backend:
                if "smolvlm" in model_name or "smol" in model_name:
                    target_models.append(model)
        
        if not target_models:
            pytest.skip(f"No SmolVLM models available for {backend} backend")
        
        # Use the first available model
        model = target_models[0]
        model_id = model["id"]
        
        print(f"Testing {backend} model: {model['name']} ({model_id})")
        
        # Encode pizza image
        pizza_base64 = self.encode_image(self.pizza_image)
        
        # Test pizza recognition with neutral prompt
        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "user", 
                    "content": "What do you see in this image?"
                }
            ],
            "images": [pizza_base64],
            "max_tokens": 50
        }
        
        response = requests.post(
            f"{vlm_test_server}/v1/chat/completions",
            json=payload,
            timeout=60  # VLM inference can be slow
        )
        
        assert response.status_code == 200, f"Request failed: {response.text}"
        
        result = response.json()
        assert "choices" in result
        assert len(result["choices"]) > 0
        
        generated_text = result["choices"][0]["message"]["content"].lower()
        print(f"Generated text: {generated_text}")
        
        # Check for pizza-related terms (prevent hallucination)
        pizza_terms = ["pizza", "food", "cheese", "crust", "slice", "italian", "dish", "meal"]
        found_terms = [term for term in pizza_terms if term in generated_text]
        
        assert len(found_terms) > 0, f"No pizza-related terms found in: {generated_text}"
        print(f"✅ {backend.upper()} pizza recognition successful! Found terms: {found_terms}")
    
    @pytest.mark.parametrize("backend", ["pytorch", "mlx"])
    def test_smolvlm_anime_recognition(self, vlm_test_server, backend):
        """Test SmolVLM anime character recognition on different backends."""
        # Get available models
        response = requests.get(f"{vlm_test_server}/v1/vision-models")
        data = response.json()
        
        # Find appropriate SmolVLM model for backend
        target_models = []
        for model in data["data"]:
            model_name = model["name"].lower()
            model_backend = model["backend"].lower()
            
            if backend == "pytorch" and ("torch" in model_backend or model_backend == "pytorch"):
                if "smolvlm" in model_name or "smol" in model_name:
                    target_models.append(model)
            elif backend == "mlx" and "mlx" in model_backend:
                if "smolvlm" in model_name or "smol" in model_name:
                    target_models.append(model)
        
        if not target_models:
            pytest.skip(f"No SmolVLM models available for {backend} backend")
        
        # Use the first available model
        model = target_models[0]
        model_id = model["id"]
        
        print(f"Testing {backend} model: {model['name']} ({model_id})")
        
        # Encode anime image
        anime_base64 = self.encode_image(self.anime_image)
        
        # Test anime recognition with neutral prompt
        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "user", 
                    "content": "Describe what you see in this image."
                }
            ],
            "images": [anime_base64],
            "max_tokens": 50
        }
        
        response = requests.post(
            f"{vlm_test_server}/v1/chat/completions",
            json=payload,
            timeout=60  # VLM inference can be slow
        )
        
        assert response.status_code == 200, f"Request failed: {response.text}"
        
        result = response.json()
        assert "choices" in result
        assert len(result["choices"]) > 0
        
        generated_text = result["choices"][0]["message"]["content"].lower()
        print(f"Generated text: {generated_text}")
        
        # Check for character/person/face-related terms (prevent hallucination)
        character_terms = ["character", "person", "girl", "woman", "face", "anime", "cartoon", "illustration", "drawing", "art", "portrait"]
        found_terms = [term for term in character_terms if term in generated_text]
        
        assert len(found_terms) > 0, f"No character-related terms found in: {generated_text}"
        print(f"✅ {backend.upper()} anime recognition successful! Found terms: {found_terms}")
    
    def test_smolvlm_model_comparison(self, vlm_test_server):
        """Test that different SmolVLM models give consistent results."""
        # Get all available SmolVLM models
        response = requests.get(f"{vlm_test_server}/v1/vision-models")
        data = response.json()
        
        smolvlm_models = []
        for model in data["data"]:
            model_name = model["name"].lower()
            if "smolvlm" in model_name or "smol" in model_name:
                smolvlm_models.append(model)
        
        if len(smolvlm_models) < 2:
            pytest.skip("Need at least 2 SmolVLM models for comparison test")
        
        # Test first 2 models with pizza image
        pizza_base64 = self.encode_image(self.pizza_image)
        results = []
        
        for model in smolvlm_models[:2]:  # Test first 2 models
            payload = {
                "model": model["id"],
                "messages": [
                    {
                        "role": "user", 
                        "content": "What food is shown in this image?"
                    }
                ],
                "images": [pizza_base64],
                "max_tokens": 30
            }
            
            response = requests.post(
                f"{vlm_test_server}/v1/chat/completions",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result["choices"][0]["message"]["content"].lower()
                results.append((model["name"], text))
                print(f"{model['name']}: {text}")
        
        # Both should mention pizza/food
        assert len(results) >= 2, "Not enough successful results for comparison"
        
        pizza_terms = ["pizza", "food", "cheese", "italian"]
        for model_name, text in results:
            found_terms = [term for term in pizza_terms if term in text]
            assert len(found_terms) > 0, f"Model {model_name} failed to recognize pizza: {text}"
        
        print("✅ Model consistency check passed!")


if __name__ == "__main__":
    # Run VLM smoke tests
    pytest.main([__file__, "-v", "-s"])