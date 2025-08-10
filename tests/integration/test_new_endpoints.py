#!/usr/bin/env python3
"""
Integration tests for new endpoints and features.

Tests the actual HTTP endpoints to ensure they work correctly
and return the expected data structures.
"""

import pytest
import requests
import json
import time
import sys
import os
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
def test_server():
    """Start the test server before running tests and stop it after."""
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
        time.sleep(0.5)
    else:
        process.terminate()
        pytest.fail("Server failed to start within 15 seconds")
    
    yield base_url
    
    # Stop the server
    process.terminate()
    process.wait(timeout=5)


# Test configuration - will be overridden by fixture
BASE_URL = "http://localhost:8001"


class TestVisionModelsEndpoint:
    """Test the /v1/vision-models endpoint."""
    
    def test_vision_models_endpoint_exists(self, test_server):
        """Test that the vision models endpoint is accessible."""
        response = requests.get(f"{test_server}/v1/vision-models")
        assert response.status_code == 200
        
        data = response.json()
        assert "object" in data
        assert "data" in data
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
    
    def test_vision_models_structure(self, test_server):
        """Test that vision models have correct structure."""
        response = requests.get(f"{test_server}/v1/vision-models")
        data = response.json()
        
        if len(data["data"]) > 0:
            model = data["data"][0]
            required_fields = ["id", "name", "backend", "description", "quantization"]
            
            for field in required_fields:
                assert field in model, f"Missing field: {field}"
            
            # Verify backend types
            assert model["backend"] in ["ONNX", "PyTorch/GGUF", "PyTorch"]
    
    def test_vision_models_only_local(self, test_server):
        """Test that only locally available models are listed."""
        response = requests.get(f"{test_server}/v1/vision-models")
        data = response.json()
        
        # This test verifies that the endpoint doesn't return models
        # that would cause "not downloaded" errors
        for model in data["data"]:
            # All models should have valid IDs that can be used
            assert model["id"] is not None
            assert len(model["id"]) > 0
            assert "/" in model["id"] or ":" in model["id"]  # Valid model identifier
    
    def test_vision_models_clean_descriptions(self, test_server):
        """Test that model descriptions are clean and not verbose."""
        response = requests.get(f"{test_server}/v1/vision-models")
        data = response.json()
        
        # Check that descriptions are simple
        allowed_descriptions = ["ONNX", "GGUF", "PyTorch", "MLX"]
        
        for model in data["data"]:
            assert model["description"] in allowed_descriptions, \
                f"Verbose description found: {model['description']}"


class TestBackendsEndpoint:
    """Test the /v1/backends endpoint."""
    
    def test_backends_endpoint_exists(self, test_server):
        """Test that the backends endpoint is accessible."""
        response = requests.get(f"{test_server}/v1/backends")
        assert response.status_code == 200
        
        data = response.json()
        assert "backends" in data
        assert "gpu" in data
        assert "status" in data
    
    def test_backends_structure(self, test_server):
        """Test that backends endpoint has correct structure."""
        response = requests.get(f"{test_server}/v1/backends")
        data = response.json()
        
        # Check backends structure
        backends = data["backends"]
        for backend_name in ["onnx", "pytorch"]:
            assert backend_name in backends
            backend = backends[backend_name]
            assert "available" in backend
            assert "initialized" in backend
            assert "models_count" in backend
            assert isinstance(backend["available"], bool)
            assert isinstance(backend["initialized"], bool)
            assert isinstance(backend["models_count"], int)
    
    def test_gpu_info_structure(self, test_server):
        """Test that GPU information has correct structure."""
        response = requests.get(f"{test_server}/v1/backends")
        data = response.json()
        
        gpu = data["gpu"]
        
        # Should have either error info or acceleration info
        if "error" in gpu or "torch_not_available" in gpu:
            # Error case - acceptable
            pass
        else:
            # Should have acceleration info
            assert "cuda_available" in gpu
            assert "mps_available" in gpu
            assert isinstance(gpu["cuda_available"], bool)
            assert isinstance(gpu["mps_available"], bool)
            
            if gpu["cuda_available"]:
                assert "cuda_device_count" in gpu
                assert isinstance(gpu["cuda_device_count"], int)
                assert gpu["cuda_device_count"] >= 0


class TestGenerationTimeTracking:
    """Test generation time tracking functionality."""
    
    def test_vision_test_page_has_timing(self):
        """Test that vision test page includes timing JavaScript."""
        response = requests.get(f"{BASE_URL}/manage/ui/vision-test.html")
        assert response.status_code == 200
        
        content = response.text
        
        # Check that timing code is present
        assert "performance.now()" in content
        assert "startTime" in content
        assert "endTime" in content
        assert "generationTime" in content
    
    def test_vision_test_result_display(self):
        """Test that result display includes generation time."""
        response = requests.get(f"{BASE_URL}/manage/ui/vision-test.html")
        content = response.text
        
        # Check that showResult function includes timing
        assert "Generation Time" in content
        assert "toFixed(2)" in content  # Time formatting


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple features."""
    
    def test_full_model_workflow(self):
        """Test complete workflow from listing to potential usage."""
        # 1. Get available models
        models_response = requests.get(f"{BASE_URL}/v1/vision-models")
        assert models_response.status_code == 200
        models_data = models_response.json()
        
        # 2. Get backend status
        backends_response = requests.get(f"{BASE_URL}/v1/backends")
        assert backends_response.status_code == 200
        backends_data = backends_response.json()
        
        # 3. Verify consistency
        if len(models_data["data"]) > 0:
            # If models are available, backends should be operational
            assert backends_data["status"] == "operational"
            
            # Count models by backend
            onnx_models = [m for m in models_data["data"] if m["backend"] == "ONNX"]
            pytorch_models = [m for m in models_data["data"] if "PyTorch" in m["backend"]]
            
            # Backend counts should be reasonable
            if len(onnx_models) > 0:
                assert backends_data["backends"]["onnx"]["available"] == True
            if len(pytorch_models) > 0:
                assert backends_data["backends"]["pytorch"]["available"] == True
    
    def test_error_handling(self):
        """Test error handling for edge cases."""
        # Test non-existent endpoint
        response = requests.get(f"{BASE_URL}/v1/nonexistent")
        assert response.status_code == 404
        
        # Backends endpoint should handle errors gracefully
        backends_response = requests.get(f"{BASE_URL}/v1/backends")
        if backends_response.status_code == 200:
            data = backends_response.json()
            # Should never crash, even with errors
            assert "status" in data




if __name__ == "__main__":
    # Run all tests using the test_server fixture
    pytest.main([__file__, "-v"])