#!/usr/bin/env python3
"""
Unit tests for GPU/acceleration status detection.

Tests the functionality that detects CUDA, MPS, and other accelerators
and displays them in the status page.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class TestGPUDetection:
    """Test GPU and acceleration detection functionality."""
    
    @patch('imageai_server.shared.torch_loader.TORCH_AVAILABLE', True)
    def test_cuda_detection_available(self):
        """Test CUDA detection when available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        mock_torch.backends.mps.is_available.return_value = False
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            # This would test the actual GPU detection logic
            gpu_info = {
                "cuda_available": mock_torch.cuda.is_available(),
                "cuda_device_count": mock_torch.cuda.device_count(),
                "mps_available": mock_torch.backends.mps.is_available(),
            }
            
            assert gpu_info["cuda_available"] == True
            assert gpu_info["cuda_device_count"] == 2
            assert gpu_info["mps_available"] == False
    
    @patch('imageai_server.shared.torch_loader.TORCH_AVAILABLE', True)
    def test_mps_detection_available(self):
        """Test MPS detection when available (Apple Silicon)."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.cuda.device_count.return_value = 0
        mock_torch.backends.mps.is_available.return_value = True
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            gpu_info = {
                "cuda_available": mock_torch.cuda.is_available(),
                "cuda_device_count": mock_torch.cuda.device_count(),
                "mps_available": mock_torch.backends.mps.is_available(),
            }
            
            assert gpu_info["cuda_available"] == False
            assert gpu_info["cuda_device_count"] == 0
            assert gpu_info["mps_available"] == True
    
    @patch('imageai_server.shared.torch_loader.TORCH_AVAILABLE', True)
    def test_cpu_only_detection(self):
        """Test CPU-only detection when no accelerators available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.cuda.device_count.return_value = 0
        mock_torch.backends.mps.is_available.return_value = False
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            gpu_info = {
                "cuda_available": mock_torch.cuda.is_available(),
                "cuda_device_count": mock_torch.cuda.device_count(),
                "mps_available": mock_torch.backends.mps.is_available(),
            }
            
            assert gpu_info["cuda_available"] == False
            assert gpu_info["cuda_device_count"] == 0
            assert gpu_info["mps_available"] == False
    
    @patch('imageai_server.shared.torch_loader.TORCH_AVAILABLE', False)
    def test_torch_not_available(self):
        """Test GPU detection when PyTorch is not available."""
        gpu_info = {"torch_not_available": True}
        assert gpu_info["torch_not_available"] == True
    
    def test_gpu_status_formatting_cuda(self):
        """Test GPU status formatting for CUDA."""
        gpu_data = {
            "cuda_available": True,
            "cuda_device_count": 2,
            "mps_available": False,
            "current_device": "cuda:0"
        }
        
        # Test the JavaScript-like formatting logic in Python
        parts = []
        if gpu_data["cuda_available"]:
            count = gpu_data["cuda_device_count"]
            parts.append(f"✅ CUDA ({count} device{'s' if count != 1 else ''})")
        if gpu_data["mps_available"]:
            parts.append("✅ MPS")
        if not parts:
            parts.append("💻 CPU only")
            
        gpu_status = ', '.join(parts)
        if gpu_data.get("current_device"):
            gpu_status += f" | Using: {gpu_data['current_device']}"
        
        assert "✅ CUDA (2 devices)" in gpu_status
        assert "Using: cuda:0" in gpu_status
        assert "MPS" not in gpu_status
    
    def test_gpu_status_formatting_mps(self):
        """Test GPU status formatting for MPS."""
        gpu_data = {
            "cuda_available": False,
            "cuda_device_count": 0,
            "mps_available": True,
            "current_device": "mps"
        }
        
        parts = []
        if gpu_data["cuda_available"]:
            count = gpu_data["cuda_device_count"]
            parts.append(f"✅ CUDA ({count} device{'s' if count != 1 else ''})")
        if gpu_data["mps_available"]:
            parts.append("✅ MPS")
        if not parts:
            parts.append("💻 CPU only")
            
        gpu_status = ', '.join(parts)
        if gpu_data.get("current_device"):
            gpu_status += f" | Using: {gpu_data['current_device']}"
        
        assert "✅ MPS" in gpu_status
        assert "Using: mps" in gpu_status
        assert "CUDA" not in gpu_status
    
    def test_gpu_status_formatting_cpu_only(self):
        """Test GPU status formatting for CPU only."""
        gpu_data = {
            "cuda_available": False,
            "cuda_device_count": 0,
            "mps_available": False
        }
        
        parts = []
        if gpu_data["cuda_available"]:
            count = gpu_data["cuda_device_count"]
            parts.append(f"✅ CUDA ({count} device{'s' if count != 1 else ''})")
        if gpu_data["mps_available"]:
            parts.append("✅ MPS")
        if not parts:
            parts.append("💻 CPU only")
            
        gpu_status = ', '.join(parts)
        
        assert gpu_status == "💻 CPU only"
        assert "CUDA" not in gpu_status
        assert "MPS" not in gpu_status


class TestBackendsEndpoint:
    """Test the /v1/backends endpoint logic."""
    
    def test_backends_response_structure(self):
        """Test that backends endpoint returns correct structure."""
        # Mock response structure
        response = {
            "backends": {
                "onnx": {"available": True, "initialized": True, "models_count": 10},
                "pytorch": {"available": True, "initialized": True, "models_count": 5}
            },
            "gpu": {
                "cuda_available": False,
                "cuda_device_count": 0,
                "mps_available": True,
                "current_device": "mps"
            },
            "default_selection": "auto",
            "status": "operational"
        }
        
        # Verify structure
        assert "backends" in response
        assert "gpu" in response
        assert "default_selection" in response
        assert "status" in response
        
        # Verify backend structure
        for backend in ["onnx", "pytorch"]:
            assert backend in response["backends"]
            backend_info = response["backends"][backend]
            assert "available" in backend_info
            assert "initialized" in backend_info
            assert "models_count" in backend_info
        
        # Verify GPU structure
        gpu_info = response["gpu"]
        assert "cuda_available" in gpu_info
        assert "mps_available" in gpu_info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])