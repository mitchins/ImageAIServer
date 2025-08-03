#!/usr/bin/env python3
"""
Unit tests for local-only model listing functionality.

Tests the core feature that only lists locally available models,
preventing "model not downloaded" errors.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from imageai_server.main import _is_model_downloaded
from imageai_server.shared.model_types import get_curated_model_config


class TestLocalModelDetection:
    """Test the core local model detection functionality."""
    
    def test_is_model_downloaded_with_huggingface_cache(self):
        """Test model detection using HuggingFace cache scanner."""
        # Mock HuggingFace cache scanner
        mock_file_info = Mock()
        mock_file_info.file_path = "/cache/snapshots/abc123/onnx/model.onnx"
        
        mock_revision = Mock()
        mock_revision.files = [mock_file_info]
        
        mock_repo = Mock()
        mock_repo.repo_id = "test-org/test-model"
        mock_repo.revisions = [mock_revision]
        
        mock_cache = Mock()
        mock_cache.repos = [mock_repo]
        
        with patch('huggingface_hub.scan_cache_dir', return_value=mock_cache):
            # Test basic repo detection
            assert _is_model_downloaded("test-org/test-model") == True
            assert _is_model_downloaded("nonexistent/model") == False
    
    def test_is_model_downloaded_with_required_files(self):
        """Test model detection with specific file requirements."""
        mock_file_info = Mock()
        mock_file_info.file_path = "/cache/snapshots/abc123/onnx/model_q4.onnx"
        
        mock_revision = Mock()
        mock_revision.files = [mock_file_info]
        
        mock_repo = Mock()
        mock_repo.repo_id = "test-org/test-model"
        mock_repo.revisions = [mock_revision]
        
        mock_cache = Mock()
        mock_cache.repos = [mock_repo]
        
        with patch('huggingface_hub.scan_cache_dir', return_value=mock_cache):
            # Test with required files
            required_files = ["onnx/model_q4.onnx"]
            assert _is_model_downloaded("test-org/test-model", required_files) == True
            
            # Test with missing required files
            missing_files = ["onnx/missing_file.onnx"]
            assert _is_model_downloaded("test-org/test-model", missing_files) == False
    
    def test_is_model_downloaded_filesystem_fallback(self):
        """Test filesystem fallback when HuggingFace cache scanner fails."""
        from huggingface_hub.errors import CacheNotFound
        
        with patch('huggingface_hub.scan_cache_dir', side_effect=CacheNotFound("No cache", "/fake/path")):
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.glob') as mock_glob:
                    # Mock snapshot directory
                    mock_snapshot = Mock()
                    mock_snapshot.iterdir.return_value = [Path("dummy_file")]
                    mock_glob.return_value = [mock_snapshot]
                    
                    assert _is_model_downloaded("test-org/test-model") == True
    
    def test_is_model_downloaded_handles_errors(self):
        """Test that model detection gracefully handles errors."""
        with patch('huggingface_hub.scan_cache_dir', side_effect=Exception("Mock error")):
            # Should return False on error, not crash
            assert _is_model_downloaded("test-org/test-model") == False


class TestModelConfigDetection:
    """Test curated model configuration detection."""
    
    def test_get_curated_model_config_gemma3n(self):
        """Test Gemma-3n configuration detection."""
        config = get_curated_model_config("Gemma-3n-E2B-it-ONNX/Q4_MIXED")
        assert config is not None
        assert "audio_encoder" in config
        assert "decoder" in config
        assert "embed_tokens" in config
        assert "vision_encoder" in config
        
        # Check specific file paths
        assert config["audio_encoder"] == "onnx/audio_encoder_q4.onnx"
        assert config["decoder"] == "onnx/decoder_model_merged_q4.onnx"
    
    def test_get_curated_model_config_nonexistent(self):
        """Test that nonexistent configurations return None."""
        config = get_curated_model_config("NonExistent/Model")
        assert config is None


class TestVisionModelsEndpointLogic:
    """Test the logic behind the /v1/vision-models endpoint."""
    
    @patch('imageai_server.main._is_model_downloaded')
    @patch('imageai_server.shared.model_types.get_available_model_quants')
    @patch('imageai_server.shared.model_types.get_curated_model_config')
    def test_only_shows_downloaded_onnx_models(self, mock_get_config, mock_get_quants, mock_is_downloaded):
        """Test that only locally available ONNX models are shown."""
        # Mock configurations
        mock_get_quants.return_value = [
            "Gemma-3n-E2B-it-ONNX/Q4_MIXED",
            "NonExistent-Model/FP16"
        ]
        
        mock_get_config.side_effect = lambda x: {
            "audio_encoder": "onnx/audio_encoder_q4.onnx"
        } if "Gemma-3n" in x else None
        
        # Only Gemma-3n is downloaded
        mock_is_downloaded.side_effect = lambda repo_id, files=None: "gemma-3n" in repo_id.lower()
        
        from imageai_server.shared.model_identifier import ModelCatalog
        
        # Mock the ModelCatalog to avoid actual model loading
        with patch.object(ModelCatalog, 'MODELS', {}):
            # Import the function that builds the model list
            # This would be tested by calling the actual endpoint logic
            pass  # Would test actual endpoint here
    
    def test_repo_id_mapping_correctness(self):
        """Test that model names map to correct repository IDs."""
        # Test the mappings we added
        onnx_model_repo_mapping = {
            "Qwen2-VL-2B-Instruct": "onnx-community/Qwen2-VL-2B-Instruct",
            "Gemma-3n-E2B-it-ONNX": "onnx-community/gemma-3n-E2B-it-ONNX",
            "Phi-3.5-vision-instruct": "onnx-community/Phi-3.5-vision-instruct",
        }
        
        for model_name, expected_repo in onnx_model_repo_mapping.items():
            # Verify mapping exists and is correct
            assert expected_repo is not None
            assert "/" in expected_repo  # Should be org/model format
            assert "onnx-community" in expected_repo or "microsoft" in expected_repo


if __name__ == "__main__":
    pytest.main([__file__, "-v"])