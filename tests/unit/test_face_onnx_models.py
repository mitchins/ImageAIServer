"""
Unit tests for face detection ONNX models.
Tests the core ONNX model functionality without external dependencies.
"""

import pytest
import numpy as np
from PIL import Image
import io
from unittest.mock import Mock, patch, MagicMock

# Import the face detection components
from imageai_server.face_api.main import (
    preprocess_for_embedding, 
    preprocess_for_clip,
    preprocess_for_detection,
    get_clip_embedding,
    ModelLoader
)
from imageai_server.face_api.presets import PRESETS


class TestFacePreprocessing:
    """Test image preprocessing functions."""
    
    def create_test_image_array(self, width=224, height=224, channels=3):
        """Create a test image array."""
        return np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)
    
    def test_preprocess_for_embedding_dimensions(self):
        """Test ArcFace preprocessing produces correct dimensions."""
        test_image = self.create_test_image_array(200, 150, 3)
        
        processed = preprocess_for_embedding(test_image, target_size=(112, 112))
        
        # Should be [batch, height, width, channels] for ArcFace
        assert processed.shape == (1, 112, 112, 3)
        assert processed.dtype == np.float32
    
    def test_preprocess_for_embedding_normalization(self):
        """Test ArcFace preprocessing normalization."""
        # Create image with known values
        test_image = np.full((100, 100, 3), 127, dtype=np.uint8)  # Mid-gray
        
        processed = preprocess_for_embedding(test_image)
        
        # Should normalize to [-1, 1] range
        # 127 -> (127 / 127.5) - 1 = 0.0
        expected_value = (127.0 / 127.5) - 1.0
        assert np.allclose(processed[0, 0, 0, :], expected_value, atol=1e-6)
    
    def test_preprocess_for_clip_dimensions(self):
        """Test CLIP preprocessing produces correct dimensions."""
        test_image = self.create_test_image_array(200, 150, 3)
        
        processed = preprocess_for_clip(test_image, target_size=(224, 224))
        
        # Should be [batch, channels, height, width] for CLIP
        assert processed.shape == (1, 3, 224, 224)
        assert processed.dtype == np.float32
    
    def test_preprocess_for_clip_normalization(self):
        """Test CLIP preprocessing uses ImageNet normalization."""
        # Create white image
        test_image = np.full((100, 100, 3), 255, dtype=np.uint8)
        
        processed = preprocess_for_clip(test_image)
        
        # Check that normalization was applied
        # White (255) -> 1.0 -> (1.0 - mean) / std
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        expected = (1.0 - mean) / std
        
        # Check first pixel of each channel
        np.testing.assert_allclose(processed[0, :, 0, 0], expected, rtol=1e-5)
    
    def test_preprocess_for_detection_dimensions(self):
        """Test detection preprocessing produces correct dimensions."""
        test_image = self.create_test_image_array(300, 200, 3)
        
        processed = preprocess_for_detection(test_image)
        
        # Should resize to 640x640 and be in CHW format
        assert processed.shape == (1, 3, 640, 640)
        assert processed.dtype == np.float32
        
        # Values should be in [0, 1] range
        assert np.all(processed >= 0.0)
        assert np.all(processed <= 1.0)


class TestCLIPEmbedding:
    """Test CLIP-based embedding generation."""
    
    def create_test_face_region(self, size=(100, 100)):
        """Create a test face region array."""
        return np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    
    def test_get_clip_embedding_dimensions(self):
        """Test CLIP embedding produces correct dimensions."""
        face_region = self.create_test_face_region()
        preset = {"threshold": 0.5}
        
        embedding = get_clip_embedding(face_region, preset)
        
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 512  # Should be padded/truncated to 512
        assert embedding.dtype == np.float32
    
    def test_get_clip_embedding_normalization(self):
        """Test CLIP embedding is properly normalized."""
        face_region = self.create_test_face_region()
        preset = {"threshold": 0.5}
        
        embedding = get_clip_embedding(face_region, preset)
        
        # Should be unit normalized
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-5
    
    def test_get_clip_embedding_consistency(self):
        """Test CLIP embedding is consistent for same input."""
        face_region = self.create_test_face_region()
        preset = {"threshold": 0.5}
        
        embedding1 = get_clip_embedding(face_region, preset)
        embedding2 = get_clip_embedding(face_region, preset)
        
        # Should be identical
        assert np.allclose(embedding1, embedding2)
    
    def test_get_clip_embedding_different_inputs(self):
        """Test CLIP embedding produces different results for different inputs."""
        face_region1 = np.full((100, 100, 3), 50, dtype=np.uint8)
        face_region2 = np.full((100, 100, 3), 200, dtype=np.uint8)
        preset = {"threshold": 0.5}
        
        embedding1 = get_clip_embedding(face_region1, preset)
        embedding2 = get_clip_embedding(face_region2, preset)
        
        # Should be different
        similarity = np.dot(embedding1, embedding2)
        assert similarity < 0.99  # Not identical


class TestModelLoader:
    """Test ModelLoader functionality."""
    
    def test_model_loader_initialization(self):
        """Test ModelLoader can be initialized."""
        loader = ModelLoader()
        assert loader is not None
        assert loader._detector is None
        assert loader._embedder is None
    
    def test_get_providers(self):
        """Test provider selection."""
        loader = ModelLoader()
        
        with patch('imageai_server.face_api.main.ort') as mock_ort:
            mock_ort.get_available_providers.return_value = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            providers = loader._get_providers()
            
            assert isinstance(providers, list)
            assert 'CPUExecutionProvider' in providers
            # Should prefer CUDA if available
            assert 'CUDAExecutionProvider' in providers
    
    @patch('imageai_server.face_api.main.ort')
    @patch('imageai_server.face_api.main.download_model')
    def test_load_detector(self, mock_download, mock_ort):
        """Test detector loading."""
        # Mock the download and ONNX session
        mock_download.return_value = "/fake/path/model.onnx"
        mock_session = Mock()
        mock_ort.InferenceSession.return_value = mock_session
        
        loader = ModelLoader()
        detector = loader.load_detector()
        
        assert detector is mock_session
        assert loader._detector is mock_session
        mock_download.assert_called_once()
        mock_ort.InferenceSession.assert_called_once()
    
    @patch('imageai_server.face_api.main.ort')
    @patch('imageai_server.face_api.main.download_model')
    def test_load_embedder(self, mock_download, mock_ort):
        """Test embedder loading."""
        # Mock the download and ONNX session
        mock_download.return_value = "/fake/path/model.onnx"
        mock_session = Mock()
        mock_ort.InferenceSession.return_value = mock_session
        
        loader = ModelLoader()
        embedder = loader.load_embedder()
        
        assert embedder is mock_session
        assert loader._embedder is mock_session
        mock_download.assert_called_once()
        mock_ort.InferenceSession.assert_called_once()
    
    def test_load_detector_caching(self):
        """Test that detector is cached after first load."""
        with patch('imageai_server.face_api.main.ort') as mock_ort, \
             patch('imageai_server.face_api.main.download_model') as mock_download:
            
            mock_download.return_value = "/fake/path/model.onnx"
            mock_session = Mock()
            mock_ort.InferenceSession.return_value = mock_session
            
            loader = ModelLoader()
            
            # First call should load
            detector1 = loader.load_detector()
            # Second call should return cached
            detector2 = loader.load_detector()
            
            assert detector1 is detector2
            # Should only call download/session creation once
            assert mock_download.call_count == 1
            assert mock_ort.InferenceSession.call_count == 1


class TestPresets:
    """Test preset configurations."""
    
    def test_preset_structure(self):
        """Test that all presets have required fields."""
        required_fields = ["detector_repo", "detector_file", "embedder_repo", "embedder_file", "threshold"]
        
        for preset_name, preset in PRESETS.items():
            for field in required_fields:
                assert field in preset, f"Preset '{preset_name}' missing field '{field}'"
            
            # Check data types
            assert isinstance(preset["threshold"], (int, float))
            assert isinstance(preset["detector_repo"], str)
            assert isinstance(preset["detector_file"], str)
            assert isinstance(preset["embedder_repo"], str)
            assert isinstance(preset["embedder_file"], str)
    
    def test_preset_threshold_ranges(self):
        """Test that thresholds are in reasonable ranges."""
        for preset_name, preset in PRESETS.items():
            threshold = preset["threshold"]
            assert 0.0 <= threshold <= 1.0, f"Preset '{preset_name}' threshold {threshold} not in [0,1]"
    
    def test_required_presets_exist(self):
        """Test that required presets exist."""
        required_presets = ["photo", "anime", "cg"]
        
        for preset_name in required_presets:
            assert preset_name in PRESETS, f"Required preset '{preset_name}' not found"
    
    def test_photo_preset_uses_arcface(self):
        """Test that photo preset uses ArcFace model."""
        photo_preset = PRESETS["photo"]
        
        # Should use the correct ArcFace model
        assert "arcface" in photo_preset["embedder_repo"].lower()
        assert photo_preset["embedder_file"] == "arc.onnx"
    
    def test_anime_preset_configuration(self):
        """Test anime preset configuration."""
        anime_preset = PRESETS["anime"]
        
        # Should use anime face detection
        assert "anime" in anime_preset["detector_repo"].lower()
        # Should use CLIP for embedding
        assert "clip" in anime_preset["embedder_repo"].lower()


@pytest.mark.unit
class TestFaceDetectionErrorHandling:
    """Test error handling in face detection components."""
    
    def test_invalid_image_preprocessing(self):
        """Test preprocessing with invalid image data."""
        # Test with wrong dimensions
        invalid_image = np.random.rand(10)  # 1D array
        
        # The function may handle this gracefully or raise an error
        try:
            result = preprocess_for_embedding(invalid_image)
            # If no error, check the result is reasonable
            assert result is not None
        except (ValueError, IndexError, AttributeError, TypeError):
            # Any of these errors are acceptable for invalid input
            pass
    
    def test_clip_embedding_error_handling(self):
        """Test CLIP embedding with invalid input."""
        # Test with invalid face region
        invalid_region = np.array([])
        preset = {"threshold": 0.5}
        
        embedding = get_clip_embedding(invalid_region, preset)
        
        # Should handle error gracefully
        assert embedding is None
    
    @patch('imageai_server.face_api.main.ort')
    def test_model_loader_download_failure(self, mock_ort):
        """Test model loader handles download failures."""
        with patch('imageai_server.face_api.main.download_model') as mock_download:
            mock_download.side_effect = Exception("Download failed")
            
            loader = ModelLoader()
            
            with pytest.raises(Exception):
                loader.load_detector()


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])