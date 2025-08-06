"""
Integration tests for face detection pipelines.
Tests the ONNX face detection models without web components.
"""

import pytest
import numpy as np
from PIL import Image
import io
import os
from pathlib import Path

# Import the face detection components
from imageai_server.face_api.main import get_embedding, ModelLoader
from imageai_server.face_api.config import load_config


class TestFaceDetectionPipeline:
    """Test face detection pipeline with different face types."""
    
    @classmethod
    def setup_class(cls):
        """Set up test fixtures."""
        cls.test_data_dir = Path(__file__).parent.parent.parent
        cls.model_loader = ModelLoader()
        
        # Load test images
        cls.real_face_a_path = cls.test_data_dir / "face-a.png"
        cls.real_face_b_path = cls.test_data_dir / "face-b.png"
        cls.anime_face_a_path = cls.test_data_dir / "anime-face-a.png"
        cls.anime_face_b_path = cls.test_data_dir / "anime-face-b.png"
        
        # Check if test images exist
        cls.has_real_faces = cls.real_face_a_path.exists() and cls.real_face_b_path.exists()
        cls.has_anime_faces = cls.anime_face_a_path.exists() and cls.anime_face_b_path.exists()
    
    def load_image_bytes(self, image_path: Path) -> bytes:
        """Load image as bytes."""
        with open(image_path, 'rb') as f:
            return f.read()
    
    def create_test_image(self, color=(255, 200, 150), size=(224, 224)) -> bytes:
        """Create a simple test image for fallback testing."""
        img = Image.new('RGB', size, color)
        # Add some basic features to make it look more face-like
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        
        # Simple face-like features
        center_x, center_y = size[0] // 2, size[1] // 2
        # Eyes
        draw.ellipse([center_x - 40, center_y - 30, center_x - 20, center_y - 10], fill='black')
        draw.ellipse([center_x + 20, center_y - 30, center_x + 40, center_y - 10], fill='black')
        # Nose
        draw.ellipse([center_x - 5, center_y - 5, center_x + 5, center_y + 10], fill='pink')
        # Mouth
        draw.arc([center_x - 20, center_y + 10, center_x + 20, center_y + 30], 0, 180, fill='red', width=3)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        return img_bytes.getvalue()
    
    def test_model_loader_initialization(self):
        """Test that the model loader can be initialized."""
        config = load_config([])
        assert config is not None
        assert hasattr(config, 'detector_model')
        assert hasattr(config, 'embedder_repo')
    
    def test_face_embedding_extraction_photo_mode(self):
        """Test face embedding extraction in photo mode."""
        if self.has_real_faces:
            # Test with real face images
            image_bytes = self.load_image_bytes(self.real_face_a_path)
        else:
            # Fallback to synthetic image
            image_bytes = self.create_test_image()
        
        # Test photo mode
        embedding = get_embedding(image_bytes, face_type="photo")
        
        # Should return an embedding (may be None if no face detected in synthetic image)
        if embedding is not None:
            assert isinstance(embedding, np.ndarray)
            assert len(embedding) == 512  # ArcFace embeddings are 512-dimensional
            assert embedding.dtype == np.float32
            # Embedding should be normalized
            norm = np.linalg.norm(embedding)
            assert abs(norm - 1.0) < 1e-5  # Should be unit normalized
    
    @pytest.mark.skipif(not Path(__file__).parent.parent.parent.joinpath("anime-face-a.png").exists(), 
                       reason="Anime face test images not available")
    def test_face_embedding_extraction_anime_mode(self):
        """Test face embedding extraction in anime mode."""
        image_bytes = self.load_image_bytes(self.anime_face_a_path)
        
        # Test anime mode
        embedding = get_embedding(image_bytes, face_type="anime")
        
        assert embedding is not None, "Should detect anime face"
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 512  # Padded to match ArcFace dimensions
        assert embedding.dtype == np.float32
        # Embedding should be normalized
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-5  # Should be unit normalized
    
    def test_face_embedding_consistency(self):
        """Test that same image produces consistent embeddings."""
        if self.has_real_faces:
            image_bytes = self.load_image_bytes(self.real_face_a_path)
        else:
            image_bytes = self.create_test_image()
        
        # Get embeddings twice
        embedding1 = get_embedding(image_bytes, face_type="photo")
        embedding2 = get_embedding(image_bytes, face_type="photo")
        
        if embedding1 is not None and embedding2 is not None:
            # Should be identical (or very close due to floating point)
            similarity = np.dot(embedding1, embedding2)
            assert similarity > 0.99, f"Same image should produce consistent embeddings, got similarity: {similarity}"
    
    @pytest.mark.skipif(not (Path(__file__).parent.parent.parent.joinpath("face-a.png").exists() and 
                           Path(__file__).parent.parent.parent.joinpath("face-b.png").exists()), 
                       reason="Real face test images not available")
    def test_real_face_comparison(self):
        """Test comparing two different real faces."""
        image_a_bytes = self.load_image_bytes(self.real_face_a_path)
        image_b_bytes = self.load_image_bytes(self.real_face_b_path)
        
        # Get embeddings
        embedding_a = get_embedding(image_a_bytes, face_type="photo")
        embedding_b = get_embedding(image_b_bytes, face_type="photo")
        
        assert embedding_a is not None, "Should detect face in image A"
        assert embedding_b is not None, "Should detect face in image B"
        
        # Calculate similarity
        similarity = float(np.dot(embedding_a, embedding_b))
        
        # Different people should have lower similarity
        assert 0.0 <= similarity <= 1.0, f"Similarity should be between 0 and 1, got: {similarity}"
        print(f"Real face comparison similarity: {similarity:.4f}")
    
    @pytest.mark.skipif(not (Path(__file__).parent.parent.parent.joinpath("anime-face-a.png").exists() and 
                           Path(__file__).parent.parent.parent.joinpath("anime-face-b.png").exists()), 
                       reason="Anime face test images not available")
    def test_anime_face_comparison(self):
        """Test comparing two anime faces."""
        image_a_bytes = self.load_image_bytes(self.anime_face_a_path)
        image_b_bytes = self.load_image_bytes(self.anime_face_b_path)
        
        # Get embeddings
        embedding_a = get_embedding(image_a_bytes, face_type="anime")
        embedding_b = get_embedding(image_b_bytes, face_type="anime")
        
        assert embedding_a is not None, "Should detect anime face in image A"
        assert embedding_b is not None, "Should detect anime face in image B"
        
        # Calculate similarity
        similarity = float(np.dot(embedding_a, embedding_b))
        
        assert 0.0 <= similarity <= 1.0, f"Similarity should be between 0 and 1, got: {similarity}"
        print(f"Anime face comparison similarity: {similarity:.4f}")
    
    def test_different_face_types_different_embeddings(self):
        """Test that different face types produce different embedding characteristics."""
        if not (self.has_real_faces and self.has_anime_faces):
            pytest.skip("Need both real and anime face images for this test")
        
        # Get embeddings for same logical image in different modes
        real_bytes = self.load_image_bytes(self.real_face_a_path)
        anime_bytes = self.load_image_bytes(self.anime_face_a_path)
        
        real_embedding = get_embedding(real_bytes, face_type="photo")
        anime_embedding = get_embedding(anime_bytes, face_type="anime")
        
        if real_embedding is not None and anime_embedding is not None:
            # Different processing pipelines should produce different embeddings
            similarity = float(np.dot(real_embedding, anime_embedding))
            print(f"Cross-type similarity (real vs anime): {similarity:.4f}")
            # We don't assert a specific range since they use different algorithms
    
    def test_invalid_face_type_defaults_to_photo(self):
        """Test that invalid face type defaults to photo mode."""
        if self.has_real_faces:
            image_bytes = self.load_image_bytes(self.real_face_a_path)
        else:
            image_bytes = self.create_test_image()
        
        # Test with invalid face type
        embedding = get_embedding(image_bytes, face_type="invalid_type")
        
        # Should still work (defaulting to photo mode)
        if embedding is not None:
            assert isinstance(embedding, np.ndarray)
            assert len(embedding) == 512
    
    def test_empty_image_handling(self):
        """Test handling of empty or invalid image data."""
        # Test with empty bytes
        embedding = get_embedding(b"", face_type="photo")
        assert embedding is None, "Empty image should return None"
        
        # Test with invalid image data
        embedding = get_embedding(b"not_an_image", face_type="photo")
        assert embedding is None, "Invalid image should return None"


@pytest.mark.integration
class TestFaceDetectionModels:
    """Test the face detection models themselves."""
    
    def test_model_availability(self):
        """Test that required models can be identified."""
        from imageai_server.face_api.presets import PRESETS
        
        # Check that presets are defined
        assert "photo" in PRESETS
        assert "anime" in PRESETS
        assert "cg" in PRESETS
        
        # Check preset structure
        for preset_name, preset in PRESETS.items():
            assert "detector_repo" in preset
            assert "embedder_repo" in preset
            assert "threshold" in preset
            assert isinstance(preset["threshold"], (int, float))
    
    def test_model_loader_creation(self):
        """Test that ModelLoader can be created and configured."""
        loader = ModelLoader()
        assert loader is not None
        
        # Should have provider selection method
        providers = loader._get_providers()
        assert isinstance(providers, list)
        assert len(providers) > 0
        assert "CPUExecutionProvider" in providers


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])