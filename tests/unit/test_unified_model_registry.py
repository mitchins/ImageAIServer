"""
Unit tests for the unified model registry.
Tests model discovery, quantization handling, and business logic.
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from imageai_server.shared.unified_model_registry import (
    UnifiedModelRegistry, 
    UnifiedModel, 
    ModelFile
)


class TestUnifiedModelRegistry(unittest.TestCase):
    """Test the unified model registry functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock registry to avoid loading real models
        self.registry = UnifiedModelRegistry.__new__(UnifiedModelRegistry)
        self.registry.models = {}
    
    def test_model_file_creation(self):
        """Test ModelFile creation and attributes."""
        model_file = ModelFile(
            path="model.onnx",
            repo_id="test/repo",
            size_mb=100.5,
            downloaded=True
        )
        
        assert model_file.path == "model.onnx"
        assert model_file.repo_id == "test/repo"
        assert model_file.size_mb == 100.5
        assert model_file.downloaded is True
    
    def test_unified_model_creation(self):
        """Test UnifiedModel creation and default values."""
        model = UnifiedModel(
            id="test-model",
            name="Test Model",
            server="chat",
            architecture="multi-component",
            description="A test model"
        )
        
        assert model.id == "test-model"
        assert model.name == "Test Model"
        assert model.server == "chat"
        assert model.architecture == "multi-component"
        assert model.description == "A test model"
        assert model.files == []
        assert model.quantizations == []
        assert model.status == "not-downloaded"
        assert model.total_size_mb == 0.0
        assert model.working_sets == []
    
    def test_get_all_models(self):
        """Test getting all models from registry."""
        # Add test models
        model1 = UnifiedModel("test1", "Test 1", "chat", "single-file", "Test model 1")
        model2 = UnifiedModel("test2", "Test 2", "face", "multi-component", "Test model 2")
        
        self.registry.models = {"test1": model1, "test2": model2}
        
        all_models = self.registry.get_all_models()
        assert len(all_models) == 2
        assert model1 in all_models
        assert model2 in all_models
    
    def test_get_model_by_id(self):
        """Test getting a specific model by ID."""
        model = UnifiedModel("test-model", "Test Model", "chat", "single-file", "Test")
        self.registry.models = {"test-model": model}
        
        found_model = self.registry.get_model("test-model")
        assert found_model is model
        
        not_found = self.registry.get_model("nonexistent")
        assert not_found is None
    
    def test_get_models_by_server(self):
        """Test filtering models by server."""
        chat_model = UnifiedModel("chat1", "Chat Model", "chat", "single-file", "Chat")
        face_model = UnifiedModel("face1", "Face Model", "face", "single-file", "Face")
        gen_model = UnifiedModel("gen1", "Gen Model", "generation", "diffusion-pipeline", "Generation")
        
        self.registry.models = {
            "chat1": chat_model,
            "face1": face_model,
            "gen1": gen_model
        }
        
        chat_models = self.registry.get_models_by_server("chat")
        assert len(chat_models) == 1
        assert chat_models[0] is chat_model
        
        face_models = self.registry.get_models_by_server("face")
        assert len(face_models) == 1
        assert face_models[0] is face_model
        
        gen_models = self.registry.get_models_by_server("generation")
        assert len(gen_models) == 1
        assert gen_models[0] is gen_model
    
    def test_get_files_for_quantization_single_file(self):
        """Test getting files for single-file models."""
        file1 = ModelFile("model.onnx", "test/repo")
        model = UnifiedModel("test", "Test", "chat", "single-file", "Test")
        model.files = [file1]
        
        files = self.registry._get_files_for_quantization(model, "FP16")
        assert files == [file1]
    
    def test_get_files_for_quantization_diffusion(self):
        """Test getting files for diffusion models."""
        file1 = ModelFile("unet/model.bin", "test/repo")
        file2 = ModelFile("vae/model.bin", "test/repo")
        model = UnifiedModel("test", "Test", "generation", "diffusion-pipeline", "Test")
        model.files = [file1, file2]
        
        # For diffusion models, all files are returned regardless of quantization
        files = self.registry._get_files_for_quantization(model, "pytorch_fp16")
        assert len(files) == 2
        assert file1 in files
        assert file2 in files
    
    def test_get_files_for_quantization_multi_component(self):
        """Test getting files for multi-component models with quantization filtering."""
        fp16_file = ModelFile("model_fp16.onnx", "test/repo")
        q4_file = ModelFile("model_q4.onnx", "test/repo")
        base_file = ModelFile("model.onnx", "test/repo")
        
        model = UnifiedModel("test", "Test", "chat", "multi-component", "Test")
        model.files = [fp16_file, q4_file, base_file]
        
        # Test FP16 filtering
        fp16_files = self.registry._get_files_for_quantization(model, "FP16")
        assert len(fp16_files) == 1
        assert fp16_files[0] is fp16_file
        
        # Test Q4 filtering
        q4_files = self.registry._get_files_for_quantization(model, "Q4")
        assert len(q4_files) == 1
        assert q4_files[0] is q4_file
        
        # Test FP32 filtering (should return base files without quantization suffixes)
        fp32_files = self.registry._get_files_for_quantization(model, "FP32")
        assert len(fp32_files) == 1
        assert fp32_files[0] is base_file
    
    @patch('huggingface_hub.list_repo_files')
    def test_discover_actual_companion_files(self, mock_list_files):
        """Test discovering companion files that actually exist."""
        # Mock repository files
        mock_list_files.return_value = [
            "model.onnx", 
            "model_data.onnx",
            "model_data_0.onnx",
            "model_data_1.onnx",
            "other_file.txt"
        ]
        
        companion_files = self.registry._discover_actual_companion_files("test/repo", "model.onnx")
        
        expected_files = ["model_data.onnx", "model_data_0.onnx", "model_data_1.onnx"]
        assert len(companion_files) == 3
        for expected in expected_files:
            assert expected in companion_files
    
    @patch('huggingface_hub.list_repo_files')
    def test_discover_actual_companion_files_error_handling(self, mock_list_files):
        """Test companion file discovery with error handling."""
        # Simulate HuggingFace API error
        mock_list_files.side_effect = Exception("API Error")
        
        companion_files = self.registry._discover_actual_companion_files("test/repo", "model.onnx")
        
        # Should fall back to conservative detection
        assert len(companion_files) == 1
        assert companion_files[0] == "model_data.onnx"
    
    def test_get_repo_id_for_model(self):
        """Test getting repository ID for known models."""
        repo_id = self.registry._get_repo_id_for_model("Gemma-3n-E2B-it-ONNX")
        assert repo_id == "onnx-community/gemma-3n-E2B-it-ONNX"
        
        unknown_repo = self.registry._get_repo_id_for_model("Unknown-Model")
        assert unknown_repo is None


class TestModelDiscovery(unittest.TestCase):
    """Test model discovery mechanisms."""
    
    @patch('imageai_server.shared.model_types.MODEL_QUANT_CONFIGS')
    @patch('imageai_server.shared.model_types.REFERENCE_MODELS')
    def test_discover_chat_models(self, mock_ref_models, mock_quant_configs):
        """Test discovery of chat models."""
        # Mock configurations
        mock_quant_configs = {
            "Gemma-3n-E2B-it-ONNX/Q4_MIXED": {
                "decoder": "decoder_model_merged_q4.onnx",
                "embedder": "embed_tokens_quantized.onnx"
            },
            "Gemma-3n-E2B-it-ONNX/FP16": {
                "decoder": "decoder_model_merged_fp16.onnx", 
                "embedder": "embed_tokens_fp16.onnx"
            }
        }
        
        mock_ref_models = {
            "gemma_3n_e2b": Mock(repo_id="onnx-community/gemma-3n-E2B-it-ONNX", description="Test model")
        }
        
        registry = UnifiedModelRegistry.__new__(UnifiedModelRegistry)
        registry.models = {}
        
        with patch.object(registry, '_get_repo_id_for_model', return_value="onnx-community/gemma-3n-E2B-it-ONNX"):
            with patch.object(registry, '_discover_actual_companion_files', return_value=[]):
                registry._discover_chat_models()
        
        # Should create one model with two quantizations
        assert len(registry.models) == 1
        model = list(registry.models.values())[0]
        assert "Q4_MIXED" in model.quantizations
        assert "FP16" in model.quantizations
        assert len(model.files) >= 2  # At least decoder and embedder files
    
    @patch('imageai_server.shared.diffusion_model_registry.diffusion_registry')
    def test_discover_diffusion_models(self, mock_diffusion_registry):
        """Test discovery of diffusion models."""
        # Create mock diffusion model
        mock_working_set = Mock()
        mock_working_set.name = "pytorch_fp16"
        mock_working_set.components = {
            "unet": Mock(repo_id="test/repo", filename="unet/model.bin", subfolder=None),
            "vae": Mock(repo_id="test/repo", filename="vae/model.bin", subfolder=None),
        }
        
        mock_model_def = Mock()
        mock_model_def.model_id = "sd15"
        mock_model_def.display_name = "Stable Diffusion 1.5"
        mock_model_def.base_repo = "runwayml/stable-diffusion-v1-5"
        mock_model_def.working_sets = [mock_working_set]
        
        mock_diffusion_registry.get_available_models.return_value = {"sd15": mock_model_def}
        
        registry = UnifiedModelRegistry.__new__(UnifiedModelRegistry)
        registry.models = {}
        registry._discover_diffusion_models()
        
        # Should create one diffusion model
        assert len(registry.models) == 1
        model = list(registry.models.values())[0]
        assert model.server == "generation"
        assert model.architecture == "diffusion-pipeline"
        assert model.name == "Stable Diffusion 1.5"
        assert "pytorch_fp16" in model.quantizations
        assert len(model.files) == 2  # unet and vae files


class TestDownloadStatusUpdate(unittest.TestCase):
    """Test download status update functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = UnifiedModelRegistry.__new__(UnifiedModelRegistry)
        self.registry.models = {}
        
        # Create test model
        file1 = ModelFile("model1.onnx", "test/repo")
        file2 = ModelFile("model2.onnx", "test/repo")  
        self.test_model = UnifiedModel("test", "Test", "chat", "multi-component", "Test")
        self.test_model.files = [file1, file2]
        self.test_model.quantizations = ["Q4", "FP16"]
        self.registry.models["test"] = self.test_model
    
    @patch('huggingface_hub.scan_cache_dir')
    def test_update_download_status_all_downloaded(self, mock_scan_cache):
        """Test status update when all files are downloaded."""
        # Mock HuggingFace cache info
        mock_file_info1 = Mock()
        mock_file_info1.file_path = "model1.onnx"
        mock_file_info1.size_on_disk = 1024 * 1024  # 1MB
        
        mock_file_info2 = Mock()
        mock_file_info2.file_path = "model2.onnx"
        mock_file_info2.size_on_disk = 2 * 1024 * 1024  # 2MB
        
        mock_revision = Mock()
        mock_revision.files = [mock_file_info1, mock_file_info2]
        
        mock_repo = Mock()
        mock_repo.repo_id = "test/repo"
        mock_repo.revisions = [mock_revision]
        
        mock_cache_info = Mock()
        mock_cache_info.repos = [mock_repo]
        mock_scan_cache.return_value = mock_cache_info
        
        with patch.object(self.registry, '_get_files_for_quantization', return_value=self.test_model.files):
            self.registry.update_download_status()
        
        # All files should be marked as downloaded
        assert all(f.downloaded for f in self.test_model.files)
        assert self.test_model.status == "downloaded"
        assert self.test_model.total_size_mb == 3.0  # 1MB + 2MB
    
    @patch('huggingface_hub.scan_cache_dir')
    def test_update_download_status_partial(self, mock_scan_cache):
        """Test status update when only some files are downloaded."""
        # Mock only one file downloaded
        mock_file_info1 = Mock()
        mock_file_info1.file_path = "model1.onnx"
        mock_file_info1.size_on_disk = 1024 * 1024  # 1MB
        
        mock_revision = Mock()
        mock_revision.files = [mock_file_info1]  # Only one file
        
        mock_repo = Mock()
        mock_repo.repo_id = "test/repo"
        mock_repo.revisions = [mock_revision]
        
        mock_cache_info = Mock()
        mock_cache_info.repos = [mock_repo]
        mock_scan_cache.return_value = mock_cache_info
        
        with patch.object(self.registry, '_get_files_for_quantization', return_value=[self.test_model.files[0]]):
            self.registry.update_download_status()
        
        # Should show quantization status for multi-component models
        assert "quants" in self.test_model.status or self.test_model.status == "partial-files"


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])