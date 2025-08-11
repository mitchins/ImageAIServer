"""
Unit tests for the diffusion model registry.
Tests diffusion model configurations, working sets, and validation.
"""

import pytest
import unittest
from unittest.mock import Mock, patch

from imageai_server.shared.diffusion_model_registry import (
    DiffusionModelRegistry,
    ModelDefinition,
    WorkingSet,
    ComponentSpec,
    Component,
    Backend,
    Quantization
)


class TestDiffusionModelRegistry(unittest.TestCase):
    """Test the diffusion model registry functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = DiffusionModelRegistry()
    
    def test_registry_initialization(self):
        """Test that registry initializes with built-in models."""
        available_models = self.registry.get_available_models()
        
        # Should have at least SD15 and SDXL
        assert "sd15" in available_models
        assert "sdxl" in available_models
        
        # Check SD15 model structure
        sd15 = available_models["sd15"]
        assert sd15.display_name == "Stable Diffusion 1.5"
        assert sd15.architecture == "sd15"
        assert len(sd15.working_sets) >= 3  # pytorch_fp16, onnx_fp16, onnx_int8
    
    def test_component_spec_creation(self):
        """Test ComponentSpec creation and defaults."""
        spec = ComponentSpec(
            repo_id="test/repo",
            filename="model.bin",
            subfolder="unet",
            quantization=Quantization.FP16,
            backend=Backend.PYTORCH
        )
        
        assert spec.repo_id == "test/repo"
        assert spec.filename == "model.bin"
        assert spec.subfolder == "unet"
        assert spec.quantization == Quantization.FP16
        assert spec.backend == Backend.PYTORCH
        assert spec.required is True  # Default value
    
    def test_working_set_creation(self):
        """Test WorkingSet creation."""
        unet_spec = ComponentSpec("test/repo", subfolder="unet")
        vae_spec = ComponentSpec("test/repo", subfolder="vae")
        
        working_set = WorkingSet(
            name="test_fp16",
            components={
                Component.UNET: unet_spec,
                Component.VAE: vae_spec
            },
            description="Test working set"
        )
        
        assert working_set.name == "test_fp16"
        assert len(working_set.components) == 2
        assert Component.UNET in working_set.components
        assert Component.VAE in working_set.components
        assert working_set.description == "Test working set"
    
    def test_model_definition_creation(self):
        """Test ModelDefinition creation."""
        working_set = WorkingSet("test", {}, description="Test")
        
        model_def = ModelDefinition(
            model_id="test_model",
            display_name="Test Model",
            base_repo="test/repo",
            architecture="test_arch",
            working_sets=[working_set],
            default_working_set="test"
        )
        
        assert model_def.model_id == "test_model"
        assert model_def.display_name == "Test Model"
        assert model_def.base_repo == "test/repo"
        assert model_def.architecture == "test_arch"
        assert len(model_def.working_sets) == 1
        assert model_def.default_working_set == "test"
        assert model_def.supports_negative_prompt is True  # Default
        assert model_def.max_resolution == 1024  # Default
    
    def test_get_model(self):
        """Test getting a specific model by ID."""
        sd15_model = self.registry.get_model("sd15")
        assert sd15_model is not None
        assert sd15_model.model_id == "sd15"
        assert sd15_model.display_name == "Stable Diffusion 1.5"
        
        # Test non-existent model
        nonexistent = self.registry.get_model("nonexistent")
        assert nonexistent is None
    
    def test_get_working_sets(self):
        """Test getting working sets for a model."""
        sd15_working_sets = self.registry.get_working_sets("sd15")
        assert len(sd15_working_sets) >= 3
        
        # Check that working sets have expected names
        working_set_names = [ws.name for ws in sd15_working_sets]
        assert "pytorch_fp16" in working_set_names
        assert "onnx_fp16" in working_set_names
        assert "onnx_int8" in working_set_names
        
        # Test non-existent model
        empty_working_sets = self.registry.get_working_sets("nonexistent")
        assert empty_working_sets == []
    
    def test_get_working_set(self):
        """Test getting a specific working set."""
        pytorch_fp16 = self.registry.get_working_set("sd15", "pytorch_fp16")
        assert pytorch_fp16 is not None
        assert pytorch_fp16.name == "pytorch_fp16"
        assert Component.UNET in pytorch_fp16.components
        
        # Test non-existent working set
        nonexistent_ws = self.registry.get_working_set("sd15", "nonexistent")
        assert nonexistent_ws is None
        
        # Test non-existent model
        nonexistent_model_ws = self.registry.get_working_set("nonexistent", "pytorch_fp16")
        assert nonexistent_model_ws is None
    
    def test_validate_working_set_valid(self):
        """Test validation of a valid working set."""
        unet_spec = ComponentSpec("test/repo", quantization=Quantization.FP16, backend=Backend.PYTORCH)
        vae_spec = ComponentSpec("test/repo", quantization=Quantization.FP16, backend=Backend.PYTORCH)
        
        working_set = WorkingSet(
            name="valid_set",
            components={
                Component.UNET: unet_spec,
                Component.VAE: vae_spec
            }
        )
        
        is_valid, issues = self.registry.validate_working_set(working_set)
        assert is_valid is True
        assert len(issues) == 0
    
    def test_validate_working_set_mixed_backends(self):
        """Test validation fails for mixed backends."""
        pytorch_spec = ComponentSpec("test/repo", backend=Backend.PYTORCH)
        onnx_spec = ComponentSpec("test/repo", backend=Backend.ONNX)
        
        working_set = WorkingSet(
            name="mixed_backends",
            components={
                Component.UNET: pytorch_spec,
                Component.VAE: onnx_spec
            }
        )
        
        is_valid, issues = self.registry.validate_working_set(working_set)
        assert is_valid is False
        assert any("Mixed backends" in issue for issue in issues)
    
    def test_validate_working_set_mixed_quantizations(self):
        """Test validation fails for problematic quantization combinations."""
        int8_spec = ComponentSpec("test/repo", quantization=Quantization.INT8)
        fp32_spec = ComponentSpec("test/repo", quantization=Quantization.FP32)
        
        working_set = WorkingSet(
            name="mixed_quants",
            components={
                Component.UNET: int8_spec,
                Component.VAE: fp32_spec
            }
        )
        
        is_valid, issues = self.registry.validate_working_set(working_set)
        assert is_valid is False
        assert any("INT8 and FP32" in issue for issue in issues)
    
    def test_validate_working_set_missing_unet(self):
        """Test validation fails when UNet is missing."""
        vae_spec = ComponentSpec("test/repo")
        
        working_set = WorkingSet(
            name="no_unet",
            components={
                Component.VAE: vae_spec
            }
        )
        
        is_valid, issues = self.registry.validate_working_set(working_set)
        assert is_valid is False
        assert any("UNet component is required" in issue for issue in issues)
    
    def test_suggest_optimal_working_set_gpu_quality(self):
        """Test optimal working set suggestion for GPU + quality preference."""
        optimal = self.registry.suggest_optimal_working_set("sd15", prefer_gpu=True, prefer_quality=True)
        assert optimal is not None
        
        # Should prefer PyTorch FP16 for GPU + quality
        assert optimal.name == "pytorch_fp16"
        
        # Check that it requires GPU
        assert optimal.constraints.get("requires_gpu", False) is True
    
    def test_suggest_optimal_working_set_cpu_speed(self):
        """Test optimal working set suggestion for CPU + speed preference."""
        optimal = self.registry.suggest_optimal_working_set("sd15", prefer_gpu=False, prefer_quality=False)
        assert optimal is not None
        
        # Should prefer ONNX INT8 for CPU + speed
        assert optimal.name == "onnx_int8"
        
        # Check that it doesn't require GPU
        assert optimal.constraints.get("requires_gpu", False) is False
    
    def test_suggest_optimal_working_set_nonexistent_model(self):
        """Test optimal working set suggestion for non-existent model."""
        optimal = self.registry.suggest_optimal_working_set("nonexistent")
        assert optimal is None
    
    def test_sd15_model_configuration(self):
        """Test SD15 model specific configuration."""
        sd15 = self.registry.get_model("sd15")
        assert sd15 is not None
        
        # Check basic properties
        assert sd15.base_repo == "stable-diffusion-v1-5/stable-diffusion-v1-5"
        assert sd15.supports_negative_prompt is True
        assert sd15.max_resolution == 768
        assert sd15.default_resolution == 512
        
        # Check working sets
        pytorch_fp16 = self.registry.get_working_set("sd15", "pytorch_fp16")
        assert pytorch_fp16 is not None
        assert pytorch_fp16.constraints["requires_gpu"] is True
        
        onnx_int8 = self.registry.get_working_set("sd15", "onnx_int8")
        assert onnx_int8 is not None
        assert onnx_int8.constraints["requires_gpu"] is False
    
    def test_sdxl_model_configuration(self):
        """Test SDXL model specific configuration."""
        sdxl = self.registry.get_model("sdxl")
        assert sdxl is not None
        
        # Check basic properties
        assert sdxl.base_repo == "stabilityai/stable-diffusion-xl-base-1.0"
        assert sdxl.max_resolution == 1536
        assert sdxl.default_resolution == 1024
        
        # SDXL should have at least PyTorch FP16
        pytorch_fp16 = self.registry.get_working_set("sdxl", "pytorch_fp16")
        assert pytorch_fp16 is not None
    
    def test_working_set_optimal_settings(self):
        """Test that working sets have appropriate optimal settings."""
        sd15_pytorch = self.registry.get_working_set("sd15", "pytorch_fp16")
        assert "num_inference_steps" in sd15_pytorch.optimal_settings
        assert "guidance_scale" in sd15_pytorch.optimal_settings
        assert sd15_pytorch.optimal_settings["num_inference_steps"] == 20
        assert sd15_pytorch.optimal_settings["guidance_scale"] == 7.5
        
        sd15_int8 = self.registry.get_working_set("sd15", "onnx_int8")
        # INT8 should have slightly more steps for quality
        assert sd15_int8.optimal_settings["num_inference_steps"] == 25


class TestEnums(unittest.TestCase):
    """Test enum definitions."""
    
    def test_backend_enum(self):
        """Test Backend enum values."""
        assert Backend.PYTORCH.value == "pytorch"
        assert Backend.ONNX.value == "onnx"
    
    def test_quantization_enum(self):
        """Test Quantization enum values."""
        assert Quantization.FP32.value == "fp32"
        assert Quantization.FP16.value == "fp16"
        assert Quantization.INT8.value == "int8"
        assert Quantization.BF16.value == "bf16"
    
    def test_component_enum(self):
        """Test Component enum values."""
        assert Component.UNET.value == "unet"
        assert Component.VAE.value == "vae"
        assert Component.TEXT_ENCODER.value == "text_encoder"
        assert Component.SCHEDULER.value == "scheduler"
        assert Component.TOKENIZER.value == "tokenizer"


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])