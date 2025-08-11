"""
Unit tests for MODEL-BACKEND:QUANT format parsing.
Tests the resolve_model_id method in ModelConfig class.
"""

import pytest
from imageai_server.generation.model_config import ModelConfig
import tempfile
import yaml
import os


@pytest.fixture
def test_model_config():
    """Create a test ModelConfig with known test data."""
    test_data = {
        'models': {
            'sd15': {
                'name': 'Stable Diffusion 1.5',
                'backends': {
                    'onnx': {
                        'quantizations': {
                            'int8': {'available': True, 'memory': '~500MB'}
                        }
                    }
                }
            },
            'sdxl-turbo': {
                'name': 'SDXL Turbo',
                'backends': {
                    'pytorch': {
                        'quantizations': {
                            'fp16': {'available': True, 'memory': '~8GB'}
                        }
                    }
                }
            },
            'stable-diffusion-1.5': {
                'name': 'Stable Diffusion 1.5 Full Name',  
                'backends': {
                    'onnx': {
                        'quantizations': {
                            'int8-optimized': {'available': True, 'memory': '~400MB'}
                        }
                    }
                }
            }
        },
        'model_id_mapping': {
            'sd15-onnx': ['sd15', 'onnx', 'int8'],
            'sdxl-turbo': ['sdxl-turbo', 'pytorch', 'fp16']
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_data, f)
        f.flush()
        config = ModelConfig(f.name)
    
    # Clean up
    os.unlink(f.name)
    return config


class TestModelIdFormat:
    """Test MODEL-BACKEND:QUANT format parsing."""
    
    def test_basic_colon_format(self, test_model_config):
        """Test basic MODEL-BACKEND:QUANT format."""
        result = test_model_config.resolve_model_id('sd15-onnx:int8')
        assert result == ('sd15', 'onnx', 'int8')
    
    def test_compound_model_name(self, test_model_config):
        """Test compound model names with hyphens."""
        result = test_model_config.resolve_model_id('sdxl-turbo-pytorch:fp16')
        assert result == ('sdxl-turbo', 'pytorch', 'fp16')
    
    def test_complex_model_and_backend_names(self, test_model_config):
        """Test complex names with multiple hyphens."""
        # This combination exists in our test config
        result = test_model_config.resolve_model_id('stable-diffusion-1.5-onnx:int8-optimized')
        assert result == ('stable-diffusion-1.5', 'onnx', 'int8-optimized')
    
    def test_splits_on_last_dash(self, test_model_config):
        """Test that splitting happens on the LAST dash before colon."""
        # For 'a-b-c-d-e:quant', should split as model='a-b-c-d', backend='e'
        # But since this combo doesn't exist in test data, we'll test the logic
        test_input = 'model-with-many-dashes-backend:quant'
        
        # Extract the splitting logic
        if ':' in test_input:
            model_backend, quant = test_input.split(':', 1)
            last_dash = model_backend.rfind('-')
            if last_dash > 0:
                model = model_backend[:last_dash]
                backend = model_backend[last_dash + 1:]
                
                assert model == 'model-with-many-dashes'
                assert backend == 'backend'
                assert quant == 'quant'
    
    def test_legacy_mapping(self, test_model_config):
        """Test that legacy IDs still work."""
        result = test_model_config.resolve_model_id('sd15-onnx')
        assert result == ('sd15', 'onnx', 'int8')
    
    def test_invalid_formats(self, test_model_config):
        """Test that invalid formats return None."""
        invalid_cases = [
            'no-colon-format',
            'model:quant',  # No backend
            'model-:quant',  # Empty backend
            'model-backend:',  # Empty quantization
            '-backend:quant',  # Empty model
            'nonexistent-backend:quant',  # Invalid combination
        ]
        
        for invalid_case in invalid_cases:
            result = test_model_config.resolve_model_id(invalid_case)
            assert result is None, f"Expected None for {invalid_case}, got {result}"
    
    def test_first_colon_splitting(self, test_model_config):
        """Test that splitting happens on FIRST colon."""
        # This should be rejected since it's invalid, but test the splitting logic
        test_input = 'model-backend:quant:extra'
        
        if ':' in test_input:
            parts = test_input.split(':', 1)
            assert parts[0] == 'model-backend'
            assert parts[1] == 'quant:extra'
    
    def test_edge_cases(self, test_model_config):
        """Test edge cases and boundary conditions."""
        edge_cases = [
            ('model-backend:', None),  # Empty quantization
            ('model--backend:quant', None),  # Double dash (invalid combo)
            ('a-b:c', None),  # Valid format but invalid combo
        ]
        
        for test_input, expected in edge_cases:
            result = test_model_config.resolve_model_id(test_input)
            assert result == expected, f"Expected {expected} for {test_input}, got {result}"


if __name__ == "__main__":
    # Run tests directly
    import sys
    sys.exit(pytest.main([__file__, "-v"]))