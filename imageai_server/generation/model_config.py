"""
Clean, data-driven model configuration system.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import os

class ModelConfig:
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent / "models.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models = self.config.get('models', {})
        self.id_mapping = self.config.get('model_id_mapping', {})
        self.ui_config = self.config.get('ui_config', {})
    
    def get_available_combinations(self) -> List[Dict[str, Any]]:
        """Get all available [model, backend, quantization] combinations."""
        combinations = []
        
        for model_key, model_data in self.models.items():
            for backend_key, backend_data in model_data.get('backends', {}).items():
                for quant_key, quant_data in backend_data.get('quantizations', {}).items():
                    if quant_data.get('available', False):
                        combo_id = f"{model_key}-{backend_key}:{quant_key}"
                        combinations.append({
                            'id': combo_id,
                            'model': model_key,
                            'backend': backend_key,
                            'quantization': quant_key,
                            'display_name': self._get_display_name(model_key, backend_key, quant_key),
                            'memory': quant_data.get('memory', 'Unknown'),
                            'description': quant_data.get('description', ''),
                            'model_data': model_data,
                            'quant_data': quant_data
                        })
        
        return combinations
    
    def get_ui_structure(self) -> Dict[str, Any]:
        """Get model data organized for UI display."""
        ui_models = {}
        
        for model_key, model_data in self.models.items():
            # Skip models with no available variants
            available_backends = {}
            
            for backend_key, backend_data in model_data.get('backends', {}).items():
                all_quants = {}
                has_available = False
                
                for quant_key, quant_data in backend_data.get('quantizations', {}).items():
                    is_available = quant_data.get('available', False)
                    if is_available:
                        has_available = True
                    
                    all_quants[quant_key] = {
                        'id': f"{model_key}-{backend_key}:{quant_key}",
                        'display_name': self._get_quant_display_name(quant_key),
                        'memory': quant_data.get('memory', 'Unknown'),
                        'description': quant_data.get('description', ''),
                        'available': is_available
                    }
                
                # Include backend if it has at least one available quantization
                if has_available:
                    available_backends[backend_key] = {
                        'display_name': self._get_backend_display_name(backend_key),
                        'quantizations': all_quants  # Include ALL quantizations, not just available
                    }
            
            if available_backends:
                ui_models[model_key] = {
                    'display_name': model_data.get('name', model_key),
                    'description': model_data.get('description', ''),
                    'supports_negative_prompt': model_data.get('supports_negative_prompt', True),
                    'default_resolution': model_data.get('default_resolution', 512),
                    'min_resolution': model_data.get('min_resolution', 256),
                    'max_resolution': model_data.get('max_resolution', 1024),
                    'backends': available_backends
                }
        
        return ui_models
    
    def resolve_model_id(self, model_id: str) -> Optional[Tuple[str, str, str]]:
        """Resolve model ID to [model, backend, quantization] tuple."""
        # Check legacy mapping first
        if model_id in self.id_mapping:
            return tuple(self.id_mapping[model_id])
        
        # Try to parse new format: model-backend:quantization
        if ':' in model_id:
            model_backend, quant = model_id.split(':', 1)
            # Split on LAST '-' to handle compound model names like "sdxl-turbo"
            last_dash = model_backend.rfind('-')
            if last_dash > 0:  # Ensure there's at least one character before the dash
                model = model_backend[:last_dash]
                backend = model_backend[last_dash + 1:]
                
                if self._is_valid_combination(model, backend, quant):
                    return (model, backend, quant)
                
                # Don't fallback for explicitly requested quantizations - they should fail
                # Only fallback when quantization is unspecified (handled separately below)
        
        # Try to parse model-backend format without quantization (default to FP16)
        if '-' in model_id and ':' not in model_id:
            last_dash = model_id.rfind('-')
            if last_dash > 0:
                model = model_id[:last_dash]
                backend = model_id[last_dash + 1:]
                
                if self._is_valid_combination(model, backend, 'fp16'):
                    print(f"ℹ️ No quantization specified for {model_id}, defaulting to FP16")
                    return (model, backend, 'fp16')
        
        # Try to parse legacy format: model-backend-quantization
        parts = model_id.split('-')
        if len(parts) >= 3:
            model = parts[0]
            backend = parts[1]
            quant = '-'.join(parts[2:])  # Handle multi-part quantization names
            
            if self._is_valid_combination(model, backend, quant):
                return (model, backend, quant)
        
        return None
    
    def get_model_info(self, model: str, backend: str, quantization: str) -> Optional[Dict[str, Any]]:
        """Get full model information for a specific combination."""
        if not self._is_valid_combination(model, backend, quantization):
            return None
        
        model_data = self.models[model]
        backend_data = model_data['backends'][backend]
        quant_data = backend_data['quantizations'][quantization]
        
        return {
            'model': model,
            'backend': backend,
            'quantization': quantization,
            'model_data': model_data,
            'backend_data': backend_data,
            'quant_data': quant_data,
            'model_path': quant_data.get('model_path', backend_data.get('base_model')),
            'subfolder': quant_data.get('subfolder'),
            'loader_function': f"get_{model.replace('-', '_')}_{backend}_{quantization}",
            'supports_negative_prompt': model_data.get('supports_negative_prompt', True),
            'resolutions': {
                'default': model_data.get('default_resolution', 512),
                'min': model_data.get('min_resolution', 256),
                'max': model_data.get('max_resolution', 1024)
            }
        }
    
    def get_legacy_model_metadata(self) -> Dict[str, Any]:
        """Generate legacy model metadata for backward compatibility."""
        metadata = {}
        
        for combo in self.get_available_combinations():
            # Map to legacy format
            legacy_id = None
            for legacy, mapping in self.id_mapping.items():
                if (mapping[0] == combo['model'] and 
                    mapping[1] == combo['backend'] and 
                    mapping[2] == combo['quantization']):
                    legacy_id = legacy
                    break
            
            if not legacy_id:
                legacy_id = combo['id']
            
            model_data = combo['model_data']
            metadata[legacy_id] = {
                'engine': combo['backend'],
                'display_name': combo['display_name'],
                'description': combo['description'],
                'memory_requirement': combo['memory'],
                'quantization': combo['quantization'].upper(),
                'supports_negative_prompt': model_data.get('supports_negative_prompt', True),
                'max_resolution': model_data.get('max_resolution', 1024),
                'default_resolution': model_data.get('default_resolution', 512),
                'min_resolution': model_data.get('min_resolution', 256)
            }
        
        return metadata
    
    def _get_display_name(self, model: str, backend: str, quantization: str) -> str:
        """Generate display name for model combination."""
        model_name = self.models[model].get('name', model)
        backend_display = self._get_backend_display_name(backend)
        quant_display = self._get_quant_display_name(quantization)
        
        return f"{model_name} ({backend_display} {quant_display})"
    
    def _get_backend_display_name(self, backend: str) -> str:
        """Get display name for backend."""
        return self.ui_config.get('backend_display', {}).get(backend, backend.upper())
    
    def _get_quant_display_name(self, quantization: str) -> str:
        """Get display name for quantization."""
        return self.ui_config.get('quantization_display', {}).get(quantization, quantization.upper())
    
    def _is_valid_combination(self, model: str, backend: str, quantization: str) -> bool:
        """Check if model/backend/quantization combination is valid and available."""
        return (model in self.models and 
                backend in self.models[model].get('backends', {}) and
                quantization in self.models[model]['backends'][backend].get('quantizations', {}) and
                self.models[model]['backends'][backend]['quantizations'][quantization].get('available', False))

# Global instance
model_config = ModelConfig()