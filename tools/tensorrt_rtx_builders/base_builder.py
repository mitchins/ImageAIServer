#!/usr/bin/env python3
"""
Base TensorRT-RTX Engine Builder

Clean, self-contained base class for building TensorRT-RTX engines.
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

# Setup TensorRT-RTX environment
def setup_tensorrt_rtx_env():
    """Setup TensorRT-RTX environment variables."""
    trt_rtx_lib = "/data/nvidia/TensorRT-RTX-1.0.0.21/targets/x86_64-linux-gnu/lib"
    if os.path.exists(trt_rtx_lib):
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if trt_rtx_lib not in current_ld_path:
            os.environ["LD_LIBRARY_PATH"] = f"{trt_rtx_lib}:{current_ld_path}"
    
    os.environ["POLYGRAPHY_USE_TENSORRT_RTX"] = "1"

# Setup environment before any TensorRT imports
setup_tensorrt_rtx_env()

logger = logging.getLogger(__name__)

class BaseTensorRTRTXBuilder(ABC):
    """Base class for TensorRT-RTX engine builders."""
    
    def __init__(self, 
                 output_dir: Path,
                 low_vram: bool = False,
                 verbose: bool = True):
        self.output_dir = Path(output_dir)
        self.hf_token = os.environ.get("HF_TOKEN")
        if not self.hf_token:
            logger.warning("HF_TOKEN environment variable not set. Some features may not work.")
        self.low_vram = low_vram
        self.verbose = verbose
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized {self.__class__.__name__}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Low VRAM mode: {self.low_vram}")
    
    @abstractmethod
    def get_supported_quantizations(self) -> List[str]:
        """Get list of supported quantizations for current GPU."""
        pass
    
    @abstractmethod
    def build_engines(self, 
                     quantization: str,
                     batch_size: int = 1,
                     height: int = 1024,
                     width: int = 1024) -> bool:
        """Build engines for specified configuration."""
        pass
    
    @abstractmethod
    def get_engine_files(self) -> List[str]:
        """Get list of expected engine files."""
        pass
    
    def validate_engines(self) -> bool:
        """Validate that all expected engines were built."""
        expected_files = self.get_engine_files()
        
        for engine_file in expected_files:
            engine_path = self.output_dir / engine_file
            if not engine_path.exists():
                logger.error(f"Missing engine file: {engine_path}")
                return False
            
            # Check file size (engines should be substantial)
            file_size_mb = engine_path.stat().st_size / (1024 * 1024)
            if file_size_mb < 10:  # Engines should be at least 10MB
                logger.error(f"Engine file too small: {engine_path} ({file_size_mb:.1f}MB)")
                return False
                
        logger.info(f"All {len(expected_files)} engine files validated successfully")
        return True
    
    def get_total_size_gb(self) -> float:
        """Get total size of built engines in GB."""
        # This is overridden by specific builders to look in the right directory
        total_bytes = 0
        for engine_file in self.get_engine_files():
            engine_path = self.output_dir / engine_file
            if engine_path.exists():
                total_bytes += engine_path.stat().st_size
        
        return total_bytes / (1024**3)
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model-specific information for README generation."""
        pass
    
    def generate_model_card(self) -> str:
        """Generate README.md content using Jinja2 template."""
        try:
            from jinja2 import Template
        except ImportError:
            raise ImportError("jinja2 is required for model card generation. Install with: pip install jinja2")
        
        # Get model-specific info
        model_info = self.get_model_info()
        
        # Get engine stats
        engine_files = self.get_engine_files()
        total_size = self.get_total_size_gb()
        
        # Detect GPU architecture
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.get_device_properties(0)
            gpu_name = device.name
            compute_cap = f"{device.major}.{device.minor}"
            arch_name = "ampere" if device.major == 8 else "ada" if device.major == 8 and device.minor >= 9 else "blackwell" if device.major >= 9 else "unknown"
        else:
            gpu_name = "Unknown GPU"
            compute_cap = "Unknown"
            arch_name = "unknown"
        
        # Template for README.md
        template_str = """---
license: {{ license }}
base_model: {{ base_model }}
tags:
{% for tag in tags %}
- {{ tag }}
{% endfor %}
---

# {{ model_name }} TensorRT-RTX {{ precision.upper() }} {{ arch_name.title() }}

{{ description }}

## Model Details

- **Base Model**: {{ base_model }}
- **Architecture**: {{ arch_name.upper() }} (Compute Capability {{ compute_cap }})
- **Precision**: {{ precision.upper() }} ({{ precision_description }})
- **TensorRT-RTX Version**: 1.0.0.21
- **Image Resolution**: {{ resolution }}
- **Batch Size**: 1 (static)

## Engine Files

This repository contains {{ engine_files|length }} TensorRT engine files:

{% for engine in engine_files %}
- `{{ engine }}` - {{ engine_descriptions.get(engine, "TensorRT engine") }}
{% endfor %}

**Total Size**: {{ "%.1f"|format(total_size) }}GB

## Hardware Requirements

- {{ hw_requirements }}
- Compute Capability {{ compute_cap }}
- Minimum {{ min_vram }}GB VRAM recommended
- TensorRT-RTX 1.0.0.21 runtime

## Usage

```python
{{ usage_example }}
```

## Performance

- **Inference Speed**: {{ performance.inference_speed }}
- **Memory Usage**: {{ performance.memory_usage }}
- **Optimizations**: {{ performance.optimizations }}

## License

{{ license_text }}

## Built With

- [TensorRT-RTX 1.0.0.21](https://developer.nvidia.com/tensorrt)
- {{ demo_source }}
- Built on {{ gpu_name }} ({{ arch_name.title() }} {{ compute_cap }})
"""
        
        template = Template(template_str)
        
        # Render with combined data
        context = {
            **model_info,
            'engine_files': engine_files,
            'total_size': total_size,
            'gpu_name': gpu_name,
            'compute_cap': compute_cap,
            'arch_name': arch_name
        }
        
        return template.render(**context)
    
    def save_model_card(self) -> None:
        """Generate and save README.md to output directory."""
        readme_content = self.generate_model_card()
        readme_path = self.output_dir / "README.md"
        readme_path.write_text(readme_content)
        logger.info(f"Generated model card: {readme_path}")