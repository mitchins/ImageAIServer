"""
Model variant registry for different quantization options.
Each base model can have multiple quantization variants.
"""

# Model variants with quantization options
MODEL_VARIANTS = {
    "sd15": {
        "display_name": "Stable Diffusion 1.5",
        "base_model": "runwayml/stable-diffusion-v1-5",
        "variants": {
            "fp16": {
                "id": "sd15-fp16",
                "display_name": "SD 1.5 (FP16)",
                "engine": "pytorch",
                "memory_requirement": "~4GB",
                "description": "Full precision, best quality",
                "loader": "get_sd15_fp16",
                "available": False  # Not implemented yet
            },
            "onnx-fp16": {
                "id": "sd15-onnx-fp16",
                "display_name": "SD 1.5 (ONNX FP16)",
                "engine": "onnx",
                "memory_requirement": "~2GB",
                "description": "ONNX optimized, good quality",
                "model_path": "sd15_onnx/fp16",
                "loader": "get_sd15_onnx_fp16",
                "available": False  # Need to implement
            },
            "onnx-int8": {
                "id": "sd15-onnx",  # Keep backward compatibility
                "display_name": "SD 1.5 (ONNX INT8)",
                "engine": "onnx",
                "memory_requirement": "~500MB",
                "description": "Highly quantized, CPU optimized",
                "model_path": "Mitchins/sd15-onnx-int8",
                "loader": "get_sd15_onnx",
                "available": True
            }
        },
        "default_variant": "onnx-int8",
        "supports_negative_prompt": True,
        "max_resolution": 768,
        "default_resolution": 512,
        "min_resolution": 256
    },
    "sdxl": {
        "display_name": "Stable Diffusion XL",
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "variants": {
            "fp16": {
                "id": "sdxl",  # Keep backward compatibility
                "display_name": "SDXL (FP16)",
                "engine": "pytorch",
                "memory_requirement": "~8GB VRAM",
                "description": "Full quality, GPU required",
                "loader": "get_sdxl",
                "available": True
            },
            "int8": {
                "id": "sdxl-int8",
                "display_name": "SDXL (INT8)",
                "engine": "pytorch",
                "memory_requirement": "~4GB VRAM",
                "description": "Quantized, reduced memory usage",
                "loader": "get_sdxl_int8",
                "available": False  # Need to implement
            },
            "onnx-fp16": {
                "id": "sdxl-onnx-fp16",
                "display_name": "SDXL (ONNX FP16)",
                "engine": "onnx",
                "memory_requirement": "~6GB",
                "description": "ONNX optimized for broader compatibility",
                "loader": "get_sdxl_onnx_fp16",
                "available": False  # Need to implement
            },
            "onnx-int8": {
                "id": "sdxl-onnx-int8",
                "display_name": "SDXL (ONNX INT8)",
                "engine": "onnx",
                "memory_requirement": "~2GB",
                "description": "Highly quantized SDXL, CPU capable",
                "loader": "get_sdxl_onnx_int8",
                "available": False  # Need to implement
            }
        },
        "default_variant": "fp16",
        "supports_negative_prompt": True,
        "max_resolution": 1024,
        "default_resolution": 1024,
        "min_resolution": 512
    },
    "sdxl-turbo": {
        "display_name": "SDXL Turbo",
        "base_model": "stabilityai/sdxl-turbo",
        "variants": {
            "fp16": {
                "id": "sdxl-turbo",  # Keep backward compatibility
                "display_name": "SDXL Turbo (FP16)",
                "engine": "pytorch",
                "memory_requirement": "~8GB VRAM",
                "description": "Fast generation, fewer steps",
                "loader": "get_sdxl_turbo",
                "available": True
            },
            "onnx": {
                "id": "sdxl-turbo-onnx",
                "display_name": "SDXL Turbo (ONNX)",
                "engine": "onnx",
                "memory_requirement": "~4GB",
                "description": "Fast generation, CPU compatible",
                "model_path": "onnxruntime/sdxl-turbo",
                "loader": "get_sdxl_turbo_onnx",
                "available": True  # Available from onnxruntime/sdxl-turbo
            }
        },
        "default_variant": "fp16",
        "supports_negative_prompt": True,
        "max_resolution": 1024,
        "default_resolution": 1024,
        "min_resolution": 512
    },
    "flux1": {
        "display_name": "FLUX.1 Schnell",
        "base_model": "black-forest-labs/FLUX.1-schnell",
        "variants": {
            "fp16": {
                "id": "flux1-schnell",  # Keep backward compatibility
                "display_name": "FLUX.1 (FP16)",
                "engine": "pytorch",
                "memory_requirement": "~12GB VRAM",
                "description": "Best quality for complex artistic prompts",
                "loader": "get_flux1",
                "available": True
            },
            "q8": {
                "id": "flux1-q8",
                "display_name": "FLUX.1 (Q8)",
                "engine": "pytorch",
                "memory_requirement": "~8GB VRAM",
                "description": "8-bit quantized, good quality with less memory",
                "model_path": "aifoundry-org/FLUX.1-schnell-Quantized",
                "variant_subfolder": "q8_0",
                "loader": "get_flux1_q8",
                "available": True
            },
            "q4": {
                "id": "flux1-q4",
                "display_name": "FLUX.1 (Q4)",
                "engine": "pytorch",
                "memory_requirement": "~6GB VRAM",
                "description": "4-bit quantized, very memory efficient",
                "model_path": "aifoundry-org/FLUX.1-schnell-Quantized",
                "variant_subfolder": "q4_0",
                "loader": "get_flux1_q4",
                "available": True
            }
        },
        "default_variant": "q8",  # Q8 as default for better memory/quality balance
        "supports_negative_prompt": False,
        "max_resolution": 1024,
        "default_resolution": 1024,
        "min_resolution": 256
    }
}

def get_all_available_models():
    """Get all available model variants as a flat list."""
    models = {}
    for model_family, family_data in MODEL_VARIANTS.items():
        for variant_key, variant in family_data["variants"].items():
            if variant["available"]:
                model_id = variant["id"]
                models[model_id] = {
                    "family": model_family,
                    "variant": variant_key,
                    "display_name": variant["display_name"],
                    "engine": variant["engine"],
                    "memory_requirement": variant["memory_requirement"],
                    "description": variant["description"],
                    "supports_negative_prompt": family_data["supports_negative_prompt"],
                    "max_resolution": family_data["max_resolution"],
                    "default_resolution": family_data["default_resolution"],
                    "min_resolution": family_data["min_resolution"],
                    "quantization": variant_key.split("-")[-1].upper() if "-" in variant_key else "FP16"
                }
    return models

def get_model_families():
    """Get model families for grouped display."""
    families = {}
    for model_family, family_data in MODEL_VARIANTS.items():
        available_variants = [
            v for v in family_data["variants"].values() 
            if v["available"]
        ]
        if available_variants:
            families[model_family] = {
                "display_name": family_data["display_name"],
                "variants": available_variants,
                "supports_negative_prompt": family_data["supports_negative_prompt"],
                "max_resolution": family_data["max_resolution"],
                "default_resolution": family_data["default_resolution"],
                "min_resolution": family_data["min_resolution"]
            }
    return families