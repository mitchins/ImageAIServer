#!/usr/bin/env python3
"""Test script to verify PyTorch backend integration."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared.model_manager import get_model_manager, BackendType
from shared.torch_loader import TORCH_AVAILABLE
from shared.onnx_loader import ONNX_AVAILABLE


def main():
    print("🔍 Testing Model Backend Detection\n")
    
    # Check available backends
    print("Backend Availability:")
    print(f"  - ONNX: {'✅ Available' if ONNX_AVAILABLE else '❌ Not Available'}")
    print(f"  - PyTorch: {'✅ Available' if TORCH_AVAILABLE else '❌ Not Available'}")
    print()
    
    # Initialize model manager
    manager = get_model_manager()
    available_backends = manager.get_available_backends()
    
    print(f"Registered Backends: {[b.value for b in available_backends]}")
    print()
    
    # List available models by backend
    all_models = manager.list_available_models()
    
    if BackendType.ONNX in all_models:
        onnx_models = all_models[BackendType.ONNX]
        print(f"ONNX Models ({len(onnx_models)} available):")
        for model in onnx_models[:5]:  # Show first 5
            print(f"  - {model}")
        if len(onnx_models) > 5:
            print(f"  ... and {len(onnx_models) - 5} more")
        print()
    
    if BackendType.PYTORCH in all_models:
        pytorch_models = all_models[BackendType.PYTORCH]
        print(f"PyTorch Models ({len(pytorch_models)} available):")
        for model in pytorch_models[:5]:  # Show first 5
            print(f"  - {model}")
        if len(pytorch_models) > 5:
            print(f"  ... and {len(pytorch_models) - 5} more")
        print()
    
    # Test backend selection
    print("Backend Selection Tests:")
    test_models = [
        "Gemma-3n-E2B-it-ONNX/Q4_MIXED",
        "meta-llama/Llama-2-7b-chat-hf",
        "microsoft/phi-2",
        "SmolVLM-256M-Instruct/UINT8"
    ]
    
    for model in test_models:
        try:
            selected_backend = manager.select_backend_for_model(model)
            print(f"  {model} → {selected_backend.value}")
        except Exception as e:
            print(f"  {model} → Error: {e}")
    
    print("\n✅ Backend detection test complete!")
    
    # Optional: Test actual model loading (commented out to avoid downloads)
    # print("\n🔄 Testing model loading (this may download models)...")
    # try:
    #     # Test ONNX model
    #     if BackendType.ONNX in available_backends:
    #         print("Loading ONNX model...")
    #         engine = manager.load_model("SmolVLM-256M-Instruct/UINT8", backend=BackendType.ONNX)
    #         result = engine.generate_text("Hello, world!", max_tokens=10)
    #         print(f"ONNX result: {result[:50]}...")
    #     
    #     # Test PyTorch model
    #     if BackendType.PYTORCH in available_backends:
    #         print("Loading PyTorch model...")
    #         engine = manager.load_model("microsoft/phi-2", backend=BackendType.PYTORCH)
    #         result = engine.generate_text("Hello, world!", max_tokens=10)
    #         print(f"PyTorch result: {result[:50]}...")
    # except Exception as e:
    #     print(f"Model loading test failed: {e}")


if __name__ == "__main__":
    main()