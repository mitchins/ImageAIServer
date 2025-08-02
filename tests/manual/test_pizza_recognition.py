#!/usr/bin/env python3
"""
Simple pizza recognition test for SmolVLM-256M INT8.

This script tests whether SmolVLM can recognize a pizza image without 
being explicitly told it's food in the prompt.

Requirements:
- pip install -r requirements-torch.txt
- Internet connection (for first-time model download)
"""

import sys
import os
import base64
import io
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
import numpy as np


def create_pizza_image():
    """Create a simple but recognizable pizza image."""
    print("🍕 Creating synthetic pizza image...")
    
    # Create a circular pizza base
    img = Image.new('RGB', (512, 512), color='white')
    pixels = np.array(img)
    
    center_x, center_y = 256, 256
    radius = 200
    
    # Draw pizza base (brown/tan crust)
    for y in range(512):
        for x in range(512):
            distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
            if distance <= radius:
                # Pizza base color
                pixels[y, x] = [210, 180, 140]  # Tan pizza base
                
                # Add cheese layer (yellowish)
                if distance <= radius - 20:
                    pixels[y, x] = [255, 240, 180]  # Cheese color
                    
                    # Add pepperoni (red circles)
                    pepperoni_positions = [
                        (center_x - 80, center_y - 60),
                        (center_x + 70, center_y - 80),
                        (center_x - 50, center_y + 70),
                        (center_x + 90, center_y + 40),
                        (center_x - 20, center_y - 100),
                        (center_x + 30, center_y + 90),
                        (center_x - 100, center_y + 20),
                        (center_x + 50, center_y - 30),
                    ]
                    
                    for px, py in pepperoni_positions:
                        pepperoni_dist = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
                        if pepperoni_dist <= 20:
                            pixels[y, x] = [150, 40, 40]  # Dark red pepperoni
                
                # Crust edge (darker brown)
                elif distance > radius - 20:
                    pixels[y, x] = [160, 120, 80]  # Darker crust
    
    img = Image.fromarray(pixels.astype('uint8'))
    
    # Save for reference
    img.save("synthetic_pizza.png")
    print("   Saved synthetic pizza image as: synthetic_pizza.png")
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return img_base64, img


def test_pizza_recognition():
    """Test pizza recognition with SmolVLM."""
    print("🧪 Testing Pizza Recognition with SmolVLM-256M INT8\n")
    
    # Check if PyTorch is available
    try:
        import torch
        import transformers
        print("✅ PyTorch and transformers available")
    except ImportError as e:
        print(f"❌ PyTorch dependencies missing: {e}")
        print("   Install with: pip install -r requirements-torch.txt")
        return False
    
    # Import our components
    from shared.model_manager import get_model_manager, BackendType
    from shared.torch_loader import TORCH_AVAILABLE
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch backend not available")
        return False
    
    # Create pizza image
    pizza_img_base64, pizza_img = create_pizza_image()
    
    # Initialize model manager
    print("\n📥 Loading SmolVLM-256M with INT8 quantization...")
    manager = get_model_manager()
    
    # Test prompts (no food hints)
    test_prompts = [
        "What is in this image?",
        "Describe what you see.",
        "What objects are visible here?",
        "Analyze this image.",
    ]
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🔍 Test {i}: '{prompt}'")
        
        try:
            result = manager.generate_text(
                model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
                text=prompt,
                max_tokens=60,
                images=[pizza_img_base64],
                backend=BackendType.PYTORCH
            )
            
            print(f"   Response: {result}")
            results.append((prompt, result))
            
        except Exception as e:
            print(f"   Error: {e}")
            results.append((prompt, f"ERROR: {e}"))
    
    # Analyze results
    print("\n📊 Analysis:")
    
    food_terms = [
        'pizza', 'food', 'meal', 'eat', 'eating', 'dish', 'cuisine', 
        'pepperoni', 'cheese', 'bread', 'dough', 'crust', 'toppings',
        'italian', 'slice', 'circular', 'round'
    ]
    
    detection_scores = []
    
    for prompt, result in results:
        if result.startswith("ERROR:"):
            continue
            
        result_lower = result.lower()
        detected_terms = [term for term in food_terms if term in result_lower]
        
        pizza_detected = 'pizza' in result_lower
        food_detected = any(term in result_lower for term in ['food', 'meal', 'eat', 'dish'])
        visual_detected = any(term in result_lower for term in ['circular', 'round', 'brown', 'red'])
        
        score = 0
        if pizza_detected:
            score = 10  # Perfect score
            status = "🎯 PIZZA DETECTED"
        elif food_detected:
            score = 7   # Good score
            status = "✅ FOOD DETECTED"
        elif len(detected_terms) > 2:
            score = 5   # Okay score
            status = "⚠️ FOOD TERMS DETECTED"
        elif visual_detected:
            score = 3   # Minimal score
            status = "🔍 VISUAL FEATURES"
        else:
            score = 0   # No recognition
            status = "❌ NO FOOD RECOGNITION"
        
        detection_scores.append(score)
        
        print(f"   '{prompt}' → {status}")
        print(f"      Terms found: {detected_terms}")
        print(f"      Score: {score}/10")
    
    # Overall assessment
    if detection_scores:
        avg_score = sum(detection_scores) / len(detection_scores)
        max_score = max(detection_scores)
        
        print(f"\n🏆 Overall Results:")
        print(f"   Average score: {avg_score:.1f}/10")
        print(f"   Best score: {max_score}/10")
        
        if max_score >= 10:
            print("   ✅ EXCELLENT: Model correctly identified pizza!")
        elif max_score >= 7:
            print("   ✅ GOOD: Model recognized it as food!")
        elif max_score >= 5:
            print("   ⚠️ OKAY: Model detected food-related elements")
        elif max_score >= 3:
            print("   ⚠️ MINIMAL: Model saw visual features but missed food context")
        else:
            print("   ❌ POOR: Model did not recognize food characteristics")
        
        success = max_score >= 5
    else:
        success = False
    
    # Cleanup
    pizza_path = Path("synthetic_pizza.png")
    if pizza_path.exists():
        print(f"\n📁 Synthetic pizza image saved as: {pizza_path.absolute()}")
        print("   You can view this image to see what the model was analyzing")
    
    return success


def main():
    """Run the pizza recognition test."""
    success = test_pizza_recognition()
    
    if success:
        print("\n🎉 Pizza recognition test PASSED!")
        print("   SmolVLM successfully recognized food elements without explicit hints")
    else:
        print("\n❌ Pizza recognition test FAILED!")
        print("   SmolVLM did not adequately recognize the pizza/food content")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())