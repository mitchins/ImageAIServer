#!/usr/bin/env python3
"""
Lightweight CLI for ImageAI Server image generation.
Calls the HTTP API and saves images directly to files.

Minimal dependencies - only needs 'requests' library.
Can be used standalone without full imageaiserver installation.
"""

import argparse
import base64
import json
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: requests library not installed", file=sys.stderr)
    print("Install with: pip install requests", file=sys.stderr)
    print("\nOr install the full package: pip install imageaiserver", file=sys.stderr)
    sys.exit(1)


def generate_image(
    prompt: str,
    output: str = "output.png",
    model: str = "sd15-onnx",
    server: str = "http://localhost:8001",
    width: int = 512,
    height: int = 512,
    negative_prompt: str = "",
    count: int = 1,
    verbose: bool = False
):
    """Generate image via ImageAI Server API and save to file."""
    
    # Prepare request
    url = f"{server}/v1/images/generations"
    payload = {
        "prompt": prompt,
        "model": model,
        "n": count,
        "width": width,
        "height": height,
        "negative_prompt": negative_prompt
    }
    
    if verbose:
        print(f"Generating with {model} at {server}...")
        print(f"Prompt: {prompt}")
        if negative_prompt:
            print(f"Negative: {negative_prompt}")
    
    try:
        # Make API request
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        data = response.json()
        images = data.get("data", [])
        
        if not images:
            print("Error: No images generated", file=sys.stderr)
            return False
        
        # Save image(s)
        output_path = Path(output)
        
        for i, img_data in enumerate(images):
            # Decode base64 image
            img_b64 = img_data.get("b64_json", "")
            img_bytes = base64.b64decode(img_b64)
            
            # Determine filename for multiple images
            if count > 1:
                stem = output_path.stem
                suffix = output_path.suffix or ".png"
                final_path = output_path.parent / f"{stem}_{i+1}{suffix}"
            else:
                final_path = output_path
            
            # Write to file
            final_path.write_bytes(img_bytes)
            print(f"✅ Saved: {final_path}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print(f"Error: Cannot connect to {server}", file=sys.stderr)
        print("Is ImageAI Server running? Start with: imageaiserver", file=sys.stderr)
        return False
    except requests.exceptions.HTTPError as e:
        print(f"Error: HTTP {e.response.status_code}", file=sys.stderr)
        try:
            error_detail = e.response.json().get("detail", "Unknown error")
            print(f"Details: {error_detail}", file=sys.stderr)
        except:
            pass
        return False
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate images using ImageAI Server",
        epilog="Examples:\n"
               "  %(prog)s -p 'cute robot in garden'\n"
               "  %(prog)s -p 'landscape' -m sdxl -o landscape.png\n"
               "  %(prog)s -p 'portrait' -W 768 -H 768 -n 4 -o portraits.png",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("-p", "--prompt", required=True,
                        help="Text prompt for image generation")
    
    # Optional arguments
    parser.add_argument("-o", "--output", default="output.png",
                        help="Output filename (default: output.png)")
    parser.add_argument("-m", "--model", default="sd15-onnx",
                        choices=["sd15-onnx", "sdxl", "sdxl-turbo", "flux1-schnell", "qwen-image"],
                        help="Model to use (default: sd15-onnx)")
    parser.add_argument("-s", "--server", default="http://localhost:8001",
                        help="ImageAI Server URL (default: http://localhost:8001)")
    parser.add_argument("-W", "--width", type=int, default=512,
                        help="Image width in pixels (default: 512)")
    parser.add_argument("-H", "--height", type=int, default=512,
                        help="Image height in pixels (default: 512)")
    parser.add_argument("--negative", default="",
                        help="Negative prompt (what to avoid)")
    parser.add_argument("-n", "--count", type=int, default=1,
                        help="Number of images to generate (default: 1)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    
    args = parser.parse_args()
    
    # Generate image
    success = generate_image(
        prompt=args.prompt,
        output=args.output,
        model=args.model,
        server=args.server,
        width=args.width,
        height=args.height,
        negative_prompt=args.negative,
        count=args.count,
        verbose=args.verbose
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()