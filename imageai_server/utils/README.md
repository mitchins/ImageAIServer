# ImageAI Server Utilities

Lightweight tools for working with ImageAI Server.

## 🖼️ generate_image.py

Simple CLI that calls the ImageAI Server API and saves images directly to files.
Perfect for bulk operations, scripting, and automation.

### Quick Start

```bash
# Basic usage
./generate_image.py -p "cute robot in garden"

# Save with specific filename
./generate_image.py -p "mountain landscape" -o landscape.png

# Use SDXL for quality
./generate_image.py -p "portrait" -m sdxl -o portrait.png

# Generate multiple images
./generate_image.py -p "abstract art" -n 4 -o art.png
# Creates: art_1.png, art_2.png, art_3.png, art_4.png

# Custom dimensions
./generate_image.py -p "banner" -W 1024 -H 256

# With negative prompt
./generate_image.py -p "photo" --negative "blurry, low quality"
```

### Options

- `-p, --prompt` (required): Text description of image
- `-o, --output`: Output filename (default: output.png)
- `-m, --model`: Model to use (sd15-onnx, sdxl, sdxl-turbo, flux1-schnell)
- `-s, --server`: Server URL (default: http://localhost:8001)
- `-W, --width`: Image width (default: 512)
- `-H, --height`: Image height (default: 512)
- `--negative`: Negative prompt
- `-n, --count`: Number of images to generate
- `-v, --verbose`: Show generation details

### Why Use This?

- **Direct file output** - No base64 parsing needed
- **Bulk operations** - Generate multiple images efficiently
- **Scriptable** - Easy to integrate in workflows
- **Token efficient** - For AI assistants, avoids large base64 strings

### Example Workflow

```bash
# Generate test images for a website
for style in "minimal" "colorful" "dark" "professional"; do
  ./generate_image.py -p "website hero $style style" -o "hero_$style.png"
done

# Batch generate with a list
cat prompts.txt | while read prompt; do
  ./generate_image.py -p "$prompt" -o "batch_$(date +%s).png"
done
```

## 🤖 mcp-server/

MCP (Model Context Protocol) server for Claude Desktop integration.

See [mcp-server/README.md](mcp-server/README.md) for setup instructions.

### Quick Setup

```json
// Add to Claude Desktop config
{
  "mcpServers": {
    "imageai": {
      "command": "node",
      "args": ["/path/to/utils/mcp-server/index.js"]
    }
  }
}
```

## Installation

### If you have ImageAI Server installed:
```bash
# The CLI tool is included! Just use:
imageai-generate -p "your prompt"
```

### Standalone usage (without server installation):
```bash
# Only needs requests library - no heavy dependencies!
pip install requests

# Download just the CLI tool
wget https://raw.githubusercontent.com/mitchins/ImageAIServer/main/imageai_server/utils/generate_image.py
chmod +x generate_image.py

# Use it (server must be running somewhere)
./generate_image.py -p "your prompt" -s http://your-server:8001
```

**Note:** These are CLIENT tools - they connect to a running ImageAI Server.
The server itself needs the full installation (`pip install imageaiserver[torch]`)