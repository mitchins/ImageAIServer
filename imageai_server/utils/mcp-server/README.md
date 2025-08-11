# ImageAI MCP Server

An MCP (Model Context Protocol) server that provides image generation capabilities through the ImageAI Server API.

## Features

- **Direct File Saving**: Images saved to disk - no base64 data in Claude's context!
- **Token Efficient**: Returns only file paths, not image data
- **Generate Images**: Create images using various AI models (Stable Diffusion, FLUX, etc.)
- **Model Selection**: Choose from available models including quantized ONNX versions
- **Flexible Parameters**: Control image size, model, negative prompts, and batch generation
- **Model Information**: List available models with their capabilities and memory requirements

## Installation

```bash
cd mcp-server
npm install
```

## Configuration

Set the ImageAI Server URL (defaults to `http://localhost:8001`):

```bash
export IMAGEAI_SERVER_URL=http://your-server:8001
```

## Usage with Claude Desktop

Add to your Claude Desktop configuration (`claude_desktop_config.json`):

**Mac/Linux:** `~/.config/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "imageai": {
      "command": "node",
      "args": ["/path/to/ImageAIServer/utils/mcp-server/index.js"],
      "env": {
        "IMAGEAI_SERVER_URL": "http://localhost:8001"
      }
    }
  }
}
```

## Available Tools

### generate_image

Generate images using AI models.

**Parameters:**
- `prompt` (required): Text description of the image to generate
- `model` (optional): Model to use - options:
  - `sd15-onnx` (default): Stable Diffusion 1.5 ONNX INT8 (~500MB)
  - `sdxl`: Stable Diffusion XL (~8GB VRAM)
  - `sdxl-turbo`: SDXL Turbo (~8GB VRAM)
  - `flux1-schnell`: FLUX.1 Schnell (requires GPU)
  - `qwen-image`: Qwen Image model
- `width` (optional): Image width in pixels (default: 512)
- `height` (optional): Image height in pixels (default: 512)
- `negative_prompt` (optional): Text describing what to avoid
- `n` (optional): Number of images to generate (1-4, default: 1)
- `output_dir` (optional): Directory to save images (default: system temp/imageai-mcp)

**Example:**
```
Generate an image of "a serene mountain landscape at sunset" using the sd15-onnx model
```

**Output:**
Images are saved to disk and the tool returns:
- File paths where images were saved
- No base64 data (keeps context clean!)
- Easy to reference saved files later

### list_models

List available image generation models and their capabilities.

**Example:**
```
List available models
```

## Model Information

The MCP server provides detailed information about each model including:

- **Display Name**: Human-readable model name
- **Description**: Model capabilities and use cases
- **Memory Requirements**: RAM/VRAM needed
- **Quantization**: Precision level (FP16, INT8)
- **Engine**: Backend (PyTorch, ONNX)
- **Resolution Support**: Min/max/default image sizes
- **Negative Prompt Support**: Whether the model supports negative prompts

## Examples

### Basic Image Generation
```
Generate an image: "A cute robot in a garden"
```

### Advanced Generation
```
Generate an image with prompt "professional headshot photo", model "sdxl", width 1024, height 1024, negative_prompt "blurry, low quality"
```

### Model Exploration
```
List all available models to see their capabilities
```

## Requirements

- Node.js 18+
- Running ImageAI Server instance
- Network access to the ImageAI Server

## Troubleshooting

1. **Connection Issues**: Ensure ImageAI Server is running and accessible at the configured URL
2. **Model Loading**: Some models require GPU/significant RAM - check model requirements
3. **Generation Failures**: Check server logs for detailed error messages

## Development

To modify or extend the MCP server:

1. Edit `index.js` to add new tools or modify existing ones
2. Update `package.json` if adding new dependencies
3. Test with a local MCP client or Claude Desktop

## API Compatibility

This MCP server is compatible with ImageAI Server API v1 endpoints:
- `/v1/images/generations` - Image generation
- `/v1/models/generation` - Model listing