# ImageAIServer <img src="imageai_server/static/icon.png" alt="ImageAIServer Icon" width="30" height="30">

**Privacy-focused AI image generation, vision, and face analysis - runs 100% offline on your hardware**

---

## 🚀 START HERE - PICK ONE:

### **Option 1: CPU/Low Memory (Works Anywhere)**
```bash
# Install and run - uses ONNX models (~500MB RAM)
pip install imageaiserver
imageaiserver

# Open http://localhost:8001 and generate images!
```

### **Option 2: GPU/High Quality (NVIDIA/Apple Silicon)**
```bash
# Install with PyTorch support for best quality
pip install imageaiserver[torch]
imageaiserver

# Open http://localhost:8001 - uses SDXL, FLUX models
```

---

## 🎨 What Can You Do?

Once running, visit **http://localhost:8001** to:

- **Generate Images** - Text-to-image with Stable Diffusion (ONNX INT8 for CPU, SDXL/FLUX for GPU)
- **Vision Chat** - Analyze images with AI models (Qwen, Llama Vision)
- **Face Comparison** - Compare face similarity between images
- **API Access** - OpenAI-compatible endpoints at `/v1/images/generations`

## 📦 Models Available

| Model | Memory | Hardware | Quality | Speed |
|-------|---------|----------|---------|-------|
| **SD 1.5 ONNX INT8** | ~500MB | CPU | Good | Fast |
| **SDXL** | ~8GB | GPU | Excellent | Medium |
| **SDXL Turbo** | ~8GB | GPU | Very Good | Very Fast |
| **FLUX.1 Schnell** | ~12GB | GPU | Excellent | Fast |

## 🔌 API Examples

### Generate Image (CLI)
```bash
# Installed with imageaiserver package!
imageai-generate -p "a serene mountain landscape" -o landscape.png
imageai-generate -p "robot" -m sdxl -n 4 -o robots.png
```

### Generate Image (curl)
```bash
curl -X POST http://localhost:8001/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a serene mountain landscape", "model": "sd15-onnx"}'
```

### Python
```python
import requests

response = requests.post("http://localhost:8001/v1/images/generations", 
    json={"prompt": "cute robot in garden", "model": "sd15-onnx"})
image_data = response.json()["data"][0]["b64_json"]
```

### Use with Claude Desktop (MCP)
```json
{
  "mcpServers": {
    "imageai": {
      "command": "node",
      "args": ["./mcp-server/index.js"],
      "env": {"IMAGEAI_SERVER_URL": "http://localhost:8001"}
    }
  }
}
```

## 🛠️ Advanced Options

### Custom Port
```bash
imageaiserver --port 8080
```

### Docker
```bash
docker run -p 8001:8001 imageaiserver:latest
```

### Development
```bash
git clone https://github.com/mitchins/ImageAIServer
cd ImageAIServer
pip install -e .[torch]
python -m imageai_server
```

## 📚 More Info

- **[Full Documentation](docs/README.md)** - Architecture, API reference, configuration
- **[PyTorch Backend](docs/PYTORCH_BACKEND.md)** - GPU setup, model details
- **[Docker Guide](docs/DOCKER.md)** - Container deployment
- **[Architecture](docs/ARCHITECTURE.md)** - System design

## 💡 Tips

- **No GPU?** Use `sd15-onnx` model - works great on CPU with INT8 quantization
- **Have GPU?** Install with `[torch]` for SDXL and FLUX models
- **Need speed?** Use `sdxl-turbo` for fast generation
- **Best quality?** Use `sdxl` or `flux1-schnell` with GPU

## 📄 License

MIT - Use freely in your projects!

---

**Problems?** [Open an issue](https://github.com/mitchins/ImageAIServer/issues) | **Questions?** See [docs](docs/)