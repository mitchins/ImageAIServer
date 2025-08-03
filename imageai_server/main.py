from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html
import importlib
import pkgutil
import logging
from .shared.manage_cache import list_cached_entries
from .shared.model_types import ModelType

logger = logging.getLogger(__name__)
app = FastAPI(
    title="ImageAIServer API", 
    version="1.0.0",
    openapi_version="3.0.2",
    docs_url=None  # Disable default docs to use custom
)

STATIC_DIR = Path(__file__).resolve().parent / "static" / "manage"

# Auto-discover and register all routers
def register_routers():
    """Auto-discover routers from apps/* directories."""
    apps_dir = Path(__file__).parent
    
    for app_path in apps_dir.iterdir():
        if (app_path.is_dir() and 
            not app_path.name.startswith('_') and 
            not app_path.name == 'shared' and
            (app_path / 'router.py').exists()):
            
            try:
                # Import the router module using relative import
                module_name = f".{app_path.name}.router"
                module = importlib.import_module(module_name, package=__package__)
                
                if hasattr(module, 'router'):
                    router = module.router
                    service_name = app_path.name.replace('_', '-')
                    
                    # Register with service prefix
                    app.include_router(router, prefix=f"/{service_name}", tags=[service_name])
                    print(f"✅ Registered {service_name} router at /{service_name}")
                    
                    # Special case: multimodal chat also gets root-level OpenAI compatibility
                    if app_path.name == 'multimodal_chat':
                        app.include_router(router, tags=["openai-compatible"])
                        print(f"✅ Registered multimodal-chat router at root level for OpenAI compatibility")
                        
            except Exception as e:
                print(f"❌ Failed to register {app_path.name} router: {e}")

register_routers()

# Root redirect to main UI
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with helpful navigation to all UIs."""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ImageAI Server</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f8f9fa;
            margin: 0;
            padding: 0;
            min-height: 100vh;
        }
        
        .main-container { 
            max-width: 1000px;
            margin: 2rem auto;
            padding: 0 2rem;
            text-align: center;
        }
        
        h1 { 
            font-size: 2.5rem; 
            margin-bottom: 1rem; 
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .subtitle { 
            color: #666; 
            font-size: 1.2rem; 
            margin-bottom: 2rem; 
        }
        
        .links { 
            display: grid; 
            gap: 1.5rem; 
            margin: 2rem 0; 
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        }
        
        .link { 
            display: block; 
            padding: 1.5rem 2rem; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px; 
            text-decoration: none; 
            color: white;
            transition: all 0.3s ease; 
            font-size: 1.1rem;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }
        
        .link:hover { 
            transform: translateY(-4px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }
        
        .link strong { 
            display: block; 
            font-size: 1.3rem; 
            margin-bottom: 0.5rem; 
        }
        
        .link small { 
            opacity: 0.9; 
            font-size: 1rem; 
        }
        
        .status { 
            margin-top: 2rem; 
            padding: 1rem 0; 
            color: #155724;
            border-left: 4px solid #28a745;
            padding-left: 1rem;
            text-align: left;
        }
        
        .api-endpoints { 
            margin-top: 2rem; 
            text-align: left; 
            color: #495057;
            padding: 1.5rem 0; 
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            border-top: 1px solid #e9ecef;
            padding-top: 1.5rem;
        }
        
        .api-endpoints strong { 
            color: #333; 
            display: block; 
            margin-bottom: 0.5rem; 
        }
        
        .endpoint { 
            margin: 0.5rem 0; 
            padding: 0.3rem 0; 
            color: #6f42c1; 
        }
        
        @media (max-width: 768px) {
            .main-container { 
                margin: 1rem; 
                padding: 1.5rem; 
            }
            .links { 
                grid-template-columns: 1fr; 
            }
        }
    </style>
</head>
<body>
    <script src="/static/manage/navigation.js"></script>
    
    <div class="main-container">
        <h1><img src="/static/icon.png" alt="ImageAIServer Icon" style="height: 1em; vertical-align: middle;"> ImageAIServer</h1>
        <p class="subtitle">Privacy-focused AI inference server monitoring and quick access</p>
        
        <!-- Server Status Section -->
        <div class="status">
            ✅ <strong>Server Status:</strong> Running and healthy<br>
            <div style="margin-top: 0.5rem; font-size: 0.9rem;">
                <span id="backend-status">Loading backend status...</span>
            </div>
        </div>
        
        <script>
        // Load backend status
        fetch('/v1/backends')
            .then(response => response.json())
            .then(data => {
                const onnxStatus = data.backends.onnx.available ? '✅ ONNX' : '❌ ONNX';
                const pytorchStatus = data.backends.pytorch.available ? '✅ PyTorch' : '❌ PyTorch';
                const onnxCount = data.backends.onnx.models_count || 0;
                const pytorchCount = data.backends.pytorch.models_count || 0;
                
                // Build GPU status
                let gpuStatus = '';
                if (data.gpu) {
                    const gpu = data.gpu;
                    if (gpu.error || gpu.torch_not_available) {
                        gpuStatus = '❌ GPU';
                    } else {
                        const parts = [];
                        if (gpu.cuda_available) {
                            parts.push(`✅ CUDA (${gpu.cuda_device_count} device${gpu.cuda_device_count !== 1 ? 's' : ''})`);
                        }
                        if (gpu.mps_available) {
                            parts.push('✅ MPS');
                        }
                        if (parts.length === 0) {
                            parts.push('💻 CPU only');
                        }
                        gpuStatus = parts.join(', ');
                        
                        if (gpu.current_device) {
                            gpuStatus += ` | Using: ${gpu.current_device}`;
                        }
                    }
                }
                
                document.getElementById('backend-status').innerHTML = 
                    `<strong>Backends:</strong> ${onnxStatus} (${onnxCount} models), ${pytorchStatus} (${pytorchCount} models)<br>` +
                    `<strong>Acceleration:</strong> ${gpuStatus}`;
            })
            .catch(() => {
                document.getElementById('backend-status').innerHTML = 
                    '<strong>Backends:</strong> Status unavailable';
            });
        </script>
        
        <!-- Quick Actions Section -->
        <h2 style="margin-top: 2rem; margin-bottom: 1rem; color: #333;">Quick Actions</h2>
        <div class="links">
            <a href="/manage/ui/" class="link">
                🌐 <strong>Model Management</strong>
                <small>Download and manage ONNX models</small>
            </a>
            <a href="/manage/ui/vision-test.html" class="link">
                🎯 <strong>Vision & Face Testing</strong>
                <small>Drag & drop testing interface</small>
            </a>
            <a href="/docs" class="link">
                📚 <strong>API Documentation</strong>
                <small>Interactive OpenAPI docs</small>
            </a>
        </div>
        
        <!-- API Information Section -->
        <h2 style="margin-top: 2rem; margin-bottom: 1rem; color: #333;">API Endpoints</h2>
        <div class="api-endpoints">
            <div class="endpoint"><strong>POST</strong> /v1/chat/completions <span style="color: #666;">• Vision + text inference (ONNX)</span></div>
            <div class="endpoint"><strong>POST</strong> /chat-server/v1/chat/completions/torch <span style="color: #666;">• PyTorch models (optional)</span></div>
            <div class="endpoint"><strong>POST</strong> /v1/image/compare_faces <span style="color: #666;">• Face comparison</span></div>
            <div class="endpoint"><strong>GET</strong> /v1/backends <span style="color: #666;">• Backend availability status</span></div>
            <div class="endpoint"><strong>GET</strong> /v1/models <span style="color: #666;">• List available models</span></div>
            <div class="endpoint"><strong>GET</strong> /health <span style="color: #666;">• Server health check</span></div>
        </div>
    </div>
</body>
</html>
    """)

# Quick redirect for common paths
@app.get("/ui")
async def ui_redirect():
    """Redirect /ui to the main management UI."""
    return RedirectResponse(url="/manage/ui/")

@app.get("/test")
async def test_redirect():
    """Redirect /test to the vision test UI."""
    return RedirectResponse(url="/manage/ui/vision-test.html")

@app.get("/docs", response_class=HTMLResponse)
async def custom_swagger_ui_html():
    """Custom docs page with navigation."""
    return HTMLResponse(content=f"""
<!DOCTYPE html>
<html>
<head>
    <title>ImageAIServer API Documentation</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui.css" />
    <style>
        body { margin: 0; padding: 0; }
        .imageai-nav {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            position: sticky;
            top: 0;
            z-index: 1000;
        }
        .nav-container {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 1rem;
            height: 60px;
        }
        .nav-brand .brand-link {
            color: white;
            text-decoration: none;
            font-size: 1.5rem;
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .nav-brand .brand-link:hover { color: rgba(255,255,255,0.9); }
        .nav-links {
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }
        .nav-link {
            color: rgba(255,255,255,0.9);
            text-decoration: none;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            transition: all 0.3s ease;
            font-size: 0.9rem;
            white-space: nowrap;
        }
        .nav-link:hover {
            background: rgba(255,255,255,0.1);
            color: white;
            transform: translateY(-1px);
        }
        .nav-link.active {
            background: rgba(255,255,255,0.2);
            color: white;
        }
        #swagger-ui { padding-top: 0; }
        .swagger-ui .topbar { display: none; }
    </style>
</head>
<body>
    <nav class="imageai-nav">
        <div class="nav-container">
            <div class="nav-brand">
                <a href="/" class="brand-link">🤖 ImageAIServer</a>
            </div>
            <div class="nav-links">
                <a href="/" class="nav-link" title="Home">🏠 Home</a>
                <a href="/manage/ui/" class="nav-link" title="Model Management">🌐 Models</a>
                <a href="/manage/ui/vision-test.html" class="nav-link" title="Vision & Face Testing">🎯 Test</a>
                <a href="/docs" class="nav-link active" title="API Documentation">📚 Docs</a>
            </div>
        </div>
    </nav>
    
    <div id="swagger-ui"></div>
    
    <script src="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui-bundle.js"></script>
    <script>
        const ui = SwaggerUIBundle({
            url: '/openapi.json',
            dom_id: '#swagger-ui',
            presets: [
                SwaggerUIBundle.presets.apis,
                SwaggerUIBundle.presets.standalone
            ],
            layout: "BaseLayout",
            deepLinking: true,
            showExtensions: true,
            showCommonExtensions: true,
            tryItOutEnabled: true
        });
    </script>
</body>
</html>
    """)

app.mount(
    "/manage/ui",
    StaticFiles(directory=STATIC_DIR, html=True),
    name="manage_ui",
)

# Mount static files for root access
app.mount(
    "/static/manage",
    StaticFiles(directory=STATIC_DIR),
    name="static_manage",
)

# Mount static files from the 'static' directory for root access (e.g., /static/icon.png)
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).resolve().parent / "static"),
    name="static",
)


@app.get("/v1/backends")
async def list_backends():
    """List available model backends and their status."""
    try:
        from .shared.model_manager import get_model_manager
        
        manager = get_model_manager()
        available_backends = manager.get_available_backends()
        
        # Get GPU/accelerator availability
        gpu_info = {}
        try:
            from .shared.torch_loader import TORCH_AVAILABLE
            if TORCH_AVAILABLE:
                import torch
                gpu_info = {
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
                    "current_device": None
                }
                
                # Get current device info from PyTorch backend if available
                if hasattr(manager, 'backends') and manager.backends:
                    from .shared.model_manager import BackendType
                    if BackendType.PYTORCH in manager.backends:
                        pytorch_backend = manager.backends[BackendType.PYTORCH]
                        if hasattr(pytorch_backend, 'device'):
                            gpu_info["current_device"] = str(pytorch_backend.device)
            else:
                gpu_info = {"torch_not_available": True}
        except Exception as e:
            gpu_info = {"error": str(e)}
        
        backend_info = {}
        for backend_type in ["onnx", "pytorch"]:
            if backend_type == "onnx":
                from .shared.onnx_loader import ONNX_AVAILABLE
                available = ONNX_AVAILABLE
            else:  # pytorch
                from .shared.torch_loader import TORCH_AVAILABLE
                available = TORCH_AVAILABLE
            
            backend_info[backend_type] = {
                "available": available,
                "initialized": backend_type in [b.value for b in available_backends],
                "models_count": 0
            }
            
            if available and backend_type in [b.value for b in available_backends]:
                # Get model count
                all_models = manager.list_available_models()
                from .shared.model_manager import BackendType
                backend_enum = BackendType(backend_type)
                if backend_enum in all_models:
                    backend_info[backend_type]["models_count"] = len(all_models[backend_enum])
        
        return {
            "backends": backend_info,
            "gpu": gpu_info,
            "default_selection": "auto",
            "status": "operational" if len(available_backends) > 0 else "no_backends"
        }
    except Exception as e:
        return {
            "backends": {
                "onnx": {"available": False, "initialized": False, "models_count": 0},
                "pytorch": {"available": False, "initialized": False, "models_count": 0}
            },
            "gpu": {"error": "Could not determine GPU status"},
            "default_selection": "auto",
            "status": "error",
            "error": str(e)
        }


@app.get("/v1/models")
async def list_models():
    """List available chat-compatible models in OpenAI-compatible format."""
    try:
        cached_entries = list_cached_entries()
        
        # Filter for ONNX models that are LLM-capable and create OpenAI-compatible response
        models = []
        chat_compatible_types = ModelType.chat_compatible_types()
        
        for entry in cached_entries:
            if (entry["path"].endswith(".onnx") and 
                entry["kind"] in [t.value for t in chat_compatible_types]):
                
                # Always create full model ID with repo and complete file path
                # This gives users the exact string they need for API calls
                model_id = f"{entry['repo']}/{entry['path']}"
                
                models.append({
                    "id": model_id,
                    "object": "model",
                    "created": int(entry["last_used"]),
                    "owned_by": entry["repo"].split("/")[0] if "/" in entry["repo"] else "huggingface",
                })
        
        return {
            "object": "list",
            "data": models
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _is_model_downloaded(repo_id: str, required_files: Optional[List[str]] = None) -> bool:
    """Check if a model is downloaded locally."""
    try:
        from huggingface_hub import scan_cache_dir
        from huggingface_hub.errors import CacheNotFound
        from pathlib import Path
        
        # Method 1: Use HuggingFace cache scanner (preferred)
        try:
            cache = scan_cache_dir()
            cached_repos = {repo.repo_id: repo for repo in cache.repos}
            
            if repo_id not in cached_repos:
                return False
                
            if required_files:
                # Check if all required files are present
                repo_info = cached_repos[repo_id]
                found_files = set()
                for revision in repo_info.revisions:
                    for file_info in revision.files:
                        # Convert absolute path to relative path for comparison
                        abs_path = str(file_info.file_path)
                        # Extract relative path from the snapshot directory
                        if '/snapshots/' in abs_path:
                            rel_path = abs_path.split('/snapshots/')[1]
                            # Remove the hash directory to get just the file path
                            if '/' in rel_path:
                                rel_path = '/'.join(rel_path.split('/')[1:])
                                found_files.add(rel_path)
                
                return all(req_file in found_files for req_file in required_files)
            else:
                # Just check if repo exists in cache
                return True
                
        except CacheNotFound:
            pass
        
        # Method 2: Direct file system check (fallback)
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        repo_cache_name = f"models--{repo_id.replace('/', '--')}"
        repo_dir = cache_dir / repo_cache_name
        
        if not repo_dir.exists():
            return False
            
        if required_files:
            # Check if all required files exist
            for snapshot_dir in repo_dir.glob("snapshots/*"):
                found_files = set()
                for file_path in snapshot_dir.rglob("*"):
                    if file_path.is_file():
                        rel_path = str(file_path.relative_to(snapshot_dir))
                        found_files.add(rel_path)
                
                if all(req_file in found_files for req_file in required_files):
                    return True
            return False
        else:
            # Check if any snapshot directory has files
            for snapshot_dir in repo_dir.glob("snapshots/*"):
                if any(snapshot_dir.iterdir()):
                    return True
            return False
            
    except Exception as e:
        logger.warning(f"Error checking if model {repo_id} is downloaded: {e}")
        return False


@app.get("/v1/vision-models")
async def list_vision_models():
    """List only locally available vision-capable models (ONNX and PyTorch/GGUF)."""
    try:
        from .shared.model_identifier import ModelCatalog
        from .shared.model_types import get_available_model_quants, get_curated_model_config
        
        models = []
        
        # Add ONNX models from curated list (only if downloaded)
        onnx_models = get_available_model_quants()
        
        # Model name to repo_id mapping for ONNX models
        onnx_model_repo_mapping = {
            "Gemma-3n-E2B-it-ONNX": "onnx-community/gemma-3n-E2B-it-ONNX",
        }
        
        for model_config in onnx_models:
            if "/" in model_config:  # Model with quantization
                model_name, quantization = model_config.split("/", 1)
                
                # Get the repo_id for this curated model
                repo_id = onnx_model_repo_mapping.get(model_name)
                if not repo_id:
                    logger.warning(f"No repo_id mapping found for ONNX model: {model_name}")
                    continue
                
                curated_config = get_curated_model_config(model_config)
                if curated_config:
                    # Get required files for this quantization
                    required_files = list(curated_config.values())
                    
                    # Only include if model is downloaded locally
                    if _is_model_downloaded(repo_id, required_files):
                        display_name = f"{model_name}/{quantization}"
                        backend = "ONNX"
                        description = "ONNX"
                        
                        models.append({
                            "id": model_config,
                            "name": display_name,
                            "backend": backend,
                            "description": description,
                            "quantization": quantization
                        })
        
        # Add PyTorch/GGUF models from ModelCatalog (only if downloaded)
        for model_id, model_info in ModelCatalog.MODELS.items():
            if model_info.backend.value == "pytorch":
                # The ModelCatalog uses GGUF repos as repo_id for efficiency, but we need to check
                # both the original PyTorch repo and GGUF repo for downloads
                
                # Mapping from GGUF repo (in catalog) to original PyTorch repo
                gguf_to_original_mapping = {
                    "ggml-org/SmolVLM-256M-Instruct-GGUF": "HuggingFaceTB/SmolVLM-256M-Instruct",
                    "ggml-org/SmolVLM-500M-Instruct-GGUF": "HuggingFaceTB/SmolVLM-500M-Instruct",
                    "bartowski/ibm-granite_granite-vision-3.2-2b-GGUF": "ibm-granite/granite-vision-3.2-2b",
                    "bartowski/mistral-community_pixtral-12b-GGUF": "mistralai/pixtral-12b",
                    "bartowski/google_gemma-3-27b-it-GGUF": "google/gemma-3-27b-it",
                }
                
                original_repo = gguf_to_original_mapping.get(model_info.repo_id)
                
                if model_info.repo_id in ModelCatalog.GGUF_QUANTIZATIONS:
                    # GGUF quantized model - check GGUF repo
                    if _is_model_downloaded(model_info.repo_id):
                        quantizations = ModelCatalog.get_available_quantizations(model_info.repo_id)
                        for quant in quantizations:
                            display_name = f"{model_info.family.title()} {model_info.size.upper()}"
                            if model_info.variant and model_info.variant != "instruct":
                                display_name += f" {model_info.variant.title()}"
                            
                            description = "GGUF"
                            
                            models.append({
                                "id": f"{model_id}:{quant.value}",
                                "name": f"{display_name} ({quant.value.upper()})",
                                "backend": "PyTorch/GGUF",
                                "description": description,
                                "quantization": quant.value.upper()
                            })
                    
                    # Also check original PyTorch repo if it exists and is downloaded
                    if original_repo and _is_model_downloaded(original_repo):
                        display_name = f"{model_info.family.title()} {model_info.size.upper()}"
                        if model_info.variant and model_info.variant != "instruct":
                            display_name += f" {model_info.variant.title()}"
                        
                        description = "PyTorch"
                        models.append({
                            "id": f"{model_id}:original",
                            "name": f"{display_name} (Original)",
                            "backend": "PyTorch",
                            "description": description,
                            "quantization": "FP16"
                        })
                else:
                    # Regular PyTorch model without GGUF - check the repo_id directly
                    if _is_model_downloaded(model_info.repo_id):
                        display_name = f"{model_info.family.title()} {model_info.size.upper()}"
                        if model_info.variant and model_info.variant != "instruct":
                            display_name += f" {model_info.variant.title()}"
                        
                        description = "PyTorch"
                        models.append({
                            "id": model_id,
                            "name": display_name,
                            "backend": "PyTorch",
                            "description": description,
                            "quantization": "FP16"
                        })
        
        return {
            "object": "list",
            "data": models
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Entry point for the ImageAIServer CLI command."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
