from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.openapi.docs import get_swagger_ui_html
import importlib
import pkgutil
import logging
import time
import psutil
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
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

# Initialize Jinja2 templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Track server start time for uptime calculation
SERVER_START_TIME = time.time()

# Auto-discover and register all routers
def register_routers():
    """Auto-discover routers from apps/* directories."""
    apps_dir = Path(__file__).parent
    
    for app_path in apps_dir.iterdir():
        if (app_path.is_dir() and 
            not app_path.name.startswith('_') and 
            not app_path.name == 'shared' and
            not app_path.name == 'multimodal_chat' and  # Skip - handled by chat_server
            (app_path / 'router.py').exists()):
            
            try:
                # Import the router module using relative import
                module_name = f".{app_path.name}.router"
                module = importlib.import_module(module_name, package=__package__)

                if hasattr(module, 'router'):
                    router = module.router
                    service_name = app_path.name.replace('_', '-')
                    # allow module to specify prefix/tag
                    prefix = getattr(module, 'router_prefix', f"/{service_name}")
                    tag    = getattr(module, 'router_tag', service_name)

                    app.include_router(router, prefix=prefix, tags=[tag])
                    print(f"✅ Registered {app_path.name} router at {prefix}")
            except Exception as e:
                print(f"❌ Failed to register {app_path.name} router: {e}")

register_routers()

# Root redirect to main UI
@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to UI."""
    return RedirectResponse(url="/ui/")

# UI Pages
@app.get("/ui/", response_class=HTMLResponse, include_in_schema=False)
async def ui_home(request: Request):
    """Main UI homepage."""
    return templates.TemplateResponse("home.html", {
        "request": request,
        "current_page": "home"
    })

@app.get("/ui/models", response_class=HTMLResponse, include_in_schema=False)
async def ui_models(request: Request):
    """Model management UI."""
    return templates.TemplateResponse("models.html", {
        "request": request,
        "current_page": "models"
    })

@app.get("/ui/generate", response_class=HTMLResponse, include_in_schema=False)
async def ui_generate(request: Request):
    """Generate images UI."""
    return templates.TemplateResponse("generate.html", {
        "request": request,
        "current_page": "generate"
    })

@app.get("/ui/vision-chat", response_class=HTMLResponse, include_in_schema=False)
async def ui_vision_chat(request: Request):
    """Vision chat UI."""
    return templates.TemplateResponse("vision-chat.html", {
        "request": request,
        "current_page": "vision-chat"
    })

@app.get("/ui/compare-faces", response_class=HTMLResponse, include_in_schema=False)
async def ui_compare_faces(request: Request):
    """Face comparison UI."""
    return templates.TemplateResponse("compare-faces.html", {
        "request": request,
        "current_page": "compare-faces"
    })

@app.get("/ui/docs", response_class=HTMLResponse, include_in_schema=False)
async def ui_docs(request: Request):
    """API documentation UI."""
    return templates.TemplateResponse("docs.html", {
        "request": request,
        "current_page": "docs"
    })

# Favicon redirect
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Redirect favicon requests to the static icon."""
    return RedirectResponse(url="/static/icon.png")


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


@app.get("/v1/backends", tags=["system"])
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




@app.get("/v1/system/status", tags=["system"])
async def get_system_status():
    """Get system status including uptime and memory usage."""
    try:
        # Calculate uptime
        current_time = time.time()
        uptime_seconds = current_time - SERVER_START_TIME
        uptime_hours = uptime_seconds / 3600
        
        # Get memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024  # Convert bytes to MB
        
        # Get system memory info
        system_memory = psutil.virtual_memory()
        system_memory_total_gb = system_memory.total / 1024 / 1024 / 1024
        system_memory_used_percent = system_memory.percent
        
        return {
            "uptime_seconds": uptime_seconds,
            "uptime_hours": uptime_hours,
            "memory_usage_mb": memory_usage_mb,
            "system_memory_total_gb": system_memory_total_gb,
            "system_memory_used_percent": system_memory_used_percent,
            "status": "healthy"
        }
    except Exception as e:
        return {
            "uptime_seconds": None,
            "uptime_hours": None,
            "memory_usage_mb": None,
            "system_memory_total_gb": None,
            "system_memory_used_percent": None,
            "status": "error",
            "error": str(e)
        }


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


@app.get("/v1/health", tags=["system"])
async def health_check():
    """Health check endpoint for system status."""
    try:
        from .shared.model_manager import get_model_manager
        from .shared.torch_loader import TORCH_AVAILABLE
        
        model_manager = get_model_manager()
        available_backends = [b.value for b in model_manager.get_available_backends()]
        
        # Check critical dependencies
        dependency_status = {}
        critical_deps = [
            "onnxruntime",
            "dghs-imgutils", 
            "transformers",
            "huggingface_hub"
        ]
        
        for dep in critical_deps:
            try:
                if dep == "dghs-imgutils":
                    import imgutils
                    dependency_status[dep] = "available"
                elif dep == "onnxruntime":
                    import onnxruntime
                    dependency_status[dep] = "available"
                elif dep == "transformers":
                    import transformers
                    dependency_status[dep] = "available"
                elif dep == "huggingface_hub":
                    import huggingface_hub
                    dependency_status[dep] = "available"
            except ImportError:
                dependency_status[dep] = "missing"
        
        return {
            "status": "ok",
            "service": "imageai-server",
            "pytorch_available": TORCH_AVAILABLE,
            "available_backends": available_backends,
            "dependencies": dependency_status
        }
    except Exception as e:
        return {
            "status": "error",
            "service": "imageai-server",
            "error": str(e)
        }

@app.get("/v1/vision-models", tags=["system"])
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
