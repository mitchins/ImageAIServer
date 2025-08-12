"""
TensorRT Engine Management System

Manages pre-built TensorRT engines with automatic downloading and architecture detection.
Engines are distributed in "model-quant-architecture" format (e.g., sdxl-fp16-ampere).
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import torch
import requests
from huggingface_hub import hf_hub_download, list_repo_files

logger = logging.getLogger(__name__)

class GPUArchitecture(Enum):
    """Supported GPU architectures for TensorRT engines."""
    AMPERE = "ampere"      # 8.0+ (A100, RTX 30xx)
    ADA_LOVELACE = "ada"   # 8.9+ (RTX 40xx)
    BLACKWELL = "blackwell" # 12.0+ (Future)
    
    @classmethod
    def from_compute_capability(cls, major: int, minor: int) -> 'GPUArchitecture':
        """Detect GPU architecture from compute capability."""
        compute_cap = major + minor / 10.0
        
        if compute_cap >= 12.0:
            return cls.BLACKWELL
        elif compute_cap >= 8.9:
            return cls.ADA_LOVELACE
        elif compute_cap >= 8.0:
            return cls.AMPERE
        else:
            raise ValueError(f"Unsupported compute capability: {compute_cap} (need 8.0+)")

@dataclass
class EngineMetadata:
    """Metadata for a TensorRT engine set."""
    model_id: str          # e.g., "sdxl"
    quantization: str      # e.g., "fp16", "fp8", "int8"
    architecture: GPUArchitecture
    batch_size: int        # Optimal batch size
    height: int           # Optimal height
    width: int            # Optimal width
    file_size: int        # Total size in bytes
    checksum: str         # SHA256 checksum
    tensorrt_version: str # TensorRT version used
    created_at: str       # ISO timestamp
    engines: Dict[str, str] # stage -> filename mapping

    @property
    def identifier(self) -> str:
        """Engine set identifier: model-quant-architecture."""
        return f"{self.model_id}-{self.quantization}-{self.architecture.value}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = self.__dict__.copy()
        data['architecture'] = self.architecture.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EngineMetadata':
        """Create from dictionary."""
        data = data.copy()
        data['architecture'] = GPUArchitecture(data['architecture'])
        return cls(**data)

class EngineManager:
    """Manages TensorRT engine downloading and caching."""
    
    def __init__(self, 
                 cache_dir: str = "./tensorrt_engines",
                 hf_repo: str = "your-org/tensorrt-engines",
                 hf_token: Optional[str] = None):
        """
        Initialize engine manager.
        
        Args:
            cache_dir: Local directory for engine cache
            hf_repo: HuggingFace repository for engine storage
            hf_token: HuggingFace API token
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hf_repo = hf_repo
        self.hf_token = hf_token
        
        # Detect current GPU architecture
        self.local_architecture = self._detect_gpu_architecture()
        logger.info(f"Detected GPU architecture: {self.local_architecture.value}")
        
        # Load cached metadata
        self.metadata_cache = self._load_metadata_cache()
    
    def _detect_gpu_architecture(self) -> GPUArchitecture:
        """Detect the local GPU architecture."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        major, minor = torch.cuda.get_device_capability()
        return GPUArchitecture.from_compute_capability(major, minor)
    
    def _load_metadata_cache(self) -> Dict[str, EngineMetadata]:
        """Load cached engine metadata."""
        cache_file = self.cache_dir / "metadata.json"
        if not cache_file.exists():
            return {}
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            return {
                key: EngineMetadata.from_dict(meta) 
                for key, meta in data.items()
            }
        except Exception as e:
            logger.warning(f"Failed to load metadata cache: {e}")
            return {}
    
    def _save_metadata_cache(self):
        """Save metadata cache to disk."""
        cache_file = self.cache_dir / "metadata.json"
        
        data = {
            key: meta.to_dict() 
            for key, meta in self.metadata_cache.items()
        }
        
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_available_engines(self, model_id: str) -> List[EngineMetadata]:
        """Get available engine variants for a model."""
        try:
            # List all files in the HF repository
            files = list_repo_files(
                repo_id=self.hf_repo,
                repo_type="dataset",  # Use dataset repo for large files
                token=self.hf_token
            )
            
            # Filter for this model
            model_files = [f for f in files if f.startswith(f"{model_id}-")]
            
            # Extract engine variants
            variants = []
            for file_path in model_files:
                if file_path.endswith("/metadata.json"):
                    # Download and parse metadata
                    metadata_path = hf_hub_download(
                        repo_id=self.hf_repo,
                        filename=file_path,
                        repo_type="dataset",
                        token=self.hf_token,
                        cache_dir=str(self.cache_dir)
                    )
                    
                    with open(metadata_path, 'r') as f:
                        metadata = EngineMetadata.from_dict(json.load(f))
                    
                    variants.append(metadata)
            
            return sorted(variants, key=lambda x: x.identifier)
            
        except Exception as e:
            logger.error(f"Failed to get available engines: {e}")
            return []
    
    def select_best_engine(self, 
                          model_id: str, 
                          preferred_quantization: str = "bf16") -> Optional[EngineMetadata]:
        """Select the best engine variant for local hardware."""
        available = self.get_available_engines(model_id)
        
        # Filter by architecture compatibility
        compatible = [e for e in available if e.architecture == self.local_architecture]
        
        if not compatible:
            logger.warning(f"No engines found for architecture: {self.local_architecture.value}")
            return None
        
        # Prefer requested quantization
        preferred = [e for e in compatible if e.quantization == preferred_quantization]
        if preferred:
            return preferred[0]
        
        # Fallback to first compatible
        return compatible[0]
    
    def download_engine(self, metadata: EngineMetadata) -> str:
        """Download engine files and return local path."""
        engine_dir = self.cache_dir / metadata.identifier
        engine_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already downloaded and valid
        if self._is_engine_cached(metadata):
            logger.info(f"Using cached engine: {metadata.identifier}")
            return str(engine_dir)
        
        logger.info(f"Downloading engine: {metadata.identifier}")
        
        # Download each engine file
        remote_base = f"{metadata.identifier}/"
        downloaded_files = []
        
        for stage, filename in metadata.engines.items():
            remote_path = remote_base + filename
            
            try:
                local_path = hf_hub_download(
                    repo_id=self.hf_repo,
                    filename=remote_path,
                    repo_type="dataset",
                    token=self.hf_token,
                    cache_dir=str(self.cache_dir)
                )
                
                # Move to our engine directory
                final_path = engine_dir / filename
                if Path(local_path) != final_path:
                    Path(local_path).rename(final_path)
                
                downloaded_files.append(final_path)
                logger.info(f"✓ Downloaded {stage}: {filename}")
                
            except Exception as e:
                logger.error(f"Failed to download {stage}: {e}")
                # Cleanup partial downloads
                for f in downloaded_files:
                    f.unlink(missing_ok=True)
                raise
        
        # Verify checksum
        if not self._verify_checksum(engine_dir, metadata.checksum):
            logger.error("Checksum verification failed")
            # Cleanup
            for f in engine_dir.glob("*"):
                f.unlink()
            raise RuntimeError("Engine download verification failed")
        
        # Cache metadata
        self.metadata_cache[metadata.identifier] = metadata
        self._save_metadata_cache()
        
        logger.info(f"✓ Engine downloaded and verified: {metadata.identifier}")
        return str(engine_dir)
    
    def _is_engine_cached(self, metadata: EngineMetadata) -> bool:
        """Check if engine is already cached and valid."""
        engine_dir = self.cache_dir / metadata.identifier
        
        if not engine_dir.exists():
            return False
        
        # Check all expected files exist
        for filename in metadata.engines.values():
            if not (engine_dir / filename).exists():
                return False
        
        # Verify checksum
        return self._verify_checksum(engine_dir, metadata.checksum)
    
    def _verify_checksum(self, engine_dir: Path, expected_checksum: str) -> bool:
        """Verify engine directory checksum."""
        try:
            hasher = hashlib.sha256()
            
            # Hash all engine files in sorted order
            for engine_file in sorted(engine_dir.glob("*.plan")):
                with open(engine_file, 'rb') as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        hasher.update(chunk)
            
            return hasher.hexdigest() == expected_checksum
            
        except Exception as e:
            logger.warning(f"Checksum verification failed: {e}")
            return False
    
    def get_engine_path(self, 
                       model_id: str, 
                       quantization: str = "bf16",
                       auto_download: bool = True) -> Optional[str]:
        """
        Get local path to engine, downloading if necessary.
        
        Args:
            model_id: Model identifier (e.g., "sdxl")
            quantization: Preferred quantization
            auto_download: Whether to download if not cached
        
        Returns:
            Local path to engine directory, or None if not available
        """
        # Check cache first
        identifier = f"{model_id}-{quantization}-{self.local_architecture.value}"
        if identifier in self.metadata_cache:
            engine_dir = self.cache_dir / identifier
            if engine_dir.exists():
                return str(engine_dir)
        
        if not auto_download:
            return None
        
        # Find and download best engine
        best_engine = self.select_best_engine(model_id, quantization)
        if not best_engine:
            logger.error(f"No compatible engine found for {model_id}")
            return None
        
        try:
            return self.download_engine(best_engine)
        except Exception as e:
            logger.error(f"Failed to download engine: {e}")
            return None
    
    def clear_cache(self, model_id: Optional[str] = None):
        """Clear engine cache (optionally for specific model)."""
        if model_id:
            # Clear specific model
            for identifier in list(self.metadata_cache.keys()):
                if identifier.startswith(f"{model_id}-"):
                    engine_dir = self.cache_dir / identifier
                    if engine_dir.exists():
                        for f in engine_dir.glob("*"):
                            f.unlink()
                        engine_dir.rmdir()
                    del self.metadata_cache[identifier]
        else:
            # Clear all
            for engine_dir in self.cache_dir.glob("*"):
                if engine_dir.is_dir():
                    for f in engine_dir.glob("*"):
                        f.unlink()
                    engine_dir.rmdir()
            self.metadata_cache.clear()
        
        self._save_metadata_cache()
        logger.info(f"Cleared engine cache for {model_id or 'all models'}")

# Global engine manager instance
_engine_manager: Optional[EngineManager] = None

def get_engine_manager(**kwargs) -> EngineManager:
    """Get the global engine manager instance."""
    global _engine_manager
    if _engine_manager is None:
        _engine_manager = EngineManager(**kwargs)
    return _engine_manager