#!/bin/bash
# Build script for ImageAIServer Docker images

set -e

echo "🐳 Building ImageAIServer Docker Images"
echo "======================================="

# Default values
BUILD_BASE=true
BUILD_TORCH=true
BUILD_GPU=false
TORCH_VERSION="cpu"
REGISTRY=""
TAG="latest"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --base-only)
      BUILD_BASE=true
      BUILD_TORCH=false
      BUILD_GPU=false
      shift
      ;;
    --torch-only)
      BUILD_BASE=false
      BUILD_TORCH=true
      BUILD_GPU=false
      shift
      ;;
    --gpu-only)
      BUILD_BASE=false
      BUILD_TORCH=false
      BUILD_GPU=true
      TORCH_VERSION="cu121"
      shift
      ;;
    --all)
      BUILD_BASE=true
      BUILD_TORCH=true
      BUILD_GPU=true
      shift
      ;;
    --torch-version)
      TORCH_VERSION="$2"
      shift 2
      ;;
    --registry)
      REGISTRY="$2/"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --base-only          Build only base (ONNX) image"
      echo "  --torch-only         Build only PyTorch image"
      echo "  --gpu-only           Build only GPU image"
      echo "  --all                Build all images (default)"
      echo "  --torch-version VER  PyTorch version (cpu|cu118|cu121, default: cpu)"
      echo "  --registry REG       Docker registry prefix"
      echo "  --tag TAG            Image tag (default: latest)"
      echo "  --help               Show this help"
      echo ""
      echo "Examples:"
      echo "  $0                                    # Build base + torch (CPU)"
      echo "  $0 --base-only                       # Build only base image"
      echo "  $0 --gpu-only --torch-version cu121  # Build GPU image with CUDA 12.1"
      echo "  $0 --registry myregistry.com --tag v1.0  # Build with custom registry/tag"
      exit 0
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

# Build base image (ONNX only)
if [ "$BUILD_BASE" = true ]; then
  echo ""
  echo "🏗️ Building Base Image (ONNX only, ~2GB)..."
  docker build -f Dockerfile.base -t "${REGISTRY}imageaiserver:base-${TAG}" .
  docker tag "${REGISTRY}imageaiserver:base-${TAG}" "${REGISTRY}imageaiserver:base"
  echo "✅ Built: ${REGISTRY}imageaiserver:base-${TAG}"
fi

# Build PyTorch image (full)
if [ "$BUILD_TORCH" = true ]; then
  echo ""
  echo "🔥 Building PyTorch Image (ONNX + PyTorch, ~8GB)..."
  echo "   PyTorch version: $TORCH_VERSION"
  docker build -f Dockerfile.torch --build-arg TORCH_VERSION="$TORCH_VERSION" -t "${REGISTRY}imageaiserver:torch-${TAG}" .
  docker tag "${REGISTRY}imageaiserver:torch-${TAG}" "${REGISTRY}imageaiserver:torch"
  echo "✅ Built: ${REGISTRY}imageaiserver:torch-${TAG}"
fi

# Build GPU image
if [ "$BUILD_GPU" = true ]; then
  echo ""
  echo "🚀 Building GPU Image (ONNX + PyTorch + CUDA, ~8GB+)..."
  echo "   CUDA version: ${TORCH_VERSION}"
  docker build -f Dockerfile.torch --build-arg TORCH_VERSION="$TORCH_VERSION" -t "${REGISTRY}imageaiserver:gpu-${TAG}" .
  docker tag "${REGISTRY}imageaiserver:gpu-${TAG}" "${REGISTRY}imageaiserver:gpu"
  echo "✅ Built: ${REGISTRY}imageaiserver:gpu-${TAG}"
fi

echo ""
echo "🎉 Build Complete!"
echo ""
echo "📋 Built Images:"
if [ "$BUILD_BASE" = true ]; then
  echo "  • ${REGISTRY}imageaiserver:base-${TAG}  (ONNX only, ~2GB)"
fi
if [ "$BUILD_TORCH" = true ]; then
  echo "  • ${REGISTRY}imageaiserver:torch-${TAG} (ONNX + PyTorch, ~8GB)"
fi
if [ "$BUILD_GPU" = true ]; then
  echo "  • ${REGISTRY}imageaiserver:gpu-${TAG}   (ONNX + PyTorch + CUDA, ~8GB+)"
fi

echo ""
echo "🚀 Quick Start:"
if [ "$BUILD_BASE" = true ]; then
  echo "  # Base (lightweight)"
  echo "  docker run -p 8000:8000 ${REGISTRY}imageaiserver:base"
fi
if [ "$BUILD_TORCH" = true ]; then
  echo "  # PyTorch (full features)"
  echo "  docker run -p 8000:8000 ${REGISTRY}imageaiserver:torch"
fi
if [ "$BUILD_GPU" = true ]; then
  echo "  # GPU (requires nvidia-docker)"
  echo "  docker run --gpus all -p 8000:8000 ${REGISTRY}imageaiserver:gpu"
fi

echo ""
echo "📊 Check backend status at: http://localhost:8000/v1/backends"
echo "📖 Full documentation: DOCKER.md"