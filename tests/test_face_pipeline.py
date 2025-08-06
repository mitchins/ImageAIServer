#!/usr/bin/env python3
"""
Face Detection Pipeline Test Runner

This script runs the comprehensive face detection pipeline tests,
including both unit tests (ONNX models) and integration tests (full pipeline).

Usage:
    python tests/test_face_pipeline.py [--unit-only] [--integration-only] [--verbose]
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_tests(test_type="all", verbose=False):
    """Run face detection pipeline tests."""
    base_dir = Path(__file__).parent.parent
    
    # Test commands
    unit_test_cmd = [
        sys.executable, "-m", "pytest", 
        "tests/unit/test_face_onnx_models.py",
        "-m", "unit"
    ]
    
    integration_test_cmd = [
        sys.executable, "-m", "pytest",
        "tests/integration/test_face_detection_pipeline.py",
        "-m", "integration"
    ]
    
    if verbose:
        unit_test_cmd.extend(["-v", "-s"])
        integration_test_cmd.extend(["-v", "-s"])
    
    # Change to project directory
    original_cwd = Path.cwd()
    try:
        import os
        os.chdir(base_dir)
        
        print("🧪 Face Detection Pipeline Tests")
        print("=" * 50)
        
        if test_type in ["all", "unit"]:
            print("\n📦 Running Unit Tests (ONNX Models)")
            print("-" * 30)
            result = subprocess.run(unit_test_cmd, capture_output=False)
            if result.returncode != 0:
                print("❌ Unit tests failed!")
                return False
            print("✅ Unit tests passed!")
        
        if test_type in ["all", "integration"]:
            print("\n🔗 Running Integration Tests (Full Pipeline)")
            print("-" * 40)
            result = subprocess.run(integration_test_cmd, capture_output=False)
            if result.returncode != 0:
                print("❌ Integration tests failed!")
                return False
            print("✅ Integration tests passed!")
        
        print("\n🎉 All face detection pipeline tests passed!")
        return True
        
    finally:
        os.chdir(original_cwd)

def main():
    parser = argparse.ArgumentParser(description="Run face detection pipeline tests")
    parser.add_argument("--unit-only", action="store_true", 
                       help="Run only unit tests")
    parser.add_argument("--integration-only", action="store_true",
                       help="Run only integration tests")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    if args.unit_only:
        test_type = "unit"
    elif args.integration_only:
        test_type = "integration"
    else:
        test_type = "all"
    
    success = run_tests(test_type, args.verbose)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()