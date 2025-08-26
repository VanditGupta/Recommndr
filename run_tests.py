#!/usr/bin/env python3
"""Test runner for Recommndr project."""

import sys
import subprocess
from pathlib import Path

def run_tests():
    """Run the test suite."""
    print("🧪 Running Recommndr Test Suite")
    print("=" * 50)
    
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("❌ pytest not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest", "pytest-cov"])
    
    # Run tests with coverage
    test_args = [
        "pytest",
        "tests/",
        "-v",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--cov-report=xml",
        "--tb=short"
    ]
    
    print(f"Running: {' '.join(test_args)}")
    print()
    
    result = subprocess.run(test_args)
    
    if result.returncode == 0:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
