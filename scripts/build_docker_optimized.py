#!/usr/bin/env python3
"""Optimized Docker build script for Recommndr."""

import subprocess
import sys
import time
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔨 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def get_image_size(image_name):
    """Get Docker image size."""
    try:
        result = subprocess.run(
            f"docker images {image_name} --format '{{{{.Size}}}}'",
            shell=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    except:
        return "Unknown"

def analyze_image_layers(image_name):
    """Analyze Docker image layers."""
    try:
        result = subprocess.run(
            f"docker history {image_name}",
            shell=True, capture_output=True, text=True
        )
        return result.stdout
    except:
        return "Could not analyze layers"

def build_optimized_images():
    """Build all optimized Docker images."""
    print("🐳 Building Optimized Docker Images for Recommndr")
    print("=" * 60)
    
    # Build base image first
    print("\n📦 Building base image...")
    base_result = run_command(
        "docker build --target base -t recommndr:base .",
        "Building base image"
    )
    
    if not base_result:
        print("❌ Base image build failed. Exiting.")
        return False
    
    # Build development image
    print("\n🔧 Building development image...")
    dev_result = run_command(
        "docker build --target development -t recommndr:dev .",
        "Building development image"
    )
    
    if not dev_result:
        print("❌ Development image build failed.")
        return False
    
    # Build testing image
    print("\n🧪 Building testing image...")
    test_result = run_command(
        "docker build --target testing -t recommndr:test .",
        "Building testing image"
    )
    
    if not test_result:
        print("❌ Testing image build failed.")
        return False
    
    # Build production image
    print("\n🚀 Building production image...")
    prod_result = run_command(
        "docker build --target production -t recommndr:prod .",
        "Building production image"
    )
    
    if not prod_result:
        print("❌ Production image build failed.")
        return False
    
    # Build minimal image for Azure
    print("\n☁️ Building minimal Azure image...")
    azure_result = run_command(
        "docker build --target minimal -t recommndr:azure .",
        "Building minimal Azure image"
    )
    
    if not azure_result:
        print("❌ Azure image build failed.")
        return False
    
    return True

def analyze_images():
    """Analyze built images."""
    print("\n📊 Image Analysis")
    print("=" * 40)
    
    images = ["recommndr:base", "recommndr:dev", "recommndr:test", "recommndr:prod", "recommndr:azure"]
    
    for image in images:
        size = get_image_size(image)
        print(f"\n🔍 {image}:")
        print(f"   Size: {size}")
        
        # Show layer analysis for production image
        if "prod" in image:
            print(f"   Layers:")
            layers = analyze_image_layers(image)
            for line in layers.split('\n')[:5]:  # Show first 5 layers
                if line.strip():
                    print(f"     {line.strip()}")

def cleanup_images():
    """Clean up intermediate images."""
    print("\n🧹 Cleaning up intermediate images...")
    
    # Remove base image (intermediate)
    run_command("docker rmi recommndr:base", "Removing base image")
    
    print("✅ Cleanup completed")

def main():
    """Main function."""
    start_time = time.time()
    
    print("🏗️ Recommndr Docker Optimization")
    print("=" * 50)
    
    # Check if running from project root
    if not (Path.cwd() / "docker" / "Dockerfile").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Check Docker
    if not run_command("docker --version", "Checking Docker"):
        print("❌ Docker not available")
        sys.exit(1)
    
    # Build images
    if not build_optimized_images():
        print("❌ Image build failed")
        sys.exit(1)
    
    # Analyze images
    analyze_images()
    
    # Cleanup
    cleanup_images()
    
    total_time = time.time() - start_time
    
    print(f"\n🎉 Docker optimization completed in {total_time:.2f} seconds!")
    print("\n📋 Available Images:")
    print("   recommndr:dev     - Development with all tools")
    print("   recommndr:test    - Testing with performance tools")
    print("   recommndr:prod    - Production optimized")
    print("   recommndr:azure   - Minimal for Azure deployment")
    
    print("\n🚀 Ready for Phase 2 testing!")

if __name__ == "__main__":
    main()
