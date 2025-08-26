#!/usr/bin/env python3
"""Advanced Docker build script with BuildKit optimizations for Recommndr."""

import subprocess
import sys
import time
import os
from pathlib import Path

def run_command(command, description, env=None):
    """Run a command and handle errors."""
    print(f"🔨 {description}...")
    try:
        cmd_env = os.environ.copy()
        if env:
            cmd_env.update(env)
        
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True,
            env=cmd_env
        )
        print(f"✅ {description} completed")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def enable_buildkit():
    """Enable Docker BuildKit for advanced optimizations."""
    print("🚀 Enabling Docker BuildKit...")
    
    # Set BuildKit environment variables
    os.environ['DOCKER_BUILDKIT'] = '1'
    os.environ['COMPOSE_DOCKER_CLI_BUILD'] = '1'
    
    # Check if BuildKit is available
    try:
        result = subprocess.run(
            "docker buildx version",
            shell=True, capture_output=True, text=True
        )
        if result.returncode == 0:
            print("✅ BuildKit is available")
            return True
        else:
            print("⚠️ BuildKit not available, using standard build")
            return False
    except:
        print("⚠️ BuildKit not available, using standard build")
        return False

def create_buildx_builder():
    """Create a multi-platform builder."""
    print("🏗️ Setting up multi-platform builder...")
    
    # Check existing builders
    result = run_command("docker buildx ls", "Listing existing builders")
    if not result:
        return False
    
    # Create new builder if needed
    if "recommndr-builder" not in result:
        run_command(
            "docker buildx create --name recommndr-builder --use",
            "Creating multi-platform builder"
        )
    else:
        run_command(
            "docker buildx use recommndr-builder",
            "Using existing builder"
        )
    
    return True

def build_optimized_images():
    """Build all optimized Docker images with advanced features."""
    print("🐳 Building Advanced Optimized Docker Images for Recommndr")
    print("=" * 70)
    
    # Build base image first with BuildKit optimizations
    print("\n📦 Building base image with BuildKit...")
    base_result = run_command(
        "docker build --target base -t recommndr:base --progress=plain .",
        "Building base image with BuildKit"
    )
    
    if not base_result:
        print("❌ Base image build failed. Exiting.")
        return False
    
    # Build development image
    print("\n🔧 Building development image...")
    dev_result = run_command(
        "docker build --target development -t recommndr:dev --progress=plain .",
        "Building development image"
    )
    
    if not dev_result:
        print("❌ Development image build failed.")
        return False
    
    # Build testing image
    print("\n🧪 Building testing image...")
    test_result = run_command(
        "docker build --target testing -t recommndr:test --progress=plain .",
        "Building testing image"
    )
    
    if not test_result:
        print("❌ Testing image build failed.")
        return False
    
    # Build production image
    print("\n🚀 Building production image...")
    prod_result = run_command(
        "docker build --target production -t recommndr:prod --progress=plain .",
        "Building production image"
    )
    
    if not prod_result:
        print("❌ Production image build failed.")
        return False
    
    # Build minimal Azure image
    print("\n☁️ Building minimal Azure image...")
    azure_result = run_command(
        "docker build --target minimal -t recommndr:azure --progress=plain .",
        "Building minimal Azure image"
    )
    
    if not azure_result:
        print("❌ Azure image build failed.")
        return False
    
    # Build secure production image
    print("\n🔒 Building secure production image...")
    secure_result = run_command(
        "docker build --target secure -t recommndr:secure --progress=plain .",
        "Building secure production image"
    )
    
    if not secure_result:
        print("❌ Secure image build failed.")
        return False
    
    return True

def build_multi_platform():
    """Build multi-platform images for production deployment."""
    print("\n🌍 Building multi-platform images...")
    
    platforms = ["linux/amd64", "linux/arm64"]
    
    for platform in platforms:
        print(f"\n🔨 Building for {platform}...")
        
        # Build minimal Azure image for platform
        result = run_command(
            f"docker buildx build --platform {platform} --target minimal -t recommndr:azure-{platform.replace('/', '-')} --load .",
            f"Building {platform} image"
        )
        
        if not result:
            print(f"⚠️ {platform} build failed, continuing...")
    
    return True

def analyze_images():
    """Analyze built images with detailed metrics."""
    print("\n📊 Advanced Image Analysis")
    print("=" * 50)
    
    images = [
        "recommndr:base", 
        "recommndr:dev", 
        "recommndr:test", 
        "recommndr:prod", 
        "recommndr:azure", 
        "recommndr:secure"
    ]
    
    for image in images:
        size = get_image_size(image)
        layers = count_image_layers(image)
        print(f"\n🔍 {image}:")
        print(f"   Size: {size}")
        print(f"   Layers: {layers}")
        
        # Show security info for secure image
        if "secure" in image:
            security_info = analyze_security(image)
            print(f"   Security: {security_info}")

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

def count_image_layers(image_name):
    """Count Docker image layers."""
    try:
        result = subprocess.run(
            f"docker history {image_name} | wc -l",
            shell=True, capture_output=True, text=True
        )
        return int(result.stdout.strip()) - 1  # Subtract header
    except:
        return "Unknown"

def analyze_security(image_name):
    """Analyze image security."""
    try:
        # Check if image runs as non-root
        result = subprocess.run(
            f"docker run --rm {image_name} id",
            shell=True, capture_output=True, text=True
        )
        if "uid=1000" in result.stdout:
            return "Non-root user ✅"
        else:
            return "Root user ⚠️"
    except:
        return "Could not analyze"

def cleanup_images():
    """Clean up intermediate images and optimize storage."""
    print("\n🧹 Advanced cleanup and optimization...")
    
    # Remove base image (intermediate)
    run_command("docker rmi recommndr:base", "Removing base image")
    
    # Prune unused images
    run_command("docker image prune -f", "Pruning unused images")
    
    # Prune unused containers
    run_command("docker container prune -f", "Pruning unused containers")
    
    # Prune unused networks
    run_command("docker network prune -f", "Pruning unused networks")
    
    # Prune unused volumes
    run_command("docker volume prune -f", "Pruning unused volumes")
    
    # Build cache pruning
    run_command("docker builder prune -f", "Pruning build cache")
    
    print("✅ Advanced cleanup completed")

def main():
    """Main function."""
    start_time = time.time()
    
    print("🏗️ Recommndr Advanced Docker Optimization")
    print("=" * 60)
    
    # Check if running from project root
    if not (Path.cwd() / "docker" / "Dockerfile").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Check Docker
    if not run_command("docker --version", "Checking Docker"):
        print("❌ Docker not available")
        sys.exit(1)
    
    # Enable BuildKit
    buildkit_available = enable_buildkit()
    
    # Setup multi-platform builder
    if buildkit_available:
        create_buildx_builder()
    
    # Build images
    if not build_optimized_images():
        print("❌ Image build failed")
        sys.exit(1)
    
    # Build multi-platform if BuildKit available
    if buildkit_available:
        build_multi_platform()
    
    # Analyze images
    analyze_images()
    
    # Cleanup
    cleanup_images()
    
    total_time = time.time() - start_time
    
    print(f"\n🎉 Advanced Docker optimization completed in {total_time:.2f} seconds!")
    print("\n📋 Available Images:")
    print("   recommndr:dev     - Development with all tools")
    print("   recommndr:test    - Testing with performance tools")
    print("   recommndr:prod    - Production optimized")
    print("   recommndr:azure   - Minimal for Azure deployment")
    print("   recommndr:secure  - Security-hardened production")
    
    if buildkit_available:
        print("   recommndr:azure-linux-amd64 - Multi-platform AMD64")
        print("   recommndr:azure-linux-arm64 - Multi-platform ARM64")
    
    print("\n🚀 Ready for Phase 2 testing with optimized images!")

if __name__ == "__main__":
    main()
