"""
Phase 6: Serving Layer Main Module

Simple server to run the recommendation API without MLflow complexity.
"""

import uvicorn
import argparse
from .simple_api import create_simple_api

def main():
    """Run the recommendation API server."""
    parser = argparse.ArgumentParser(description="Phase 6: Recommendation API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print("🚀 Starting Recommndr API Server...")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Auto-reload: {args.reload}")
    print("   API will be available at:")
    print(f"   - Main: http://{args.host}:{args.port}")
    print(f"   - Docs: http://{args.host}:{args.port}/docs")
    print(f"   - Health: http://{args.host}:{args.port}/health")
    print()
    
    # Create and run the API
    app = create_simple_api()
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()
