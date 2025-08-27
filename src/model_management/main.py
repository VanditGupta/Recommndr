"""
Phase 6: Model Management Main Pipeline

Main entry point for model management operations:
1. Register models in MLflow registry
2. Setup performance monitoring
3. Configure automated rollback system
4. Start model management API
5. Initialize feature metadata

Usage:
    python -m src.model_management.main --register-models
    python -m src.model_management.main --start-monitoring
    python -m src.model_management.main --serve-api
    python -m src.model_management.main --full-setup
"""

import argparse
import time
import uvicorn
from pathlib import Path

from .model_manager import ModelManager
from .feast_metadata import FeatureMetadata
from .management_api import create_model_management_api
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Phase6Pipeline:
    """Main pipeline for Phase 6 model management setup."""
    
    def __init__(self, mlflow_tracking_uri: str = "http://localhost:5001"):
        """Initialize Phase 6 pipeline."""
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.model_manager = ModelManager(mlflow_tracking_uri)
        self.feature_metadata = FeatureMetadata()
        
        logger.info("🎯 Phase 6 Model Management Pipeline initialized")
    
    def register_all_models(self) -> dict:
        """Register all current models in MLflow registry."""
        logger.info("📝 Registering all models in MLflow registry...")
        
        start_time = time.time()
        
        # Register models
        results = self.model_manager.register_current_models()
        
        registration_time = time.time() - start_time
        
        logger.info(f"✅ Model registration completed in {registration_time:.2f}s")
        
        # Print summary
        print("\n📊 MODEL REGISTRATION SUMMARY")
        print("=" * 60)
        
        for model_name, result in results['results'].items():
            status = result.get('status', 'unknown')
            if status == 'success':
                version = result.get('version', 'unknown')
                print(f"✅ {model_name}: v{version}")
            else:
                message = result.get('message', 'unknown error')
                print(f"❌ {model_name}: {message}")
        
        return results
    
    def promote_models_to_production(self) -> dict:
        """Promote all staging models to production."""
        logger.info("🚀 Promoting models to production...")
        
        results = self.model_manager.promote_models_to_production()
        
        # Print summary
        print("\n🚀 MODEL PROMOTION SUMMARY")
        print("=" * 60)
        
        for model_name, result in results['results'].items():
            status = result.get('status', 'unknown')
            if status == 'success':
                version = result.get('version', 'unknown')
                print(f"✅ {model_name}: v{version} → Production")
            else:
                message = result.get('message', 'unknown error')
                print(f"❌ {model_name}: {message}")
        
        return results
    
    def setup_feature_metadata(self):
        """Setup comprehensive feature metadata."""
        logger.info("📊 Setting up feature metadata...")
        
        # Register all feature metadata
        self.feature_metadata.register_all_feature_metadata()
        
        # Generate documentation
        documentation = self.feature_metadata.generate_feature_documentation()
        
        # Print summary
        summary = self.feature_metadata.get_feature_summary()
        
        print("\n📊 FEATURE METADATA SUMMARY")
        print("=" * 60)
        print(f"✅ Total features: {summary['total_features']}")
        print(f"✅ User features: {summary['feature_categories']['user_features']}")
        print(f"✅ Item features: {summary['feature_categories']['item_features']}")
        print(f"✅ Contextual features: {summary['feature_categories']['contextual_features']}")
        print(f"✅ Interaction features: {summary['feature_categories']['interaction_features']}")
        
        logger.info("✅ Feature metadata setup completed")
    
    def start_monitoring_systems(self):
        """Start all monitoring systems."""
        logger.info("🔄 Starting monitoring systems...")
        
        # Start monitoring
        self.model_manager.start_monitoring()
        
        print("\n🔄 MONITORING SYSTEMS STARTED")
        print("=" * 60)
        print("✅ Performance monitoring active")
        print("✅ Automated rollback system active")
        print("✅ MLflow registry connected")
        
        logger.info("✅ All monitoring systems started")
    
    def perform_system_health_check(self):
        """Perform comprehensive system health check."""
        logger.info("🏥 Performing system health check...")
        
        health_status = self.model_manager.perform_health_check()
        
        print("\n🏥 SYSTEM HEALTH CHECK")
        print("=" * 60)
        print(f"Overall Status: {health_status['overall_status'].upper()}")
        print()
        
        for component, status in health_status['components'].items():
            component_status = status.get('status', 'unknown')
            if component_status == 'healthy':
                print(f"✅ {component}: Healthy")
            else:
                print(f"❌ {component}: {component_status}")
                if 'error' in status:
                    print(f"   Error: {status['error']}")
        
        return health_status
    
    def run_rollback_test(self):
        """Test the automated rollback system."""
        logger.info("🧪 Testing automated rollback system...")
        
        print("\n🧪 ROLLBACK SYSTEM TEST")
        print("=" * 60)
        
        # Test latency degradation
        print("Testing latency degradation simulation...")
        results = self.model_manager.simulate_performance_degradation("latency")
        
        print(f"✅ Simulation completed: {results['alerts_triggered']} alerts triggered")
        
        for alert in results.get('alert_details', []):
            print(f"   🚨 {alert['severity'].upper()}: {alert['message']}")
        
        # Check rollback stats
        rollback_stats = self.model_manager.rollback_system.get_rollback_stats()
        print(f"✅ Rollback system status: {rollback_stats['system_active']}")
        print(f"✅ Total rollbacks: {rollback_stats['total_rollbacks']}")
        
        return results
    
    def serve_management_api(self, host: str = "0.0.0.0", port: int = 8002, reload: bool = False):
        """Serve the model management API."""
        logger.info(f"🌐 Starting Model Management API on {host}:{port}")
        
        # Create FastAPI app
        app = create_model_management_api(self.mlflow_tracking_uri)
        
        print("\n🌐 MODEL MANAGEMENT API")
        print("=" * 60)
        print(f"🚀 Starting server on http://{host}:{port}")
        print("📚 API Documentation: http://localhost:8002/docs")
        print("🏥 Health Check: http://localhost:8002/health")
        print("📊 System Status: http://localhost:8002/status")
        
        # Run server
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    
    def run_full_setup(self):
        """Run complete Phase 6 setup."""
        logger.info("🚀 Running complete Phase 6 setup...")
        
        print("\n🎯 PHASE 6: MODEL MANAGEMENT SETUP")
        print("=" * 80)
        
        # Step 1: Register models
        print("\n📝 Step 1: Registering models...")
        self.register_all_models()
        
        # Step 2: Setup feature metadata
        print("\n📊 Step 2: Setting up feature metadata...")
        self.setup_feature_metadata()
        
        # Step 3: Promote models to production
        print("\n🚀 Step 3: Promoting models to production...")
        self.promote_models_to_production()
        
        # Step 4: Start monitoring
        print("\n🔄 Step 4: Starting monitoring systems...")
        self.start_monitoring_systems()
        
        # Step 5: Health check
        print("\n🏥 Step 5: Performing health check...")
        health_status = self.perform_system_health_check()
        
        # Step 6: Test rollback system
        print("\n🧪 Step 6: Testing rollback system...")
        self.run_rollback_test()
        
        print("\n🎉 PHASE 6 SETUP COMPLETE!")
        print("=" * 80)
        
        if health_status['overall_status'] == 'healthy':
            print("✅ All systems operational and ready for production")
        else:
            print("⚠️  Some components need attention - check health status")
        
        print("\n🌐 To start the management API:")
        print("   python -m src.model_management.main --serve-api")
        
        return {
            'setup_completed': True,
            'overall_health': health_status['overall_status'],
            'timestamp': time.time()
        }
    
    def run_interactive_demo(self):
        """Run interactive demo of Phase 6 capabilities."""
        print("\n🎭 PHASE 6 INTERACTIVE DEMO")
        print("=" * 80)
        print("This demo showcases model management capabilities:")
        print("  📝 Model registration and versioning")
        print("  🔄 Performance monitoring")
        print("  🚨 Automated rollback triggers")
        print("  📊 Feature metadata management")
        
        while True:
            try:
                print("\n📋 Available commands:")
                print("  1. health - Perform health check")
                print("  2. status - Show system status")
                print("  3. models - List registered models")
                print("  4. metrics - Show performance metrics")
                print("  5. simulate - Simulate performance degradation")
                print("  6. rollback - Show rollback history")
                print("  7. features - Show feature metadata")
                print("  8. quit - Exit demo")
                
                command = input("\n> ").strip().lower()
                
                if command == 'quit':
                    break
                elif command == 'health':
                    self.perform_system_health_check()
                elif command == 'status':
                    status = self.model_manager.get_system_status()
                    print(f"\n📊 System Status: {status.get('timestamp', time.time())}")
                elif command == 'models':
                    models = self.model_manager.registry.list_all_models()
                    print(f"\n📝 Registered Models: {len(models)}")
                    for name, info in models.items():
                        print(f"   {name}: {len(info.get('versions', []))} versions")
                elif command == 'metrics':
                    metrics = self.model_manager.monitor.get_current_metrics(60)
                    print(f"\n📈 Performance Metrics (last 60 min):")
                    for metric, data in metrics.get('metrics', {}).items():
                        if isinstance(data, dict) and 'count' in data:
                            print(f"   {metric}: {data['count']} events")
                elif command == 'simulate':
                    print("Simulating latency degradation...")
                    results = self.run_rollback_test()
                elif command == 'rollback':
                    history = self.model_manager.rollback_system.get_rollback_history(24)
                    print(f"\n🔄 Rollback History (24h): {len(history)} events")
                    for event in history[-5:]:  # Last 5
                        print(f"   {event.model_name}: {event.trigger.value}")
                elif command == 'features':
                    summary = self.feature_metadata.get_feature_summary()
                    print(f"\n📊 Feature Summary: {summary['total_features']} features")
                    for category, count in summary['feature_categories'].items():
                        print(f"   {category}: {count}")
                else:
                    print("❌ Unknown command")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n👋 Demo ended!")


def main():
    """Main entry point for Phase 6 pipeline."""
    parser = argparse.ArgumentParser(description="Phase 6: Model Management Pipeline")
    
    # Main actions
    parser.add_argument('--register-models', action='store_true', help='Register models in MLflow registry')
    parser.add_argument('--promote-models', action='store_true', help='Promote staging models to production')
    parser.add_argument('--setup-metadata', action='store_true', help='Setup feature metadata')
    parser.add_argument('--start-monitoring', action='store_true', help='Start monitoring systems')
    parser.add_argument('--health-check', action='store_true', help='Perform system health check')
    parser.add_argument('--test-rollback', action='store_true', help='Test rollback system')
    parser.add_argument('--serve-api', action='store_true', help='Serve model management API')
    parser.add_argument('--full-setup', action='store_true', help='Run complete Phase 6 setup')
    parser.add_argument('--demo', action='store_true', help='Run interactive demo')
    
    # Configuration options
    parser.add_argument('--mlflow-uri', type=str, default="http://localhost:5001", help='MLflow tracking URI')
    parser.add_argument('--api-port', type=int, default=8002, help='API server port')
    parser.add_argument('--api-host', type=str, default="0.0.0.0", help='API server host')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = Phase6Pipeline(args.mlflow_uri)
    
    if args.register_models:
        results = pipeline.register_all_models()
        print(f"\n✅ Model registration completed!")
    
    if args.promote_models:
        results = pipeline.promote_models_to_production()
        print(f"\n✅ Model promotion completed!")
    
    if args.setup_metadata:
        pipeline.setup_feature_metadata()
        print(f"\n✅ Feature metadata setup completed!")
    
    if args.start_monitoring:
        pipeline.start_monitoring_systems()
        print(f"\n✅ Monitoring systems started!")
        print("Press Ctrl+C to stop monitoring...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pipeline.model_manager.stop_monitoring()
            print("\n⏹️ Monitoring stopped")
    
    if args.health_check:
        health_status = pipeline.perform_system_health_check()
        print(f"\n✅ Health check completed!")
    
    if args.test_rollback:
        results = pipeline.run_rollback_test()
        print(f"\n✅ Rollback test completed!")
    
    if args.serve_api:
        pipeline.serve_management_api(host=args.api_host, port=args.api_port)
    
    if args.full_setup:
        results = pipeline.run_full_setup()
        print(f"\n🎉 Phase 6 setup completed!")
    
    if args.demo:
        pipeline.run_interactive_demo()
    
    # Default action if no specific command given
    if not any([args.register_models, args.promote_models, args.setup_metadata,
                args.start_monitoring, args.health_check, args.test_rollback,
                args.serve_api, args.full_setup, args.demo]):
        logger.info("🎯 No specific action provided. Running full setup...")
        results = pipeline.run_full_setup()
        print(f"\n🎉 Phase 6 completed! Model management system is ready.")


if __name__ == "__main__":
    main()
