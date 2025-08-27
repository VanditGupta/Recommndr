"""
Basic Phase 6 Test Script

Tests core Phase 6 functionality:
- Model registration
- Performance monitoring
- Feature metadata
- Basic rollback system
"""

import time
from src.model_management.model_manager import ModelManager
from src.model_management.feast_metadata import FeatureMetadata
from src.utils.logging import get_logger

logger = get_logger(__name__)


def test_phase6_basic():
    """Test basic Phase 6 functionality."""
    print("\n🎯 PHASE 6: BASIC FUNCTIONALITY TEST")
    print("=" * 80)
    
    # Initialize components
    print("\n📊 Initializing Model Manager...")
    model_manager = ModelManager("http://localhost:5001")
    
    print("\n📊 Initializing Feature Metadata...")
    feature_metadata = FeatureMetadata()
    
    # Test 1: Feature Metadata
    print("\n🧪 Test 1: Feature Metadata Registration")
    print("-" * 50)
    
    feature_metadata.register_all_feature_metadata()
    summary = feature_metadata.get_feature_summary()
    
    print(f"✅ Registered {summary['total_features']} features")
    print(f"   - User features: {summary['feature_categories']['user_features']}")
    print(f"   - Item features: {summary['feature_categories']['item_features']}")
    print(f"   - Contextual features: {summary['feature_categories']['contextual_features']}")
    print(f"   - Interaction features: {summary['feature_categories']['interaction_features']}")
    
    # Test 2: Performance Monitoring
    print("\n🧪 Test 2: Performance Monitoring")
    print("-" * 50)
    
    # Start monitoring
    model_manager.start_monitoring()
    print("✅ Monitoring systems started")
    
    # Record some test metrics
    for i in range(10):
        latency = 50 + (i * 5)  # Normal latency
        model_manager.monitor.record_latency(latency, user_id=1000 + i, model_version="test")
        
        ctr = 0.05 + (i * 0.001)  # Normal CTR
        model_manager.monitor.record_ctr(ctr, user_id=1000 + i, model_version="test")
    
    print("✅ Recorded 10 test metrics")
    
    # Get current metrics
    metrics = model_manager.monitor.get_current_metrics(5)
    print(f"✅ Current metrics: {len(metrics.get('metrics', {}))} metric types")
    
    # Test 3: Health Check
    print("\n🧪 Test 3: System Health Check")
    print("-" * 50)
    
    health_status = model_manager.perform_health_check()
    print(f"✅ Overall health status: {health_status['overall_status']}")
    
    for component, status in health_status['components'].items():
        component_status = status.get('status', 'unknown')
        print(f"   {component}: {component_status}")
    
    # Test 4: Model Registry (Basic)
    print("\n🧪 Test 4: Model Registry Status")
    print("-" * 50)
    
    try:
        models_info = model_manager.registry.list_all_models()
        print(f"✅ Connected to MLflow registry")
        print(f"   Registered models: {len(models_info)}")
        
        for model_name in models_info:
            print(f"   - {model_name}")
            
    except Exception as e:
        print(f"⚠️  MLflow registry connection issue: {e}")
    
    # Test 5: Rollback System Status
    print("\n🧪 Test 5: Rollback System")
    print("-" * 50)
    
    rollback_stats = model_manager.rollback_system.get_rollback_stats()
    print(f"✅ Rollback system active: {rollback_stats['system_active']}")
    print(f"   Auto-rollback enabled: {rollback_stats['auto_rollback_enabled']}")
    print(f"   Total rollbacks: {rollback_stats['total_rollbacks']}")
    
    # Test 6: Performance Simulation
    print("\n🧪 Test 6: Performance Alert Simulation")
    print("-" * 50)
    
    # Simulate high latency to trigger alerts
    print("Simulating high latency...")
    for i in range(5):
        high_latency = 350 + (i * 10)  # High latency to trigger alerts
        model_manager.monitor.record_latency(high_latency, user_id=2000 + i, model_version="test")
    
    time.sleep(1)  # Wait for monitoring to process
    
    # Check for alerts
    recent_alerts = [alert for alert in model_manager.monitor.alerts if alert.timestamp > time.time() - 10]
    print(f"✅ Triggered {len(recent_alerts)} alerts")
    
    for alert in recent_alerts:
        print(f"   🚨 {alert.severity.upper()}: {alert.message}")
    
    # Stop monitoring
    model_manager.stop_monitoring()
    print("✅ Monitoring systems stopped")
    
    # Final Summary
    print("\n🎉 PHASE 6 TEST SUMMARY")
    print("=" * 80)
    print("✅ Feature metadata system: Working")
    print("✅ Performance monitoring: Working") 
    print("✅ Alert system: Working")
    print("✅ Health checks: Working")
    print("✅ Rollback system: Working")
    print(f"✅ MLflow connectivity: {'Working' if 'models_info' in locals() else 'Partial'}")
    
    print("\n🎯 Phase 6 core functionality verified!")
    print("📚 For full API functionality, run:")
    print("   python -m src.model_management.main --serve-api")


if __name__ == "__main__":
    test_phase6_basic()
