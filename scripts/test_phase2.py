#!/usr/bin/env python3
"""Test script for Phase 2: Streaming & Feature Pipeline."""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.streaming.clickstream_simulator import ClickstreamSimulator
from src.features.feature_engineering import FeatureEngineeringPipeline
from src.utils.logging import get_logger

logger = get_logger(__name__)


def test_phase2_pipeline():
    """Test the complete Phase 2 pipeline."""
    print("🚀 Testing Phase 2: Streaming & Feature Pipeline")
    print("=" * 60)
    
    # Initialize components
    simulator = None
    pipeline = None
    
    try:
        print("\n📊 Step 1: Testing Clickstream Simulator...")
        simulator = ClickstreamSimulator()
        print("✅ Clickstream simulator initialized")
        
        print("\n🔧 Step 2: Testing Feature Engineering Pipeline...")
        pipeline = FeatureEngineeringPipeline()
        print("✅ Feature engineering pipeline initialized")
        
        print("\n🎯 Step 3: Running Small Simulation...")
        # Run a small simulation (5 users, 2 minutes each, 1 event per minute)
        simulator.run_simulation(
            num_users=5,
            session_duration_minutes=2,
            events_per_minute=1
        )
        print("✅ Clickstream simulation completed")
        
        print("\n📈 Step 4: Checking Feature Stats...")
        stats = pipeline.get_feature_stats()
        print(f"📊 Feature Statistics: {stats}")
        
        print("\n🔍 Step 5: Checking Individual Features...")
        # Check user features
        user_features = pipeline.get_user_features(1)
        if user_features:
            print(f"👤 User 1 features: {len(user_features.feature_vector)} features")
            print(f"   Last updated: {user_features.last_updated}")
            print(f"   Version: {user_features.feature_version}")
        
        # Check product features
        product_features = pipeline.get_product_features(1)
        if product_features:
            print(f"🛍️  Product 1 features: {len(product_features.feature_vector)} features")
            print(f"   Last updated: {product_features.last_updated}")
            print(f"   Version: {product_features.feature_version}")
        
        print("\n🎉 Phase 2 Pipeline Test Completed Successfully!")
        print("\n📋 What We've Built:")
        print("   ✅ Kafka streaming simulation")
        print("   ✅ Real-time feature engineering")
        print("   ✅ Redis-based feature store")
        print("   ✅ User and product feature vectors")
        print("   ✅ Realistic clickstream events")
        
        print("\n🚀 Ready for Phase 3: ML Model Training!")
        
    except Exception as e:
        print(f"❌ Error during Phase 2 test: {e}")
        logger.error(f"Phase 2 test failed: {e}")
        return False
    
    finally:
        # Clean up
        if simulator:
            simulator.close()
        if pipeline:
            pipeline.close()
    
    return True


def main():
    """Main function."""
    print("🏗️ Recommndr Phase 2 Test")
    print("=" * 50)
    
    # Check if running from project root
    if not (project_root / "src").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Run the test
    success = test_phase2_pipeline()
    
    if success:
        print("\n🎯 Next Steps:")
        print("   1. Start the full stack: docker-compose up -d")
        print("   2. Run larger simulations")
        print("   3. Move to Phase 3: ML Model Training")
        sys.exit(0)
    else:
        print("\n⚠️  Phase 2 test completed with issues")
        print("🔧 Please check the logs and try again")
        sys.exit(1)


if __name__ == "__main__":
    main()
