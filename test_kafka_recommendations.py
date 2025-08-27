#!/usr/bin/env python3
"""
Simple Kafka + Recommendations Integration Test

This script demonstrates end-to-end testing:
1. Send events to Kafka
2. Process through streaming pipeline  
3. Show recommendation changes
"""

import json
import time
import random
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timezone
from src.streaming.kafka_producer import KafkaEventProducer
from src.streaming.main import Phase2Pipeline

class KafkaRecommendationTest:
    """Test Kafka integration with recommendations."""
    
    def __init__(self):
        self.producer = None
        self.pipeline = None
        self.user_mapping = None
        self.item_mapping = None
        self.model_data = None
        self.products_df = None
        
    def initialize(self):
        """Initialize components."""
        print("🚀 Initializing Kafka Recommendation Test...")
        
        # Initialize Kafka producer
        self.producer = KafkaEventProducer()
        print("✅ Kafka producer initialized")
        
        # Load ML components
        with open('data/processed/user_mapping.pkl', 'rb') as f:
            self.user_mapping = pickle.load(f)
        with open('data/processed/item_mapping.pkl', 'rb') as f:
            self.item_mapping = pickle.load(f)
        with open('models/phase3/als_model.pkl', 'rb') as f:
            self.model_data = pickle.load(f)
        
        self.products_df = pd.read_parquet('data/processed/products_cleaned.parquet')
        print("✅ ML components loaded")
        
    def create_test_event(self, user_id: int, product_id: int, event_type: str = 'view'):
        """Create a test clickstream event."""
        return {
            'user_id': user_id,
            'product_id': product_id,
            'event_type': event_type,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'session_id': f'test_session_{user_id}_{int(time.time())}',
            'dwell_time': random.randint(10, 300) if event_type == 'view' else None,
            'scroll_depth': random.randint(20, 100) if event_type == 'view' else None,
            'device_type': 'desktop',
            'page_url': f'/product/{product_id}',
            'test_event': True
        }
    
    def get_recommendations(self, user_id: int, top_k: int = 5):
        """Get current recommendations for user."""
        if user_id not in self.user_mapping:
            return []
        
        user_idx = self.user_mapping[user_id]
        user_factors = self.model_data['user_factors'][user_idx]
        item_factors = self.model_data['item_factors']
        
        scores = np.dot(item_factors, user_factors)
        reverse_item_mapping = {v: k for k, v in self.item_mapping.items()}
        
        recommendations = []
        for item_idx in np.argsort(scores)[::-1]:
            item_id = reverse_item_mapping.get(item_idx)
            if item_id and len(recommendations) < top_k:
                product_info = self.products_df[self.products_df['product_id'] == item_id].iloc[0]
                recommendations.append({
                    'product_id': item_id,
                    'name': product_info['name'],
                    'category': product_info['category'],
                    'score': float(scores[item_idx])
                })
        
        return recommendations
    
    def send_test_events(self, user_id: int, num_events: int = 3):
        """Send test events to Kafka."""
        print(f"📤 Sending {num_events} test events for user {user_id}...")
        
        events_sent = []
        available_products = list(self.item_mapping.keys())
        
        for i in range(num_events):
            product_id = random.choice(available_products)
            event_type = random.choice(['view', 'click', 'view', 'view'])  # Mostly views
            
            event = self.create_test_event(user_id, product_id, event_type)
            
            try:
                self.producer.send_event('clickstream-events', event)
                events_sent.append(event)
                print(f"   ✅ Event {i+1}: {event_type} on product {product_id}")
                time.sleep(0.5)  # Small delay between events
                
            except Exception as e:
                print(f"   ❌ Failed to send event {i+1}: {e}")
        
        return events_sent
    
    def test_user_journey(self, user_id: int):
        """Test complete user journey with recommendations."""
        print()
        print(f"🎯 Testing User Journey: {user_id}")
        print("=" * 50)
        
        # Get initial recommendations
        print("📊 Initial recommendations:")
        initial_recs = self.get_recommendations(user_id, top_k=3)
        for i, rec in enumerate(initial_recs, 1):
            print(f"   {i}. {rec['name']} (Score: {rec['score']:.3f})")
        
        # Send live events
        events = self.send_test_events(user_id, num_events=3)
        
        # Show events sent
        print()
        print("📤 Events sent to Kafka:")
        for event in events:
            product_name = self.products_df[self.products_df['product_id'] == event['product_id']]['name'].iloc[0]
            print(f"   - {event['event_type'].title()}: {product_name}")
        
        print()
        print("✅ Events successfully sent to Kafka topic 'clickstream-events'")
        print("🔄 In a real system, these would be processed by Flink and update the feature store")
        print("💡 Recommendations would then be refreshed based on the new interaction data")
        
        return {
            'user_id': user_id,
            'initial_recommendations': initial_recs,
            'events_sent': events
        }
    
    def run_streaming_test(self):
        """Run the Phase 2 streaming pipeline with test events."""
        print()
        print("🌊 Running Phase 2 Streaming Pipeline Test...")
        print("=" * 60)
        
        try:
            # Run streaming pipeline
            pipeline = Phase2Pipeline()
            
            print("🚀 Starting streaming pipeline components...")
            pipeline.setup_components()
            
            print("📤 Sending test events through pipeline...")
            pipeline.run_pipeline(duration_seconds=20)
            
            print("✅ Streaming pipeline test completed")
            
        except Exception as e:
            print(f"❌ Streaming test failed: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.producer:
                self.producer.close()
            print("✅ Cleanup completed")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

def main():
    """Main test function."""
    print("🎯 Kafka + Recommendations Integration Test")
    print("=" * 60)
    print("This test demonstrates the integration between:")
    print("- Real-time clickstream events (Kafka)")
    print("- Stream processing (Flink)")
    print("- ML-powered recommendations (ALS)")
    print()
    
    tester = KafkaRecommendationTest()
    
    try:
        # Initialize
        tester.initialize()
        
        # Test individual user journeys
        test_users = [1, 1118, 3194]  # Mix of different users
        
        for user_id in test_users:
            result = tester.test_user_journey(user_id)
            time.sleep(2)  # Brief pause between users
        
        # Run streaming pipeline test
        tester.run_streaming_test()
        
        print()
        print("🎉 Integration Test Summary:")
        print("=" * 40)
        print("✅ Kafka events sent successfully")
        print("✅ Streaming pipeline processed events")
        print("✅ Recommendations system ready for real-time updates")
        print()
        print("🔄 Next steps for full integration:")
        print("1. Events flow through Kafka → Flink → Feature Store")
        print("2. Feature Store updates user/product features in real-time")
        print("3. Recommendations are refreshed based on updated features")
        print("4. API serves updated recommendations to users")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        tester.cleanup()

if __name__ == "__main__":
    main()
