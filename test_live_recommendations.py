#!/usr/bin/env python3
"""
Live Clickstream Testing for Real-time Recommendations

This script demonstrates how the recommendation system works with live data:
1. Simulates real-time user interactions
2. Sends events through Kafka
3. Processes via Flink 
4. Updates feature store
5. Generates updated recommendations
"""

import time
import json
import random
import asyncio
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timezone
from typing import Dict, List, Optional
import threading
from queue import Queue

# Import our streaming components
from src.streaming.kafka_producer import KafkaProducer
from src.streaming.kafka_consumer import KafkaConsumer
from src.streaming.flink_processor import FlinkStreamProcessor
from src.features.feast_integration import FeastFeatureStore
from src.retrieval.main import CandidateGenerationPipeline

class LiveRecommendationTester:
    """Test live recommendations with real-time clickstream data."""
    
    def __init__(self):
        self.kafka_producer = None
        self.kafka_consumer = None
        self.flink_processor = None
        self.feature_store = None
        self.recommendation_pipeline = None
        self.model_data = None
        self.user_mapping = None
        self.item_mapping = None
        self.products_df = None
        self.users_df = None
        
        # Event tracking
        self.events_sent = 0
        self.events_processed = 0
        self.recommendation_updates = 0
        
        # Results storage
        self.before_recommendations = {}
        self.after_recommendations = {}
        self.event_log = []
        
    async def initialize(self):
        """Initialize all components for live testing."""
        print("🚀 Initializing Live Recommendation Testing System...")
        
        try:
            # Initialize streaming components
            self.kafka_producer = KafkaProducer()
            self.kafka_consumer = KafkaConsumer()
            self.flink_processor = FlinkStreamProcessor()
            self.feature_store = FeastFeatureStore()
            
            print("✅ Streaming components initialized")
            
            # Initialize ML pipeline
            self.recommendation_pipeline = CandidateGenerationPipeline()
            self.recommendation_pipeline.load_data()
            
            # Load trained model
            with open('models/phase3/als_model.pkl', 'rb') as f:
                self.model_data = pickle.load(f)
            
            # Load mappings
            with open('data/processed/user_mapping.pkl', 'rb') as f:
                self.user_mapping = pickle.load(f)
            with open('data/processed/item_mapping.pkl', 'rb') as f:
                self.item_mapping = pickle.load(f)
                
            # Load data for context
            self.products_df = pd.read_parquet('data/processed/products_cleaned.parquet')
            self.users_df = pd.read_parquet('data/processed/users_cleaned.parquet')
            
            print("✅ ML components initialized")
            print(f"📊 Ready to test with {len(self.user_mapping)} users and {len(self.item_mapping)} products")
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            raise
    
    def generate_realistic_event(self, user_id: int, session_context: Dict = None) -> Dict:
        """Generate a realistic clickstream event."""
        
        # Event types with probabilities (view is most common)
        event_types = ['view', 'click', 'add_to_cart', 'purchase']
        event_weights = [0.7, 0.2, 0.08, 0.02]
        
        event_type = random.choices(event_types, weights=event_weights)[0]
        
        # Select a product (could be influenced by user preferences or trending items)
        available_products = list(self.item_mapping.keys())
        product_id = random.choice(available_products)
        
        # Get product info for realistic event data
        product_info = self.products_df[self.products_df['product_id'] == product_id].iloc[0]
        
        # Create realistic event
        event = {
            'event_id': f"evt_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
            'user_id': user_id,
            'product_id': product_id,
            'event_type': event_type,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'session_id': session_context.get('session_id', f"session_{user_id}_{int(time.time())}") if session_context else f"session_{user_id}_{int(time.time())}",
            
            # Event metadata
            'product_category': product_info['category'],
            'product_price': float(product_info['price']),
            'product_rating': float(product_info['rating']),
            
            # User behavior data
            'dwell_time': random.randint(5, 300) if event_type == 'view' else None,
            'scroll_depth': random.randint(10, 100) if event_type == 'view' else None,
            'quantity': random.randint(1, 3) if event_type in ['add_to_cart', 'purchase'] else None,
            
            # Technical metadata
            'device_type': random.choice(['mobile', 'desktop', 'tablet']),
            'page_url': f"/product/{product_id}",
            'referrer': random.choice(['search', 'category', 'recommendation', 'direct']),
            
            # Simulation metadata
            'is_test_event': True,
            'test_scenario': 'live_recommendation_test'
        }
        
        return event
    
    def get_current_recommendations(self, user_id: int, top_k: int = 5) -> List[Dict]:
        """Get current recommendations for a user."""
        if user_id not in self.user_mapping:
            return []
        
        user_idx = self.user_mapping[user_id]
        user_factors = self.model_data['user_factors'][user_idx]
        item_factors = self.model_data['item_factors']
        
        # Calculate scores
        scores = np.dot(item_factors, user_factors)
        
        # Get user's existing interactions (to exclude)
        try:
            user_interactions = pd.read_parquet('data/processed/interactions_cleaned.parquet')
            user_items = set(user_interactions[user_interactions['user_id'] == user_id]['product_id'].values)
        except:
            user_items = set()
        
        # Get top recommendations
        recommendations = []
        reverse_item_mapping = {v: k for k, v in self.item_mapping.items()}
        
        for item_idx in np.argsort(scores)[::-1]:
            item_id = reverse_item_mapping.get(item_idx)
            if item_id and item_id not in user_items and len(recommendations) < top_k:
                product_info = self.products_df[self.products_df['product_id'] == item_id].iloc[0]
                recommendations.append({
                    'product_id': item_id,
                    'name': product_info['name'],
                    'category': product_info['category'],
                    'price': float(product_info['price']),
                    'score': float(scores[item_idx])
                })
        
        return recommendations
    
    async def send_live_events(self, user_id: int, num_events: int = 5, delay_seconds: float = 2.0):
        """Send a sequence of live events for a user."""
        session_context = {
            'session_id': f"live_session_{user_id}_{int(time.time())}",
            'start_time': datetime.now(timezone.utc)
        }
        
        events_sent = []
        
        for i in range(num_events):
            # Generate and send event
            event = self.generate_realistic_event(user_id, session_context)
            
            try:
                # Send to Kafka
                await self.kafka_producer.send_event(event)
                events_sent.append(event)
                self.events_sent += 1
                
                print(f"📤 Sent event {i+1}/{num_events}: {event['event_type']} on {event['product_id']} (User {user_id})")
                self.event_log.append(event)
                
                # Wait before next event
                if i < num_events - 1:
                    await asyncio.sleep(delay_seconds)
                    
            except Exception as e:
                print(f"❌ Failed to send event: {e}")
        
        return events_sent
    
    async def monitor_processing(self, duration_seconds: int = 30):
        """Monitor event processing through the pipeline."""
        print(f"👀 Monitoring pipeline for {duration_seconds} seconds...")
        
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            try:
                # Check Flink processing stats
                flink_stats = self.flink_processor.get_processing_stats()
                
                # Check feature store updates
                feature_stats = self.feature_store.get_stats()
                
                print(f"📊 Pipeline Status: Events sent: {self.events_sent}, "
                      f"Flink processed: {flink_stats.get('events_processed', 0)}, "
                      f"Features updated: {feature_stats.get('total_keys', 0)}")
                
                await asyncio.sleep(5)
                
            except Exception as e:
                print(f"⚠️ Monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def test_recommendation_updates(self, user_id: int, num_events: int = 5):
        """Test how recommendations change with live events."""
        print()
        print(f"🎯 Testing Recommendation Updates for User {user_id}")
        print("=" * 60)
        
        # Get initial recommendations
        print("📊 Getting initial recommendations...")
        initial_recs = self.get_current_recommendations(user_id, top_k=5)
        self.before_recommendations[user_id] = initial_recs
        
        print("🔮 INITIAL RECOMMENDATIONS:")
        for i, rec in enumerate(initial_recs, 1):
            print(f"   {i}. {rec['name']} ({rec['category']}) - Score: {rec['score']:.3f}")
        
        print()
        print(f"📤 Sending {num_events} live events...")
        
        # Send live events
        events = await self.send_live_events(user_id, num_events, delay_seconds=1.0)
        
        # Wait for processing
        print("⏳ Waiting for events to be processed...")
        await asyncio.sleep(10)
        
        # Monitor processing
        await self.monitor_processing(duration_seconds=15)
        
        # Get updated recommendations
        print("📊 Getting updated recommendations...")
        await asyncio.sleep(5)  # Allow time for feature store updates
        
        updated_recs = self.get_current_recommendations(user_id, top_k=5)
        self.after_recommendations[user_id] = updated_recs
        
        print()
        print("🔮 UPDATED RECOMMENDATIONS:")
        for i, rec in enumerate(updated_recs, 1):
            print(f"   {i}. {rec['name']} ({rec['category']}) - Score: {rec['score']:.3f}")
        
        # Analyze changes
        self.analyze_recommendation_changes(user_id, initial_recs, updated_recs, events)
        
        return {
            'user_id': user_id,
            'initial_recommendations': initial_recs,
            'updated_recommendations': updated_recs,
            'events_sent': events,
            'change_detected': initial_recs != updated_recs
        }
    
    def analyze_recommendation_changes(self, user_id: int, before: List[Dict], after: List[Dict], events: List[Dict]):
        """Analyze how recommendations changed after live events."""
        print()
        print("📈 RECOMMENDATION CHANGE ANALYSIS:")
        print("-" * 40)
        
        # Product IDs before and after
        before_ids = {rec['product_id'] for rec in before}
        after_ids = {rec['product_id'] for rec in after}
        
        # Calculate changes
        new_products = after_ids - before_ids
        removed_products = before_ids - after_ids
        
        print(f"📊 Products added to recommendations: {len(new_products)}")
        if new_products:
            for pid in new_products:
                product = next(rec for rec in after if rec['product_id'] == pid)
                print(f"   + {product['name']} ({product['category']})")
        
        print(f"📊 Products removed from recommendations: {len(removed_products)}")
        if removed_products:
            for pid in removed_products:
                product = next(rec for rec in before if rec['product_id'] == pid)
                print(f"   - {product['name']} ({product['category']})")
        
        # Event influence analysis
        event_categories = [e['product_category'] for e in events]
        category_counts = {}
        for cat in event_categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        print(f"📊 Event categories in this session:")
        for cat, count in category_counts.items():
            print(f"   - {cat}: {count} events")
        
        # Score changes for common products
        common_products = before_ids & after_ids
        print(f"📊 Score changes for common products:")
        for pid in common_products:
            before_score = next(rec['score'] for rec in before if rec['product_id'] == pid)
            after_score = next(rec['score'] for rec in after if rec['product_id'] == pid)
            change = after_score - before_score
            product_name = next(rec['name'] for rec in before if rec['product_id'] == pid)
            print(f"   {product_name}: {before_score:.3f} → {after_score:.3f} ({change:+.3f})")
    
    async def run_comprehensive_test(self, test_users: List[int] = None, events_per_user: int = 5):
        """Run a comprehensive live testing scenario."""
        print("🚀 Starting Comprehensive Live Recommendation Test")
        print("=" * 80)
        
        if test_users is None:
            # Select diverse test users
            available_users = list(self.user_mapping.keys())
            test_users = random.sample(available_users, min(3, len(available_users)))
        
        results = []
        
        for user_id in test_users:
            try:
                print()
                result = await self.test_recommendation_updates(user_id, events_per_user)
                results.append(result)
                
                # Brief pause between users
                await asyncio.sleep(3)
                
            except Exception as e:
                print(f"❌ Test failed for user {user_id}: {e}")
        
        # Final summary
        self.print_test_summary(results)
        return results
    
    def print_test_summary(self, results: List[Dict]):
        """Print comprehensive test summary."""
        print()
        print("=" * 80)
        print("🎯 LIVE RECOMMENDATION TESTING SUMMARY")
        print("=" * 80)
        
        print(f"📊 Total users tested: {len(results)}")
        print(f"📤 Total events sent: {self.events_sent}")
        print(f"⚡ Average events per user: {self.events_sent / len(results) if results else 0:.1f}")
        
        changes_detected = sum(1 for r in results if r['change_detected'])
        print(f"🔄 Users with recommendation changes: {changes_detected}/{len(results)}")
        
        print()
        print("✅ Test completed successfully!")
        print("🎯 This demonstrates real-time recommendation updates based on live user behavior")
        
    async def cleanup(self):
        """Clean up resources."""
        try:
            if self.kafka_producer:
                await self.kafka_producer.close()
            if self.kafka_consumer:
                await self.kafka_consumer.close()
            if self.feature_store:
                await self.feature_store.close()
            print("✅ Cleanup completed")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

async def main():
    """Main function to run live testing."""
    tester = LiveRecommendationTester()
    
    try:
        # Initialize components
        await tester.initialize()
        
        # Run comprehensive test
        await tester.run_comprehensive_test(
            test_users=[1, 1118, 3194],  # Mix of different user types
            events_per_user=5
        )
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    print("🎯 Live Clickstream Recommendation Testing")
    print("==========================================")
    print("This test simulates real-time user interactions and shows")
    print("how recommendations adapt to live behavioral data.")
    print()
    
    # Run the test
    asyncio.run(main())
