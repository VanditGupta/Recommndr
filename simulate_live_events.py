#!/usr/bin/env python3
"""
Live Event Simulator for Recommendation Testing

This script continuously generates realistic clickstream events
to test the real-time recommendation system.
"""

import json
import time
import random
import asyncio
import pandas as pd
from datetime import datetime, timezone
from src.streaming.kafka_producer import KafkaProducer

class LiveEventSimulator:
    """Simulate realistic live events for testing."""
    
    def __init__(self):
        self.producer = None
        self.products_df = None
        self.users_df = None
        self.user_sessions = {}
        self.events_sent = 0
        
    def initialize(self):
        """Initialize the simulator."""
        print("🎬 Initializing Live Event Simulator...")
        
        # Initialize Kafka producer
        self.producer = KafkaProducer()
        
        # Load data for realistic simulation
        self.products_df = pd.read_parquet('data/processed/products_cleaned.parquet')
        self.users_df = pd.read_parquet('data/processed/users_cleaned.parquet')
        
        print(f"✅ Simulator ready with {len(self.users_df)} users and {len(self.products_df)} products")
    
    def get_user_preferences(self, user_id: int):
        """Get user preferences for realistic event generation."""
        user_info = self.users_df[self.users_df['user_id'] == user_id].iloc[0]
        
        # Categories user might be interested in based on preference
        preferred_category = user_info['preference_category']
        
        # Get products in preferred category
        preferred_products = self.products_df[
            self.products_df['category'] == preferred_category
        ]['product_id'].tolist()
        
        # Also include some random products (exploration)
        all_products = self.products_df['product_id'].tolist()
        random_products = random.sample(all_products, min(20, len(all_products)))
        
        # Combine with preference weighting
        candidate_products = preferred_products * 3 + random_products  # 3x weight for preferred
        
        return {
            'preferred_category': preferred_category,
            'candidate_products': candidate_products,
            'age': user_info['age'],
            'income_level': user_info['income_level'],
            'device_type': user_info['device_type']
        }
    
    def generate_session_events(self, user_id: int, session_length: int = None):
        """Generate a realistic user session with multiple events."""
        if session_length is None:
            # Realistic session lengths (most are short)
            session_length = random.choices([1, 2, 3, 4, 5, 8, 12], weights=[30, 25, 20, 15, 5, 3, 2])[0]
        
        user_prefs = self.get_user_preferences(user_id)
        session_id = f"session_{user_id}_{int(time.time())}_{random.randint(100, 999)}"
        
        events = []
        current_product = None
        
        for i in range(session_length):
            # Event type probabilities change based on session progression
            if i == 0:
                # First event is always a view
                event_type = 'view'
            else:
                # Later events can be more engaging
                event_types = ['view', 'click', 'add_to_cart', 'purchase']
                # More likely to engage with current product
                if current_product and random.random() < 0.3:
                    event_weights = [0.4, 0.4, 0.15, 0.05]
                else:
                    event_weights = [0.7, 0.2, 0.08, 0.02]
                event_type = random.choices(event_types, weights=event_weights)[0]
            
            # Product selection
            if current_product and random.random() < 0.4:
                # Continue with current product (realistic browsing)
                product_id = current_product
            else:
                # Select new product from preferences
                product_id = random.choice(user_prefs['candidate_products'])
                current_product = product_id
            
            # Get product info
            product_info = self.products_df[self.products_df['product_id'] == product_id].iloc[0]
            
            # Create event
            event = {
                'event_id': f"evt_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
                'user_id': user_id,
                'product_id': product_id,
                'event_type': event_type,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'session_id': session_id,
                
                # Product context
                'product_name': product_info['name'],
                'product_category': product_info['category'],
                'product_price': float(product_info['price']),
                'product_brand': product_info['brand'],
                'product_rating': float(product_info['rating']),
                
                # User behavior
                'dwell_time': self.generate_dwell_time(event_type, product_info),
                'scroll_depth': random.randint(10, 100) if event_type == 'view' else None,
                'quantity': random.randint(1, 3) if event_type in ['add_to_cart', 'purchase'] else None,
                
                # Technical details
                'device_type': user_prefs['device_type'],
                'page_url': f"/product/{product_id}",
                'referrer': self.get_referrer(i == 0),
                'user_agent': self.get_user_agent(user_prefs['device_type']),
                
                # Simulation metadata
                'simulation': True,
                'simulation_time': datetime.now().isoformat()
            }
            
            events.append(event)
        
        return events
    
    def generate_dwell_time(self, event_type: str, product_info: pd.Series):
        """Generate realistic dwell time based on event and product."""
        if event_type != 'view':
            return None
        
        # Base dwell time
        base_time = random.randint(5, 60)
        
        # Longer for expensive products
        if product_info['price'] > 500:
            base_time += random.randint(10, 30)
        
        # Longer for high-rated products
        if product_info['rating'] > 4.0:
            base_time += random.randint(5, 15)
        
        return base_time
    
    def get_referrer(self, is_first_event: bool):
        """Get realistic referrer."""
        if is_first_event:
            return random.choice(['google', 'direct', 'facebook', 'email', 'recommendation'])
        else:
            return random.choice(['internal', 'search', 'category', 'recommendation'])
    
    def get_user_agent(self, device_type: str):
        """Get realistic user agent string."""
        agents = {
            'mobile': 'Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15',
            'desktop': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'tablet': 'Mozilla/5.0 (iPad; CPU OS 15_0 like Mac OS X) AppleWebKit/605.1.15'
        }
        return agents.get(device_type, agents['desktop'])
    
    async def send_user_session(self, user_id: int):
        """Send a complete user session to Kafka."""
        events = self.generate_session_events(user_id)
        
        print(f"👤 User {user_id} session: {len(events)} events")
        
        for i, event in enumerate(events):
            try:
                await self.producer.send_event(event)
                self.events_sent += 1
                
                print(f"   📤 {i+1}/{len(events)}: {event['event_type']} → {event['product_name'][:30]}...")
                
                # Realistic delay between events in a session
                await asyncio.sleep(random.uniform(1.0, 5.0))
                
            except Exception as e:
                print(f"   ❌ Failed to send event: {e}")
        
        return events
    
    async def simulate_continuous_traffic(self, duration_minutes: int = 5, users_per_minute: int = 2):
        """Simulate continuous user traffic."""
        print(f"🚦 Simulating continuous traffic for {duration_minutes} minutes...")
        print(f"📊 Target: {users_per_minute} users per minute")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        available_users = self.users_df['user_id'].tolist()
        
        while time.time() < end_time:
            # Select random users for this batch
            batch_users = random.sample(available_users, min(users_per_minute, len(available_users)))
            
            # Send sessions for these users concurrently
            tasks = [self.send_user_session(user_id) for user_id in batch_users]
            await asyncio.gather(*tasks)
            
            remaining_time = end_time - time.time()
            print(f"⏱️  {remaining_time/60:.1f} minutes remaining | Events sent: {self.events_sent}")
            
            # Wait until next minute
            await asyncio.sleep(max(0, 60 - (time.time() % 60)))
        
        print(f"✅ Simulation completed! Total events sent: {self.events_sent}")
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            if self.producer:
                await self.producer.close()
            print("✅ Simulator cleanup completed")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

async def main():
    """Main simulation function."""
    print("🎬 Live Event Simulator for Recommendation Testing")
    print("=" * 60)
    print("This simulator generates realistic clickstream events to test")
    print("the real-time recommendation system with Kafka and Flink.")
    print()
    
    simulator = LiveEventSimulator()
    
    try:
        # Initialize
        simulator.initialize()
        
        # Choose simulation mode
        print("📋 Simulation Options:")
        print("1. Single user session test")
        print("2. Multiple user batch test")
        print("3. Continuous traffic simulation")
        
        # For this demo, let's run a batch test
        print("🚀 Running batch test with multiple users...")
        
        # Test multiple users
        test_users = random.sample(simulator.users_df['user_id'].tolist(), 5)
        
        for user_id in test_users:
            await simulator.send_user_session(user_id)
            await asyncio.sleep(2)  # Brief pause between users
        
        print()
        print("🎉 Batch simulation completed!")
        print(f"📊 Total events sent: {simulator.events_sent}")
        print()
        print("💡 To run continuous simulation, uncomment the line below:")
        print("# await simulator.simulate_continuous_traffic(duration_minutes=10)")
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await simulator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
