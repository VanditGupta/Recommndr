"""Clickstream simulator for Phase 2 streaming pipeline."""

import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from kafka import KafkaProducer
from pydantic import BaseModel, Field

from src.utils.logging import get_logger

logger = get_logger(__name__)


class ClickstreamEvent(BaseModel):
    """Clickstream event model."""
    
    event_id: str = Field(..., description="Unique event identifier")
    user_id: int = Field(..., description="User ID")
    product_id: int = Field(..., description="Product ID")
    event_type: str = Field(..., description="Event type (view, click, add_to_cart, purchase)")
    timestamp: datetime = Field(..., description="Event timestamp")
    session_id: str = Field(..., description="User session ID")
    page_url: str = Field(..., description="Page URL")
    referrer_url: Optional[str] = Field(None, description="Referrer URL")
    user_agent: str = Field(..., description="User agent string")
    ip_address: str = Field(..., description="IP address")
    dwell_time: Optional[int] = Field(None, description="Time spent on page (seconds)")
    scroll_depth: Optional[int] = Field(None, description="Scroll depth percentage")
    device_type: str = Field(..., description="Device type (mobile, desktop, tablet)")
    location: str = Field(..., description="User location")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ClickstreamSimulator:
    """Simulates realistic clickstream events for Phase 2."""
    
    def __init__(self, kafka_bootstrap_servers: str = "localhost:9092"):
        """Initialize clickstream simulator.
        
        Args:
            kafka_bootstrap_servers: Kafka bootstrap servers
        """
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
            key_serializer=lambda k: str(k).encode('utf-8')
        )
        
        # Event type probabilities (realistic distribution)
        self.event_probabilities = {
            "view": 0.70,      # 70% of events are views
            "click": 0.20,     # 20% are clicks
            "add_to_cart": 0.08,  # 8% are add to cart
            "purchase": 0.02   # 2% are purchases
        }
        
        # Page URLs for realistic navigation
        self.page_urls = [
            "/",
            "/products",
            "/products/electronics",
            "/products/clothing", 
            "/products/home-garden",
            "/products/sports-outdoors",
            "/search",
            "/cart",
            "/checkout",
            "/account"
        ]
        
        # User agent strings
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15",
            "Mozilla/5.0 (Android 13; Mobile; rv:109.0) Gecko/118.0 Firefox/118.0"
        ]
        
        # IP address ranges (simplified)
        self.ip_ranges = [
            "192.168.1.", "10.0.0.", "172.16.0.", "203.0.113."
        ]
        
        logger.info("Clickstream simulator initialized")
    
    def generate_event(self, user_id: int, product_id: int, session_id: str) -> ClickstreamEvent:
        """Generate a realistic clickstream event.
        
        Args:
            user_id: User ID
            product_id: Product ID
            session_id: Session ID
            
        Returns:
            ClickstreamEvent
        """
        # Determine event type based on probabilities
        event_type = random.choices(
            list(self.event_probabilities.keys()),
            weights=list(self.event_probabilities.values())
        )[0]
        
        # Generate timestamp (within last hour)
        timestamp = datetime.now() - timedelta(
            seconds=random.randint(0, 3600)
        )
        
        # Generate realistic dwell time and scroll depth for certain events
        dwell_time = None
        scroll_depth = None
        
        if event_type in ["view", "click"]:
            dwell_time = random.randint(5, 300)  # 5 seconds to 5 minutes
            scroll_depth = random.randint(10, 100)  # 10% to 100%
        
        # Generate event
        event = ClickstreamEvent(
            event_id=f"event_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
            user_id=user_id,
            product_id=product_id,
            event_type=event_type,
            timestamp=timestamp,
            session_id=session_id,
            page_url=random.choice(self.page_urls),
            referrer_url=random.choice(self.page_urls) if random.random() > 0.3 else None,
            user_agent=random.choice(self.user_agents),
            ip_address=f"{random.choice(self.ip_ranges)}{random.randint(1, 254)}",
            dwell_time=dwell_time,
            scroll_depth=scroll_depth,
            device_type=random.choice(["mobile", "desktop", "tablet"]),
            location=random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"])
        )
        
        return event
    
    def send_event(self, event: ClickstreamEvent, topic: str = "clickstream-events") -> None:
        """Send event to Kafka topic.
        
        Args:
            event: ClickstreamEvent to send
            topic: Kafka topic name
        """
        try:
            # Use user_id as key for partitioning
            future = self.kafka_producer.send(
                topic,
                key=event.user_id,
                value=event.model_dump()
            )
            
            # Wait for send to complete
            record_metadata = future.get(timeout=10)
            
            logger.debug(
                f"Event sent to {topic}",
                extra={
                    "event_id": event.event_id,
                    "user_id": event.user_id,
                    "partition": record_metadata.partition,
                    "offset": record_metadata.offset
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to send event: {e}")
            raise
    
    def simulate_user_session(
        self, 
        user_id: int, 
        session_duration_minutes: int = 15,
        events_per_minute: int = 2
    ) -> None:
        """Simulate a complete user session.
        
        Args:
            user_id: User ID
            session_duration_minutes: Session duration in minutes
            events_per_minute: Average events per minute
        """
        session_id = f"session_{user_id}_{int(time.time())}"
        total_events = session_duration_minutes * events_per_minute
        
        logger.info(
            f"Starting session simulation",
            extra={
                "user_id": user_id,
                "session_id": session_id,
                "duration_minutes": session_duration_minutes,
                "total_events": total_events
            }
        )
        
        # Generate events throughout the session
        for i in range(total_events):
            # Random product ID (1-1000)
            product_id = random.randint(1, 1000)
            
            # Generate and send event
            event = self.generate_event(user_id, product_id, session_id)
            self.send_event(event)
            
            # Wait between events (realistic timing)
            time.sleep(random.uniform(0.5, 2.0))
        
        logger.info(
            f"Session simulation completed",
            extra={
                "user_id": user_id,
                "session_id": session_id,
                "events_sent": total_events
            }
        )
    
    def run_simulation(
        self, 
        num_users: int = 100,
        session_duration_minutes: int = 15,
        events_per_minute: int = 2
    ) -> None:
        """Run the complete clickstream simulation.
        
        Args:
            num_users: Number of users to simulate
            session_duration_minutes: Session duration per user
            events_per_minute: Events per minute per user
        """
        logger.info(
            f"Starting clickstream simulation",
            extra={
                "num_users": num_users,
                "session_duration": session_duration_minutes,
                "events_per_minute": events_per_minute
            }
        )
        
        start_time = time.time()
        
        for user_id in range(1, num_users + 1):
            try:
                self.simulate_user_session(
                    user_id, 
                    session_duration_minutes, 
                    events_per_minute
                )
                
                # Small delay between users
                time.sleep(random.uniform(1, 3))
                
            except Exception as e:
                logger.error(f"Error simulating user {user_id}: {e}")
                continue
        
        total_time = time.time() - start_time
        logger.info(
            f"Clickstream simulation completed",
            extra={
                "total_users": num_users,
                "total_time_seconds": total_time,
                "events_per_second": (num_users * session_duration_minutes * events_per_minute) / total_time
            }
        )
    
    def close(self) -> None:
        """Close Kafka producer."""
        self.kafka_producer.close()
        logger.info("Clickstream simulator closed")


def main():
    """Main function for testing."""
    simulator = ClickstreamSimulator()
    
    try:
        # Run a small simulation for testing
        simulator.run_simulation(
            num_users=10,
            session_duration_minutes=5,
            events_per_minute=1
        )
    finally:
        simulator.close()


if __name__ == "__main__":
    main()
