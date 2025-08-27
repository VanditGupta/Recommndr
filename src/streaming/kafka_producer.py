"""Kafka producer for streaming clickstream events."""

import json
import time
from typing import Dict, Any, Optional
from kafka import KafkaProducer
from kafka.errors import KafkaError

from src.utils.logging import get_logger

logger = get_logger(__name__)


class KafkaEventProducer:
    """Produces events to Kafka topics."""
    
    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        """Initialize Kafka producer.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
        """
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        self._connect()
    
    def _connect(self) -> None:
        """Connect to Kafka."""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                key_serializer=lambda k: str(k).encode('utf-8'),
                acks='all',  # Wait for all replicas
                retries=3,   # Retry failed sends
                request_timeout_ms=30000
            )
            logger.info(f"Connected to Kafka at {self.bootstrap_servers}")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise
    
    def send_event(self, topic: str, event: Dict[str, Any], key: Optional[str] = None) -> bool:
        """Send event to Kafka topic.
        
        Args:
            topic: Kafka topic name
            event: Event data to send
            key: Message key for partitioning
            
        Returns:
            True if successful, False otherwise
        """
        if not self.producer:
            logger.error("Producer not connected")
            return False
        
        try:
            # Use event_id as key if no key provided
            if key is None:
                key = event.get('event_id', str(int(time.time() * 1000)))
            
            # Send event
            future = self.producer.send(topic, key=key, value=event)
            
            # Wait for send to complete
            record_metadata = future.get(timeout=10)
            
            logger.debug(
                f"Event sent to {topic}",
                extra={
                    "topic": topic,
                    "partition": record_metadata.partition,
                    "offset": record_metadata.offset,
                    "key": key
                }
            )
            
            return True
            
        except KafkaError as e:
            logger.error(f"Kafka error sending event: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to send event: {e}")
            return False
    
    def send_batch(self, topic: str, events: list, key_field: str = 'event_id') -> Dict[str, int]:
        """Send multiple events to Kafka topic.
        
        Args:
            topic: Kafka topic name
            events: List of events to send
            key_field: Field to use as message key
            
        Returns:
            Dictionary with success/failure counts
        """
        results = {"success": 0, "failed": 0}
        
        for event in events:
            key = event.get(key_field, str(int(time.time() * 1000)))
            if self.send_event(topic, event, key):
                results["success"] += 1
            else:
                results["failed"] += 1
        
        logger.info(
            f"Batch send completed",
            extra={
                "topic": topic,
                "total_events": len(events),
                "success": results["success"],
                "failed": results["failed"]
            }
        )
        
        return results
    
    def flush(self) -> None:
        """Flush all pending messages."""
        if self.producer:
            self.producer.flush()
            logger.debug("Producer flushed")
    
    def close(self) -> None:
        """Close the producer."""
        if self.producer:
            self.producer.close()
            logger.info("Kafka producer closed")


def main():
    """Test the Kafka producer."""
    producer = KafkaEventProducer()
    
    try:
        # Test event
        test_event = {
            "event_id": f"test_{int(time.time() * 1000)}",
            "user_id": 123,
            "event_type": "test",
            "timestamp": time.time()
        }
        
        # Send test event
        success = producer.send_event("test-topic", test_event)
        if success:
            print("✅ Test event sent successfully")
        else:
            print("❌ Failed to send test event")
            
    finally:
        producer.close()


if __name__ == "__main__":
    main()
