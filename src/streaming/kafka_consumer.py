"""Kafka consumer for streaming clickstream events."""

import json
import time
from typing import Dict, Any, Callable, Optional
from kafka import KafkaConsumer
from kafka.errors import KafkaError

from src.utils.logging import get_logger

logger = get_logger(__name__)


class KafkaEventConsumer:
    """Consumes events from Kafka topics."""
    
    def __init__(
        self, 
        bootstrap_servers: str = "localhost:9092",
        group_id: str = "recommndr-consumer-group"
    ):
        """Initialize Kafka consumer.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            group_id: Consumer group ID
        """
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.consumer = None
        self._connect()
    
    def _connect(self) -> None:
        """Connect to Kafka."""
        try:
            self.consumer = KafkaConsumer(
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset='earliest',  # Start from beginning
                enable_auto_commit=True,
                auto_commit_interval_ms=1000,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                key_deserializer=lambda x: x.decode('utf-8') if x else None,
                consumer_timeout_ms=1000  # Timeout for polling
            )
            logger.info(f"Connected to Kafka at {self.bootstrap_servers}")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise
    
    def subscribe(self, topics: list) -> None:
        """Subscribe to Kafka topics.
        
        Args:
            topics: List of topic names to subscribe to
        """
        if not self.consumer:
            logger.error("Consumer not connected")
            return
        
        try:
            self.consumer.subscribe(topics)
            logger.info(f"Subscribed to topics: {topics}")
        except Exception as e:
            logger.error(f"Failed to subscribe to topics: {e}")
            raise
    
    def consume_events(
        self, 
        topic: str, 
        message_handler: Callable[[Dict[str, Any]], None],
        max_messages: Optional[int] = None,
        timeout_seconds: int = 30
    ) -> Dict[str, int]:
        """Consume events from a topic.
        
        Args:
            topic: Topic to consume from
            message_handler: Function to handle each message
            max_messages: Maximum number of messages to consume (None for unlimited)
            timeout_seconds: Timeout in seconds
            
        Returns:
            Dictionary with consumption statistics
        """
        if not self.consumer:
            logger.error("Consumer not connected")
            return {"consumed": 0, "errors": 0}
        
        # Subscribe to topic
        self.subscribe([topic])
        
        stats = {"consumed": 0, "errors": 0}
        start_time = time.time()
        
        try:
            while True:
                # Check timeout
                if time.time() - start_time > timeout_seconds:
                    logger.info(f"Consumption timeout reached ({timeout_seconds}s)")
                    break
                
                # Check max messages
                if max_messages and stats["consumed"] >= max_messages:
                    logger.info(f"Max messages reached ({max_messages})")
                    break
                
                # Poll for messages
                messages = self.consumer.poll(timeout_ms=1000, max_records=100)
                
                for topic_partition, partition_messages in messages.items():
                    for message in partition_messages:
                        try:
                            # Handle message
                            message_handler(message.value)
                            stats["consumed"] += 1
                            
                            logger.debug(
                                f"Message consumed",
                                extra={
                                    "topic": message.topic,
                                    "partition": message.partition,
                                    "offset": message.offset,
                                    "key": message.key
                                }
                            )
                            
                        except Exception as e:
                            logger.error(f"Error handling message: {e}")
                            stats["errors"] += 1
                
                # Small delay to prevent busy waiting
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("Consumption interrupted by user")
        except Exception as e:
            logger.error(f"Error during consumption: {e}")
        
        logger.info(
            f"Consumption completed",
            extra={
                "topic": topic,
                "consumed": stats["consumed"],
                "errors": stats["errors"],
                "duration_seconds": time.time() - start_time
            }
        )
        
        return stats
    
    def get_topic_info(self, topic: str) -> Dict[str, Any]:
        """Get information about a Kafka topic.
        
        Args:
            topic: Topic name
            
        Returns:
            Topic information
        """
        if not self.consumer:
            return {}
        
        try:
            # Get topic partitions
            partitions = self.consumer.partitions_for_topic(topic)
            if not partitions:
                return {"topic": topic, "partitions": 0}
            
            # Get beginning and end offsets for each partition
            topic_info = {
                "topic": topic,
                "partitions": len(partitions),
                "partition_details": {}
            }
            
            for partition in partitions:
                beginning = self.consumer.beginning_offsets([(topic, partition)])[(topic, partition)]
                end = self.consumer.end_offsets([(topic, partition)])[(topic, partition)]
                
                topic_info["partition_details"][partition] = {
                    "beginning_offset": beginning,
                    "end_offset": end,
                    "total_messages": end - beginning
                }
            
            return topic_info
            
        except Exception as e:
            logger.error(f"Failed to get topic info: {e}")
            return {"topic": topic, "error": str(e)}
    
    def close(self) -> None:
        """Close the consumer."""
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer closed")


def main():
    """Test the Kafka consumer."""
    consumer = KafkaEventConsumer()
    
    try:
        # Test message handler
        def handle_message(message):
            print(f"📨 Received: {message}")
        
        # Get topic info
        topic_info = consumer.get_topic_info("test-topic")
        print(f"📊 Topic info: {topic_info}")
        
        # Consume messages (will timeout after 5 seconds)
        stats = consumer.consume_events(
            topic="test-topic",
            message_handler=handle_message,
            timeout_seconds=5
        )
        
        print(f"📈 Consumption stats: {stats}")
        
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
