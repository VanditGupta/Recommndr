"""Kafka topic management for Phase 2 streaming pipeline."""

import time
from typing import List, Dict, Any
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError, NoBrokersAvailable

from src.utils.logging import get_logger

logger = get_logger(__name__)


class KafkaTopicManager:
    """Manages Kafka topics for the streaming pipeline."""
    
    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        """Initialize Kafka topic manager.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
        """
        self.bootstrap_servers = bootstrap_servers
        self.admin_client = None
        self._connect()
    
    def _connect(self) -> None:
        """Connect to Kafka admin client."""
        try:
            self.admin_client = KafkaAdminClient(
                bootstrap_servers=self.bootstrap_servers,
                client_id='recommndr-topic-manager'
            )
            logger.info(f"Connected to Kafka admin at {self.bootstrap_servers}")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka admin: {e}")
            raise
    
    def create_topics(self, topics: List[Dict[str, Any]]) -> bool:
        """Create Kafka topics.
        
        Args:
            topics: List of topic configurations
            
        Returns:
            True if all topics created successfully
        """
        if not self.admin_client:
            logger.error("Admin client not connected")
            return False
        
        try:
            # Convert topic configs to NewTopic objects
            new_topics = []
            for topic_config in topics:
                new_topic = NewTopic(
                    name=topic_config['name'],
                    num_partitions=topic_config.get('partitions', 1),
                    replication_factor=topic_config.get('replication_factor', 1),
                    topic_configs=topic_config.get('configs', {})
                )
                new_topics.append(new_topic)
            
            # Create topics
            self.admin_client.create_topics(new_topics)
            logger.info(f"Successfully created {len(new_topics)} topics")
            return True
            
        except TopicAlreadyExistsError:
            logger.info("Some topics already exist (this is fine)")
            return True
        except Exception as e:
            logger.error(f"Failed to create topics: {e}")
            return False
    
    def list_topics(self) -> List[str]:
        """List all Kafka topics.
        
        Returns:
            List of topic names
        """
        if not self.admin_client:
            return []
        
        try:
            topics = self.admin_client.list_topics()
            return topics
        except Exception as e:
            logger.error(f"Failed to list topics: {e}")
            return []
    
    def delete_topics(self, topic_names: List[str]) -> bool:
        """Delete Kafka topics.
        
        Args:
            topic_names: List of topic names to delete
            
        Returns:
            True if all topics deleted successfully
        """
        if not self.admin_client:
            return False
        
        try:
            self.admin_client.delete_topics(topic_names)
            logger.info(f"Successfully deleted {len(topic_names)} topics")
            return True
        except Exception as e:
            logger.error(f"Failed to delete topics: {e}")
            return False
    
    def close(self) -> None:
        """Close the admin client."""
        if self.admin_client:
            self.admin_client.close()
            logger.info("Kafka admin client closed")


def get_default_topics() -> List[Dict[str, Any]]:
    """Get default topic configurations for Phase 2.
    
    Returns:
        List of topic configurations
    """
    return [
        {
            'name': 'clickstream-events',
            'partitions': 3,
            'replication_factor': 1,
            'configs': {
                'retention.ms': '604800000',  # 7 days
                'cleanup.policy': 'delete'
            }
        },
        {
            'name': 'processed-events',
            'partitions': 3,
            'replication_factor': 1,
            'configs': {
                'retention.ms': '604800000',  # 7 days
                'cleanup.policy': 'delete'
            }
        },
        {
            'name': 'user-features',
            'partitions': 2,
            'replication_factor': 1,
            'configs': {
                'retention.ms': '2592000000',  # 30 days
                'cleanup.policy': 'delete'
            }
        },
        {
            'name': 'product-features',
            'partitions': 2,
            'replication_factor': 1,
            'configs': {
                'retention.ms': '2592000000',  # 30 days
                'cleanup.policy': 'delete'
            }
        }
    ]


def main():
    """Create default Kafka topics for Phase 2."""
    topic_manager = KafkaTopicManager()
    
    try:
        # Get default topics
        topics = get_default_topics()
        
        # Create topics
        success = topic_manager.create_topics(topics)
        if success:
            print("✅ Kafka topics created successfully")
        else:
            print("❌ Failed to create some topics")
        
        # List existing topics
        existing_topics = topic_manager.list_topics()
        print(f"📋 Existing topics: {existing_topics}")
        
    finally:
        topic_manager.close()


if __name__ == "__main__":
    main()
