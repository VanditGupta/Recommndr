"""Phase 2: Streaming Ingestion & Feature Pipeline Main Module."""

import time
import json
from typing import Dict, Any, List
from datetime import datetime

from src.streaming.kafka_producer import KafkaEventProducer
from src.streaming.kafka_consumer import KafkaEventConsumer
from src.streaming.flink_processor import FlinkStreamProcessor
from src.streaming.kafka_manager import KafkaTopicManager, get_default_topics
from src.features.feast_integration import FeastFeatureStore
from src.streaming.clickstream_simulator import ClickstreamSimulator
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Phase2Pipeline:
    """Phase 2: Streaming Ingestion & Feature Pipeline."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Phase 2 pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.kafka_producer = None
        self.kafka_consumer = None
        self.flink_processor = None
        self.feature_store = None
        self.clickstream_simulator = None
        
        # Pipeline state
        self.is_running = False
        self.pipeline_stats = {
            "events_generated": 0,
            "events_processed": 0,
            "features_stored": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None
        }
        
        logger.info("Phase 2 pipeline initialized")
    
    def setup_components(self) -> bool:
        """Setup all pipeline components.
        
        Returns:
            True if all components setup successfully
        """
        try:
            logger.info("Setting up Phase 2 pipeline components...")
            
            # Setup Kafka producer
            self.kafka_producer = KafkaEventProducer(
                bootstrap_servers=self.config.get('kafka_bootstrap_servers', 'localhost:9092')
            )
            logger.info("✅ Kafka producer setup complete")
            
            # Setup Kafka consumer
            self.kafka_consumer = KafkaEventConsumer(
                bootstrap_servers=self.config.get('kafka_bootstrap_servers', 'localhost:9092'),
                group_id=self.config.get('kafka_group_id', 'recommndr-phase2-group')
            )
            logger.info("✅ Kafka consumer setup complete")
            
            # Setup Flink processor
            self.flink_processor = FlinkStreamProcessor()
            logger.info("✅ Flink processor setup complete")
            
            # Setup feature store
            self.feature_store = FeastFeatureStore(
                redis_host=self.config.get('redis_host', 'localhost'),
                redis_port=self.config.get('redis_port', 6380)
            )
            logger.info("✅ Feature store setup complete")
            
            # Setup clickstream simulator
            self.clickstream_simulator = ClickstreamSimulator()
            logger.info("✅ Clickstream simulator setup complete")
            
            logger.info("🎉 All Phase 2 components setup successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup Phase 2 components: {e}")
            return False
    
    def run_pipeline(self, duration_seconds: int = 60, events_per_second: int = 10) -> Dict[str, Any]:
        """Run the complete Phase 2 pipeline.
        
        Args:
            duration_seconds: How long to run the pipeline
            events_per_second: Events to generate per second
            
        Returns:
            Pipeline execution results
        """
        if not all([self.kafka_producer, self.kafka_consumer, self.flink_processor, 
                   self.feature_store, self.clickstream_simulator]):
            logger.error("Pipeline components not setup. Call setup_components() first.")
            return {}
        
        try:
            logger.info(f"🚀 Starting Phase 2 pipeline for {duration_seconds} seconds...")
            self.is_running = True
            self.pipeline_stats["start_time"] = datetime.now()
            
            # Start pipeline components
            self._start_pipeline_components()
            
            # Run pipeline for specified duration
            start_time = time.time()
            while time.time() - start_time < duration_seconds and self.is_running:
                # Generate and process events
                self._process_pipeline_iteration(events_per_second)
                time.sleep(1)  # Wait 1 second between iterations
            
            # Stop pipeline components
            self._stop_pipeline_components()
            
            # Finalize pipeline
            self.pipeline_stats["end_time"] = datetime.now()
            self.is_running = False
            
            # Generate final report
            final_stats = self._generate_pipeline_report()
            
            logger.info("🎉 Phase 2 pipeline completed successfully!")
            return final_stats
            
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
            self._stop_pipeline_components()
            self.is_running = False
            return self._generate_pipeline_report()
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self._stop_pipeline_components()
            self.is_running = False
            return self._generate_pipeline_report()
    
    def _start_pipeline_components(self) -> None:
        """Start all pipeline components."""
        logger.info("Starting pipeline components...")
        
        # Create Kafka topics if they don't exist
        self._ensure_kafka_topics()
        
        # Start consumer in background (simulated)
        logger.info("Pipeline components started")
    
    def _stop_pipeline_components(self) -> None:
        """Stop all pipeline components."""
        logger.info("Stopping pipeline components...")
        
        if self.kafka_producer:
            self.kafka_producer.flush()
            self.kafka_producer.close()
        
        if self.kafka_consumer:
            self.kafka_consumer.close()
        
        if self.feature_store:
            self.feature_store.close()
        
        logger.info("Pipeline components stopped")
    
    def _ensure_kafka_topics(self) -> None:
        """Ensure required Kafka topics exist."""
        try:
            # Create topic manager
            topic_manager = KafkaTopicManager(
                bootstrap_servers=self.config.get('kafka_bootstrap_servers', 'localhost:9092')
            )
            
            # Get default topics
            required_topics = get_default_topics()
            
            # Create topics
            success = topic_manager.create_topics(required_topics)
            if success:
                logger.info("✅ Kafka topics created/verified successfully")
            else:
                logger.warning("⚠️ Some Kafka topics may not be available")
            
            # List existing topics
            existing_topics = topic_manager.list_topics()
            logger.info(f"📋 Available Kafka topics: {existing_topics}")
            
            # Close topic manager
            topic_manager.close()
            
        except Exception as e:
            logger.error(f"Failed to ensure Kafka topics: {e}")
            logger.warning("⚠️ Pipeline will continue but may have topic issues")
    
    def _process_pipeline_iteration(self, events_per_second: int) -> None:
        """Process one iteration of the pipeline.
        
        Args:
            events_per_second: Number of events to process
        """
        try:
            # Generate clickstream events
            events = self.clickstream_simulator.generate_events(events_per_second)
            self.pipeline_stats["events_generated"] += len(events)
            
            # Send events to Kafka
            for event in events:
                # Convert Pydantic model to dictionary
                event_dict = event.model_dump()
                success = self.kafka_producer.send_event('clickstream-events', event_dict)
                if not success:
                    self.pipeline_stats["errors"] += 1
            
            # Process events through Flink
            processed_events = []
            for event in events:
                try:
                    # Convert Pydantic model to dictionary for processing
                    event_dict = event.model_dump()
                    processed_event = self.flink_processor.process_event(event_dict)
                    processed_events.append(processed_event)
                    self.pipeline_stats["events_processed"] += 1
                except Exception as e:
                    logger.error(f"Error processing event: {e}")
                    self.pipeline_stats["errors"] += 1
            
            # Store features in feature store
            for processed_event in processed_events:
                self._store_event_features(processed_event)
            
            # Send processed events to output topic
            for processed_event in processed_events:
                self.kafka_producer.send_event('processed-events', processed_event)
            
            logger.debug(
                f"Pipeline iteration completed",
                extra={
                    "events_generated": len(events),
                    "events_processed": len(processed_events),
                    "total_events": self.pipeline_stats["events_generated"]
                }
            )
            
        except Exception as e:
            logger.error(f"Error in pipeline iteration: {e}")
            self.pipeline_stats["errors"] += 1
    
    def _store_event_features(self, processed_event: Dict[str, Any]) -> None:
        """Store features from processed event.
        
        Args:
            processed_event: Processed event with features
        """
        try:
            user_id = processed_event.get('user_id')
            product_id = processed_event.get('product_id')
            features = processed_event.get('features', {})
            
            # Store user features
            if user_id and features:
                user_features = self._extract_user_features(processed_event)
                if user_features:
                    self.feature_store.store_user_features(user_id, user_features)
                    self.pipeline_stats["features_stored"] += 1
            
            # Store product features
            if product_id and features:
                product_features = self._extract_product_features(processed_event)
                if product_features:
                    self.feature_store.store_product_features(product_id, product_features)
                    self.pipeline_stats["features_stored"] += 1
                    
        except Exception as e:
            logger.error(f"Error storing event features: {e}")
            self.pipeline_stats["errors"] += 1
    
    def _extract_user_features(self, processed_event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract user features from processed event.
        
        Args:
            processed_event: Processed event
            
        Returns:
            User features dictionary
        """
        features = processed_event.get('features', {})
        
        user_features = {
            'total_events': 1,  # This would be aggregated in real implementation
            'last_event_type': processed_event.get('event_type'),
            'last_device_type': features.get('device_mobile', 0) or features.get('device_desktop', 0),
            'last_activity': processed_event.get('timestamp'),
            'engagement_score': self._calculate_engagement_score(features)
        }
        
        return user_features
    
    def _extract_product_features(self, processed_event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract product features from processed event.
        
        Args:
            processed_event: Processed event
            
        Returns:
            Product features dictionary
        """
        features = processed_event.get('features', {})
        
        product_features = {
            'view_count': 1 if processed_event.get('event_type') == 'view' else 0,
            'click_count': 1 if processed_event.get('event_type') == 'click' else 0,
            'last_activity': processed_event.get('timestamp'),
            'popularity_score': self._calculate_popularity_score(features)
        }
        
        return product_features
    
    def _calculate_engagement_score(self, features: Dict[str, Any]) -> float:
        """Calculate user engagement score.
        
        Args:
            features: Event features
            
        Returns:
            Engagement score (0-1)
        """
        score = 0.0
        
        # Dwell time contribution
        dwell_time = features.get('dwell_time_seconds', 0)
        if dwell_time > 0:
            score += min(dwell_time / 300.0, 1.0) * 0.4  # Max 40% from dwell time
        
        # Scroll depth contribution
        scroll_depth = features.get('scroll_depth_percentage', 0)
        if scroll_depth > 0:
            score += (scroll_depth / 100.0) * 0.3  # Max 30% from scroll depth
        
        # Event type contribution
        event_type = features.get('event_type_purchase', 0)
        if event_type:
            score += 0.3  # 30% bonus for purchase events
        
        return min(score, 1.0)
    
    def _calculate_popularity_score(self, features: Dict[str, Any]) -> float:
        """Calculate product popularity score.
        
        Args:
            features: Event features
            
        Returns:
            Popularity score (0-1)
        """
        score = 0.0
        
        # Event type weights
        event_weights = {
            'event_type_view': 0.1,
            'event_type_click': 0.3,
            'event_type_add_to_cart': 0.5,
            'event_type_purchase': 1.0
        }
        
        for event_type, weight in event_weights.items():
            if features.get(event_type, 0):
                score += weight
        
        return min(score, 1.0)
    
    def _generate_pipeline_report(self) -> Dict[str, Any]:
        """Generate final pipeline execution report.
        
        Returns:
            Pipeline execution report
        """
        report = self.pipeline_stats.copy()
        
        # Calculate execution time
        if report["start_time"] and report["end_time"]:
            execution_time = (report["end_time"] - report["start_time"]).total_seconds()
            report["execution_time_seconds"] = execution_time
            report["events_per_second"] = round(report["events_generated"] / execution_time, 2)
        
        # Add component stats
        if self.flink_processor:
            report["flink_stats"] = self.flink_processor.get_processing_stats()
        
        if self.feature_store:
            report["feature_store_stats"] = self.feature_store.get_feature_store_stats()
        
        # Calculate success rate
        total_operations = report["events_generated"] + report["errors"]
        if total_operations > 0:
            report["success_rate"] = round((total_operations - report["errors"]) / total_operations * 100, 2)
        else:
            report["success_rate"] = 100.0
        
        return report
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status.
        
        Returns:
            Pipeline status dictionary
        """
        status = {
            "is_running": self.is_running,
            "components_status": {},
            "current_stats": self.pipeline_stats.copy()
        }
        
        # Check component status
        status["components_status"]["kafka_producer"] = self.kafka_producer is not None
        status["components_status"]["kafka_consumer"] = self.kafka_consumer is not None
        status["components_status"]["flink_processor"] = self.flink_processor is not None
        status["components_status"]["feature_store"] = self.feature_store is not None
        status["components_status"]["clickstream_simulator"] = self.clickstream_simulator is not None
        
        return status


def main():
    """Test the Phase 2 pipeline."""
    # Configuration
    config = {
        'kafka_bootstrap_servers': 'localhost:9092',
        'kafka_group_id': 'recommndr-phase2-test',
        'redis_host': 'localhost',
        'redis_port': 6380
    }
    
    # Initialize pipeline
    pipeline = Phase2Pipeline(config)
    
    try:
        # Setup components
        if not pipeline.setup_components():
            print("❌ Failed to setup pipeline components")
            return
        
        print("✅ Pipeline components setup complete")
        
        # Run pipeline for 30 seconds
        print("🚀 Starting pipeline...")
        results = pipeline.run_pipeline(duration_seconds=30, events_per_second=5)
        
        # Display results
        print("\n📊 Pipeline Results:")
        print(f"Events Generated: {results.get('events_generated', 0)}")
        print(f"Events Processed: {results.get('events_processed', 0)}")
        print(f"Features Stored: {results.get('features_stored', 0)}")
        print(f"Success Rate: {results.get('success_rate', 0)}%")
        print(f"Execution Time: {results.get('execution_time_seconds', 0)}s")
        
        if 'flink_stats' in results:
            print(f"\n🔧 Flink Stats:")
            flink_stats = results['flink_stats']
            print(f"  Events Processed: {flink_stats.get('events_processed', 0)}")
            print(f"  Features Generated: {flink_stats.get('features_generated', 0)}")
            print(f"  Unique Users: {flink_stats.get('unique_users', 0)}")
            print(f"  Unique Products: {flink_stats.get('unique_products', 0)}")
        
        if 'feature_store_stats' in results:
            print(f"\n🏪 Feature Store Stats:")
            store_stats = results['feature_store_stats']
            print(f"  Total Keys: {store_stats.get('total_keys', 0)}")
            print(f"  User Features: {store_stats.get('user_features_count', 0)}")
            print(f"  Product Features: {store_stats.get('product_features_count', 0)}")
        
        print("\n🎉 Phase 2 pipeline test completed!")
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        logger.error(f"Pipeline test failed: {e}")


if __name__ == "__main__":
    main()
