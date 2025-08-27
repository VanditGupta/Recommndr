"""Flink processor for streaming clickstream events."""

import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from src.utils.logging import get_logger

logger = get_logger(__name__)


class FlinkStreamProcessor:
    """Simulates Flink stream processing for Phase 2."""
    
    def __init__(self):
        """Initialize Flink stream processor."""
        self.processing_stats = {
            "events_processed": 0,
            "features_generated": 0,
            "processing_time_ms": 0,
            "errors": 0
        }
        
        # Feature aggregation windows
        self.user_session_features = {}
        self.product_features = {}
        self.global_features = {}
        
        # Time windows for aggregation
        self.session_window_minutes = 30
        self.product_window_minutes = 60
        self.global_window_minutes = 120
        
        logger.info("Flink stream processor initialized")
    
    def process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single clickstream event.
        
        Args:
            event: Clickstream event to process
            
        Returns:
            Processed event with features
        """
        start_time = time.time()
        
        try:
            # Extract event details
            user_id = event.get('user_id')
            product_id = event.get('product_id')
            event_type = event.get('event_type')
            timestamp = event.get('timestamp')
            session_id = event.get('session_id')
            
            # Generate features
            features = self._generate_features(event)
            
            # Update aggregations
            self._update_user_features(user_id, event, features)
            self._update_product_features(product_id, event, features)
            self._update_global_features(event, features)
            
            # Create processed event
            processed_event = {
                **event,
                "features": features,
                "processed_at": datetime.now().isoformat(),
                "processing_latency_ms": round((time.time() - start_time) * 1000, 2)
            }
            
            # Update stats
            self.processing_stats["events_processed"] += 1
            self.processing_stats["features_generated"] += len(features)
            self.processing_stats["processing_time_ms"] += round((time.time() - start_time) * 1000, 2)
            
            logger.debug(
                f"Event processed successfully",
                extra={
                    "event_id": event.get('event_id'),
                    "user_id": user_id,
                    "features_count": len(features),
                    "processing_time_ms": round((time.time() - start_time) * 1000, 2)
                }
            )
            
            return processed_event
            
        except Exception as e:
            self.processing_stats["errors"] += 1
            logger.error(f"Error processing event: {e}")
            raise
    
    def _generate_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Generate features from event data.
        
        Args:
            event: Raw event data
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Event type features
        event_type = event.get('event_type', '')
        features['event_type_view'] = 1 if event_type == 'view' else 0
        features['event_type_click'] = 1 if event_type == 'click' else 0
        features['event_type_add_to_cart'] = 1 if event_type == 'add_to_cart' else 0
        features['event_type_purchase'] = 1 if event_type == 'purchase' else 0
        
        # Temporal features
        timestamp = event.get('timestamp')
        if timestamp:
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                dt = timestamp
            
            features['hour_of_day'] = dt.hour
            features['day_of_week'] = dt.weekday()
            features['is_weekend'] = 1 if dt.weekday() >= 5 else 0
            features['is_business_hours'] = 1 if 9 <= dt.hour <= 17 else 0
        
        # User interaction features
        dwell_time = event.get('dwell_time')
        if dwell_time:
            features['dwell_time_seconds'] = dwell_time
            features['dwell_time_category'] = self._categorize_dwell_time(dwell_time)
        
        scroll_depth = event.get('scroll_depth')
        if scroll_depth:
            features['scroll_depth_percentage'] = scroll_depth
            features['scroll_depth_category'] = self._categorize_scroll_depth(scroll_depth)
        
        # Device features
        device_type = event.get('device_type', '')
        features['device_mobile'] = 1 if device_type == 'mobile' else 0
        features['device_desktop'] = 1 if device_type == 'desktop' else 0
        features['device_tablet'] = 1 if device_type == 'tablet' else 0
        
        return features
    
    def _categorize_dwell_time(self, dwell_time: int) -> str:
        """Categorize dwell time into buckets.
        
        Args:
            dwell_time: Dwell time in seconds
            
        Returns:
            Dwell time category
        """
        if dwell_time < 10:
            return 'very_short'
        elif dwell_time < 30:
            return 'short'
        elif dwell_time < 120:
            return 'medium'
        elif dwell_time < 300:
            return 'long'
        else:
            return 'very_long'
    
    def _categorize_scroll_depth(self, scroll_depth: int) -> str:
        """Categorize scroll depth into buckets.
        
        Args:
            scroll_depth: Scroll depth percentage
            
        Returns:
            Scroll depth category
        """
        if scroll_depth < 25:
            return 'low'
        elif scroll_depth < 50:
            return 'medium'
        elif scroll_depth < 75:
            return 'high'
        else:
            return 'very_high'
    
    def _update_user_features(self, user_id: Any, event: Dict[str, Any], features: Dict[str, Any]) -> None:
        """Update user-level feature aggregations.
        
        Args:
            user_id: User identifier
            event: Event data
            features: Generated features
        """
        if user_id not in self.user_session_features:
            self.user_session_features[user_id] = {
                'session_count': 0,
                'total_events': 0,
                'event_types': {},
                'total_dwell_time': 0,
                'total_scroll_depth': 0,
                'last_activity': None
            }
        
        user_data = self.user_session_features[user_id]
        user_data['total_events'] += 1
        user_data['last_activity'] = event.get('timestamp')
        
        # Update event type counts
        event_type = event.get('event_type', 'unknown')
        user_data['event_types'][event_type] = user_data['event_types'].get(event_type, 0) + 1
        
        # Update dwell time and scroll depth
        if 'dwell_time_seconds' in features:
            user_data['total_dwell_time'] += features['dwell_time_seconds']
        if 'scroll_depth_percentage' in features:
            user_data['total_scroll_depth'] += features['scroll_depth_percentage']
    
    def _update_product_features(self, product_id: Any, event: Dict[str, Any], features: Dict[str, Any]) -> None:
        """Update product-level feature aggregations.
        
        Args:
            product_id: Product identifier
            event: Event data
            features: Generated features
        """
        if product_id not in self.product_features:
            self.product_features[product_id] = {
                'view_count': 0,
                'click_count': 0,
                'add_to_cart_count': 0,
                'purchase_count': 0,
                'total_dwell_time': 0,
                'total_scroll_depth': 0,
                'unique_users': set(),
                'last_activity': None
            }
        
        product_data = self.product_features[product_id]
        product_data['last_activity'] = event.get('timestamp')
        
        # Update event type counts
        event_type = event.get('event_type', '')
        if event_type == 'view':
            product_data['view_count'] += 1
        elif event_type == 'click':
            product_data['click_count'] += 1
        elif event_type == 'add_to_cart':
            product_data['add_to_cart_count'] += 1
        elif event_type == 'purchase':
            product_data['purchase_count'] += 1
        
        # Update user tracking
        user_id = event.get('user_id')
        if user_id:
            product_data['unique_users'].add(user_id)
        
        # Update dwell time and scroll depth
        if 'dwell_time_seconds' in features:
            product_data['total_dwell_time'] += features['dwell_time_seconds']
        if 'scroll_depth_percentage' in features:
            product_data['total_scroll_depth'] += features['scroll_depth_percentage']
    
    def _update_global_features(self, event: Dict[str, Any], features: Dict[str, Any]) -> None:
        """Update global feature aggregations.
        
        Args:
            event: Event data
            features: Generated features
        """
        if 'total_events' not in self.global_features:
            self.global_features = {
                'total_events': 0,
                'total_users': set(),
                'total_products': set(),
                'event_type_distribution': {},
                'device_distribution': {},
                'hourly_distribution': {},
                'last_activity': None
            }
        
        global_data = self.global_features
        global_data['total_events'] += 1
        global_data['last_activity'] = event.get('timestamp')
        
        # Update user and product counts
        user_id = event.get('user_id')
        product_id = event.get('product_id')
        if user_id:
            global_data['total_users'].add(user_id)
        if product_id:
            global_data['total_products'].add(product_id)
        
        # Update event type distribution
        event_type = event.get('event_type', 'unknown')
        global_data['event_type_distribution'][event_type] = global_data['event_type_distribution'].get(event_type, 0) + 1
        
        # Update device distribution
        device_type = event.get('device_type', 'unknown')
        global_data['device_distribution'][device_type] = global_data['device_distribution'].get(device_type, 0) + 1
        
        # Update hourly distribution
        timestamp = event.get('timestamp')
        if timestamp:
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                dt = timestamp
            
            hour = dt.hour
            global_data['hourly_distribution'][hour] = global_data['hourly_distribution'].get(hour, 0) + 1
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        stats = self.processing_stats.copy()
        
        # Add aggregation stats
        stats['unique_users'] = len(self.user_session_features)
        stats['unique_products'] = len(self.product_features)
        stats['total_sessions'] = sum(1 for user in self.user_session_features.values() if user['session_count'] > 0)
        
        # Calculate averages
        if stats['events_processed'] > 0:
            stats['avg_processing_time_ms'] = round(stats['processing_time_ms'] / stats['events_processed'], 2)
            stats['avg_features_per_event'] = round(stats['features_generated'] / stats['events_processed'], 2)
        
        return stats
    
    def get_user_features(self, user_id: Any) -> Dict[str, Any]:
        """Get features for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            User features dictionary
        """
        return self.user_session_features.get(user_id, {})
    
    def get_product_features(self, product_id: Any) -> Dict[str, Any]:
        """Get features for a specific product.
        
        Args:
            product_id: Product identifier
            
        Returns:
            Product features dictionary
        """
        product_data = self.product_features.get(product_id, {})
        
        # Convert set to count for JSON serialization
        if 'unique_users' in product_data:
            product_data['unique_user_count'] = len(product_data['unique_users'])
            product_data['unique_users'] = list(product_data['unique_users'])
        
        return product_data
    
    def get_global_features(self) -> Dict[str, Any]:
        """Get global feature aggregations.
        
        Returns:
            Global features dictionary
        """
        global_data = self.global_features.copy()
        
        # Convert sets to counts for JSON serialization
        if 'total_users' in global_data:
            global_data['total_user_count'] = len(global_data['total_users'])
            global_data['total_users'] = list(global_data['total_users'])
        
        if 'total_products' in global_data:
            global_data['total_product_count'] = len(global_data['total_products'])
            global_data['total_products'] = list(global_data['total_products'])
        
        return global_data


def main():
    """Test the Flink stream processor."""
    processor = FlinkStreamProcessor()
    
    # Test events
    test_events = [
        {
            "event_id": "test_1",
            "user_id": 123,
            "product_id": 456,
            "event_type": "view",
            "timestamp": datetime.now().isoformat(),
            "session_id": "session_123",
            "dwell_time": 45,
            "scroll_depth": 75,
            "device_type": "desktop"
        },
        {
            "event_id": "test_2",
            "user_id": 123,
            "product_id": 456,
            "event_type": "click",
            "timestamp": datetime.now().isoformat(),
            "session_id": "session_123",
            "dwell_time": 120,
            "scroll_depth": 90,
            "device_type": "desktop"
        }
    ]
    
    # Process events
    for event in test_events:
        processed_event = processor.process_event(event)
        print(f"✅ Processed: {processed_event['event_id']}")
    
    # Get stats
    stats = processor.get_processing_stats()
    print(f"📊 Processing stats: {stats}")
    
    # Get features
    user_features = processor.get_user_features(123)
    print(f"👤 User features: {user_features}")
    
    product_features = processor.get_product_features(456)
    print(f"🛍️ Product features: {product_features}")
    
    global_features = processor.get_global_features()
    print(f"🌍 Global features: {global_features}")


if __name__ == "__main__":
    main()
