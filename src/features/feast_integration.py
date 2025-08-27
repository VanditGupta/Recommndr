"""Feast integration for feature store and feature serving."""

import json
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import redis

from src.utils.logging import get_logger

logger = get_logger(__name__)


class FeastFeatureStore:
    """Feature store using Feast for real-time feature serving."""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6380):
        """Initialize Feast feature store.
        
        Args:
            redis_host: Redis host
            redis_port: Redis port
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_client = None
        self._connect()
        
        # Feature definitions
        self.feature_definitions = {
            "user_features": [
                "total_events", "session_count", "avg_dwell_time", "avg_scroll_depth",
                "event_type_view_count", "event_type_click_count", "event_type_purchase_count",
                "device_preference", "activity_recency", "engagement_score"
            ],
            "product_features": [
                "view_count", "click_count", "add_to_cart_count", "purchase_count",
                "conversion_rate", "avg_dwell_time", "avg_scroll_depth", "unique_user_count",
                "popularity_score", "trending_score"
            ],
            "interaction_features": [
                "event_type", "dwell_time", "scroll_depth", "device_type",
                "hour_of_day", "day_of_week", "is_weekend", "is_business_hours",
                "session_duration", "page_sequence"
            ]
        }
        
        logger.info("Feast feature store initialized")
    
    def _connect(self) -> None:
        """Connect to Redis."""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.redis_host}:{self.redis_port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def store_user_features(self, user_id: Union[str, int], features: Dict[str, Any], ttl_seconds: int = 3600) -> bool:
        """Store user features in the feature store.
        
        Args:
            user_id: User identifier
            features: User features dictionary
            ttl_seconds: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            key = f"user_features:{user_id}"
            
            # Add metadata
            features_with_metadata = {
                **features,
                "stored_at": datetime.now().isoformat(),
                "ttl_seconds": ttl_seconds
            }
            
            # Store in Redis
            self.redis_client.setex(
                key,
                ttl_seconds,
                json.dumps(features_with_metadata)
            )
            
            logger.debug(
                f"User features stored",
                extra={
                    "user_id": user_id,
                    "features_count": len(features),
                    "ttl_seconds": ttl_seconds
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store user features: {e}")
            return False
    
    def store_product_features(self, product_id: Union[str, int], features: Dict[str, Any], ttl_seconds: int = 7200) -> bool:
        """Store product features in the feature store.
        
        Args:
            product_id: Product identifier
            features: Product features dictionary
            ttl_seconds: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            key = f"product_features:{product_id}"
            
            # Add metadata
            features_with_metadata = {
                **features,
                "stored_at": datetime.now().isoformat(),
                "ttl_seconds": ttl_seconds
            }
            
            # Store in Redis
            self.redis_client.setex(
                key,
                ttl_seconds,
                json.dumps(features_with_metadata)
            )
            
            logger.debug(
                f"Product features stored",
                extra={
                    "product_id": product_id,
                    "features_count": len(features),
                    "ttl_seconds": ttl_seconds
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store product features: {e}")
            return False
    
    def get_user_features(self, user_id: Union[str, int]) -> Optional[Dict[str, Any]]:
        """Get user features from the feature store.
        
        Args:
            user_id: User identifier
            
        Returns:
            User features dictionary or None if not found
        """
        try:
            key = f"user_features:{user_id}"
            features_json = self.redis_client.get(key)
            
            if features_json:
                features = json.loads(features_json)
                logger.debug(f"User features retrieved for user {user_id}")
                return features
            else:
                logger.debug(f"No features found for user {user_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get user features: {e}")
            return None
    
    def get_product_features(self, product_id: Union[str, int]) -> Optional[Dict[str, Any]]:
        """Get product features from the feature store.
        
        Args:
            product_id: Product identifier
            
        Returns:
            Product features dictionary or None if not found
        """
        try:
            key = f"product_features:{product_id}"
            features_json = self.redis_client.get(key)
            
            if features_json:
                features = json.loads(features_json)
                logger.debug(f"Product features retrieved for product {product_id}")
                return features
            else:
                logger.debug(f"No features found for product {product_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get product features: {e}")
            return None
    
    def get_features_batch(self, entity_type: str, entity_ids: List[Union[str, int]]) -> Dict[Union[str, int], Dict[str, Any]]:
        """Get features for multiple entities in batch.
        
        Args:
            entity_type: Type of entity ('user' or 'product')
            entity_ids: List of entity identifiers
            
        Returns:
            Dictionary mapping entity IDs to their features
        """
        try:
            batch_features = {}
            
            # Use Redis pipeline for batch operations
            with self.redis_client.pipeline() as pipe:
                for entity_id in entity_ids:
                    key = f"{entity_type}_features:{entity_id}"
                    pipe.get(key)
                
                # Execute pipeline
                results = pipe.execute()
                
                # Process results
                for entity_id, result in zip(entity_ids, results):
                    if result:
                        features = json.loads(result)
                        batch_features[entity_id] = features
                    else:
                        batch_features[entity_id] = None
            
            logger.debug(
                f"Batch features retrieved",
                extra={
                    "entity_type": entity_type,
                    "requested_count": len(entity_ids),
                    "found_count": sum(1 for f in batch_features.values() if f is not None)
                }
            )
            
            return batch_features
            
        except Exception as e:
            logger.error(f"Failed to get batch features: {e}")
            return {}
    
    def update_feature_metadata(self, entity_type: str, entity_id: Union[str, int], metadata: Dict[str, Any]) -> bool:
        """Update metadata for features.
        
        Args:
            entity_type: Type of entity ('user' or 'product')
            entity_id: Entity identifier
            metadata: Metadata to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            key = f"{entity_type}_features:{entity_id}"
            features_json = self.redis_client.get(key)
            
            if features_json:
                features = json.loads(features_json)
                features.update(metadata)
                features["updated_at"] = datetime.now().isoformat()
                
                # Get TTL
                ttl = self.redis_client.ttl(key)
                if ttl > 0:
                    # Update with new data
                    self.redis_client.setex(key, ttl, json.dumps(features))
                    logger.debug(f"Feature metadata updated for {entity_type} {entity_id}")
                    return True
                else:
                    logger.warning(f"Feature expired for {entity_type} {entity_id}")
                    return False
            else:
                logger.warning(f"No features found for {entity_type} {entity_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update feature metadata: {e}")
            return False
    
    def get_feature_store_stats(self) -> Dict[str, Any]:
        """Get feature store statistics.
        
        Returns:
            Dictionary with feature store statistics
        """
        try:
            stats = {
                "total_keys": 0,
                "user_features_count": 0,
                "product_features_count": 0,
                "memory_usage_bytes": 0,
                "connected_clients": 0
            }
            
            # Get Redis info
            info = self.redis_client.info()
            stats["memory_usage_bytes"] = info.get("used_memory", 0)
            stats["connected_clients"] = info.get("connected_clients", 0)
            
            # Count keys by pattern
            user_keys = self.redis_client.keys("user_features:*")
            product_keys = self.redis_client.keys("product_features:*")
            
            stats["user_features_count"] = len(user_keys)
            stats["product_features_count"] = len(product_keys)
            stats["total_keys"] = stats["user_features_count"] + stats["product_features_count"]
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get feature store stats: {e}")
            return {}
    
    def cleanup_expired_features(self) -> Dict[str, int]:
        """Clean up expired features.
        
        Returns:
            Dictionary with cleanup statistics
        """
        try:
            cleanup_stats = {
                "user_features_cleaned": 0,
                "product_features_cleaned": 0,
                "total_cleaned": 0
            }
            
            # Get all keys
            user_keys = self.redis_client.keys("user_features:*")
            product_keys = self.redis_client.keys("product_features:*")
            
            # Check TTL for each key
            for key in user_keys:
                ttl = self.redis_client.ttl(key)
                if ttl <= 0:
                    self.redis_client.delete(key)
                    cleanup_stats["user_features_cleaned"] += 1
            
            for key in product_keys:
                ttl = self.redis_client.ttl(key)
                if ttl <= 0:
                    self.redis_client.delete(key)
                    cleanup_stats["product_features_cleaned"] += 1
            
            cleanup_stats["total_cleaned"] = (
                cleanup_stats["user_features_cleaned"] + 
                cleanup_stats["product_features_cleaned"]
            )
            
            if cleanup_stats["total_cleaned"] > 0:
                logger.info(
                    f"Cleanup completed",
                    extra=cleanup_stats
                )
            
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired features: {e}")
            return {}
    
    def close(self) -> None:
        """Close Redis connection."""
        if self.redis_client:
            self.redis_client.close()
            logger.info("Feast feature store connection closed")


def main():
    """Test the Feast feature store."""
    feature_store = FeastFeatureStore()
    
    try:
        # Test user features
        user_features = {
            "total_events": 25,
            "session_count": 3,
            "avg_dwell_time": 45.2,
            "avg_scroll_depth": 67.8,
            "engagement_score": 0.85
        }
        
        # Store user features
        success = feature_store.store_user_features(123, user_features)
        print(f"✅ User features stored: {success}")
        
        # Retrieve user features
        retrieved_features = feature_store.get_user_features(123)
        print(f"📥 Retrieved features: {retrieved_features}")
        
        # Test product features
        product_features = {
            "view_count": 150,
            "click_count": 45,
            "conversion_rate": 0.30,
            "popularity_score": 0.78
        }
        
        # Store product features
        success = feature_store.store_product_features(456, product_features)
        print(f"✅ Product features stored: {success}")
        
        # Get batch features
        batch_features = feature_store.get_features_batch("user", [123, 124, 125])
        print(f"📦 Batch features: {batch_features}")
        
        # Get store stats
        stats = feature_store.get_feature_store_stats()
        print(f"📊 Store stats: {stats}")
        
    finally:
        feature_store.close()


if __name__ == "__main__":
    main()
