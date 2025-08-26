"""Feature engineering pipeline for Phase 2."""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import redis
from pydantic import BaseModel, Field

from src.utils.logging import get_logger

logger = get_logger(__name__)


class UserFeature(BaseModel):
    """User feature model."""
    
    user_id: int = Field(..., description="User ID")
    feature_vector: Dict[str, float] = Field(..., description="Feature vector")
    last_updated: datetime = Field(..., description="Last update timestamp")
    feature_version: str = Field(..., description="Feature version")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ProductFeature(BaseModel):
    """Product feature model."""
    
    product_id: int = Field(..., description="Product ID")
    feature_vector: Dict[str, float] = Field(..., description="Feature vector")
    last_updated: datetime = Field(..., description="Last update timestamp")
    feature_version: str = Field(..., description="Feature version")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class FeatureEngineeringPipeline:
    """Feature engineering pipeline for Phase 2."""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        """Initialize feature engineering pipeline.
        
        Args:
            redis_host: Redis host
            redis_port: Redis port
        """
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
        
        # Feature version for tracking
        self.feature_version = f"v1.0.{int(time.time())}"
        
        # Feature names
        self.user_features = [
            "total_views", "total_clicks", "total_adds_to_cart", "total_purchases",
            "avg_dwell_time", "avg_scroll_depth", "session_count", "avg_session_duration",
            "category_preferences", "price_sensitivity", "device_preference_mobile",
            "device_preference_desktop", "device_preference_tablet", "location_preferences"
        ]
        
        self.product_features = [
            "total_views", "total_clicks", "total_adds_to_cart", "total_purchases",
            "view_to_click_rate", "click_to_cart_rate", "cart_to_purchase_rate",
            "avg_dwell_time", "avg_scroll_depth", "category_popularity",
            "price_tier", "seasonal_trend", "user_engagement_score"
        ]
        
        logger.info("Feature engineering pipeline initialized")
    
    def process_clickstream_event(self, event_data: Dict) -> None:
        """Process a single clickstream event and update features.
        
        Args:
            event_data: Clickstream event data
        """
        try:
            user_id = event_data["user_id"]
            product_id = event_data["product_id"]
            event_type = event_data["event_type"]
            timestamp = datetime.fromisoformat(event_data["timestamp"])
            
            # Update user features
            self._update_user_features(user_id, event_type, event_data, timestamp)
            
            # Update product features
            self._update_product_features(product_id, event_type, event_data, timestamp)
            
            logger.debug(
                f"Event processed successfully",
                extra={
                    "user_id": user_id,
                    "product_id": product_id,
                    "event_type": event_type
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing event: {e}")
            raise
    
    def _update_user_features(
        self, 
        user_id: int, 
        event_type: str, 
        event_data: Dict, 
        timestamp: datetime
    ) -> None:
        """Update user features based on event.
        
        Args:
            user_id: User ID
            event_type: Event type
            event_data: Event data
            timestamp: Event timestamp
        """
        # Get existing user features
        user_key = f"user_features:{user_id}"
        existing_features = self.redis_client.hgetall(user_key)
        
        if not existing_features:
            # Initialize new user features
            features = {feature: 0.0 for feature in self.user_features}
            features["last_updated"] = timestamp.isoformat()
            features["feature_version"] = self.feature_version
        else:
            # Parse existing features
            features = {}
            for key, value in existing_features.items():
                if key in ["last_updated", "feature_version"]:
                    features[key] = value
                else:
                    features[key] = float(value) if value else 0.0
        
        # Update features based on event type
        if event_type == "view":
            features["total_views"] += 1
            if "dwell_time" in event_data and event_data["dwell_time"]:
                features["avg_dwell_time"] = self._update_average(
                    features["avg_dwell_time"], 
                    event_data["dwell_time"], 
                    features["total_views"]
                )
            if "scroll_depth" in event_data and event_data["scroll_depth"]:
                features["avg_scroll_depth"] = self._update_average(
                    features["avg_scroll_depth"], 
                    event_data["scroll_depth"], 
                    features["total_views"]
                )
        
        elif event_type == "click":
            features["total_clicks"] += 1
        
        elif event_type == "add_to_cart":
            features["total_adds_to_cart"] += 1
        
        elif event_type == "purchase":
            features["total_purchases"] += 1
        
        # Update device preference
        device_type = event_data.get("device_type", "desktop")
        if device_type == "mobile":
            features["device_preference_mobile"] += 1
        elif device_type == "desktop":
            features["device_preference_desktop"] += 1
        elif device_type == "tablet":
            features["device_preference_tablet"] += 1
        
        # Update location preferences
        location = event_data.get("location", "Unknown")
        if "location_preferences" not in features:
            features["location_preferences"] = {}
        if location not in features["location_preferences"]:
            features["location_preferences"][location] = 0
        features["location_preferences"][location] += 1
        
        # Update timestamp and version
        features["last_updated"] = timestamp.isoformat()
        features["feature_version"] = self.feature_version
        
        # Store updated features
        self.redis_client.hset(user_key, mapping=features)
        
        # Set expiration (30 days)
        self.redis_client.expire(user_key, 30 * 24 * 3600)
    
    def _update_product_features(
        self, 
        product_id: int, 
        event_type: str, 
        event_data: Dict, 
        timestamp: datetime
    ) -> None:
        """Update product features based on event.
        
        Args:
            product_id: Product ID
            event_type: Event type
            event_data: Event data
            timestamp: Event timestamp
        """
        # Get existing product features
        product_key = f"product_features:{product_id}"
        existing_features = self.redis_client.hgetall(product_key)
        
        if not existing_features:
            # Initialize new product features
            features = {feature: 0.0 for feature in self.product_features}
            features["last_updated"] = timestamp.isoformat()
            features["feature_version"] = self.feature_version
        else:
            # Parse existing features
            features = {}
            for key, value in existing_features.items():
                if key in ["last_updated", "feature_version"]:
                    features[key] = value
                else:
                    features[key] = float(value) if value else 0.0
        
        # Update features based on event type
        if event_type == "view":
            features["total_views"] += 1
            if "dwell_time" in event_data and event_data["dwell_time"]:
                features["avg_dwell_time"] = self._update_average(
                    features["avg_dwell_time"], 
                    event_data["dwell_time"], 
                    features["total_views"]
                )
            if "scroll_depth" in event_data and event_data["scroll_depth"]:
                features["avg_scroll_depth"] = self._update_average(
                    features["avg_scroll_depth"], 
                    event_data["scroll_depth"], 
                    features["total_views"]
                )
        
        elif event_type == "click":
            features["total_clicks"] += 1
        
        elif event_type == "add_to_cart":
            features["total_adds_to_cart"] += 1
        
        elif event_type == "purchase":
            features["total_purchases"] += 1
        
        # Calculate derived features
        if features["total_views"] > 0:
            features["view_to_click_rate"] = features["total_clicks"] / features["total_views"]
        
        if features["total_clicks"] > 0:
            features["click_to_cart_rate"] = features["total_adds_to_cart"] / features["total_clicks"]
        
        if features["total_adds_to_cart"] > 0:
            features["cart_to_purchase_rate"] = features["total_purchases"] / features["total_adds_to_cart"]
        
        # Calculate user engagement score
        engagement_score = (
            features["total_views"] * 0.1 +
            features["total_clicks"] * 0.3 +
            features["total_adds_to_cart"] * 0.5 +
            features["total_purchases"] * 1.0
        )
        features["user_engagement_score"] = engagement_score
        
        # Update timestamp and version
        features["last_updated"] = timestamp.isoformat()
        features["feature_version"] = self.feature_version
        
        # Store updated features
        self.redis_client.hset(product_key, mapping=features)
        
        # Set expiration (30 days)
        self.redis_client.expire(product_key, 30 * 24 * 3600)
    
    def _update_average(self, current_avg: float, new_value: float, count: int) -> float:
        """Update running average.
        
        Args:
            current_avg: Current average
            new_value: New value
            count: Total count
            
        Returns:
            Updated average
        """
        if count == 1:
            return new_value
        return (current_avg * (count - 1) + new_value) / count
    
    def get_user_features(self, user_id: int) -> Optional[UserFeature]:
        """Get user features.
        
        Args:
            user_id: User ID
            
        Returns:
            UserFeature or None if not found
        """
        user_key = f"user_features:{user_id}"
        features_data = self.redis_client.hgetall(user_key)
        
        if not features_data:
            return None
        
        # Parse features
        feature_vector = {}
        for key, value in features_data.items():
            if key not in ["last_updated", "feature_version"]:
                feature_vector[key] = float(value) if value else 0.0
        
        return UserFeature(
            user_id=user_id,
            feature_vector=feature_vector,
            last_updated=datetime.fromisoformat(features_data["last_updated"]),
            feature_version=features_data["feature_version"]
        )
    
    def get_product_features(self, product_id: int) -> Optional[ProductFeature]:
        """Get product features.
        
        Args:
            product_id: Product ID
            
        Returns:
            ProductFeature or None if not found
        """
        product_key = f"product_features:{product_id}"
        features_data = self.redis_client.hgetall(product_key)
        
        if not features_data:
            return None
        
        # Parse features
        feature_vector = {}
        for key, value in features_data.items():
            if key not in ["last_updated", "feature_version"]:
                feature_vector[key] = float(value) if value else 0.0
        
        return ProductFeature(
            product_id=product_id,
            feature_vector=feature_vector,
            last_updated=datetime.fromisoformat(features_data["last_updated"]),
            feature_version=features_data["feature_version"]
        )
    
    def get_feature_stats(self) -> Dict[str, int]:
        """Get feature statistics.
        
        Returns:
            Dictionary with feature counts
        """
        # Count user features
        user_keys = self.redis_client.keys("user_features:*")
        user_count = len(user_keys)
        
        # Count product features
        product_keys = self.redis_client.keys("product_features:*")
        product_count = len(product_keys)
        
        return {
            "total_users": user_count,
            "total_products": product_count,
            "feature_version": self.feature_version
        }
    
    def close(self) -> None:
        """Close Redis connection."""
        self.redis_client.close()
        logger.info("Feature engineering pipeline closed")


def main():
    """Main function for testing."""
    pipeline = FeatureEngineeringPipeline()
    
    try:
        # Test feature stats
        stats = pipeline.get_feature_stats()
        print(f"Feature stats: {stats}")
        
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
