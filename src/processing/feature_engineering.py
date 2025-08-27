"""Feature engineering for recommendation system."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

from config.settings import settings
from src.utils.logging import get_logger, log_performance_metrics

logger = get_logger(__name__)


class FeatureEngineer:
    """Create features for recommendation models."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.label_encoders = {}
        self.scalers = {}
        self.feature_stats = {}
        
    def create_user_features(self, users_df: pd.DataFrame, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Create user features for recommendation models."""
        logger.info("Creating user features")
        start_time = datetime.now()
        
        user_features = users_df.copy()
        
        # User interaction features
        user_interactions = interactions_df.groupby('user_id').agg({
            'interaction_id': 'count',
            'rating': ['mean', 'count'],
            'total_amount': ['sum', 'mean'],
            'dwell_time': ['mean', 'sum'],
            'scroll_depth': ['mean', 'max'],
            'timestamp': ['min', 'max']
        }).reset_index()
        
        # Flatten column names
        user_interactions.columns = [
            'user_id', 'total_interactions', 'avg_rating', 'rating_count',
            'total_spent', 'avg_order_value', 'avg_dwell_time', 'total_dwell_time',
            'avg_scroll_depth', 'max_scroll_depth', 'first_interaction', 'last_interaction'
        ]
        
        # Merge with user data
        user_features = user_features.merge(user_interactions, on='user_id', how='left')
        
        # Fill missing values
        user_features['total_interactions'] = user_features['total_interactions'].fillna(0)
        user_features['avg_rating'] = user_features['avg_rating'].fillna(0)
        user_features['rating_count'] = user_features['rating_count'].fillna(0)
        user_features['total_spent'] = user_features['total_spent'].fillna(0)
        user_features['avg_order_value'] = user_features['avg_order_value'].fillna(0)
        
        # User behavior features
        user_features['interaction_frequency'] = user_features['total_interactions'] / user_features['days_since_created'].clip(1)
        user_features['avg_rating_weighted'] = (user_features['avg_rating'] * user_features['rating_count']) / user_features['rating_count'].clip(1)
        user_features['spending_power'] = user_features['total_spent'] / user_features['days_since_created'].clip(1)
        
        # Categorical encoding
        categorical_cols = ['gender', 'location', 'income_level', 'preference_category', 'device_type', 'language_preference', 'timezone']
        for col in categorical_cols:
            if col in user_features.columns:
                le = LabelEncoder()
                user_features[f'{col}_encoded'] = le.fit_transform(user_features[col].fillna('unknown'))
                self.label_encoders[f'user_{col}'] = le
        
        # Numerical features
        numerical_cols = ['age', 'total_interactions', 'avg_rating', 'total_spent', 'interaction_frequency']
        for col in numerical_cols:
            if col in user_features.columns:
                scaler = StandardScaler()
                user_features[f'{col}_scaled'] = scaler.fit_transform(user_features[[col]])
                self.scalers[f'user_{col}'] = scaler
        
        # Log feature creation statistics
        duration = (datetime.now() - start_time).total_seconds()
        self.feature_stats['user_features'] = {
            'original_columns': len(users_df.columns),
            'feature_columns': len(user_features.columns),
            'duration_seconds': duration
        }
        
        logger.info(f"User features created: {len(user_features.columns)} columns in {duration:.2f}s")
        return user_features
    
    def create_product_features(self, products_df: pd.DataFrame, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Create product features for recommendation models."""
        logger.info("Creating product features")
        start_time = datetime.now()
        
        product_features = products_df.copy()
        
        # Product interaction features
        product_interactions = interactions_df.groupby('product_id').agg({
            'interaction_id': 'count',
            'rating': ['mean', 'count'],
            'total_amount': ['sum', 'mean'],
            'dwell_time': ['mean', 'sum'],
            'scroll_depth': ['mean', 'max'],
            'timestamp': ['min', 'max']
        }).reset_index()
        
        # Flatten column names
        product_interactions.columns = [
            'product_id', 'total_interactions', 'avg_rating', 'rating_count',
            'total_revenue', 'avg_order_value', 'avg_dwell_time', 'total_dwell_time',
            'avg_scroll_depth', 'max_scroll_depth', 'first_interaction', 'last_interaction'
        ]
        
        # Merge with product data
        product_features = product_features.merge(product_interactions, on='product_id', how='left')
        
        # Fill missing values
        product_features['total_interactions'] = product_features['total_interactions'].fillna(0)
        product_features['avg_rating'] = product_features['avg_rating'].fillna(0)
        product_features['rating_count'] = product_features['rating_count'].fillna(0)
        product_features['total_revenue'] = product_features['total_revenue'].fillna(0)
        
        # Product popularity features
        product_features['popularity_score'] = (
            product_features['total_interactions'] * 0.4 +
            product_features['avg_rating'] * 0.3 +
            product_features['rating_count'] * 0.3
        )
        
        product_features['engagement_score'] = (
            product_features['avg_dwell_time'] * 0.5 +
            product_features['avg_scroll_depth'] * 0.5
        )
        
        # Price features
        product_features['price_category'] = pd.cut(
            product_features['price'],
            bins=[0, 50, 200, 500, 1000, float('inf')],
            labels=['budget', 'affordable', 'mid-range', 'premium', 'luxury']
        )
        
        product_features['discount_impact'] = product_features['discount_percentage'] / 100
        
        # Categorical encoding
        categorical_cols = ['category', 'subcategory', 'brand', 'color', 'size', 'availability_status']
        for col in categorical_cols:
            if col in product_features.columns:
                le = LabelEncoder()
                product_features[f'{col}_encoded'] = le.fit_transform(product_features[col].fillna('unknown'))
                self.label_encoders[f'product_{col}'] = le
        
        # Numerical features
        numerical_cols = ['price', 'rating', 'review_count', 'stock_quantity', 'popularity_score']
        for col in numerical_cols:
            if col in product_features.columns:
                scaler = StandardScaler()
                product_features[f'{col}_scaled'] = scaler.fit_transform(product_features[[col]])
                self.scalers[f'product_{col}'] = scaler
        
        # Log feature creation statistics
        duration = (datetime.now() - start_time).total_seconds()
        self.feature_stats['product_features'] = {
            'original_columns': len(products_df.columns),
            'feature_columns': len(product_features.columns),
            'duration_seconds': duration
        }
        
        logger.info(f"Product features created: {len(product_features.columns)} columns in {duration:.2f}s")
        return product_features
    
    def create_interaction_features(self, interactions_df: pd.DataFrame, users_df: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features for recommendation models."""
        logger.info("Creating interaction features")
        start_time = datetime.now()
        
        interaction_features = interactions_df.copy()
        
        # Merge user and product features
        user_cols = ['user_id', 'age', 'income_level', 'preference_category', 'device_type']
        product_cols = ['product_id', 'category', 'subcategory', 'brand', 'price', 'rating']
        
        interaction_features = interaction_features.merge(
            users_df[user_cols], on='user_id', how='left'
        )
        interaction_features = interaction_features.merge(
            products_df[product_cols], on='product_id', how='left'
        )
        
        # Interaction type features
        interaction_features['is_purchase'] = interaction_features['interaction_type'] == 'purchase'
        interaction_features['is_view'] = interaction_features['interaction_type'] == 'view'
        interaction_features['is_cart'] = interaction_features['interaction_type'] == 'add_to_cart'
        interaction_features['is_review'] = interaction_features['interaction_type'] == 'review'
        
        # Time-based features
        interaction_features['hour_of_day'] = interaction_features['timestamp'].dt.hour
        interaction_features['day_of_week'] = interaction_features['timestamp'].dt.dayofweek
        interaction_features['is_weekend'] = interaction_features['day_of_week'].isin([5, 6])
        interaction_features['is_business_hours'] = interaction_features['hour_of_day'].between(9, 17)
        
        # User-product compatibility features
        interaction_features['category_match'] = (
            interaction_features['preference_category'] == interaction_features['category']
        ).astype(int)
        
        interaction_features['price_affordability'] = (
            interaction_features['price'] <= interaction_features['income_level'].map({
                'low': 100, 'medium': 500, 'high': 1000
            })
        ).astype(int)
        
        # Session features
        session_features = interaction_features.groupby('session_id').agg({
            'interaction_id': 'count',
            'product_id': 'nunique',
            'total_amount': 'sum'
        }).reset_index()
        session_features.columns = ['session_id', 'session_length', 'unique_products', 'session_value']
        
        interaction_features = interaction_features.merge(session_features, on='session_id', how='left')
        
        # Categorical encoding
        categorical_cols = ['interaction_type', 'payment_method', 'preference_category', 'category', 'subcategory']
        for col in categorical_cols:
            if col in interaction_features.columns:
                le = LabelEncoder()
                interaction_features[f'{col}_encoded'] = le.fit_transform(interaction_features[col].fillna('unknown'))
                self.label_encoders[f'interaction_{col}'] = le
        
        # Numerical features
        numerical_cols = ['rating', 'quantity', 'total_amount', 'dwell_time', 'scroll_depth']
        for col in numerical_cols:
            if col in interaction_features.columns:
                scaler = StandardScaler()
                interaction_features[f'{col}_scaled'] = scaler.fit_transform(interaction_features[[col]])
                self.scalers[f'interaction_{col}'] = scaler
        
        # Log feature creation statistics
        duration = (datetime.now() - start_time).total_seconds()
        self.feature_stats['interaction_features'] = {
            'original_columns': len(interactions_df.columns),
            'feature_columns': len(interaction_features.columns),
            'duration_seconds': duration
        }
        
        logger.info(f"Interaction features created: {len(interaction_features.columns)} columns in {duration:.2f}s")
        return interaction_features
    
    def get_feature_stats(self) -> Dict:
        """Get feature engineering statistics."""
        return self.feature_stats
    
    def get_encoders_and_scalers(self) -> Tuple[Dict, Dict]:
        """Get fitted encoders and scalers for inference."""
        return self.label_encoders, self.scalers
