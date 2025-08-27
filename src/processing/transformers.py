"""Data transformation and preparation for ML models."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from config.settings import settings
from src.utils.logging import get_logger, log_performance_metrics

logger = get_logger(__name__)


class DataTransformer:
    """Transform data for ML model training and inference."""
    
    def __init__(self):
        """Initialize data transformer."""
        self.transformation_stats = {}
        
    def prepare_user_item_matrix(self, interactions_df: pd.DataFrame, users_df: pd.DataFrame, products_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Prepare user-item matrix for collaborative filtering."""
        logger.info("Preparing user-item matrix")
        start_time = pd.Timestamp.now()
        
        # Ensure we have the rating column, create it if missing
        if 'rating' not in interactions_df.columns:
            logger.warning("Rating column not found, creating binary interaction matrix")
            # Create binary interaction matrix (1 if interaction exists, 0 otherwise)
            user_item_matrix = interactions_df.pivot_table(
                index='user_id',
                columns='product_id',
                values='interaction_id',
                aggfunc='count',
                fill_value=0
            )
            # Convert to binary (any interaction = 1)
            user_item_matrix = (user_item_matrix > 0).astype(int)
        else:
            # Create rating-based matrix
            user_item_matrix = interactions_df.pivot_table(
                index='user_id',
                columns='product_id',
                values='rating',
                fill_value=0
            )
        
        # Create user features matrix
        user_features = users_df.set_index('user_id')
        
        # Create product features matrix
        product_features = products_df.set_index('product_id')
        
        # Log transformation statistics
        duration = (pd.Timestamp.now() - start_time).total_seconds()
        self.transformation_stats['user_item_matrix'] = {
            'users': len(user_item_matrix),
            'products': len(user_item_matrix.columns),
            'sparsity': 1 - (user_item_matrix != 0).sum().sum() / (len(user_item_matrix) * len(user_item_matrix.columns)),
            'duration_seconds': duration
        }
        
        logger.info(f"User-item matrix prepared: {len(user_item_matrix)} users × {len(user_item_matrix.columns)} products in {duration:.2f}s")
        return user_item_matrix, user_features, product_features
    
    def create_training_data(self, interactions_df: pd.DataFrame, users_df: pd.DataFrame, products_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Create training data for recommendation models."""
        logger.info("Creating training data")
        start_time = pd.Timestamp.now()
        
        # Merge all features
        training_data = interactions_df.merge(users_df, on='user_id', how='left')
        training_data = training_data.merge(products_df, on='product_id', how='left', suffixes=('_user', '_product'))
        
        # Create target variable (rating or interaction success)
        if 'rating' in training_data.columns:
            training_data['target'] = training_data['rating'].fillna(0)
        else:
            # If no rating, use interaction success (1 for any interaction)
            training_data['target'] = 1
        
        training_data['interaction_success'] = (training_data['target'] > 0).astype(int)
        
        # Select features for training - be more flexible about available columns
        potential_feature_columns = [
            # User features
            'age', 'income_level_encoded', 'preference_category_encoded', 'device_type_encoded',
            'days_since_created', 'days_since_active', 'is_active',
            'total_interactions', 'avg_rating', 'total_spent', 'interaction_frequency',
            
            # Product features
            'price', 'rating', 'review_count', 'stock_quantity',
            'popularity_score', 'engagement_score', 'discount_percentage',
            'category_encoded', 'subcategory_encoded', 'brand_encoded',
            
            # Interaction features
            'interaction_type_encoded', 'quantity', 'total_amount',
            'dwell_time', 'scroll_depth', 'hour_of_day', 'day_of_week',
            'is_weekend', 'is_business_hours', 'category_match', 'price_affordability',
            'session_length', 'unique_products', 'session_value'
        ]
        
        # Filter available features
        available_features = [col for col in potential_feature_columns if col in training_data.columns]
        logger.info(f"Using {len(available_features)} available features out of {len(potential_feature_columns)} potential features")
        
        X = training_data[available_features].fillna(0)
        y = training_data['target']
        
        # Log transformation statistics
        duration = (pd.Timestamp.now() - start_time).total_seconds()
        self.transformation_stats['training_data'] = {
            'samples': len(X),
            'features': len(X.columns),
            'target_distribution': y.value_counts().to_dict(),
            'duration_seconds': duration
        }
        
        logger.info(f"Training data created: {len(X)} samples × {len(X.columns)} features in {duration:.2f}s")
        return X, y
    
    def create_validation_split(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Create train/validation split for model evaluation."""
        logger.info("Creating train/validation split")
        
        # Sort by index to maintain temporal order
        sorted_indices = X.index.sort_values()
        split_idx = int(len(sorted_indices) * (1 - test_size))
        
        train_indices = sorted_indices[:split_idx]
        val_indices = sorted_indices[split_idx:]
        
        X_train = X.loc[train_indices]
        X_val = X.loc[val_indices]
        y_train = y.loc[train_indices]
        y_val = y.loc[val_indices]
        
        logger.info(f"Train/validation split: {len(X_train)} train, {len(X_val)} validation")
        return X_train, X_val, y_train, y_val
    
    def prepare_inference_data(self, user_id: int, product_ids: List[int], users_df: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for model inference."""
        logger.info(f"Preparing inference data for user {user_id}")
        
        # Get user features
        user_features = users_df[users_df['user_id'] == user_id].iloc[0]
        
        # Get product features for all candidate products
        product_features = products_df[products_df['product_id'].isin(product_ids)]
        
        # Create inference dataframe
        inference_data = []
        for _, product in product_features.iterrows():
            row = {}
            
            # User features
            for col in users_df.columns:
                if col != 'user_id':
                    row[f'user_{col}'] = user_features[col]
            
            # Product features
            for col in products_df.columns:
                if col != 'product_id':
                    row[f'product_{col}'] = product[col]
            
            # Interaction features (default values for inference)
            row['interaction_type_encoded'] = 0
            row['quantity'] = 1
            row['dwell_time'] = 0
            row['scroll_depth'] = 0
            row['hour_of_day'] = pd.Timestamp.now().hour
            row['day_of_week'] = pd.Timestamp.now().dayofweek
            row['is_weekend'] = pd.Timestamp.now().dayofweek in [5, 6]
            row['is_business_hours'] = pd.Timestamp.now().hour in range(9, 18)
            
            # Compatibility features
            if 'preference_category' in user_features and 'category' in product:
                row['category_match'] = int(user_features['preference_category'] == product['category'])
            else:
                row['category_match'] = 0
            
            inference_data.append(row)
        
        inference_df = pd.DataFrame(inference_data)
        
        # Fill missing values
        inference_df = inference_df.fillna(0)
        
        logger.info(f"Inference data prepared: {len(inference_df)} products × {len(inference_df.columns)} features")
        return inference_df
    
    def get_transformation_stats(self) -> Dict:
        """Get transformation statistics."""
        return self.transformation_stats
