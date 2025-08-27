"""Data cleaning and preprocessing for e-commerce data."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

from config.settings import settings
from src.utils.logging import get_logger, log_performance_metrics

logger = get_logger(__name__)


class DataCleaner:
    """Clean and preprocess e-commerce data for ML models."""
    
    def __init__(self):
        """Initialize data cleaner."""
        self.cleaning_stats = {}
        
    def clean_users(self, users_df: pd.DataFrame) -> pd.DataFrame:
        """Clean users data."""
        logger.info("Cleaning users data")
        start_time = datetime.now()
        
        # Create a copy to avoid modifying original
        users_clean = users_df.copy()
        
        # Handle missing values
        users_clean['age'] = users_clean['age'].fillna(users_clean['age'].median())
        users_clean['gender'] = users_clean['gender'].fillna('unknown')
        users_clean['location'] = users_clean['location'].fillna('Unknown')
        users_clean['income_level'] = users_clean['income_level'].fillna('medium')
        
        # Clean and standardize text fields
        users_clean['location'] = users_clean['location'].str.strip().str.title()
        users_clean['preference_category'] = users_clean['preference_category'].str.strip()
        
        # Ensure email format consistency
        users_clean['email'] = users_clean['email'].str.lower().str.strip()
        
        # Convert timestamps to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(users_clean['created_at']):
            users_clean['created_at'] = pd.to_datetime(users_clean['created_at'])
        if not pd.api.types.is_datetime64_any_dtype(users_clean['last_active']):
            users_clean['last_active'] = pd.to_datetime(users_clean['last_active'])
        
        # Calculate user activity features
        users_clean['days_since_created'] = (datetime.now() - users_clean['created_at']).dt.days
        users_clean['days_since_active'] = (datetime.now() - users_clean['last_active']).dt.days
        users_clean['is_active'] = users_clean['days_since_active'] <= 30
        
        # Log cleaning statistics
        duration = (datetime.now() - start_time).total_seconds()
        self.cleaning_stats['users'] = {
            'original_rows': len(users_df),
            'cleaned_rows': len(users_clean),
            'duration_seconds': duration
        }
        
        logger.info(f"Users cleaning completed: {len(users_clean)} rows in {duration:.2f}s")
        return users_clean
    
    def clean_products(self, products_df: pd.DataFrame) -> pd.DataFrame:
        """Clean products data."""
        logger.info("Cleaning products data")
        start_time = datetime.now()
        
        products_clean = products_df.copy()
        
        # Handle missing values
        products_clean['price'] = products_clean['price'].fillna(products_clean['price'].median())
        products_clean['rating'] = products_clean['rating'].fillna(products_clean['rating'].median())
        products_clean['review_count'] = products_clean['review_count'].fillna(0)
        products_clean['stock_quantity'] = products_clean['stock_quantity'].fillna(0)
        
        # Clean text fields
        products_clean['name'] = products_clean['name'].str.strip()
        products_clean['description'] = products_clean['description'].str.strip()
        products_clean['category'] = products_clean['category'].str.strip()
        products_clean['subcategory'] = products_clean['subcategory'].str.strip()
        products_clean['brand'] = products_clean['brand'].str.strip()
        
        # Ensure numeric fields are valid
        products_clean['price'] = pd.to_numeric(products_clean['price'], errors='coerce')
        products_clean['rating'] = pd.to_numeric(products_clean['rating'], errors='coerce')
        products_clean['review_count'] = pd.to_numeric(products_clean['review_count'], errors='coerce')
        products_clean['stock_quantity'] = pd.to_numeric(products_clean['stock_quantity'], errors='coerce')
        
        # Remove invalid data
        products_clean = products_clean[products_clean['price'] > 0]
        products_clean = products_clean[products_clean['rating'] >= 0]
        products_clean = products_clean[products_clean['rating'] <= 5]
        
        # Convert timestamps
        if not pd.api.types.is_datetime64_any_dtype(products_clean['created_at']):
            products_clean['created_at'] = pd.to_datetime(products_clean['created_at'])
        
        # Calculate product features
        products_clean['days_since_created'] = (datetime.now() - products_clean['created_at']).dt.days
        products_clean['has_discount'] = products_clean['discount_percentage'] > 0
        products_clean['final_price'] = products_clean['price'] * (1 - products_clean['discount_percentage'] / 100)
        
        # Log cleaning statistics
        duration = (datetime.now() - start_time).total_seconds()
        self.cleaning_stats['products'] = {
            'original_rows': len(products_df),
            'cleaned_rows': len(products_clean),
            'duration_seconds': duration
        }
        
        logger.info(f"Products cleaning completed: {len(products_clean)} rows in {duration:.2f}s")
        return products_clean
    
    def clean_interactions(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Clean interactions data."""
        logger.info("Cleaning interactions data")
        start_time = datetime.now()
        
        interactions_clean = interactions_df.copy()
        
        # Handle missing values
        interactions_clean['rating'] = interactions_clean['rating'].fillna(0)
        interactions_clean['review_text'] = interactions_clean['review_text'].fillna('')
        interactions_clean['quantity'] = interactions_clean['quantity'].fillna(1)
        interactions_clean['total_amount'] = interactions_clean['total_amount'].fillna(0)
        
        # Clean text fields
        interactions_clean['interaction_type'] = interactions_clean['interaction_type'].str.strip()
        interactions_clean['session_id'] = interactions_clean['session_id'].str.strip()
        interactions_clean['payment_method'] = interactions_clean['payment_method'].fillna('unknown')
        
        # Ensure numeric fields are valid
        interactions_clean['rating'] = pd.to_numeric(interactions_clean['rating'], errors='coerce')
        interactions_clean['quantity'] = pd.to_numeric(interactions_clean['quantity'], errors='coerce')
        interactions_clean['total_amount'] = pd.to_numeric(interactions_clean['total_amount'], errors='coerce')
        interactions_clean['dwell_time'] = pd.to_numeric(interactions_clean['dwell_time'], errors='coerce')
        interactions_clean['scroll_depth'] = pd.to_numeric(interactions_clean['scroll_depth'], errors='coerce')
        
        # Remove invalid data
        interactions_clean = interactions_clean[interactions_clean['rating'] >= 0]
        interactions_clean = interactions_clean[interactions_clean['rating'] <= 5]
        interactions_clean = interactions_clean[interactions_clean['quantity'] > 0]
        
        # Convert timestamps
        if not pd.api.types.is_datetime64_any_dtype(interactions_clean['timestamp']):
            interactions_clean['timestamp'] = pd.to_datetime(interactions_clean['timestamp'])
        
        # Calculate interaction features
        interactions_clean['hour_of_day'] = interactions_clean['timestamp'].dt.hour
        interactions_clean['day_of_week'] = interactions_clean['timestamp'].dt.dayofweek
        interactions_clean['is_weekend'] = interactions_clean['day_of_week'].isin([5, 6])
        
        # Log cleaning statistics
        duration = (datetime.now() - start_time).total_seconds()
        self.cleaning_stats['interactions'] = {
            'original_rows': len(interactions_df),
            'cleaned_rows': len(interactions_clean),
            'duration_seconds': duration
        }
        
        logger.info(f"Interactions cleaning completed: {len(interactions_clean)} rows in {duration:.2f}s")
        return interactions_clean
    
    def get_cleaning_stats(self) -> Dict:
        """Get cleaning statistics."""
        return self.cleaning_stats
