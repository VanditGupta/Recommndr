"""
Ranking Feature Engineering for Phase 4

Creates contextual features for the LightGBM ranking model by combining:
- User features (demographics, interaction history)
- Item features (content, popularity, ratings)
- User-Item interaction features (compatibility, similarity)
- Real-time features (from streaming pipeline)
- Temporal features (time of day, recency)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone
import pickle
from pathlib import Path
import logging

from src.utils.logging import get_logger

logger = get_logger(__name__)


class RankingFeatureEngineer:
    """Feature engineering for ranking pipeline."""
    
    def __init__(self):
        self.user_features = None
        self.item_features = None
        self.interaction_features = None
        self.user_mapping = None
        self.item_mapping = None
        self.als_model_data = None
        
    def load_base_features(self, data_dir: str = "data"):
        """Load base features from processed data."""
        logger.info("Loading base features for ranking...")
        
        data_path = Path(data_dir)
        
        # Load engineered features
        self.user_features = pd.read_parquet(data_path / "features" / "users_features.parquet")
        self.item_features = pd.read_parquet(data_path / "features" / "products_features.parquet") 
        self.interaction_features = pd.read_parquet(data_path / "features" / "interactions_features.parquet")
        
        # Load mappings
        with open(data_path / "processed" / "user_mapping.pkl", 'rb') as f:
            self.user_mapping = pickle.load(f)
        with open(data_path / "processed" / "item_mapping.pkl", 'rb') as f:
            self.item_mapping = pickle.load(f)
            
        # Load ALS model for embeddings
        with open(data_path.parent / "models" / "phase3" / "als_model.pkl", 'rb') as f:
            self.als_model_data = pickle.load(f)
            
        logger.info(f"✅ Loaded features: {len(self.user_features)} users, {len(self.item_features)} items")
        
    def create_user_item_features(self, user_id: int, item_id: int) -> Dict:
        """Create features for a specific user-item pair."""
        features = {}
        
        # User features
        user_row = self.user_features[self.user_features['user_id'] == user_id]
        if len(user_row) > 0:
            user_row = user_row.iloc[0]
            features.update({
                'user_age': user_row.get('age', 0),
                'user_income_encoded': self._encode_income(user_row.get('income_level', 'medium')),
                'user_total_interactions': user_row.get('total_interactions', 0),
                'user_avg_rating': user_row.get('avg_rating_given', 3.5),
                'user_category_diversity': user_row.get('category_diversity', 0.5),
                'user_avg_price': user_row.get('avg_product_price', 500),
                'user_premium_ratio': user_row.get('premium_ratio', 0.3),
            })
        
        # Item features  
        item_row = self.item_features[self.item_features['product_id'] == item_id]
        if len(item_row) > 0:
            item_row = item_row.iloc[0]
            features.update({
                'item_price': item_row.get('price', 500),
                'item_rating': item_row.get('rating', 3.5),
                'item_review_count': item_row.get('review_count', 100),
                'item_popularity_score': item_row.get('popularity_score', 0.5),
                'item_engagement_score': item_row.get('engagement_score', 0.5),
                'item_conversion_rate': item_row.get('conversion_rate', 0.02),
                'item_category_encoded': self._encode_category(item_row.get('category', 'Other')),
                'item_brand_popularity': item_row.get('brand_popularity', 0.5),
            })
        
        # User-Item compatibility features
        if user_id in self.user_mapping and item_id in self.item_mapping:
            features.update(self._create_compatibility_features(user_id, item_id))
            
        # ALS embedding features
        features.update(self._create_embedding_features(user_id, item_id))
        
        # Temporal features
        features.update(self._create_temporal_features())
        
        return features
    
    def _create_compatibility_features(self, user_id: int, item_id: int) -> Dict:
        """Create user-item compatibility features."""
        features = {}
        
        user_row = self.user_features[self.user_features['user_id'] == user_id]
        item_row = self.item_features[self.item_features['product_id'] == item_id]
        
        if len(user_row) > 0 and len(item_row) > 0:
            user_row = user_row.iloc[0]
            item_row = item_row.iloc[0]
            
            # Price compatibility
            user_avg_price = user_row.get('avg_product_price', 500)
            item_price = item_row.get('price', 500)
            features['price_compatibility'] = 1.0 / (1.0 + abs(np.log(item_price + 1) - np.log(user_avg_price + 1)))
            
            # Rating compatibility  
            user_avg_rating = user_row.get('avg_rating_given', 3.5)
            item_rating = item_row.get('rating', 3.5)
            features['rating_compatibility'] = 1.0 - abs(user_avg_rating - item_rating) / 5.0
            
            # Category preference match
            user_pref_category = user_row.get('preference_category', 'Other')
            item_category = item_row.get('category', 'Other')
            features['category_match'] = 1.0 if user_pref_category == item_category else 0.0
            
            # Premium compatibility
            user_premium_ratio = user_row.get('premium_ratio', 0.3)
            item_is_premium = 1.0 if item_price > 500 else 0.0
            features['premium_compatibility'] = 1.0 - abs(user_premium_ratio - item_is_premium)
            
        return features
    
    def _create_embedding_features(self, user_id: int, item_id: int) -> Dict:
        """Create features from ALS embeddings."""
        features = {}
        
        if (self.als_model_data and 
            user_id in self.user_mapping and 
            item_id in self.item_mapping):
            
            user_idx = self.user_mapping[user_id]
            item_idx = self.item_mapping[item_id]
            
            user_factors = self.als_model_data['user_factors'][user_idx]
            item_factors = self.als_model_data['item_factors'][item_idx]
            
            # ALS prediction score
            features['als_score'] = float(np.dot(user_factors, item_factors))
            
            # Embedding similarity metrics
            features['embedding_cosine'] = float(
                np.dot(user_factors, item_factors) / 
                (np.linalg.norm(user_factors) * np.linalg.norm(item_factors) + 1e-8)
            )
            
            # Embedding magnitude features
            features['user_embedding_norm'] = float(np.linalg.norm(user_factors))
            features['item_embedding_norm'] = float(np.linalg.norm(item_factors))
            
            # Top factor values (most important latent dimensions)
            features['user_top_factor'] = float(np.max(user_factors))
            features['item_top_factor'] = float(np.max(item_factors))
            
        else:
            # Default values if embeddings not available
            features.update({
                'als_score': 0.0,
                'embedding_cosine': 0.0, 
                'user_embedding_norm': 1.0,
                'item_embedding_norm': 1.0,
                'user_top_factor': 0.0,
                'item_top_factor': 0.0
            })
            
        return features
    
    def _create_temporal_features(self) -> Dict:
        """Create time-based features."""
        now = datetime.now()
        
        features = {
            'hour_of_day': now.hour,
            'day_of_week': now.weekday(),
            'is_weekend': 1.0 if now.weekday() >= 5 else 0.0,
            'is_evening': 1.0 if 17 <= now.hour <= 22 else 0.0,
            'is_morning': 1.0 if 6 <= now.hour <= 11 else 0.0,
        }
        
        return features
    
    def _encode_income(self, income_level: str) -> float:
        """Encode income level to numerical value."""
        income_map = {
            'low': 0.2,
            'medium': 0.5, 
            'high': 0.8
        }
        return income_map.get(income_level, 0.5)
    
    def _encode_category(self, category: str) -> float:
        """Encode category to numerical value (could be improved with embeddings)."""
        # Simple hash-based encoding for now
        return hash(category) % 100 / 100.0
    
    def create_training_features(self, interactions_df: pd.DataFrame, 
                               include_negative_samples: bool = True,
                               negative_ratio: float = 2.0) -> pd.DataFrame:
        """Create training dataset with features for all user-item pairs."""
        logger.info("Creating training features for ranking model...")
        
        # Positive samples from interactions
        positive_samples = []
        for _, interaction in interactions_df.iterrows():
            user_id = interaction['user_id']
            item_id = interaction['product_id']
            
            features = self.create_user_item_features(user_id, item_id)
            features['user_id'] = user_id
            features['item_id'] = item_id
            features['label'] = 1.0  # Positive interaction
            features['rating'] = interaction.get('rating', 1.0)
            
            positive_samples.append(features)
        
        training_data = positive_samples
        
        # Add negative samples
        if include_negative_samples:
            logger.info("Generating negative samples...")
            negative_samples = self._generate_negative_samples(
                interactions_df, int(len(positive_samples) * negative_ratio)
            )
            training_data.extend(negative_samples)
        
        # Convert to DataFrame
        training_df = pd.DataFrame(training_data)
        
        logger.info(f"✅ Created training features: {len(training_df)} samples")
        logger.info(f"   Positive samples: {sum(training_df['label'] == 1.0)}")
        logger.info(f"   Negative samples: {sum(training_df['label'] == 0.0)}")
        
        return training_df
    
    def _generate_negative_samples(self, interactions_df: pd.DataFrame, 
                                 num_samples: int) -> List[Dict]:
        """Generate negative samples for training."""
        negative_samples = []
        
        # Get all user-item pairs that exist (positive interactions)
        positive_pairs = set(zip(interactions_df['user_id'], interactions_df['product_id']))
        
        # Get all possible users and items
        all_users = list(self.user_mapping.keys())
        all_items = list(self.item_mapping.keys())
        
        attempts = 0
        max_attempts = num_samples * 10  # Avoid infinite loop
        
        while len(negative_samples) < num_samples and attempts < max_attempts:
            user_id = np.random.choice(all_users)
            item_id = np.random.choice(all_items)
            
            # Skip if this is a positive interaction
            if (user_id, item_id) not in positive_pairs:
                features = self.create_user_item_features(user_id, item_id)
                features['user_id'] = user_id
                features['item_id'] = item_id
                features['label'] = 0.0  # Negative sample
                features['rating'] = 0.0
                
                negative_samples.append(features)
            
            attempts += 1
        
        logger.info(f"Generated {len(negative_samples)} negative samples")
        return negative_samples
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names for model training."""
        # Create a sample feature dict to get names
        sample_features = self.create_user_item_features(1, 1)
        
        # Remove ID columns and label
        feature_names = [k for k in sample_features.keys() 
                        if k not in ['user_id', 'item_id', 'label', 'rating']]
        
        return feature_names
