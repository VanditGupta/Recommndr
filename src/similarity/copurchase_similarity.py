"""
Co-Purchase Similarity Module for Phase 5

Specialized module for computing item-item similarity based on 
co-purchase patterns and market basket analysis.

Implements various co-purchase similarity metrics:
- Jaccard coefficient
- Cosine similarity
- Lift/confidence measures
- Frequent itemset mining
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import time
from collections import defaultdict, Counter
from itertools import combinations
import logging

from src.utils.logging import get_logger

logger = get_logger(__name__)


class CoPurchaseSimilarity:
    """Specialized co-purchase similarity engine."""
    
    def __init__(self):
        """Initialize co-purchase similarity engine."""
        self.interactions_df = None
        self.user_baskets = None
        self.item_frequencies = None
        self.cooccurrence_matrix = None
        self.similarity_matrix = None
        
    def load_interactions(self, interactions_df: pd.DataFrame):
        """Load interaction data for co-purchase analysis."""
        self.interactions_df = interactions_df.copy()
        logger.info(f"📊 Loaded {len(interactions_df)} interactions for co-purchase analysis")
        
    def extract_user_baskets(self, interaction_types: List[str] = None):
        """Extract user purchase baskets from interactions."""
        if interaction_types is None:
            interaction_types = ['purchase', 'add_to_cart']
        
        logger.info(f"🛒 Extracting user baskets for interaction types: {interaction_types}")
        
        # Filter to relevant interaction types
        purchase_data = self.interactions_df[
            self.interactions_df['interaction_type'].isin(interaction_types)
        ].copy()
        
        logger.info(f"   Using {len(purchase_data)} purchase interactions")
        
        # Group by user to create baskets
        self.user_baskets = {}
        
        for user_id, user_interactions in purchase_data.groupby('user_id'):
            basket = set(user_interactions['product_id'].unique())
            if len(basket) > 1:  # Only include users with multiple items
                self.user_baskets[user_id] = basket
        
        logger.info(f"✅ Extracted {len(self.user_baskets)} user baskets")
        logger.info(f"   Average basket size: {np.mean([len(basket) for basket in self.user_baskets.values()]):.1f}")
        
        return self.user_baskets
    
    def compute_item_frequencies(self):
        """Compute item purchase frequencies."""
        logger.info("📊 Computing item frequencies...")
        
        self.item_frequencies = Counter()
        
        for basket in self.user_baskets.values():
            for item in basket:
                self.item_frequencies[item] += 1
        
        logger.info(f"✅ Computed frequencies for {len(self.item_frequencies)} items")
        logger.info(f"   Most frequent item: {self.item_frequencies.most_common(1)[0] if self.item_frequencies else 'None'}")
        
        return self.item_frequencies
    
    def compute_cooccurrence_matrix(self, min_basket_size: int = 2, max_basket_size: int = 50):
        """Compute item co-occurrence matrix."""
        logger.info(f"🔗 Computing co-occurrence matrix (basket size: {min_basket_size}-{max_basket_size})...")
        
        start_time = time.time()
        
        # Filter baskets by size
        filtered_baskets = {
            user_id: basket for user_id, basket in self.user_baskets.items()
            if min_basket_size <= len(basket) <= max_basket_size
        }
        
        logger.info(f"   Using {len(filtered_baskets)} baskets after size filtering")
        
        # Initialize co-occurrence matrix
        self.cooccurrence_matrix = defaultdict(lambda: defaultdict(int))
        
        # Count co-occurrences
        total_pairs = 0
        
        for basket in filtered_baskets.values():
            # Generate all pairs in this basket
            for item1, item2 in combinations(basket, 2):
                self.cooccurrence_matrix[item1][item2] += 1
                self.cooccurrence_matrix[item2][item1] += 1  # Symmetric
                total_pairs += 2
        
        computation_time = time.time() - start_time
        
        logger.info(f"✅ Co-occurrence matrix computed in {computation_time:.2f}s")
        logger.info(f"   Total item pairs: {total_pairs:,}")
        logger.info(f"   Unique item pairs: {len(self.cooccurrence_matrix):,}")
        
        return self.cooccurrence_matrix
    
    def compute_jaccard_similarity(self, min_cooccurrence: int = 2) -> Dict:
        """Compute Jaccard similarity between items."""
        logger.info(f"📐 Computing Jaccard similarity (min_cooccurrence={min_cooccurrence})...")
        
        if not self.cooccurrence_matrix:
            raise ValueError("Co-occurrence matrix not computed")
        
        if not self.item_frequencies:
            raise ValueError("Item frequencies not computed")
        
        start_time = time.time()
        
        jaccard_similarities = {}
        total_comparisons = 0
        
        # Compute Jaccard for each item pair
        for item1 in self.cooccurrence_matrix:
            jaccard_similarities[item1] = {}
            
            for item2, cooccurrence in self.cooccurrence_matrix[item1].items():
                if cooccurrence >= min_cooccurrence:
                    # Jaccard = |A ∩ B| / |A ∪ B|
                    intersection = cooccurrence
                    union = (self.item_frequencies[item1] + 
                            self.item_frequencies[item2] - 
                            intersection)
                    
                    if union > 0:
                        jaccard_score = intersection / union
                        jaccard_similarities[item1][item2] = jaccard_score
                        total_comparisons += 1
        
        computation_time = time.time() - start_time
        
        logger.info(f"✅ Jaccard similarity computed in {computation_time:.2f}s")
        logger.info(f"   Valid similarities: {total_comparisons:,}")
        
        return jaccard_similarities
    
    def compute_lift_similarity(self, min_cooccurrence: int = 2) -> Dict:
        """Compute lift-based similarity between items."""
        logger.info(f"📈 Computing lift similarity (min_cooccurrence={min_cooccurrence})...")
        
        if not self.cooccurrence_matrix:
            raise ValueError("Co-occurrence matrix not computed")
        
        total_baskets = len(self.user_baskets)
        lift_similarities = {}
        
        for item1 in self.cooccurrence_matrix:
            lift_similarities[item1] = {}
            
            item1_support = self.item_frequencies[item1] / total_baskets
            
            for item2, cooccurrence in self.cooccurrence_matrix[item1].items():
                if cooccurrence >= min_cooccurrence:
                    item2_support = self.item_frequencies[item2] / total_baskets
                    joint_support = cooccurrence / total_baskets
                    
                    expected_joint = item1_support * item2_support
                    
                    if expected_joint > 0:
                        lift = joint_support / expected_joint
                        lift_similarities[item1][item2] = lift
        
        logger.info(f"✅ Lift similarity computed")
        
        return lift_similarities
    
    def compute_confidence_similarity(self, min_cooccurrence: int = 2) -> Dict:
        """Compute confidence-based similarity between items."""
        logger.info(f"🎯 Computing confidence similarity (min_cooccurrence={min_cooccurrence})...")
        
        confidence_similarities = {}
        
        for item1 in self.cooccurrence_matrix:
            confidence_similarities[item1] = {}
            
            for item2, cooccurrence in self.cooccurrence_matrix[item1].items():
                if cooccurrence >= min_cooccurrence:
                    # Confidence = P(item2|item1) = |item1 ∩ item2| / |item1|
                    confidence = cooccurrence / self.item_frequencies[item1]
                    confidence_similarities[item1][item2] = confidence
        
        logger.info(f"✅ Confidence similarity computed")
        
        return confidence_similarities
    
    def find_frequent_itemsets(self, min_support: float = 0.01, max_itemset_size: int = 3) -> Dict:
        """Find frequent itemsets using Apriori-like algorithm."""
        logger.info(f"🔍 Finding frequent itemsets (min_support={min_support}, max_size={max_itemset_size})...")
        
        total_baskets = len(self.user_baskets)
        min_basket_count = int(min_support * total_baskets)
        
        logger.info(f"   Minimum basket count: {min_basket_count}")
        
        # Find frequent 1-itemsets
        frequent_itemsets = {1: {}}
        
        for item, count in self.item_frequencies.items():
            if count >= min_basket_count:
                frequent_itemsets[1][frozenset([item])] = count
        
        logger.info(f"   Frequent 1-itemsets: {len(frequent_itemsets[1])}")
        
        # Find frequent k-itemsets for k > 1
        for k in range(2, max_itemset_size + 1):
            frequent_itemsets[k] = {}
            
            # Generate candidates from frequent (k-1)-itemsets
            prev_itemsets = list(frequent_itemsets[k-1].keys())
            
            for i in range(len(prev_itemsets)):
                for j in range(i + 1, len(prev_itemsets)):
                    # Union of two frequent (k-1)-itemsets
                    candidate = prev_itemsets[i] | prev_itemsets[j]
                    
                    if len(candidate) == k:
                        # Count support for this candidate
                        support_count = 0
                        
                        for basket in self.user_baskets.values():
                            if candidate.issubset(basket):
                                support_count += 1
                        
                        if support_count >= min_basket_count:
                            frequent_itemsets[k][candidate] = support_count
            
            logger.info(f"   Frequent {k}-itemsets: {len(frequent_itemsets[k])}")
            
            # Stop if no frequent itemsets found
            if not frequent_itemsets[k]:
                break
        
        return frequent_itemsets
    
    def get_top_similar_items(self, item_id: int, similarity_dict: Dict, 
                             top_k: int = 10) -> List[Tuple[int, float]]:
        """Get top-k similar items for a given item."""
        if item_id not in similarity_dict:
            return []
        
        similarities = similarity_dict[item_id]
        
        # Sort by similarity score
        sorted_items = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_items[:top_k]
    
    def analyze_market_basket(self, item_id: int, top_k: int = 5) -> Dict:
        """Analyze market basket patterns for a specific item."""
        if item_id not in self.cooccurrence_matrix:
            return {'error': f'Item {item_id} not found in co-occurrence data'}
        
        # Get co-occurring items
        cooccurring_items = self.cooccurrence_matrix[item_id]
        
        # Sort by co-occurrence frequency
        top_cooccurring = sorted(cooccurring_items.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Calculate additional metrics
        item_frequency = self.item_frequencies[item_id]
        total_baskets = len(self.user_baskets)
        
        analysis = {
            'item_id': item_id,
            'item_frequency': item_frequency,
            'item_support': item_frequency / total_baskets,
            'top_cooccurring_items': []
        }
        
        for cooc_item, cooc_count in top_cooccurring:
            cooc_frequency = self.item_frequencies[cooc_item]
            
            # Calculate metrics
            jaccard = cooc_count / (item_frequency + cooc_frequency - cooc_count)
            confidence = cooc_count / item_frequency
            lift = (cooc_count / total_baskets) / ((item_frequency / total_baskets) * (cooc_frequency / total_baskets))
            
            analysis['top_cooccurring_items'].append({
                'item_id': cooc_item,
                'cooccurrence_count': cooc_count,
                'jaccard_similarity': jaccard,
                'confidence': confidence,
                'lift': lift
            })
        
        return analysis
    
    def get_recommendations_via_copurchase(self, user_basket: Set[int], 
                                         top_k: int = 10, 
                                         similarity_type: str = 'jaccard') -> List[Dict]:
        """Get recommendations based on current user basket using co-purchase patterns."""
        if not user_basket:
            return []
        
        # Choose similarity metric
        if similarity_type == 'jaccard':
            similarity_dict = self.compute_jaccard_similarity()
        elif similarity_type == 'lift':
            similarity_dict = self.compute_lift_similarity()
        elif similarity_type == 'confidence':
            similarity_dict = self.compute_confidence_similarity()
        else:
            raise ValueError(f"Unknown similarity type: {similarity_type}")
        
        # Aggregate recommendations from all items in basket
        recommendation_scores = defaultdict(float)
        
        for basket_item in user_basket:
            if basket_item in similarity_dict:
                for similar_item, score in similarity_dict[basket_item].items():
                    if similar_item not in user_basket:  # Don't recommend items already in basket
                        recommendation_scores[similar_item] += score
        
        # Sort by aggregated score
        sorted_recommendations = sorted(
            recommendation_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        # Format recommendations
        recommendations = []
        for item_id, score in sorted_recommendations:
            recommendations.append({
                'item_id': item_id,
                'copurchase_score': score,
                'similarity_type': similarity_type
            })
        
        return recommendations
