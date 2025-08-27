"""
Item-Item Similarity Engine for Phase 5

Computes and serves item-item similarities using:
1. ALS latent embeddings (cosine similarity)
2. Co-purchase patterns
3. Hybrid combination of both approaches

Provides fast similarity search for the /similar_items endpoint.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import pickle
from pathlib import Path
import time
import logging
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import heapq

from src.utils.logging import get_logger

logger = get_logger(__name__)


class ItemSimilarityEngine:
    """Main engine for computing and serving item-item similarities."""
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models"):
        """Initialize the similarity engine."""
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        
        # Data components
        self.item_embeddings = None
        self.item_mapping = None
        self.reverse_item_mapping = None
        self.products_df = None
        self.interactions_df = None
        
        # Similarity matrices
        self.als_similarity_matrix = None
        self.copurchase_similarity_matrix = None
        self.hybrid_similarity_matrix = None
        
        # Precomputed similarity index
        self.similarity_index = None
        
        logger.info("🔄 Item Similarity Engine initialized")
    
    def load_data(self):
        """Load all required data for similarity computation."""
        logger.info("📊 Loading data for similarity computation...")
        
        try:
            # Load ALS model and embeddings
            als_model_path = self.models_dir / "phase3" / "als_model.pkl"
            with open(als_model_path, 'rb') as f:
                als_data = pickle.load(f)
            
            self.item_embeddings = als_data['item_factors']
            logger.info(f"✅ Loaded ALS item embeddings: {self.item_embeddings.shape}")
            
            # Load mappings
            with open(self.data_dir / "processed" / "item_mapping.pkl", 'rb') as f:
                self.item_mapping = pickle.load(f)
            
            self.reverse_item_mapping = {v: k for k, v in self.item_mapping.items()}
            logger.info(f"✅ Loaded item mappings: {len(self.item_mapping)} items")
            
            # Load product data
            self.products_df = pd.read_parquet(self.data_dir / "processed" / "products_cleaned.parquet")
            logger.info(f"✅ Loaded product data: {len(self.products_df)} products")
            
            # Load interaction data for co-purchase analysis
            self.interactions_df = pd.read_parquet(self.data_dir / "processed" / "interactions_cleaned.parquet")
            logger.info(f"✅ Loaded interactions: {len(self.interactions_df)} interactions")
            
            return self
            
        except Exception as e:
            logger.error(f"Failed to load similarity data: {e}")
            raise
    
    def compute_als_similarity(self, metric: str = "cosine", top_k: int = 100):
        """Compute item-item similarity using ALS embeddings."""
        logger.info(f"🧮 Computing ALS-based similarity using {metric} metric...")
        
        start_time = time.time()
        
        if metric == "cosine":
            # Compute cosine similarity matrix
            similarity_matrix = cosine_similarity(self.item_embeddings)
            
            # Set diagonal to 0 (item is not similar to itself for recommendations)
            np.fill_diagonal(similarity_matrix, 0)
            
        else:
            raise ValueError(f"Metric {metric} not supported")
        
        self.als_similarity_matrix = similarity_matrix
        
        computation_time = time.time() - start_time
        logger.info(f"✅ ALS similarity computed in {computation_time:.2f}s")
        logger.info(f"   Matrix shape: {similarity_matrix.shape}")
        logger.info(f"   Average similarity: {np.mean(similarity_matrix):.4f}")
        
        return similarity_matrix
    
    def compute_copurchase_similarity(self, min_support: int = 2, top_k: int = 100):
        """Compute item-item similarity based on co-purchase patterns."""
        logger.info(f"🛒 Computing co-purchase similarity (min_support={min_support})...")
        
        start_time = time.time()
        
        # Filter to purchase interactions only
        purchase_interactions = self.interactions_df[
            self.interactions_df['interaction_type'].isin(['purchase', 'add_to_cart'])
        ].copy()
        
        logger.info(f"   Using {len(purchase_interactions)} purchase/cart interactions")
        
        # Create user-item matrix for purchases
        user_item_matrix = purchase_interactions.pivot_table(
            index='user_id', 
            columns='product_id', 
            values='rating',
            fill_value=0,
            aggfunc='count'
        )
        
        # Convert to binary matrix (purchased or not)
        user_item_binary = (user_item_matrix > 0).astype(int)
        
        # Compute item-item co-occurrence matrix
        item_cooccurrence = user_item_binary.T.dot(user_item_binary)
        
        # Convert to similarity using Jaccard coefficient
        # Jaccard(A,B) = |A ∩ B| / |A ∪ B|
        item_counts = np.array(user_item_binary.sum(axis=0)).flatten()
        
        # Initialize similarity matrix
        n_items = len(item_counts)
        copurchase_similarity = np.zeros((n_items, n_items))
        
        for i in range(n_items):
            for j in range(n_items):
                if i != j:
                    intersection = item_cooccurrence.iloc[i, j]
                    union = item_counts[i] + item_counts[j] - intersection
                    
                    if union > 0 and intersection >= min_support:
                        copurchase_similarity[i, j] = intersection / union
        
        # Map to full item space (handle items not in purchase data)
        full_similarity_matrix = np.zeros((len(self.item_mapping), len(self.item_mapping)))
        
        # Map indices from purchase matrix to full item mapping
        purchase_items = list(user_item_matrix.columns)
        for i, item_i in enumerate(purchase_items):
            if item_i in self.item_mapping:
                idx_i = self.item_mapping[item_i]
                for j, item_j in enumerate(purchase_items):
                    if item_j in self.item_mapping:
                        idx_j = self.item_mapping[item_j]
                        full_similarity_matrix[idx_i, idx_j] = copurchase_similarity[i, j]
        
        self.copurchase_similarity_matrix = full_similarity_matrix
        
        computation_time = time.time() - start_time
        logger.info(f"✅ Co-purchase similarity computed in {computation_time:.2f}s")
        logger.info(f"   Matrix shape: {full_similarity_matrix.shape}")
        logger.info(f"   Non-zero similarities: {np.count_nonzero(full_similarity_matrix)}")
        logger.info(f"   Average similarity: {np.mean(full_similarity_matrix):.4f}")
        
        return full_similarity_matrix
    
    def compute_hybrid_similarity(self, als_weight: float = 0.7, copurchase_weight: float = 0.3):
        """Compute hybrid similarity combining ALS and co-purchase."""
        logger.info(f"🔀 Computing hybrid similarity (ALS: {als_weight}, Co-purchase: {copurchase_weight})...")
        
        if self.als_similarity_matrix is None:
            raise ValueError("ALS similarity not computed. Call compute_als_similarity() first.")
        
        if self.copurchase_similarity_matrix is None:
            raise ValueError("Co-purchase similarity not computed. Call compute_copurchase_similarity() first.")
        
        # Normalize both matrices to [0, 1] range
        als_normalized = self._normalize_matrix(self.als_similarity_matrix)
        copurchase_normalized = self._normalize_matrix(self.copurchase_similarity_matrix)
        
        # Compute weighted combination
        self.hybrid_similarity_matrix = (
            als_weight * als_normalized + 
            copurchase_weight * copurchase_normalized
        )
        
        logger.info(f"✅ Hybrid similarity computed")
        logger.info(f"   Average similarity: {np.mean(self.hybrid_similarity_matrix):.4f}")
        
        return self.hybrid_similarity_matrix
    
    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Normalize similarity matrix to [0, 1] range."""
        matrix_min = np.min(matrix)
        matrix_max = np.max(matrix)
        
        if matrix_max - matrix_min == 0:
            return matrix
        
        return (matrix - matrix_min) / (matrix_max - matrix_min)
    
    def build_similarity_index(self, similarity_type: str = "hybrid", top_k: int = 50):
        """Build fast similarity index for serving."""
        logger.info(f"🏗️ Building similarity index ({similarity_type}, top_k={top_k})...")
        
        # Select similarity matrix
        if similarity_type == "als":
            similarity_matrix = self.als_similarity_matrix
        elif similarity_type == "copurchase":
            similarity_matrix = self.copurchase_similarity_matrix
        elif similarity_type == "hybrid":
            similarity_matrix = self.hybrid_similarity_matrix
        else:
            raise ValueError(f"Unknown similarity type: {similarity_type}")
        
        if similarity_matrix is None:
            raise ValueError(f"{similarity_type} similarity not computed")
        
        # Build index: for each item, store top-k most similar items
        self.similarity_index = {}
        
        for item_idx in range(len(similarity_matrix)):
            # Get similarities for this item
            similarities = similarity_matrix[item_idx]
            
            # Get top-k most similar items (excluding self)
            top_indices = np.argsort(similarities)[::-1][:top_k]
            top_similarities = similarities[top_indices]
            
            # Filter out zero similarities
            valid_mask = top_similarities > 0
            top_indices = top_indices[valid_mask]
            top_similarities = top_similarities[valid_mask]
            
            # Convert to item IDs
            item_id = self.reverse_item_mapping[item_idx]
            similar_items = []
            
            for idx, sim_score in zip(top_indices, top_similarities):
                similar_item_id = self.reverse_item_mapping[idx]
                similar_items.append({
                    'item_id': similar_item_id,
                    'similarity_score': float(sim_score)
                })
            
            self.similarity_index[item_id] = similar_items
        
        logger.info(f"✅ Similarity index built for {len(self.similarity_index)} items")
        logger.info(f"   Average similar items per item: {np.mean([len(items) for items in self.similarity_index.values()]):.1f}")
        
        return self.similarity_index
    
    def get_similar_items(self, item_id: int, top_k: int = 10, 
                         include_metadata: bool = True) -> List[Dict]:
        """Get similar items for a given item."""
        if self.similarity_index is None:
            raise ValueError("Similarity index not built. Call build_similarity_index() first.")
        
        if item_id not in self.similarity_index:
            logger.warning(f"Item {item_id} not found in similarity index")
            return []
        
        # Get similar items from index
        similar_items = self.similarity_index[item_id][:top_k]
        
        if include_metadata:
            # Enrich with product metadata
            enriched_items = []
            for item in similar_items:
                product_info = self.products_df[
                    self.products_df['product_id'] == item['item_id']
                ]
                
                if len(product_info) > 0:
                    product = product_info.iloc[0]
                    enriched_items.append({
                        'item_id': item['item_id'],
                        'similarity_score': item['similarity_score'],
                        'name': product['name'],
                        'category': product['category'],
                        'brand': product['brand'],
                        'price': float(product['price']),
                        'rating': float(product['rating']),
                        'description': product.get('description', '')[:100] + '...'
                    })
                else:
                    enriched_items.append(item)
            
            return enriched_items
        
        return similar_items
    
    def get_similarity_stats(self) -> Dict[str, Any]:
        """Get statistics about the similarity engine."""
        stats = {
            'n_items': len(self.item_mapping) if self.item_mapping else 0,
            'n_products': len(self.products_df) if self.products_df is not None else 0,
            'n_interactions': len(self.interactions_df) if self.interactions_df is not None else 0,
        }
        
        if self.als_similarity_matrix is not None:
            stats['als_similarity'] = {
                'shape': self.als_similarity_matrix.shape,
                'avg_similarity': float(np.mean(self.als_similarity_matrix)),
                'max_similarity': float(np.max(self.als_similarity_matrix)),
                'non_zero_count': int(np.count_nonzero(self.als_similarity_matrix))
            }
        
        if self.copurchase_similarity_matrix is not None:
            stats['copurchase_similarity'] = {
                'shape': self.copurchase_similarity_matrix.shape,
                'avg_similarity': float(np.mean(self.copurchase_similarity_matrix)),
                'max_similarity': float(np.max(self.copurchase_similarity_matrix)),
                'non_zero_count': int(np.count_nonzero(self.copurchase_similarity_matrix))
            }
        
        if self.hybrid_similarity_matrix is not None:
            stats['hybrid_similarity'] = {
                'shape': self.hybrid_similarity_matrix.shape,
                'avg_similarity': float(np.mean(self.hybrid_similarity_matrix)),
                'max_similarity': float(np.max(self.hybrid_similarity_matrix)),
                'non_zero_count': int(np.count_nonzero(self.hybrid_similarity_matrix))
            }
        
        if self.similarity_index is not None:
            stats['similarity_index'] = {
                'n_items_indexed': len(self.similarity_index),
                'avg_similar_items': float(np.mean([len(items) for items in self.similarity_index.values()]))
            }
        
        return stats
    
    def save_similarity_data(self, output_dir: str = "models/phase5"):
        """Save computed similarity matrices and index."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Saving similarity data to {output_path}")
        
        # Save similarity matrices
        if self.als_similarity_matrix is not None:
            np.save(output_path / "als_similarity_matrix.npy", self.als_similarity_matrix)
            logger.info("   ✅ ALS similarity matrix saved")
        
        if self.copurchase_similarity_matrix is not None:
            np.save(output_path / "copurchase_similarity_matrix.npy", self.copurchase_similarity_matrix)
            logger.info("   ✅ Co-purchase similarity matrix saved")
        
        if self.hybrid_similarity_matrix is not None:
            np.save(output_path / "hybrid_similarity_matrix.npy", self.hybrid_similarity_matrix)
            logger.info("   ✅ Hybrid similarity matrix saved")
        
        # Save similarity index
        if self.similarity_index is not None:
            with open(output_path / "similarity_index.pkl", 'wb') as f:
                pickle.dump(self.similarity_index, f)
            logger.info("   ✅ Similarity index saved")
        
        # Save metadata
        metadata = {
            'timestamp': time.time(),
            'n_items': len(self.item_mapping),
            'stats': self.get_similarity_stats()
        }
        
        with open(output_path / "similarity_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"✅ All similarity data saved to {output_path}")
    
    def load_similarity_data(self, input_dir: str = "models/phase5"):
        """Load pre-computed similarity data."""
        input_path = Path(input_dir)
        
        logger.info(f"📂 Loading similarity data from {input_path}")
        
        try:
            # Load similarity matrices
            als_path = input_path / "als_similarity_matrix.npy"
            if als_path.exists():
                self.als_similarity_matrix = np.load(als_path)
                logger.info("   ✅ ALS similarity matrix loaded")
            
            copurchase_path = input_path / "copurchase_similarity_matrix.npy"
            if copurchase_path.exists():
                self.copurchase_similarity_matrix = np.load(copurchase_path)
                logger.info("   ✅ Co-purchase similarity matrix loaded")
            
            hybrid_path = input_path / "hybrid_similarity_matrix.npy"
            if hybrid_path.exists():
                self.hybrid_similarity_matrix = np.load(hybrid_path)
                logger.info("   ✅ Hybrid similarity matrix loaded")
            
            # Load similarity index
            index_path = input_path / "similarity_index.pkl"
            if index_path.exists():
                with open(index_path, 'rb') as f:
                    self.similarity_index = pickle.load(f)
                logger.info("   ✅ Similarity index loaded")
            
            logger.info(f"✅ Similarity data loaded successfully")
            return self
            
        except Exception as e:
            logger.error(f"Failed to load similarity data: {e}")
            raise
    
    def run_similarity_pipeline(self, als_weight: float = 0.7, copurchase_weight: float = 0.3,
                              similarity_type: str = "hybrid", top_k: int = 50,
                              save_results: bool = True):
        """Run the complete similarity computation pipeline."""
        logger.info("🚀 Running complete similarity pipeline...")
        
        start_time = time.time()
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Compute ALS similarity
        self.compute_als_similarity()
        
        # Step 3: Compute co-purchase similarity
        self.compute_copurchase_similarity()
        
        # Step 4: Compute hybrid similarity
        self.compute_hybrid_similarity(als_weight, copurchase_weight)
        
        # Step 5: Build similarity index
        self.build_similarity_index(similarity_type, top_k)
        
        # Step 6: Save results
        if save_results:
            self.save_similarity_data()
        
        total_time = time.time() - start_time
        
        logger.info(f"🎉 Similarity pipeline completed in {total_time:.2f}s")
        logger.info(f"   📊 Pipeline stats: {self.get_similarity_stats()}")
        
        return self
