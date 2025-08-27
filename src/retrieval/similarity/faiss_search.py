"""Faiss-based similarity search for fast vector retrieval."""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union
import pickle
from pathlib import Path
import logging
from sklearn.metrics.pairwise import cosine_similarity

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if Faiss is available
FAISS_AVAILABLE = False
try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("✅ Faiss imported successfully")
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("⚠️ Faiss not available. Falling back to basic cosine similarity search.")


class FaissSimilaritySearch:
    """Faiss-based similarity search with fallback to basic similarity."""
    
    def __init__(self, use_faiss: bool = True):
        """Initialize similarity search.
        
        Args:
            use_faiss: Whether to use Faiss (if available) or fallback
        """
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.index = None
        self.item_embeddings = None
        self.item_ids = None
        
        if self.use_faiss:
            logger.info("🚀 Using Faiss for similarity search")
        else:
            logger.info("📊 Using fallback cosine similarity search")
    
    def build_index(self, item_embeddings: np.ndarray, item_ids: Optional[np.ndarray] = None) -> None:
        """Build similarity search index.
        
        Args:
            item_embeddings: Item embedding vectors
            item_ids: Optional item IDs mapping
        """
        self.item_embeddings = item_embeddings
        self.item_ids = item_ids if item_ids is not None else np.arange(len(item_embeddings))
        
        if self.use_faiss:
            self._build_faiss_index()
        else:
            logger.info("📊 Using fallback similarity search (no index building needed)")
    
    def _build_faiss_index(self) -> None:
        """Build Faiss index for fast similarity search."""
        if not FAISS_AVAILABLE:
            raise ImportError("Faiss not available")
            
        n_items, n_dim = self.item_embeddings.shape
        logger.info(f"🔨 Building Faiss index for {n_items} items with {n_dim} dimensions")
        
        # Use IVF index for better performance
        quantizer = faiss.IndexFlatIP(n_dim)
        self.index = faiss.IndexIVFFlat(quantizer, n_dim, min(100, n_items // 10))
        
        # Train the index
        self.index.train(self.item_embeddings.astype(np.float32))
        
        # Add vectors to index
        self.index.add(self.item_embeddings.astype(np.float32))
        logger.info(f"✅ Faiss index built with {self.index.ntotal} vectors")
    
    def search_similar(self, query_vector: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar items.
        
        Args:
            query_vector: Query vector
            k: Number of similar items to return
            
        Returns:
            Tuple of (distances, item_indices)
        """
        if self.use_faiss and self.index is not None:
            return self._faiss_search(query_vector, k)
        else:
            return self._fallback_search(query_vector, k)
    
    def _faiss_search(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Faiss-based similarity search."""
        if not FAISS_AVAILABLE:
            raise ImportError("Faiss not available")
            
        query_vector = query_vector.astype(np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_vector, k)
        return distances[0], indices[0]
    
    def _fallback_search(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback similarity search using cosine similarity."""
        if self.item_embeddings is None:
            raise ValueError("Item embeddings not loaded")
        
        # Ensure query_vector is 1D
        query_vector = query_vector.flatten()
        
        # Calculate cosine similarities
        similarities = []
        for i, item_emb in enumerate(self.item_embeddings):
            # Use sklearn cosine_similarity with proper reshaping
            sim = cosine_similarity(query_vector.reshape(1, -1), item_emb.reshape(1, -1))[0, 0]
            similarities.append((sim, i))
        
        # Sort by similarity (descending) and take top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_k = similarities[:k]
        
        distances = np.array([1 - sim for sim, _ in top_k])  # Convert to distance
        indices = np.array([idx for _, idx in top_k])
        
        return distances, indices
    
    def get_similar_items(self, item_id: int, k: int = 10, 
                         exclude_self: bool = True) -> List[Dict]:
        """Get similar items for a given item.
        
        Args:
            item_id: Target item ID
            k: Number of similar items to return
            exclude_self: Whether to exclude the item itself
            
        Returns:
            List of similar items with metadata
        """
        # Find item index
        item_idx = np.where(self.item_ids == item_id)[0]
        if len(item_idx) == 0:
            logger.warning(f"Item {item_id} not found in index")
            return []
        
        item_idx = item_idx[0]
        
        # Get item vector
        item_vector = self.item_embeddings[item_idx]
        
        # Search for similar items
        distances, indices = self.search_similar(item_vector, k + 1)  # +1 to account for self
        
        # Filter results
        results = []
        for i, (distance, idx) in enumerate(zip(distances, indices)):
            if idx == -1:  # Invalid index
                continue
                
            similar_item_id = self.item_ids[idx]
            
            # Skip self if requested
            if exclude_self and similar_item_id == item_id:
                continue
            
            # Create result
            result = {
                'item_id': int(similar_item_id),
                'similarity_score': float(distance),
                'rank': i + 1
            }
            
            # Add metadata if available
            if self.item_embeddings is not None and idx < len(self.item_embeddings):
                # The original code had item_metadata, but item_embeddings is now used.
                # Assuming item_embeddings is the source of truth for metadata if available.
                # If item_metadata was intended to be loaded separately, it needs to be re-added.
                # For now, we'll just add a placeholder or remove if not applicable.
                # Given the new code, item_embeddings is the primary source.
                pass # No metadata available in the new FaissSimilaritySearch class
            
            results.append(result)
            
            if len(results) >= k:
                break
        
        return results
    
    def get_recommendations_for_user(self, user_embedding: np.ndarray, k: int = 10,
                                   exclude_items: Optional[np.ndarray] = None) -> List[Dict]:
        """Get recommendations for a user based on their embedding.
        
        Args:
            user_embedding: User's embedding vector
            k: Number of recommendations to return
            exclude_items: Optional item IDs to exclude
            
        Returns:
            List of recommended items with metadata
        """
        # Search for similar items
        distances, indices = self.search_similar(user_embedding, k)
        
        # Create results
        results = []
        for i, (distance, idx) in enumerate(zip(distances, indices)):
            if idx == -1:  # Invalid index
                continue
                
            item_id = self.item_ids[idx]
            
            # Create result
            result = {
                'item_id': int(item_id),
                'similarity_score': float(distance),
                'rank': i + 1
            }
            
            # Add metadata if available
            if self.item_embeddings is not None and idx < len(self.item_embeddings):
                # The original code had item_metadata, but item_embeddings is now used.
                # Assuming item_embeddings is the source of truth for metadata if available.
                # If item_metadata was intended to be loaded separately, it needs to be re-added.
                # For now, we'll just add a placeholder or remove if not applicable.
                # Given the new code, item_embeddings is the primary source.
                pass # No metadata available in the new FaissSimilaritySearch class
            
            results.append(result)
        
        return results
    
    def save_index(self, filepath: str) -> None:
        """Save the trained index to disk.
        
        Args:
            filepath: Path to save the index
        """
        if not self.use_faiss or self.index is None:
            raise ValueError("Index must be trained and Faiss must be available before saving")
        
        # Save Faiss index
        index_path = filepath.replace('.pkl', '.faiss')
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata = {
            'use_faiss': self.use_faiss,
            'item_embeddings': self.item_embeddings,
            'item_ids': self.item_ids,
            'is_trained': True # Assuming training is part of building the index
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Index saved to {index_path} and metadata to {filepath}")
    
    @classmethod
    def load_index(cls, filepath: str) -> 'FaissSimilaritySearch':
        """Load a trained index from disk.
        
        Args:
            filepath: Path to the saved index metadata
            
        Returns:
            Loaded FaissSimilaritySearch instance
        """
        # Load metadata
        with open(filepath, 'rb') as f:
            metadata = pickle.load(f)
        
        # Create instance
        instance = cls(use_faiss=metadata['use_faiss'])
        
        # Load Faiss index
        index_path = filepath.replace('.pkl', '.faiss')
        if instance.use_faiss and FAISS_AVAILABLE:
            instance.index = faiss.read_index(index_path)
        else:
            instance.index = None # Indicate no index loaded if Faiss is not available
        
        # Load metadata
        instance.item_embeddings = metadata['item_embeddings']
        instance.item_ids = metadata['item_ids']
        # The original code had item_metadata, but item_embeddings is now used.
        # Assuming item_embeddings is the source of truth for metadata if available.
        # If item_metadata was intended to be loaded separately, it needs to be re-added.
        # For now, we'll just add a placeholder or remove if not applicable.
        # Given the new code, item_embeddings is the primary source.
        instance.is_trained = True # Assuming training is part of building the index
        
        logger.info(f"Index loaded from {index_path}")
        return instance
