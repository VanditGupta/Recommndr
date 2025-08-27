"""Faiss-based similarity search for fast vector retrieval."""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union
import faiss
import pickle
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)


class FaissSimilaritySearch:
    """Faiss-based similarity search for fast vector retrieval."""
    
    def __init__(self, index_type: str = "IVF", n_lists: int = 100, 
                 nprobe: int = 10, metric: str = "cosine"):
        """Initialize Faiss similarity search.
        
        Args:
            index_type: Type of Faiss index ("IVF", "HNSW", "Flat")
            n_lists: Number of clusters for IVF index
            nprobe: Number of clusters to probe during search
            metric: Distance metric ("cosine", "euclidean", "ip")
        """
        self.index_type = index_type
        self.n_lists = n_lists
        self.nprobe = nprobe
        self.metric = metric
        
        # Faiss index
        self.index = None
        self.dimension = None
        self.is_trained = False
        
        # Metadata
        self.item_ids = None
        self.item_metadata = None
        
        logger.info(f"Initialized Faiss search with {index_type} index, {metric} metric")
    
    def build_index(self, vectors: np.ndarray, item_ids: np.ndarray, 
                    item_metadata: Optional[pd.DataFrame] = None) -> 'FaissSimilaritySearch':
        """Build and train the Faiss index.
        
        Args:
            vectors: Input vectors (n_items, n_dimensions)
            item_ids: Corresponding item IDs
            item_metadata: Optional item metadata DataFrame
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Building Faiss index for {len(vectors)} vectors...")
        
        self.dimension = vectors.shape[1]
        self.item_ids = item_ids
        self.item_metadata = item_metadata
        
        # Normalize vectors for cosine similarity
        if self.metric == "cosine":
            vectors = self._normalize_vectors(vectors)
        
        # Create appropriate index type
        if self.index_type == "IVF":
            self.index = self._create_ivf_index(vectors)
        elif self.index_type == "HNSW":
            self.index = self._create_hnsw_index(vectors)
        elif self.index_type == "Flat":
            self.index = self._create_flat_index(vectors)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        # Train the index
        if hasattr(self.index, 'train'):
            logger.info("Training Faiss index...")
            self.index.train(vectors)
        
        # Add vectors to index
        self.index.add(vectors)
        self.is_trained = True
        
        logger.info(f"✅ Faiss index built successfully! Index size: {self.index.ntotal}")
        return self
    
    def _create_ivf_index(self, vectors: np.ndarray) -> faiss.Index:
        """Create IVF (Inverted File) index."""
        if self.metric == "cosine":
            index = faiss.IndexIVFFlat(faiss.IndexFlatIP(self.dimension), 
                                     self.dimension, self.n_lists)
        elif self.metric == "euclidean":
            index = faiss.IndexIVFFlat(faiss.IndexFlatL2(self.dimension), 
                                     self.dimension, self.n_lists)
        else:
            index = faiss.IndexIVFFlat(faiss.IndexFlatIP(self.dimension), 
                                     self.dimension, self.n_lists)
        
        index.nprobe = self.nprobe
        return index
    
    def _create_hnsw_index(self, vectors: np.ndarray) -> faiss.Index:
        """Create HNSW (Hierarchical Navigable Small World) index."""
        if self.metric == "cosine":
            index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 neighbors
        elif self.metric == "euclidean":
            index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            index = faiss.IndexHNSWFlat(self.dimension, 32)
        
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 100
        return index
    
    def _create_flat_index(self, vectors: np.ndarray) -> faiss.Index:
        """Create Flat index (exact search)."""
        if self.metric == "cosine":
            return faiss.IndexFlatIP(self.dimension)
        elif self.metric == "euclidean":
            return faiss.IndexFlatL2(self.dimension)
        else:
            return faiss.IndexFlatIP(self.dimension)
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return vectors / norms
    
    def search(self, query_vector: np.ndarray, k: int = 10, 
               filter_ids: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            filter_ids: Optional item IDs to filter results
            
        Returns:
            Tuple of (distances, item_indices)
        """
        if not self.is_trained:
            raise ValueError("Index must be trained before searching")
        
        # Normalize query vector for cosine similarity
        if self.metric == "cosine":
            query_vector = self._normalize_vectors(query_vector.reshape(1, -1)).flatten()
        
        # Perform search
        if filter_ids is not None:
            # Filtered search
            distances, indices = self._filtered_search(query_vector, k, filter_ids)
        else:
            # Regular search
            distances, indices = self.index.search(query_vector.reshape(1, -1), k)
            distances = distances.flatten()
            indices = indices.flatten()
        
        return distances, indices
    
    def _filtered_search(self, query_vector: np.ndarray, k: int, 
                        filter_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform filtered search using item IDs."""
        # Create a mask for filtered items
        filter_mask = np.isin(self.item_ids, filter_ids)
        filtered_indices = np.where(filter_mask)[0]
        
        if len(filtered_indices) == 0:
            return np.array([]), np.array([])
        
        # Get filtered vectors
        filtered_vectors = self.index.reconstruct_n(0, self.index.ntotal)[filtered_indices]
        
        # Calculate similarities manually for filtered subset
        if self.metric == "cosine":
            similarities = np.dot(filtered_vectors, query_vector)
        elif self.metric == "euclidean":
            distances = np.linalg.norm(filtered_vectors - query_vector, axis=1)
            similarities = -distances  # Convert to similarities
        else:
            similarities = np.dot(filtered_vectors, query_vector)
        
        # Get top-k results
        top_k_indices = np.argsort(similarities)[::-1][:k]
        top_k_similarities = similarities[top_k_indices]
        top_k_filtered_indices = filtered_indices[top_k_indices]
        
        return top_k_similarities, top_k_filtered_indices
    
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
        item_vector = self.index.reconstruct(item_idx)
        
        # Search for similar items
        distances, indices = self.search(item_vector, k + 1)  # +1 to account for self
        
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
            if self.item_metadata is not None and idx < len(self.item_metadata):
                metadata = self.item_metadata.iloc[idx]
                result['metadata'] = metadata.to_dict()
            
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
        distances, indices = self.search(user_embedding, k, exclude_items)
        
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
            if self.item_metadata is not None and idx < len(self.item_metadata):
                metadata = self.item_metadata.iloc[idx]
                result['metadata'] = metadata.to_dict()
            
            results.append(result)
        
        return results
    
    def save_index(self, filepath: str) -> None:
        """Save the trained index to disk.
        
        Args:
            filepath: Path to save the index
        """
        if not self.is_trained:
            raise ValueError("Index must be trained before saving")
        
        # Save Faiss index
        index_path = filepath.replace('.pkl', '.faiss')
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata = {
            'index_type': self.index_type,
            'n_lists': self.n_lists,
            'nprobe': self.nprobe,
            'metric': self.metric,
            'dimension': self.dimension,
            'item_ids': self.item_ids,
            'item_metadata': self.item_metadata,
            'is_trained': self.is_trained
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
        instance = cls(
            index_type=metadata['index_type'],
            n_lists=metadata['n_lists'],
            nprobe=metadata['nprobe'],
            metric=metadata['metric']
        )
        
        # Load Faiss index
        index_path = filepath.replace('.pkl', '.faiss')
        instance.index = faiss.read_index(index_path)
        
        # Load metadata
        instance.dimension = metadata['dimension']
        instance.item_ids = metadata['item_ids']
        instance.item_metadata = metadata['item_metadata']
        instance.is_trained = metadata['is_trained']
        
        logger.info(f"Index loaded from {index_path}")
        return instance
