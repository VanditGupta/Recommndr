"""
Phase 5: Similarity Layer

This module implements item-item similarity computation using:
- ALS latent embeddings (cosine similarity)
- Co-purchase matrix patterns  
- Fast similarity search and API serving
"""

from .item_similarity import ItemSimilarityEngine
from .copurchase_similarity import CoPurchaseSimilarity
from .similarity_api import SimilarityAPI

__all__ = [
    'ItemSimilarityEngine',
    'CoPurchaseSimilarity', 
    'SimilarityAPI'
]
