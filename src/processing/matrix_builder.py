"""Build user-item matrix and mappings for Phase 3: Candidate Generation."""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import pickle
from pathlib import Path
from typing import Tuple, Dict

from config.settings import get_data_path
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_user_item_matrix() -> Tuple[csr_matrix, Dict, Dict]:
    """Build user-item interaction matrix and mappings.
    
    Returns:
        Tuple of (user_item_matrix, user_mapping, item_mapping)
    """
    logger.info("🔨 Building user-item matrix for Phase 3...")
    
    # Load processed data
    data_dir = get_data_path("processed")
    
    interactions_path = data_dir / "interactions_cleaned.parquet"
    users_path = data_dir / "users_cleaned.parquet"
    products_path = data_dir / "products_cleaned.parquet"
    
    if not interactions_path.exists():
        raise FileNotFoundError(f"Interactions file not found: {interactions_path}")
    
    logger.info("📊 Loading processed data...")
    interactions = pd.read_parquet(interactions_path)
    users = pd.read_parquet(users_path)
    products = pd.read_parquet(products_path)
    
    logger.info(f"✅ Loaded data:")
    logger.info(f"   Interactions: {len(interactions):,}")
    logger.info(f"   Users: {len(users):,}")
    logger.info(f"   Products: {len(products):,}")
    
    # Create mappings
    logger.info("🔗 Creating ID mappings...")
    user_ids = sorted(users['user_id'].unique())
    product_ids = sorted(products['product_id'].unique())
    
    user_mapping = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_mapping = {product_id: idx for idx, product_id in enumerate(product_ids)}
    
    logger.info(f"   User mapping: {len(user_mapping):,} users")
    logger.info(f"   Item mapping: {len(item_mapping):,} items")
    
    # Create sparse matrix
    logger.info("📐 Building sparse user-item matrix...")
    
    # Map user and product IDs to indices
    interactions['user_idx'] = interactions['user_id'].map(user_mapping)
    interactions['product_idx'] = interactions['product_id'].map(item_mapping)
    
    # Remove any interactions with missing mappings
    valid_interactions = interactions.dropna(subset=['user_idx', 'product_idx'])
    
    if len(valid_interactions) < len(interactions):
        logger.warning(f"Removed {len(interactions) - len(valid_interactions)} invalid interactions")
    
    # Create interaction values (use rating if available, otherwise 1)
    if 'rating' in valid_interactions.columns:
        interaction_values = valid_interactions['rating'].fillna(1)
        logger.info("Using rating values for interactions")
    else:
        interaction_values = np.ones(len(valid_interactions))
        logger.info("Using binary interaction values (1 for any interaction)")
    
    # Build sparse matrix
    matrix = csr_matrix(
        (interaction_values, 
         (valid_interactions['user_idx'], valid_interactions['product_idx'])),
        shape=(len(user_mapping), len(item_mapping))
    )
    
    # Calculate sparsity
    sparsity = 1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])
    
    logger.info(f"✅ Matrix built successfully!")
    logger.info(f"   Shape: {matrix.shape[0]:,} × {matrix.shape[1]:,}")
    logger.info(f"   Non-zero elements: {matrix.nnz:,}")
    logger.info(f"   Sparsity: {sparsity:.4f}")
    
    return matrix, user_mapping, item_mapping


def save_matrix_and_mappings(matrix: csr_matrix, user_mapping: Dict, item_mapping: Dict, 
                           output_dir: Path = None) -> None:
    """Save the matrix and mappings to disk.
    
    Args:
        matrix: User-item interaction matrix
        user_mapping: User ID to index mapping
        item_mapping: Product ID to index mapping
        output_dir: Directory to save files
    """
    if output_dir is None:
        output_dir = get_data_path("processed")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"💾 Saving matrix and mappings to {output_dir}...")
    
    # Save sparse matrix
    matrix_path = output_dir / "user_item_matrix.npz"
    from scipy.sparse import save_npz
    save_npz(str(matrix_path), matrix)
    logger.info(f"   ✅ Matrix saved: {matrix_path}")
    
    # Save mappings
    user_mapping_path = output_dir / "user_mapping.pkl"
    with open(user_mapping_path, 'wb') as f:
        pickle.dump(user_mapping, f)
    logger.info(f"   ✅ User mapping saved: {user_mapping_path}")
    
    item_mapping_path = output_dir / "item_mapping.pkl"
    with open(item_mapping_path, 'wb') as f:
        pickle.dump(item_mapping, f)
    logger.info(f"   ✅ Item mapping saved: {item_mapping_path}")
    
    # Save matrix info
    matrix_info = {
        'shape': matrix.shape,
        'nnz': matrix.nnz,
        'sparsity': 1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1]),
        'dtype': str(matrix.dtype),
        'format': matrix.format
    }
    
    info_path = output_dir / "matrix_info.json"
    import json
    with open(info_path, 'w') as f:
        json.dump(matrix_info, f, indent=2)
    logger.info(f"   ✅ Matrix info saved: {info_path}")


def main():
    """Main function to build and save the matrix."""
    try:
        logger.info("🚀 Starting matrix building process...")
        
        # Build matrix and mappings
        matrix, user_mapping, item_mapping = build_user_item_matrix()
        
        # Save to disk
        save_matrix_and_mappings(matrix, user_mapping, item_mapping)
        
        logger.info("🎉 Matrix building completed successfully!")
        logger.info(f"📊 Final matrix: {matrix.shape[0]:,} users × {matrix.shape[1]:,} products")
        logger.info(f"🔗 Ready for Phase 3: Candidate Generation!")
        
    except Exception as e:
        logger.error(f"❌ Matrix building failed: {e}")
        raise


if __name__ == "__main__":
    main()
