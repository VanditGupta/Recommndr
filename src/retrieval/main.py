"""Phase 3: Candidate Generation Pipeline using ALS and Faiss."""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import pickle

from src.retrieval.models.als_model import ALSModel
from src.retrieval.similarity.faiss_search import FaissSimilaritySearch
from src.utils.logging import get_logger
from config.settings import get_data_path

logger = get_logger(__name__)


class CandidateGenerationPipeline:
    """Main pipeline for candidate generation using ALS and Faiss."""
    
    def __init__(self, data_dir: Optional[Path] = None, 
                 models_dir: Optional[Path] = None):
        """Initialize the candidate generation pipeline.
        
        Args:
            data_dir: Directory containing processed data
            models_dir: Directory to save trained models
        """
        self.data_dir = data_dir or get_data_path("processed")
        self.models_dir = models_dir or Path("models/phase3")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Pipeline components
        self.als_model = None
        self.faiss_search = None
        
        # Data
        self.user_item_matrix = None
        self.user_mapping = None
        self.item_mapping = None
        self.user_features = None
        self.item_features = None
        
        logger.info("🚀 Candidate Generation Pipeline initialized")
    
    def load_data(self) -> 'CandidateGenerationPipeline':
        """Load processed data for training."""
        logger.info("Loading processed data...")
        
        try:
            # Load user-item matrix
            matrix_path = self.data_dir / "user_item_matrix.npz"
            if matrix_path.exists():
                self.user_item_matrix = csr_matrix.load_npz(str(matrix_path))
                logger.info(f"✅ Loaded user-item matrix: {self.user_item_matrix.shape}")
            else:
                raise FileNotFoundError(f"User-item matrix not found at {matrix_path}")
            
            # Load mappings
            with open(self.data_dir / "user_mapping.pkl", "rb") as f:
                self.user_mapping = pickle.load(f)
            with open(self.data_dir / "item_mapping.pkl", "rb") as f:
                self.item_mapping = pickle.load(f)
            
            # Load features if available
            features_path = self.data_dir / "user_features.parquet"
            if features_path.exists():
                self.user_features = pd.read_parquet(features_path)
                logger.info(f"✅ Loaded user features: {self.user_features.shape}")
            
            features_path = self.data_dir / "item_features.parquet"
            if features_path.exists():
                self.item_features = pd.read_parquet(features_path)
                logger.info(f"✅ Loaded item features: {self.item_features.shape}")
            
            logger.info(f"📊 Data loaded successfully!")
            logger.info(f"   Users: {len(self.user_mapping)}")
            logger.info(f"   Items: {len(self.item_mapping)}")
            logger.info(f"   Interactions: {self.user_item_matrix.nnz}")
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
        
        return self
    
    def train_als_model(self, n_factors: int = 100, n_iterations: int = 20,
                       regularization: float = 0.1) -> 'CandidateGenerationPipeline':
        """Train the ALS model for collaborative filtering."""
        logger.info("🎯 Training ALS model...")
        
        try:
            # Initialize ALS model
            self.als_model = ALSModel(
                n_factors=n_factors,
                n_iterations=n_iterations,
                regularization=regularization
            )
            
            # Train the model
            start_time = time.time()
            self.als_model.fit(self.user_item_matrix)
            training_time = time.time() - start_time
            
            logger.info(f"✅ ALS model trained successfully in {training_time:.2f}s")
            logger.info(f"   Final loss: {self.als_model.training_loss[-1]:.4f}")
            
            # Save the model
            model_path = self.models_dir / "als_model.pkl"
            self.als_model.save_model(str(model_path))
            
        except Exception as e:
            logger.error(f"Failed to train ALS model: {e}")
            raise
        
        return self
    
    def build_faiss_index(self, index_type: str = "IVF", n_lists: int = 100,
                         metric: str = "cosine") -> 'CandidateGenerationPipeline':
        """Build Faiss index for fast similarity search."""
        logger.info("🔍 Building Faiss similarity index...")
        
        try:
            # Get item embeddings from ALS model
            if self.als_model is None:
                raise ValueError("ALS model must be trained before building Faiss index")
            
            item_embeddings = self.als_model.get_item_embeddings()
            item_ids = np.array(list(self.item_mapping.keys()))
            
            # Initialize Faiss search
            self.faiss_search = FaissSimilaritySearch(
                index_type=index_type,
                n_lists=min(n_lists, len(item_embeddings) // 10),  # Ensure n_lists < n_items
                metric=metric
            )
            
            # Build the index
            start_time = time.time()
            self.faiss_search.build_index(
                vectors=item_embeddings,
                item_ids=item_ids,
                item_metadata=self.item_features
            )
            build_time = time.time() - start_time
            
            logger.info(f"✅ Faiss index built successfully in {build_time:.2f}s")
            logger.info(f"   Index type: {index_type}")
            logger.info(f"   Metric: {metric}")
            logger.info(f"   Vectors: {len(item_embeddings)}")
            
            # Save the index
            index_path = self.models_dir / "faiss_index.pkl"
            self.faiss_search.save_index(str(index_path))
            
        except Exception as e:
            logger.error(f"Failed to build Faiss index: {e}")
            raise
        
        return self
    
    def generate_candidates(self, user_id: int, k: int = 20,
                          exclude_interacted: bool = True) -> List[Dict]:
        """Generate candidate items for a user.
        
        Args:
            user_id: Target user ID
            k: Number of candidates to generate
            exclude_interacted: Whether to exclude items the user has interacted with
            
        Returns:
            List of candidate items with metadata
        """
        if self.als_model is None or self.faiss_search is None:
            raise ValueError("Both ALS model and Faiss index must be available")
        
        try:
            # Get user embedding
            user_idx = self.user_mapping.get(user_id)
            if user_idx is None:
                logger.warning(f"User {user_id} not found in mapping")
                return []
            
            user_embedding = self.als_model.get_user_embeddings([user_idx])[0]
            
            # Get items to exclude
            exclude_items = None
            if exclude_interacted:
                user_interactions = self.user_item_matrix[user_idx].nonzero()[1]
                exclude_items = np.array([list(self.item_mapping.keys())[idx] for idx in user_interactions])
            
            # Get recommendations
            candidates = self.faiss_search.get_recommendations_for_user(
                user_embedding=user_embedding,
                k=k,
                exclude_items=exclude_items
            )
            
            logger.info(f"Generated {len(candidates)} candidates for user {user_id}")
            return candidates
            
        except Exception as e:
            logger.error(f"Failed to generate candidates for user {user_id}: {e}")
            return []
    
    def evaluate_model(self, test_users: Optional[List[int]] = None,
                      n_test_users: int = 100) -> Dict:
        """Evaluate the candidate generation model."""
        logger.info("📊 Evaluating candidate generation model...")
        
        if test_users is None:
            # Sample test users
            all_users = list(self.user_mapping.keys())
            test_users = np.random.choice(all_users, 
                                        size=min(n_test_users, len(all_users)), 
                                        replace=False)
        
        metrics = {
            'n_test_users': len(test_users),
            'avg_candidates_per_user': 0,
            'avg_similarity_score': 0,
            'coverage': set(),
            'diversity': 0
        }
        
        total_candidates = 0
        total_similarity = 0
        all_recommended_items = set()
        
        for user_id in test_users:
            candidates = self.generate_candidates(user_id, k=20)
            
            if candidates:
                total_candidates += len(candidates)
                total_similarity += sum(c['similarity_score'] for c in candidates)
                all_recommended_items.update(c['item_id'] for c in candidates)
        
        # Calculate metrics
        if total_candidates > 0:
            metrics['avg_candidates_per_user'] = total_candidates / len(test_users)
            metrics['avg_similarity_score'] = total_similarity / total_candidates
            metrics['coverage'] = len(all_recommended_items)
            metrics['diversity'] = len(all_recommended_items) / len(self.item_mapping)
        
        logger.info(f"✅ Model evaluation completed!")
        logger.info(f"   Average candidates per user: {metrics['avg_candidates_per_user']:.2f}")
        logger.info(f"   Average similarity score: {metrics['avg_similarity_score']:.4f}")
        logger.info(f"   Coverage: {metrics['coverage']} items")
        logger.info(f"   Diversity: {metrics['diversity']:.4f}")
        
        return metrics
    
    def run_pipeline(self, n_factors: int = 100, n_iterations: int = 20,
                    regularization: float = 0.1, index_type: str = "IVF",
                    n_lists: int = 100, metric: str = "cosine") -> 'CandidateGenerationPipeline':
        """Run the complete candidate generation pipeline."""
        logger.info("🚀 Starting Phase 3: Candidate Generation Pipeline")
        
        start_time = time.time()
        
        try:
            # Load data
            self.load_data()
            
            # Train ALS model
            self.train_als_model(n_factors, n_iterations, regularization)
            
            # Build Faiss index
            self.build_faiss_index(index_type, n_lists, metric)
            
            # Evaluate model
            self.evaluate_model()
            
            pipeline_time = time.time() - start_time
            logger.info(f"🎉 Pipeline completed successfully in {pipeline_time:.2f}s!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        
        return self


def main():
    """Main entry point for the candidate generation pipeline."""
    parser = argparse.ArgumentParser(description="Phase 3: Candidate Generation Pipeline")
    parser.add_argument("--data-dir", type=str, help="Directory containing processed data")
    parser.add_argument("--models-dir", type=str, help="Directory to save trained models")
    parser.add_argument("--n-factors", type=int, default=100, help="Number of latent factors")
    parser.add_argument("--n-iterations", type=int, default=20, help="Number of training iterations")
    parser.add_argument("--regularization", type=float, default=0.1, help="Regularization parameter")
    parser.add_argument("--index-type", type=str, default="IVF", choices=["IVF", "HNSW", "Flat"], 
                       help="Faiss index type")
    parser.add_argument("--n-lists", type=int, default=100, help="Number of clusters for IVF")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean", "ip"],
                       help="Distance metric")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CandidateGenerationPipeline(
        data_dir=Path(args.data_dir) if args.data_dir else None,
        models_dir=Path(args.models_dir) if args.models_dir else None
    )
    
    # Run pipeline
    pipeline.run_pipeline(
        n_factors=args.n_factors,
        n_iterations=args.n_iterations,
        regularization=args.regularization,
        index_type=args.index_type,
        n_lists=args.n_lists,
        metric=args.metric
    )


if __name__ == "__main__":
    main()
