"""
Phase 4: Ranking Pipeline Main Module

Integrates the complete ranking pipeline:
1. ALS Candidate Generation (from Phase 3)
2. Feature Engineering for contextual ranking  
3. LightGBM training and inference
4. ONNX export for production serving
5. End-to-end recommendation pipeline (Retrieval + Ranking)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import argparse
from pathlib import Path
import logging
import pickle

from .feature_engineering import RankingFeatureEngineer
from .lightgbm_ranker import LightGBMRanker
from src.retrieval.main import CandidateGenerationPipeline
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RankingPipeline:
    """Complete ranking pipeline integrating ALS + LightGBM."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.feature_engineer = RankingFeatureEngineer()
        self.ranker = LightGBMRanker()
        self.candidate_generator = CandidateGenerationPipeline()
        
        # Load base data
        self.interactions_df = None
        self.users_df = None
        self.products_df = None
        
    def initialize(self):
        """Initialize the ranking pipeline."""
        logger.info("🚀 Initializing Phase 4 Ranking Pipeline...")
        
        # Load base features
        self.feature_engineer.load_base_features(self.data_dir)
        
        # Load candidate generation pipeline
        self.candidate_generator.load_data()
        
        # Load the trained ALS model and Faiss index
        try:
            # Load the trained ALS model
            als_model_path = Path("models/phase3/als_model.pkl")
            if als_model_path.exists():
                with open(als_model_path, 'rb') as f:
                    self.candidate_generator.als_model = pickle.load(f)
                logger.info("✅ ALS model loaded for candidate generation")
            else:
                logger.warning("ALS model file not found")
        except Exception as e:
            logger.warning(f"Could not load ALS model: {e}")
            
        try:
            # Load the trained Faiss index
            faiss_index_path = Path("models/phase3/faiss_index.pkl")
            if faiss_index_path.exists():
                with open(faiss_index_path, 'rb') as f:
                    self.candidate_generator.faiss_search = pickle.load(f)
                logger.info("✅ Faiss index loaded for candidate generation")
            else:
                logger.warning("Faiss index file not found")
        except Exception as e:
            logger.warning(f"Could not load Faiss index: {e}")
        
        # Load interaction data for training
        self.interactions_df = pd.read_parquet(f"{self.data_dir}/processed/interactions_cleaned.parquet")
        self.users_df = pd.read_parquet(f"{self.data_dir}/processed/users_cleaned.parquet")
        self.products_df = pd.read_parquet(f"{self.data_dir}/processed/products_cleaned.parquet")
        
        logger.info(f"✅ Pipeline initialized with {len(self.interactions_df)} interactions")
    
    def create_training_data(self, num_samples: Optional[int] = None, 
                           negative_ratio: float = 2.0) -> pd.DataFrame:
        """Create training data for the ranking model."""
        logger.info("📊 Creating training data for ranking model...")
        
        # Use subset for faster training if specified
        interactions_sample = self.interactions_df
        if num_samples:
            interactions_sample = self.interactions_df.sample(n=min(num_samples, len(self.interactions_df)))
            logger.info(f"Using {len(interactions_sample)} interaction samples")
        
        # Create training features
        training_df = self.feature_engineer.create_training_features(
            interactions_sample, 
            include_negative_samples=True,
            negative_ratio=negative_ratio
        )
        
        return training_df
    
    def train_ranking_model(self, training_df: Optional[pd.DataFrame] = None,
                          num_samples: int = 10000,
                          model_params: Optional[Dict] = None) -> Dict:
        """Train the LightGBM ranking model."""
        logger.info("🎯 Training LightGBM ranking model...")
        
        # Create training data if not provided
        if training_df is None:
            training_df = self.create_training_data(num_samples=num_samples)
        
        # Set custom model parameters if provided
        if model_params:
            self.ranker.model_params.update(model_params)
        
        # Train the model
        training_results = self.ranker.train(training_df, track_experiment=True)
        
        # Save the trained model
        model_path = Path("models/phase4/lightgbm_ranker.pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.ranker.save_model(str(model_path))
        
        return training_results
    
    def export_to_onnx(self, output_path: str = "models/phase4/lightgbm_ranker.onnx") -> str:
        """Export the trained model to ONNX format."""
        logger.info("📦 Exporting model to ONNX...")
        
        # Create sample input for ONNX export
        sample_features = self.feature_engineer.create_user_item_features(1, 1)
        feature_names = self.feature_engineer.get_feature_names()
        sample_input = np.array([[sample_features.get(name, 0.0) for name in feature_names]], dtype=np.float32)
        
        # Export to ONNX
        onnx_path = self.ranker.export_to_onnx(output_path, sample_input)
        
        return onnx_path
    
    def get_recommendations(self, user_id: int, top_k: int = 10, 
                          candidate_k: int = 100) -> List[Dict]:
        """Get top-k recommendations for a user using the complete pipeline."""
        logger.info(f"🎯 Getting recommendations for user {user_id}")
        
        start_time = time.time()
        
        # Step 1: Generate candidates using ALS (Phase 3)
        candidates = self.candidate_generator.generate_candidates(
            user_id, k=candidate_k
        )
        
        if not candidates:
            logger.warning(f"No candidates found for user {user_id}")
            return []
        
        candidate_items = [item['item_id'] for item in candidates]
        logger.info(f"📋 Generated {len(candidate_items)} candidates")
        
        # Step 2: Rank candidates using LightGBM (Phase 4)
        ranked_candidates = self.ranker.rank_candidates(
            user_id, candidate_items, self.feature_engineer
        )
        
        # Step 3: Enrich with product details and format results
        recommendations = []
        for item_id, score in ranked_candidates[:top_k]:
            product_info = self.products_df[self.products_df['product_id'] == item_id]
            if len(product_info) > 0:
                product_info = product_info.iloc[0]
                recommendations.append({
                    'product_id': item_id,
                    'name': product_info['name'],
                    'category': product_info['category'],
                    'brand': product_info['brand'],
                    'price': float(product_info['price']),
                    'rating': float(product_info['rating']),
                    'ranking_score': float(score),
                    'description': product_info.get('description', '')[:100] + '...'
                })
        
        total_time = time.time() - start_time
        logger.info(f"✅ Generated {len(recommendations)} recommendations in {total_time:.3f}s")
        
        return recommendations
    
    def evaluate_ranking_model(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate the ranking model performance."""
        logger.info("📊 Evaluating ranking model...")
        
        # Prepare test data
        X_test, y_test, _ = self.ranker.prepare_training_data(test_df)
        
        # Make predictions
        predictions = self.ranker.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mae': mean_absolute_error(y_test, predictions),
            'r2': r2_score(y_test, predictions)
        }
        
        logger.info(f"📈 Evaluation Results:")
        for metric, value in metrics.items():
            logger.info(f"   {metric.upper()}: {value:.4f}")
        
        return metrics
    
    def benchmark_inference_speed(self, num_requests: int = 1000) -> Dict[str, float]:
        """Benchmark inference speed for the ranking pipeline."""
        logger.info(f"⚡ Benchmarking inference speed with {num_requests} requests...")
        
        # Random user-item pairs for testing
        random_users = np.random.choice(list(self.feature_engineer.user_mapping.keys()), num_requests)
        random_items = np.random.choice(list(self.feature_engineer.item_mapping.keys()), num_requests)
        
        # Benchmark feature engineering
        start_time = time.time()
        for user_id, item_id in zip(random_users, random_items):
            self.feature_engineer.create_user_item_features(user_id, item_id)
        feature_time = time.time() - start_time
        
        # Benchmark model inference
        features_batch = []
        for user_id, item_id in zip(random_users[:100], random_items[:100]):  # Smaller batch for model
            features = self.feature_engineer.create_user_item_features(user_id, item_id)
            feature_vector = [features.get(name, 0.0) for name in self.ranker.feature_names]
            features_batch.append(feature_vector)
        
        features_array = np.array(features_batch)
        
        start_time = time.time()
        predictions = self.ranker.predict(features_array)
        inference_time = time.time() - start_time
        
        # Calculate metrics
        benchmark_results = {
            'feature_engineering_ms_per_request': (feature_time / num_requests) * 1000,
            'model_inference_ms_per_request': (inference_time / 100) * 1000,
            'total_requests': num_requests,
            'feature_requests_per_second': num_requests / feature_time,
            'model_requests_per_second': 100 / inference_time
        }
        
        logger.info("⚡ Benchmark Results:")
        logger.info(f"   Feature Engineering: {benchmark_results['feature_engineering_ms_per_request']:.2f} ms/request")
        logger.info(f"   Model Inference: {benchmark_results['model_inference_ms_per_request']:.2f} ms/request")
        logger.info(f"   Feature RPS: {benchmark_results['feature_requests_per_second']:.0f}")
        logger.info(f"   Model RPS: {benchmark_results['model_requests_per_second']:.0f}")
        
        return benchmark_results
    
    def run_end_to_end_demo(self, test_users: List[int] = None) -> Dict:
        """Run an end-to-end demonstration of the ranking pipeline."""
        logger.info("🎭 Running end-to-end ranking pipeline demo...")
        
        if test_users is None:
            test_users = [1, 1118, 3194]  # Default test users
        
        demo_results = []
        
        for user_id in test_users:
            logger.info(f"👤 Testing recommendations for User {user_id}")
            
            # Get user info
            user_info = self.users_df[self.users_df['user_id'] == user_id]
            if len(user_info) > 0:
                user_info = user_info.iloc[0]
                logger.info(f"   Profile: {user_info['age']}y {user_info['gender']} from {user_info['location']}")
            
            # Get recommendations
            start_time = time.time()
            recommendations = self.get_recommendations(user_id, top_k=5)
            recommendation_time = time.time() - start_time
            
            # Display results
            logger.info(f"   🎯 Top recommendations (generated in {recommendation_time:.3f}s):")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"      {i}. {rec['name']} ({rec['category']}) - Score: {rec['ranking_score']:.3f}")
            
            demo_results.append({
                'user_id': user_id,
                'recommendations': recommendations,
                'generation_time': recommendation_time
            })
        
        return {
            'demo_results': demo_results,
            'avg_generation_time': np.mean([r['generation_time'] for r in demo_results])
        }


def main():
    """Main function to run Phase 4 ranking pipeline."""
    parser = argparse.ArgumentParser(description="Phase 4: Ranking Pipeline")
    parser.add_argument('--train', action='store_true', help='Train the ranking model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--export-onnx', action='store_true', help='Export model to ONNX')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark inference speed')
    parser.add_argument('--demo', action='store_true', help='Run end-to-end demo')
    parser.add_argument('--user-id', type=int, help='Get recommendations for specific user')
    parser.add_argument('--num-samples', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--top-k', type=int, default=10, help='Number of recommendations')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RankingPipeline()
    pipeline.initialize()
    
    if args.train:
        logger.info("🎯 Training ranking model...")
        results = pipeline.train_ranking_model(num_samples=args.num_samples)
        logger.info(f"✅ Training completed with RMSE: {results['metrics']['val_rmse']:.4f}")
    
    if args.export_onnx:
        logger.info("📦 Exporting to ONNX...")
        # Load the trained model first
        model_path = "models/phase4/lightgbm_ranker.pkl"
        pipeline.ranker.load_model(model_path)
        onnx_path = pipeline.export_to_onnx()
        logger.info(f"✅ Model exported to: {onnx_path}")
    
    if args.evaluate:
        logger.info("📊 Evaluating model...")
        test_df = pipeline.create_training_data(num_samples=1000)
        metrics = pipeline.evaluate_ranking_model(test_df)
    
    if args.benchmark:
        logger.info("⚡ Benchmarking inference speed...")
        benchmark_results = pipeline.benchmark_inference_speed()
    
    if args.demo:
        logger.info("🎭 Running end-to-end demo...")
        demo_results = pipeline.run_end_to_end_demo()
        logger.info(f"✅ Demo completed. Avg generation time: {demo_results['avg_generation_time']:.3f}s")
    
    if args.user_id:
        logger.info(f"🎯 Getting recommendations for User {args.user_id}...")
        # Load the trained model first
        model_path = "models/phase4/lightgbm_ranker.pkl"
        pipeline.ranker.load_model(model_path)
        recommendations = pipeline.get_recommendations(args.user_id, top_k=args.top_k)
        
        print(f"\n🎁 Top {len(recommendations)} Recommendations for User {args.user_id}:")
        print("=" * 80)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i:2d}. {rec['name']}")
            print(f"    Category: {rec['category']} | Brand: {rec['brand']}")
            print(f"    Price: ${rec['price']:.2f} | Rating: {rec['rating']:.1f}/5")
            print(f"    Ranking Score: {rec['ranking_score']:.3f}")
            print()


if __name__ == "__main__":
    main()
