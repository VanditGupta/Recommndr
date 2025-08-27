"""Optimized ALS Model for Collaborative Filtering with MLflow Integration."""

import numpy as np
import numpy.linalg as LA
from scipy.sparse import csr_matrix
import time
from typing import Optional, List, Dict, Any
import logging
import mlflow
import mlflow.pytorch
from pathlib import Path

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimizedALSModel:
    """Optimized Alternating Least Squares for Collaborative Filtering with MLflow tracking."""
    
    def __init__(self, n_factors: int = 100, n_iterations: int = 20, 
                 regularization: float = 0.1, random_state: int = 42,
                 use_blas: bool = True, batch_size: int = 1000,
                 experiment_name: str = "als_training", run_name: str = None):
        """Initialize optimized ALS model with MLflow tracking.
        
        Args:
            n_factors: Number of latent factors
            n_iterations: Number of training iterations
            regularization: L2 regularization parameter
            random_state: Random seed for reproducibility
            use_blas: Whether to use BLAS-optimized operations
            batch_size: Batch size for vectorized updates
            experiment_name: MLflow experiment name
            run_name: MLflow run name (auto-generated if None)
        """
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.random_state = random_state
        self.use_blas = use_blas
        self.batch_size = batch_size
        self.experiment_name = experiment_name
        self.run_name = run_name or f"als_{n_factors}f_{n_iterations}i_{regularization}r"
        self.is_trained = False
        
        # Set random seed
        np.random.seed(random_state)
        
        # Training history
        self.training_loss = []
        self.training_times = []
        
        # MLflow setup
        self._setup_mlflow()
        
    def _setup_mlflow(self):
        """Setup MLflow experiment and run."""
        try:
            # Set experiment
            mlflow.set_experiment(self.experiment_name)
            
            # Start run
            self.mlflow_run = mlflow.start_run(run_name=self.run_name)
            
            # Log parameters
            mlflow.log_params({
                "n_factors": self.n_factors,
                "n_iterations": self.n_iterations,
                "regularization": self.regularization,
                "random_state": self.random_state,
                "use_blas": self.use_blas,
                "batch_size": self.batch_size
            })
            
            logger.info(f"✅ MLflow run started: {self.run_name}")
            
        except Exception as e:
            logger.warning(f"⚠️ MLflow setup failed: {e}")
            self.mlflow_run = None
        
    def fit(self, user_item_matrix: csr_matrix) -> 'OptimizedALSModel':
        """Train the optimized ALS model with MLflow tracking."""
        start_time = time.time()
        logger.info(f"🚀 Training optimized ALS with {self.n_factors} factors...")
        
        n_users, n_items = user_item_matrix.shape
        logger.info(f"📊 Matrix dimensions: {n_users:,} users × {n_items:,} items")
        
        # Log dataset info to MLflow
        if self.mlflow_run:
            mlflow.log_params({
                "n_users": n_users,
                "n_items": n_items,
                "n_interactions": user_item_matrix.nnz,
                "sparsity": 1 - user_item_matrix.nnz / (n_users * n_items)
            })
        
        # Initialize factors and biases with better initialization
        self.user_factors = np.random.randn(n_users, self.n_factors) * 0.1
        self.item_factors = np.random.randn(n_items, self.n_factors) * 0.1
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        self.global_bias = 0.0
        
        # Pre-compute regularization matrix
        reg_matrix = self.regularization * np.eye(self.n_factors)
        
        # Training loop with vectorized operations and MLflow tracking
        for iteration in range(self.n_iterations):
            iter_start = time.time()
            
            # Vectorized user factor updates
            self._update_user_factors_vectorized(user_item_matrix, reg_matrix)
            
            # Vectorized item factor updates
            self._update_item_factors_vectorized(user_item_matrix, reg_matrix)
            
            # Calculate loss (vectorized)
            loss = self._calculate_loss_vectorized(user_item_matrix)
            self.training_loss.append(loss)
            
            iter_time = time.time() - iter_start
            self.training_times.append(iter_time)
            
            # Log metrics to MLflow
            if self.mlflow_run:
                mlflow.log_metrics({
                    "loss": loss,
                    "iteration_time": iter_time,
                    "cumulative_time": sum(self.training_times)
                }, step=iteration)
            
            if iteration % 5 == 0 or iteration == self.n_iterations - 1:
                logger.info(f"   Iteration {iteration + 1}/{self.n_iterations}, "
                           f"Loss: {loss:.4f}, Time: {iter_time:.2f}s")
        
        self.is_trained = True
        total_time = time.time() - start_time
        avg_iter_time = np.mean(self.training_times)
        
        # Log final metrics to MLflow
        if self.mlflow_run:
            mlflow.log_metrics({
                "final_loss": loss,
                "total_training_time": total_time,
                "avg_iteration_time": avg_iter_time,
                "fastest_iteration": min(self.training_times),
                "slowest_iteration": max(self.training_times)
            })
            
            # Log training curves
            for i, (loss_val, iter_time) in enumerate(zip(self.training_loss, self.training_times)):
                mlflow.log_metrics({
                    "loss_history": loss_val,
                    "iteration_time_history": iter_time
                }, step=i)
        
        logger.info(f"🎉 Training completed in {total_time:.2f}s")
        logger.info(f"⚡ Average iteration time: {avg_iter_time:.2f}s")
        logger.info(f"📈 Final loss: {loss:.4f}")
        
        return self
    
    def _update_user_factors_vectorized(self, user_item_matrix: csr_matrix, reg_matrix: np.ndarray) -> None:
        """Vectorized user factor updates using batch processing."""
        n_users = user_item_matrix.shape[0]
        
        # Process users in batches for better memory efficiency
        for start_idx in range(0, n_users, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_users)
            batch_users = range(start_idx, end_idx)
            
            # Get batch data
            batch_matrix = user_item_matrix[batch_users]
            
            # Vectorized update for batch
            for user_idx in batch_users:
                user_items = user_item_matrix[user_idx].nonzero()[1]
                if len(user_items) == 0:
                    continue
                
                # Get ratings and adjust for biases
                ratings = user_item_matrix[user_idx, user_items].toarray().flatten()
                adjusted_ratings = (ratings - self.global_bias - 
                                  self.item_biases[user_items])
                
                # Vectorized factor update using BLAS
                item_factors_subset = self.item_factors[user_items]
                
                if self.use_blas:
                    # Use BLAS-optimized operations
                    A = item_factors_subset.T @ item_factors_subset + reg_matrix
                    b = item_factors_subset.T @ adjusted_ratings
                    self.user_factors[user_idx] = LA.solve(A, b)
                else:
                    # Fallback to standard numpy
                    A = item_factors_subset.T @ item_factors_subset + reg_matrix
                    b = item_factors_subset.T @ adjusted_ratings
                    self.user_factors[user_idx] = LA.solve(A, b)
                
                # Update bias
                self.user_biases[user_idx] = np.mean(adjusted_ratings - 
                                                    self.user_factors[user_idx] @ 
                                                    item_factors_subset.T)
    
    def _update_item_factors_vectorized(self, user_item_matrix: csr_matrix, reg_matrix: np.ndarray) -> None:
        """Vectorized item factor updates using batch processing."""
        n_items = user_item_matrix.shape[1]
        
        # Process items in batches
        for start_idx in range(0, n_items, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_items)
            batch_items = range(start_idx, end_idx)
            
            # Vectorized update for batch
            for item_idx in batch_items:
                item_users = user_item_matrix[:, item_idx].nonzero()[0]
                if len(item_users) == 0:
                    continue
                
                # Get ratings and adjust for biases
                ratings = user_item_matrix[item_users, item_idx].toarray().flatten()
                adjusted_ratings = (ratings - self.global_bias - 
                                  self.user_biases[item_users])
                
                # Vectorized factor update using BLAS
                user_factors_subset = self.user_factors[item_users]
                
                if self.use_blas:
                    # Use BLAS-optimized operations
                    A = user_factors_subset.T @ user_factors_subset + reg_matrix
                    b = user_factors_subset.T @ adjusted_ratings
                    self.item_factors[item_idx] = LA.solve(A, b)
                else:
                    # Fallback to standard numpy
                    A = user_factors_subset.T @ user_factors_subset + reg_matrix
                    b = user_factors_subset.T @ adjusted_ratings
                    self.item_factors[item_idx] = LA.solve(A, b)
                
                # Update bias
                self.item_biases[item_idx] = np.mean(adjusted_ratings - 
                                                    user_factors_subset @ 
                                                    self.item_factors[item_idx])
    
    def _calculate_loss_vectorized(self, user_item_matrix: csr_matrix) -> float:
        """Vectorized loss calculation for better performance."""
        total_loss = 0
        n_interactions = 0
        
        # Process in batches to avoid memory issues
        n_users = user_item_matrix.shape[0]
        
        for start_idx in range(0, n_users, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_users)
            batch_users = range(start_idx, end_idx)
            
            # Vectorized loss calculation for batch
            for user_idx in batch_users:
                user_items = user_item_matrix[user_idx].nonzero()[1]
                if len(user_items) == 0:
                    continue
                
                ratings = user_item_matrix[user_idx, user_items].toarray().flatten()
                
                # Vectorized prediction calculation
                user_factor = self.user_factors[user_idx]
                item_factors = self.item_factors[user_items]
                
                if self.use_blas:
                    # Use BLAS for matrix multiplication
                    predictions = (user_factor @ item_factors.T + 
                                 self.user_biases[user_idx] + 
                                 self.item_biases[user_items] + 
                                 self.global_bias)
                else:
                    # Standard numpy
                    predictions = (user_factor @ item_factors.T + 
                                 self.user_biases[user_idx] + 
                                 self.item_biases[user_items] + 
                                 self.global_bias)
                
                # Vectorized MSE calculation
                loss = np.mean((ratings - predictions) ** 2)
                total_loss += loss * len(user_items)
                n_interactions += len(user_items)
        
        # Vectorized regularization loss
        reg_loss = (self.regularization * 
                   (np.sum(self.user_factors ** 2) + np.sum(self.item_factors ** 2)))
        
        return (total_loss / n_interactions) + reg_loss
    
    def get_user_embeddings(self, user_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Get user embeddings."""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting embeddings")
        if user_ids is None:
            return self.user_factors
        return self.user_factors[user_ids]
    
    def get_item_embeddings(self, item_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Get item embeddings."""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting embeddings")
        if item_ids is None:
            return self.item_factors
        return self.item_factors[item_ids]
    
    def predict(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        """Predict ratings for user-item pairs."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        user_idx = user_id if isinstance(user_id, int) else user_id
        user_factor = self.user_factors[user_idx]
        item_factors = self.item_factors[item_ids]
        
        # Vectorized prediction
        predictions = (user_factor @ item_factors.T + 
                      self.user_biases[user_idx] + 
                      self.item_biases[item_ids] + 
                      self.global_bias)
        
        return predictions
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk and MLflow."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Save locally
        import pickle
        model_data = {
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'user_biases': self.user_biases,
            'item_biases': self.item_biases,
            'global_bias': self.global_bias,
            'n_factors': self.n_factors,
            'training_loss': self.training_loss,
            'training_times': self.training_times,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"💾 Model saved to {filepath}")
        
        # Log model to MLflow
        if self.mlflow_run:
            try:
                # Log model artifacts
                mlflow.log_artifact(filepath, "model")
                
                # Log model info
                mlflow.log_dict({
                    "model_type": "OptimizedALSModel",
                    "model_path": filepath,
                    "model_size_mb": Path(filepath).stat().st_size / (1024 * 1024),
                    "training_summary": {
                        "final_loss": self.training_loss[-1],
                        "total_time": sum(self.training_times),
                        "avg_iteration_time": np.mean(self.training_times)
                    }
                }, "model_info.json")
                
                logger.info("✅ Model logged to MLflow")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to log model to MLflow: {e}")
    
    def end_run(self):
        """End the MLflow run."""
        if self.mlflow_run:
            mlflow.end_run()
            logger.info("✅ MLflow run ended")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'OptimizedALSModel':
        """Load a trained model from disk."""
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(n_factors=model_data['n_factors'])
        model.user_factors = model_data['user_factors']
        model.item_factors = model_data['item_factors']
        model.user_biases = model_data['user_biases']
        model.item_biases = model_data['item_biases']
        model.global_bias = model_data['global_bias']
        model.training_loss = model_data.get('training_loss', [])
        model.training_times = model_data.get('training_times', [])
        model.is_trained = model_data['is_trained']
        
        logger.info(f"📂 Model loaded from {filepath}")
        return model


# Performance comparison function (updated for MLflow)
def compare_performance(user_item_matrix: csr_matrix, n_factors: int = 100, 
                       n_iterations: int = 10, experiment_name: str = "als_comparison") -> dict:
    """Compare performance between different model configurations with MLflow tracking."""
    print("🔍 Performance Comparison with MLflow Tracking")
    print("=" * 50)
    
    # Start MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    results = {}
    
    # Test different configurations
    configs = [
        {"n_factors": 50, "n_iterations": n_iterations, "name": "Small Model"},
        {"n_factors": 100, "n_iterations": n_iterations, "name": "Medium Model"},
        {"n_factors": 200, "n_iterations": n_iterations, "name": "Large Model"}
    ]
    
    for config in configs:
        print(f"\n📊 Testing {config['name']}...")
        
        with mlflow.start_run(run_name=f"{config['name']}_{config['n_factors']}f"):
            # Log configuration
            mlflow.log_params(config)
            
            # Train model
            start_time = time.time()
            model = OptimizedALSModel(
                n_factors=config['n_factors'],
                n_iterations=config['n_iterations'],
                experiment_name=experiment_name
            )
            model.fit(user_item_matrix)
            training_time = time.time() - start_time
            
            # Log results
            mlflow.log_metrics({
                "training_time": training_time,
                "final_loss": model.training_loss[-1],
                "avg_iteration_time": np.mean(model.training_times)
            })
            
            # Save results
            results[config['name']] = {
                'model': model,
                'training_time': training_time,
                'final_loss': model.training_loss[-1],
                'avg_iteration_time': np.mean(model.training_times)
            }
            
            print(f"   ✅ {config['name']} completed in {training_time:.2f}s")
            print(f"   📊 Final Loss: {model.training_loss[-1]:.4f}")
    
    # Print comparison
    print(f"\n🏆 Performance Comparison Results:")
    print("=" * 50)
    
    for name, result in results.items():
        print(f"   {name}:")
        print(f"      ⏱️  Time: {result['training_time']:.2f}s")
        print(f"      📊 Loss: {result['final_loss']:.4f}")
        print(f"      ⚡ Avg Iter: {result['avg_iteration_time']:.2f}s")
    
    return results


# Utility function for MLflow experiment management
def list_experiments():
    """List all MLflow experiments."""
    try:
        experiments = mlflow.search_experiments()
        print("📋 Available MLflow Experiments:")
        print("=" * 40)
        
        for exp in experiments:
            print(f"   🧪 {exp.name}")
            print(f"      ID: {exp.experiment_id}")
            print(f"      Artifact Location: {exp.artifact_location}")
            print()
            
    except Exception as e:
        print(f"❌ Failed to list experiments: {e}")


def get_best_run(experiment_name: str, metric: str = "final_loss", ascending: bool = True):
    """Get the best run from an experiment based on a metric."""
    try:
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"]
        )
        
        if len(runs) > 0:
            best_run = runs.iloc[0]
            print(f"🏆 Best Run in {experiment_name}:")
            print(f"   Run ID: {best_run['run_id']}")
            print(f"   {metric}: {best_run[f'metrics.{metric}']}")
            print(f"   Status: {best_run['status']}")
            return best_run
        else:
            print(f"❌ No runs found in experiment: {experiment_name}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to get best run: {e}")
        return None
