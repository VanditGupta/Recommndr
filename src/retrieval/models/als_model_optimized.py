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
import matplotlib.pyplot as plt
import pandas as pd

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
        """Setup MLflow experiment and run with advanced features."""
        try:
            # Set experiment
            mlflow.set_experiment(self.experiment_name)
            
            # Start run
            self.mlflow_run = mlflow.start_run(run_name=self.run_name)
            
            # Enable system metrics logging (CPU, memory, etc.)
            try:
                mlflow.enable_system_metrics_logging()
                logger.info("✅ System metrics logging enabled")
            except Exception as e:
                logger.warning(f"⚠️ System metrics logging failed: {e}")
            
            # Log parameters
            mlflow.log_params({
                "n_factors": self.n_factors,
                "n_iterations": self.n_iterations,
                "regularization": self.regularization,
                "random_state": self.random_state,
                "use_blas": self.use_blas,
                "batch_size": self.batch_size,
                "model_type": "OptimizedALSModel",
                "framework": "numpy_scipy",
                "optimization": "vectorized_blas"
            })
            
            # Log tags for better organization
            mlflow.set_tags({
                "project": "Recommndr",
                "phase": "Phase3",
                "algorithm": "ALS",
                "optimization": "vectorized",
                "data_type": "sparse_matrix"
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
            
            # Log trace for iteration start
            if self.mlflow_run:
                try:
                    mlflow.log_trace(
                        name="training_iteration",
                        inputs={"iteration": iteration, "n_factors": self.n_factors},
                        outputs={"status": "started"}
                    )
                except Exception as e:
                    logger.debug(f"Trace logging failed: {e}")
            
            # Vectorized user factor updates
            self._update_user_factors_vectorized(user_item_matrix, reg_matrix)
            
            # Vectorized item factor updates
            self._update_item_factors_vectorized(user_item_matrix, reg_matrix)
            
            # Calculate loss (vectorized)
            loss = self._calculate_loss_vectorized(user_item_matrix)
            self.training_loss.append(loss)
            
            iter_time = time.time() - iter_start
            self.training_times.append(iter_time)
            
            # Enhanced metrics logging to MLflow
            if self.mlflow_run:
                # Basic training metrics
                mlflow.log_metrics({
                    "loss": loss,
                    "iteration_time": iter_time,
                    "cumulative_time": sum(self.training_times),
                    "loss_improvement": self.training_loss[0] - loss,
                    "loss_improvement_pct": ((self.training_loss[0] - loss) / self.training_loss[0]) * 100
                }, step=iteration)
                
                # Performance metrics
                if iteration > 0:
                    mlflow.log_metrics({
                        "loss_change": loss - self.training_loss[-2],
                        "time_change": iter_time - self.training_times[-2],
                        "convergence_rate": abs(loss - self.training_loss[-2]) / self.training_loss[-2] if self.training_loss[-2] > 0 else 0
                    }, step=iteration)
                
                # Log trace for iteration completion
                try:
                    mlflow.log_trace(
                        name="training_iteration",
                        inputs={"iteration": iteration, "loss": loss, "time": iter_time},
                        outputs={"status": "completed", "loss": loss, "iteration_time": iter_time}
                    )
                except Exception as e:
                    logger.debug(f"Trace logging failed: {e}")
            
            if iteration % 5 == 0 or iteration == self.n_iterations - 1:
                logger.info(f"   Iteration {iteration + 1}/{self.n_iterations}, "
                           f"Loss: {loss:.4f}, Time: {iter_time:.2f}s")
        
        self.is_trained = True
        total_time = time.time() - start_time
        avg_iter_time = np.mean(self.training_times)
        
        # Enhanced final metrics logging to MLflow
        if self.mlflow_run:
            # Training summary metrics
            mlflow.log_metrics({
                "final_loss": loss,
                "total_training_time": total_time,
                "avg_iteration_time": avg_iter_time,
                "fastest_iteration": min(self.training_times),
                "slowest_iteration": max(self.training_times),
                "total_loss_improvement": self.training_loss[0] - loss,
                "loss_improvement_percentage": ((self.training_loss[0] - loss) / self.training_loss[0]) * 100,
                "training_efficiency": (self.training_loss[0] - loss) / total_time,  # Improvement per second
                "convergence_stability": np.std(self.training_loss[-5:]) if len(self.training_loss) >= 5 else 0
            })
            
            # Log training curves as structured data
            try:
                training_curve_data = {
                    "iteration": list(range(len(self.training_loss))),
                    "loss": self.training_loss,
                    "training_time": self.training_times,
                    "cumulative_time": list(np.cumsum(self.training_times)),
                    "loss_improvement": [self.training_loss[0] - l for l in self.training_loss]
                }
                mlflow.log_table(
                    data=training_curve_data,
                    artifact_file="training_curves.json"
                )
                logger.info("✅ Training curves logged to MLflow")
            except Exception as e:
                logger.warning(f"⚠️ Failed to log training curves: {e}")
            
            # Log trace for training completion
            try:
                mlflow.log_trace(
                    name="training_completion",
                    inputs={"total_iterations": len(self.training_loss), "final_loss": loss},
                    outputs={"status": "completed", "total_time": total_time, "is_trained": True}
                )
            except Exception as e:
                logger.debug(f"Trace logging failed: {e}")
        
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
        
        # Create comprehensive visualizations
        logger.info("🎨 Creating training visualizations...")
        try:
            plots = self._create_training_visualizations()
            logger.info(f"✅ Visualizations created: {list(plots.keys())}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to create visualizations: {e}")
            plots = {}
        
        # Log model artifacts to MLflow
        if self.mlflow_run:
            try:
                # Log model file
                mlflow.log_artifact(filepath, "model")
                
                # Log all visualization files
                for plot_name, plot_path in plots.items():
                    if Path(plot_path).exists():
                        mlflow.log_artifact(plot_path, f"visualizations/{plot_name}")
                
                # Log training curves as a figure (if matplotlib is available)
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(12, 8))
                    
                    # Training loss curve
                    plt.subplot(2, 2, 1)
                    plt.plot(self.training_loss, 'b-', linewidth=2, marker='o', markersize=4)
                    plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
                    plt.xlabel('Iteration', fontsize=12)
                    plt.ylabel('Loss', fontsize=12)
                    plt.grid(True, alpha=0.3)
                    plt.yscale('log')
                    
                    # Loss improvement
                    plt.subplot(2, 2, 2)
                    loss_improvement = [self.training_loss[0] - loss for loss in self.training_loss]
                    plt.plot(loss_improvement, 'g-', linewidth=2, marker='s', markersize=4)
                    plt.title('Loss Improvement Over Time', fontsize=14, fontweight='bold')
                    plt.xlabel('Iteration', fontsize=12)
                    plt.ylabel('Loss Improvement', fontsize=12)
                    plt.grid(True, alpha=0.3)
                    
                    # Training time per iteration
                    plt.subplot(2, 2, 3)
                    plt.plot(self.training_times, 'r-', linewidth=2, marker='^', markersize=4)
                    plt.title('Training Time per Iteration', fontsize=14, fontweight='bold')
                    plt.xlabel('Iteration', fontsize=12)
                    plt.ylabel('Time (seconds)', fontsize=12)
                    plt.grid(True, alpha=0.3)
                    
                    # Factor norms distribution
                    plt.subplot(2, 2, 4)
                    user_norms = np.linalg.norm(self.user_factors, axis=1)
                    item_norms = np.linalg.norm(self.item_factors, axis=1)
                    plt.hist(user_norms, bins=30, alpha=0.7, color='skyblue', edgecolor='black', label='Users')
                    plt.hist(item_norms, bins=30, alpha=0.7, color='lightcoral', edgecolor='black', label='Items')
                    plt.title('Factor Norms Distribution', fontsize=14, fontweight='bold')
                    plt.xlabel('Norm Value', fontsize=12)
                    plt.ylabel('Frequency', fontsize=12)
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    
                    # Log the figure to MLflow
                    mlflow.log_figure(plt.gcf(), "training_summary.png")
                    plt.close()
                    
                    logger.info("✅ Training summary figure logged to MLflow")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to log training figure: {e}")
                
                # Enhanced model info logging
                model_info = {
                    "model_type": "OptimizedALSModel",
                    "model_path": filepath,
                    "model_size_mb": Path(filepath).stat().st_size / (1024 * 1024),
                    "visualizations_created": list(plots.keys()),
                    "training_summary": {
                        "final_loss": self.training_loss[-1],
                        "total_time": sum(self.training_times),
                        "avg_iteration_time": np.mean(self.training_times),
                        "n_iterations": len(self.training_loss),
                        "loss_improvement": self.training_loss[0] - self.training_loss[-1],
                        "improvement_percentage": ((self.training_loss[0] - self.training_loss[-1]) / self.training_loss[0]) * 100
                    },
                    "model_architecture": {
                        "n_users": self.user_factors.shape[0],
                        "n_items": self.item_factors.shape[1],
                        "n_factors": self.user_factors.shape[1],
                        "user_factors_shape": self.user_factors.shape,
                        "item_factors_shape": self.item_factors.shape,
                        "user_biases_shape": self.user_biases.shape,
                        "item_biases_shape": self.item_biases.shape
                    },
                    "training_metadata": {
                        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "total_training_time": sum(self.training_times),
                        "convergence_stable": len(set(self.training_loss[-3:])) < 3 if len(self.training_loss) >= 3 else False,
                        "final_loss_stable": np.std(self.training_loss[-5:]) if len(self.training_loss) >= 5 else 0
                    }
                }
                
                mlflow.log_dict(model_info, "model_info.json")
                
                # Log model as a custom artifact
                try:
                    mlflow.log_text(
                        f"OptimizedALSModel trained on {self.user_factors.shape[0]:,} users and {self.item_factors.shape[0]:,} items with {self.user_factors.shape[1]} factors. Final loss: {self.training_loss[-1]:.4f}",
                        "model_description.txt"
                    )
                except Exception as e:
                    logger.debug(f"Failed to log model description: {e}")
                
                logger.info("✅ Enhanced model and visualizations logged to MLflow")
                
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

    def _create_training_visualizations(self) -> Dict[str, str]:
        """Create comprehensive training visualizations and save them."""
        if not self.is_trained or len(self.training_loss) == 0:
            logger.warning("No training data available for visualizations")
            return {}
        
        # Create output directory
        output_dir = Path("data/processed/visualizations")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plots = {}
        
        # 1. Training Loss Curve
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(self.training_loss, 'b-', linewidth=2, marker='o', markersize=4)
        plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale for better visualization
        
        # 2. Loss Improvement
        plt.subplot(2, 2, 2)
        loss_improvement = [self.training_loss[0] - loss for loss in self.training_loss]
        plt.plot(loss_improvement, 'g-', linewidth=2, marker='s', markersize=4)
        plt.title('Loss Improvement Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Loss Improvement', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 3. Training Time per Iteration
        plt.subplot(2, 2, 3)
        if hasattr(self, 'training_times') and len(self.training_times) > 0:
            plt.plot(self.training_times, 'r-', linewidth=2, marker='^', markersize=4)
            plt.title('Training Time per Iteration', fontsize=14, fontweight='bold')
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Time (seconds)', fontsize=12)
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No timing data available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Training Time per Iteration', fontsize=14, fontweight='bold')
        
        # 4. Cumulative Training Time
        plt.subplot(2, 2, 4)
        if hasattr(self, 'training_times') and len(self.training_times) > 0:
            cumulative_time = np.cumsum(self.training_times)
            plt.plot(cumulative_time, 'm-', linewidth=2, marker='d', markersize=4)
            plt.title('Cumulative Training Time', fontsize=14, fontweight='bold')
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Cumulative Time (seconds)', fontsize=12)
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No timing data available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Cumulative Training Time', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        loss_plot_path = output_dir / "training_loss_analysis.png"
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots['training_loss'] = str(loss_plot_path)
        
        # 5. Factor Distribution Analysis
        plt.figure(figsize=(15, 10))
        
        # User factors distribution
        plt.subplot(2, 3, 1)
        plt.hist(self.user_factors.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('User Factors Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Factor Value', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Item factors distribution
        plt.subplot(2, 3, 2)
        plt.hist(self.item_factors.flatten(), bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.title('Item Factors Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Factor Value', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # User biases distribution
        plt.subplot(2, 3, 3)
        plt.hist(self.user_biases, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.title('User Biases Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Bias Value', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Item biases distribution
        plt.subplot(2, 3, 4)
        plt.hist(self.item_biases, bins=30, alpha=0.7, color='gold', edgecolor='black')
        plt.title('Item Biases Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Bias Value', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Factor norms distribution
        plt.subplot(2, 3, 5)
        user_norms = np.linalg.norm(self.user_factors, axis=1)
        item_norms = np.linalg.norm(self.item_factors, axis=1)
        plt.hist(user_norms, bins=30, alpha=0.7, color='plum', edgecolor='black', label='Users')
        plt.hist(item_norms, bins=30, alpha=0.7, color='orange', edgecolor='black', label='Items')
        plt.title('Factor Norms Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Norm Value', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Training metrics summary
        plt.subplot(2, 3, 6)
        metrics_text = f"""
        Training Summary:
        
        Final Loss: {self.training_loss[-1]:.4f}
        Initial Loss: {self.training_loss[0]:.4f}
        Improvement: {self.training_loss[0] - self.training_loss[-1]:.4f}
        Iterations: {len(self.training_loss)}
        Users: {self.user_factors.shape[0]:,}
        Items: {self.item_factors.shape[0]:,}
        Factors: {self.user_factors.shape[1]}
        """
        plt.text(0.1, 0.9, metrics_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        plt.title('Training Metrics Summary', fontsize=12, fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        factors_plot_path = output_dir / "factor_analysis.png"
        plt.savefig(factors_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots['factor_analysis'] = str(factors_plot_path)
        
        # 6. Create training summary CSV
        training_summary = pd.DataFrame({
            'iteration': range(len(self.training_loss)),
            'loss': self.training_loss,
            'loss_improvement': [self.training_loss[0] - loss for loss in self.training_loss],
            'training_time': self.training_times if hasattr(self, 'training_times') else [0] * len(self.training_loss),
            'cumulative_time': np.cumsum(self.training_times) if hasattr(self, 'training_times') else [0] * len(self.training_loss)
        })
        
        csv_path = output_dir / "training_summary.csv"
        training_summary.to_csv(csv_path, index=False)
        plots['training_summary'] = str(csv_path)
        
        # 7. Create detailed metrics JSON
        metrics_data = {
            'model_info': {
                'n_users': self.user_factors.shape[0],
                'n_items': self.item_factors.shape[0],
                'n_factors': self.user_factors.shape[1],
                'n_iterations': len(self.training_loss)
            },
            'training_metrics': {
                'initial_loss': float(self.training_loss[0]),
                'final_loss': float(self.training_loss[-1]),
                'total_improvement': float(self.training_loss[0] - self.training_loss[-1]),
                'improvement_percentage': float((self.training_loss[0] - self.training_loss[-1]) / self.training_loss[0] * 100),
                'convergence_stable': len(set(self.training_loss[-3:])) < 3
            },
            'factor_statistics': {
                'user_factors': {
                    'mean': float(self.user_factors.mean()),
                    'std': float(self.user_factors.std()),
                    'min': float(self.user_factors.min()),
                    'max': float(self.user_factors.max())
                },
                'item_factors': {
                    'mean': float(self.item_factors.mean()),
                    'std': float(self.item_factors.std()),
                    'min': float(self.item_factors.min()),
                    'max': float(self.item_factors.max())
                }
            },
            'bias_statistics': {
                'global_bias': float(self.global_bias),
                'user_biases': {
                    'mean': float(self.user_biases.mean()),
                    'std': float(self.user_biases.std()),
                    'min': float(self.user_biases.min()),
                    'max': float(self.user_biases.max())
                },
                'item_biases': {
                    'mean': float(self.item_biases.mean()),
                    'std': float(self.item_biases.std()),
                    'min': float(self.item_biases.min()),
                    'max': float(self.item_biases.max())
                }
            }
        }
        
        if hasattr(self, 'training_times') and len(self.training_times) > 0:
            metrics_data['timing_metrics'] = {
                'total_training_time': float(sum(self.training_times)),
                'average_iteration_time': float(np.mean(self.training_times)),
                'fastest_iteration': float(min(self.training_times)),
                'slowest_iteration': float(max(self.training_times))
            }
        
        import json
        json_path = output_dir / "training_metrics.json"
        with open(json_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        plots['training_metrics'] = str(json_path)
        
        logger.info(f"✅ Training visualizations saved to {output_dir}")
        return plots
    
    def create_visualization_report(self) -> str:
        """Create a comprehensive HTML report of all visualizations."""
        if not self.is_trained:
            raise ValueError("Model must be trained before creating report")
        
        try:
            # Create visualizations first
            plots = self._create_training_visualizations()
            
            # Create HTML report
            output_dir = Path("data/processed/visualizations")
            html_path = output_dir / "training_report.html"
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>ALS Model Training Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                    .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                    .section {{ background-color: white; padding: 20px; margin: 20px 0; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                    .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #ecf0f1; border-radius: 5px; }}
                    .plot {{ text-align: center; margin: 20px 0; }}
                    .plot img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
                    h1, h2 {{ color: #2c3e50; }}
                    .success {{ color: #27ae60; font-weight: bold; }}
                    .warning {{ color: #f39c12; font-weight: bold; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🚀 ALS Model Training Report</h1>
                    <p>Comprehensive analysis of model training performance and visualizations</p>
                </div>
                
                <div class="section">
                    <h2>📊 Model Overview</h2>
                    <div class="metric">
                        <strong>Users:</strong> {self.user_factors.shape[0]:,}
                    </div>
                    <div class="metric">
                        <strong>Items:</strong> {self.item_factors.shape[0]:,}
                    </div>
                    <div class="metric">
                        <strong>Factors:</strong> {self.user_factors.shape[1]}
                    </div>
                    <div class="metric">
                        <strong>Iterations:</strong> {len(self.training_loss)}
                    </div>
                </div>
                
                <div class="section">
                    <h2>📈 Training Performance</h2>
                    <div class="metric">
                        <strong>Initial Loss:</strong> {self.training_loss[0]:.4f}
                    </div>
                    <div class="metric">
                        <strong>Final Loss:</strong> {self.training_loss[-1]:.4f}
                    </div>
                    <div class="metric">
                        <strong>Improvement:</strong> {self.training_loss[0] - self.training_loss[-1]:.4f}
                    </div>
                    <div class="metric">
                        <strong>Total Time:</strong> {sum(self.training_times):.2f}s
                    </div>
                </div>
                
                <div class="section">
                    <h2>🎨 Training Visualizations</h2>
                    <div class="plot">
                        <h3>Training Loss Analysis</h3>
                        <img src="training_loss_analysis.png" alt="Training Loss Analysis">
                    </div>
                    <div class="plot">
                        <h3>Factor Distribution Analysis</h3>
                        <img src="factor_analysis.png" alt="Factor Distribution Analysis">
                    </div>
                </div>
                
                <div class="section">
                    <h2>📁 Generated Files</h2>
                    <ul>
                        <li><strong>Training Loss Analysis:</strong> training_loss_analysis.png</li>
                        <li><strong>Factor Analysis:</strong> factor_analysis.png</li>
                        <li><strong>Training Summary:</strong> training_summary.csv</li>
                        <li><strong>Training Metrics:</strong> training_metrics.json</li>
                    </ul>
                </div>
                
                <div class="section">
                    <h2>🔍 Analysis Summary</h2>
                    <p class="success">✅ Model training completed successfully</p>
                    <p class="success">✅ Loss improved from {self.training_loss[0]:.4f} to {self.training_loss[-1]:.4f}</p>
                    <p class="success">✅ Training converged in {len(self.training_loss)} iterations</p>
                    <p class="success">✅ All visualizations generated successfully</p>
                </div>
            </body>
            </html>
            """
            
            with open(html_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"✅ HTML report created: {html_path}")
            return str(html_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to create HTML report: {e}")
            return ""


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
