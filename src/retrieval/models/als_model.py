"""ALS (Alternating Least Squares) Model for Collaborative Filtering."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)


class ALSModel:
    """Alternating Least Squares model for collaborative filtering."""
    
    def __init__(self, n_factors: int = 100, n_iterations: int = 20, 
                 regularization: float = 0.1, random_state: int = 42):
        """Initialize ALS model.
        
        Args:
            n_factors: Number of latent factors
            n_iterations: Number of iterations for training
            regularization: Regularization parameter (lambda)
            random_state: Random seed for reproducibility
        """
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.random_state = random_state
        
        # Model parameters
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_bias = None
        
        # Training metadata
        self.training_loss = []
        self.is_trained = False
        
        # Set random seed
        np.random.seed(random_state)
        
    def fit(self, user_item_matrix: csr_matrix, 
            user_features: Optional[pd.DataFrame] = None,
            item_features: Optional[pd.DataFrame] = None) -> 'ALSModel':
        """Fit the ALS model.
        
        Args:
            user_item_matrix: Sparse user-item interaction matrix
            user_features: Optional user features DataFrame
            item_features: Optional item features DataFrame
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training ALS model with {self.n_factors} factors...")
        
        n_users, n_items = user_item_matrix.shape
        logger.info(f"Matrix dimensions: {n_users} users × {n_items} items")
        
        # Initialize factors randomly
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        # Initialize biases
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        self.global_bias = np.mean(user_item_matrix.data) if user_item_matrix.data.size > 0 else 0
        
        # Training loop
        for iteration in range(self.n_iterations):
            # Update user factors
            self._update_user_factors(user_item_matrix)
            
            # Update item factors
            self._update_item_factors(user_item_matrix)
            
            # Calculate loss
            loss = self._calculate_loss(user_item_matrix)
            self.training_loss.append(loss)
            
            if iteration % 5 == 0:
                logger.info(f"Iteration {iteration + 1}/{self.n_iterations}, Loss: {loss:.4f}")
        
        self.is_trained = True
        logger.info(f"✅ ALS model training completed! Final loss: {loss:.4f}")
        
        return self
    
    def _update_user_factors(self, user_item_matrix: csr_matrix) -> None:
        """Update user factors using ALS algorithm."""
        for user_id in range(user_item_matrix.shape[0]):
            # Get items rated by this user
            user_items = user_item_matrix[user_id].nonzero()[1]
            if len(user_items) == 0:
                continue
                
            # Get ratings for this user
            ratings = user_item_matrix[user_id, user_items].toarray().flatten()
            
            # Adjust ratings by removing biases
            adjusted_ratings = ratings - self.global_bias - self.item_biases[user_items]
            
            # Update user factors
            item_factors_subset = self.item_factors[user_items]
            A = item_factors_subset.T @ item_factors_subset + self.regularization * np.eye(self.n_factors)
            b = item_factors_subset.T @ adjusted_ratings
            self.user_factors[user_id] = np.linalg.solve(A, b)
            
            # Update user bias
            self.user_biases[user_id] = np.mean(adjusted_ratings - self.user_factors[user_id] @ item_factors_subset.T)
    
    def _update_item_factors(self, user_item_matrix: csr_matrix) -> None:
        """Update item factors using ALS algorithm."""
        for item_id in range(user_item_matrix.shape[1]):
            # Get users who rated this item
            item_users = user_item_matrix[:, item_id].nonzero()[0]
            if len(item_users) == 0:
                continue
                
            # Get ratings for this item
            ratings = user_item_matrix[item_users, item_id].toarray().flatten()
            
            # Adjust ratings by removing biases
            adjusted_ratings = ratings - self.global_bias - self.user_biases[item_users]
            
            # Update item factors
            user_factors_subset = self.user_factors[item_users]
            A = user_factors_subset.T @ user_factors_subset + self.regularization * np.eye(self.n_factors)
            b = user_factors_subset.T @ adjusted_ratings
            self.item_factors[item_id] = np.linalg.solve(A, b)
            
            # Update item bias
            self.item_biases[item_id] = np.mean(adjusted_ratings - user_factors_subset @ self.item_factors[item_id])
    
    def _calculate_loss(self, user_item_matrix: csr_matrix) -> float:
        """Calculate training loss."""
        total_loss = 0
        n_interactions = 0
        
        for user_id in range(user_item_matrix.shape[0]):
            user_items = user_item_matrix[user_id].nonzero()[1]
            if len(user_items) == 0:
                continue
                
            ratings = user_item_matrix[user_id, user_items].toarray().flatten()
            predictions = self.predict(user_id, user_items)
            
            # MSE loss
            loss = np.mean((ratings - predictions) ** 2)
            total_loss += loss * len(user_items)
            n_interactions += len(user_items)
        
        # Add regularization
        reg_loss = (self.regularization * 
                   (np.sum(self.user_factors ** 2) + np.sum(self.item_factors ** 2)))
        
        return (total_loss / n_interactions) + reg_loss
    
    def predict(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        """Predict ratings for user-item pairs.
        
        Args:
            user_id: User ID
            item_ids: Array of item IDs
            
        Returns:
            Array of predicted ratings
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get user and item factors
        user_factor = self.user_factors[user_id]
        item_factors = self.item_factors[item_ids]
        
        # Calculate predictions
        predictions = (user_factor @ item_factors.T + 
                      self.user_biases[user_id] + 
                      self.item_biases[item_ids] + 
                      self.global_bias)
        
        return predictions
    
    def get_user_embeddings(self, user_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Get user embeddings.
        
        Args:
            user_ids: Specific user IDs, or None for all users
            
        Returns:
            User embeddings matrix
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting embeddings")
        
        if user_ids is None:
            return self.user_factors
        return self.user_factors[user_ids]
    
    def get_item_embeddings(self, item_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Get item embeddings.
        
        Args:
            item_ids: Specific item IDs, or None for all items
            
        Returns:
            Item embeddings matrix
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting embeddings")
        
        if item_ids is None:
            return self.item_factors
        return self.item_factors[item_ids]
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'user_biases': self.user_biases,
            'item_biases': self.item_biases,
            'global_bias': self.global_bias,
            'n_factors': self.n_factors,
            'training_loss': self.training_loss,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'ALSModel':
        """Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded ALS model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create model instance
        model = cls(n_factors=model_data['n_factors'])
        
        # Load model parameters
        model.user_factors = model_data['user_factors']
        model.item_factors = model_data['item_factors']
        model.user_biases = model_data['user_biases']
        model.item_biases = model_data['item_biases']
        model.global_bias = model_data['global_bias']
        model.training_loss = model_data['training_loss']
        model.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")
        return model
