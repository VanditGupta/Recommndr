"""
LightGBM Ranking Model for Phase 4

Implements a LightGBM-based ranking model that:
- Takes ALS candidate items and contextual features
- Ranks items by relevance score
- Exports to ONNX for fast inference
- Integrates with MLflow for experiment tracking
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import pickle
import json
from pathlib import Path
import time
import logging

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score, precision_score, recall_score
import mlflow
import mlflow.lightgbm
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from src.utils.logging import get_logger

logger = get_logger(__name__)


class LightGBMRanker:
    """LightGBM-based ranking model for recommendation scoring."""
    
    def __init__(self, model_params: Optional[Dict] = None):
        """Initialize LightGBM ranker."""
        self.model = None
        self.feature_names = None
        self.model_params = model_params or self._get_default_params()
        self.training_metrics = {}
        self.onnx_model = None
        self.onnx_session = None
        
    def _get_default_params(self) -> Dict:
        """Get default LightGBM parameters optimized for ranking."""
        return {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 128,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 50,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 200,
            'early_stopping_rounds': 20,
        }
    
    def prepare_training_data(self, training_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data for LightGBM."""
        logger.info("Preparing training data for LightGBM...")
        
        # Get feature columns (exclude IDs and labels)
        feature_cols = [col for col in training_df.columns 
                       if col not in ['user_id', 'item_id', 'label', 'rating']]
        
        X = training_df[feature_cols].fillna(0.0)
        y = training_df['rating'].fillna(0.0)  # Use rating as target for regression
        
        self.feature_names = feature_cols
        
        logger.info(f"✅ Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"   Features: {feature_cols[:5]}...")
        
        return X.values, y.values, feature_cols
    
    def train(self, training_df: pd.DataFrame, 
              validation_split: float = 0.2,
              track_experiment: bool = True) -> Dict[str, Any]:
        """Train the LightGBM ranking model."""
        logger.info("🚀 Starting LightGBM ranking model training...")
        
        # Start MLflow experiment
        if track_experiment:
            mlflow.set_experiment("lightgbm_ranking")
            mlflow.start_run()
        
        start_time = time.time()
        
        try:
            # Prepare data
            X, y, feature_names = self.prepare_training_data(training_df)
            
            # Train-validation split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=None
            )
            
            logger.info(f"Train size: {X_train.shape[0]}, Validation size: {X_val.shape[0]}")
            
            # Create LightGBM datasets
            train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_names)
            
            # Train model
            logger.info("Training LightGBM model...")
            self.model = lgb.train(
                self.model_params,
                train_data,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'val'],
                callbacks=[lgb.log_evaluation(100)]
            )
            
            # Evaluate model
            train_predictions = self.model.predict(X_train)
            val_predictions = self.model.predict(X_val)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_train, train_predictions, y_val, val_predictions)
            self.training_metrics = metrics
            
            training_time = time.time() - start_time
            
            # Log results
            logger.info(f"✅ Training completed in {training_time:.2f}s")
            logger.info(f"   Train RMSE: {metrics['train_rmse']:.4f}")
            logger.info(f"   Val RMSE: {metrics['val_rmse']:.4f}")
            
            # MLflow logging
            if track_experiment:
                mlflow.log_params(self.model_params)
                mlflow.log_metrics(metrics)
                mlflow.log_metric("training_time_seconds", training_time)
                mlflow.lightgbm.log_model(self.model, "model")
                
                # Log feature importance
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': self.model.feature_importance()
                }).sort_values('importance', ascending=False)
                
                mlflow.log_text(importance_df.to_string(), "feature_importance.txt")
            
            return {
                'model': self.model,
                'metrics': metrics,
                'training_time': training_time,
                'feature_importance': self.model.feature_importance()
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            if track_experiment:
                mlflow.end_run()
    
    def _calculate_metrics(self, y_train: np.ndarray, train_preds: np.ndarray,
                          y_val: np.ndarray, val_preds: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_preds)),
            'train_mae': mean_absolute_error(y_train, train_preds),
            'train_r2': r2_score(y_train, train_preds),
            'val_rmse': np.sqrt(mean_squared_error(y_val, val_preds)),
            'val_mae': mean_absolute_error(y_val, val_preds),
            'val_r2': r2_score(y_val, val_preds),
        }
        
        return metrics
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(features)
    
    def predict_single(self, features: Dict[str, float]) -> float:
        """Make prediction for a single user-item pair."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Convert features dict to array in correct order
        feature_vector = np.array([features.get(name, 0.0) for name in self.feature_names])
        prediction = self.model.predict([feature_vector])[0]
        
        return float(prediction)
    
    def rank_candidates(self, user_id: int, candidate_items: List[int], 
                       feature_engineer: Any) -> List[Tuple[int, float]]:
        """Rank a list of candidate items for a user."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Generate features for each candidate
        candidates_with_scores = []
        
        for item_id in candidate_items:
            features = feature_engineer.create_user_item_features(user_id, item_id)
            score = self.predict_single(features)
            candidates_with_scores.append((item_id, score))
        
        # Sort by score (descending)
        ranked_candidates = sorted(candidates_with_scores, key=lambda x: x[1], reverse=True)
        
        return ranked_candidates
    
    def export_to_onnx(self, output_path: str, sample_input: Optional[np.ndarray] = None) -> str:
        """Export the trained model to ONNX format."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info("🔄 Exporting LightGBM model to ONNX...")
        
        try:
            # Create a sample input if not provided
            if sample_input is None:
                sample_input = np.zeros((1, len(self.feature_names)), dtype=np.float32)
            
            # Convert LightGBM to ONNX using lightgbm's built-in converter
            # Note: For LightGBM, we need to use a different approach
            import onnxmltools
            from onnxmltools.convert import convert_lightgbm
            from onnxmltools.utils import save_model
            
            # Convert to ONNX
            onnx_model = convert_lightgbm(
                self.model,
                initial_types=[('input', FloatTensorType([None, len(self.feature_names)]))],
                target_opset=11
            )
            
            # Save ONNX model
            onnx_path = Path(output_path)
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            
            save_model(onnx_model, str(onnx_path))
            self.onnx_model = onnx_model
            
            logger.info(f"✅ ONNX model exported to: {onnx_path}")
            
            # Test ONNX model
            self._test_onnx_model(str(onnx_path), sample_input)
            
            return str(onnx_path)
            
        except ImportError:
            logger.warning("onnxmltools not available. Installing...")
            import subprocess
            subprocess.check_call(["pip", "install", "onnxmltools"])
            
            # Retry export
            return self.export_to_onnx(output_path, sample_input)
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            logger.info("Falling back to pickle export...")
            
            # Fallback to pickle
            pickle_path = str(onnx_path).replace('.onnx', '.pkl')
            with open(pickle_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'feature_names': self.feature_names,
                    'model_params': self.model_params
                }, f)
            
            logger.info(f"✅ Model saved as pickle: {pickle_path}")
            return pickle_path
    
    def _test_onnx_model(self, onnx_path: str, sample_input: np.ndarray):
        """Test the exported ONNX model."""
        try:
            # Create ONNX Runtime session
            session = ort.InferenceSession(onnx_path)
            self.onnx_session = session
            
            # Test prediction
            input_name = session.get_inputs()[0].name
            onnx_prediction = session.run(None, {input_name: sample_input.astype(np.float32)})[0]
            
            # Compare with original model
            lgb_prediction = self.model.predict(sample_input)
            
            # Check if predictions are close
            diff = np.abs(onnx_prediction.flatten() - lgb_prediction).max()
            if diff < 1e-5:
                logger.info(f"✅ ONNX model test passed (max diff: {diff:.2e})")
            else:
                logger.warning(f"⚠️ ONNX model differs from original (max diff: {diff:.2e})")
                
        except Exception as e:
            logger.error(f"ONNX model test failed: {e}")
    
    def predict_onnx(self, features: np.ndarray) -> np.ndarray:
        """Make predictions using ONNX model for fast inference."""
        if self.onnx_session is None:
            raise ValueError("ONNX model not loaded. Call export_to_onnx() first.")
        
        input_name = self.onnx_session.get_inputs()[0].name
        predictions = self.onnx_session.run(None, {input_name: features.astype(np.float32)})[0]
        
        return predictions.flatten()
    
    def save_model(self, output_path: str):
        """Save the complete model with metadata."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_params': self.model_params,
            'training_metrics': self.training_metrics,
            'timestamp': time.time()
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"✅ Model saved to: {output_path}")
    
    def load_model(self, model_path: str):
        """Load a saved model."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.model_params = model_data.get('model_params', {})
        self.training_metrics = model_data.get('training_metrics', {})
        
        logger.info(f"✅ Model loaded from: {model_path}")
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """Get feature importance as DataFrame."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importance(importance_type=importance_type)
        }).sort_values('importance', ascending=False)
        
        return importance_df
