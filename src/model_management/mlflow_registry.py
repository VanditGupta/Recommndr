"""
MLflow Model Registry Integration

Manages model registration, versioning, and staging in MLflow:
- Register ALS and LightGBM models
- Handle model transitions (Dev -> Staging -> Production)
- Track model metadata and lineage
- Enable model comparison and rollback
"""

import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import pandas as pd
import numpy as np
import pickle
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
from datetime import datetime

from src.utils.logging import get_logger

logger = get_logger(__name__)


class MLflowModelRegistry:
    """MLflow Model Registry for managing recommendation system models."""
    
    def __init__(self, tracking_uri: str = "http://localhost:5001"):
        """Initialize MLflow Model Registry."""
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri)
        
        # Model names in registry
        self.als_model_name = "recommndr-als-model"
        self.lightgbm_model_name = "recommndr-lightgbm-ranker"
        self.similarity_model_name = "recommndr-similarity-engine"
        
        # Model stages
        self.stages = ["Development", "Staging", "Production", "Archived"]
        
        logger.info(f"🏗️ MLflow Model Registry initialized with URI: {tracking_uri}")
        
        # Ensure model names are registered
        self._ensure_registered_models()
    
    def _ensure_registered_models(self):
        """Ensure all model names are registered in MLflow."""
        model_names = [
            self.als_model_name,
            self.lightgbm_model_name, 
            self.similarity_model_name
        ]
        
        for model_name in model_names:
            try:
                self.client.get_registered_model(model_name)
                logger.info(f"✅ Model '{model_name}' already registered")
            except Exception:
                try:
                    self.client.create_registered_model(
                        model_name,
                        description=f"Recommndr {model_name.split('-')[-1]} model"
                    )
                    logger.info(f"✅ Created registered model: {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to create model {model_name}: {e}")
    
    def register_als_model(self, model_path: str, run_id: Optional[str] = None,
                          model_metadata: Optional[Dict] = None) -> str:
        """Register ALS model in MLflow registry."""
        logger.info(f"📝 Registering ALS model from {model_path}")
        
        # Start MLflow run if not provided
        if run_id is None:
            with mlflow.start_run(experiment_id=self._get_or_create_experiment("als_training")) as run:
                run_id = run.info.run_id
                
                # Load and log the ALS model
                with open(model_path, 'rb') as f:
                    als_data = pickle.load(f)
                
                # Log model artifacts
                mlflow.log_artifact(model_path, "model")
                
                # Log model parameters
                if 'factors' in als_data:
                    mlflow.log_param("n_factors", als_data.get('factors', 50))
                if 'iterations' in als_data:
                    mlflow.log_param("n_iterations", als_data.get('iterations', 10))
                
                # Log model metadata
                if model_metadata:
                    for key, value in model_metadata.items():
                        mlflow.log_param(f"metadata_{key}", value)
                
                # Log model info
                if 'user_factors' in als_data and 'item_factors' in als_data:
                    n_users, n_factors = als_data['user_factors'].shape
                    n_items = als_data['item_factors'].shape[0]
                    
                    mlflow.log_metric("n_users", n_users)
                    mlflow.log_metric("n_items", n_items)
                    mlflow.log_metric("n_factors", n_factors)
                
                # Create a custom PyFunc model for ALS
                class ALSModel(mlflow.pyfunc.PythonModel):
                    def load_context(self, context):
                        import pickle
                        with open(context.artifacts["model"], 'rb') as f:
                            self.als_data = pickle.load(f)
                    
                    def predict(self, context, model_input):
                        # Simple prediction interface for ALS
                        if isinstance(model_input, pd.DataFrame):
                            user_ids = model_input['user_id'].values
                            n_recommendations = model_input.get('n_recommendations', [10] * len(user_ids))
                        else:
                            user_ids = [model_input]
                            n_recommendations = [10]
                        
                        results = []
                        for user_id, n_recs in zip(user_ids, n_recommendations):
                            # This would call your actual ALS recommendation logic
                            results.append({
                                'user_id': user_id,
                                'recommendations': list(range(n_recs))  # Placeholder
                            })
                        return results
                
                # Log the PyFunc model
                artifacts = {"model": model_path}
                mlflow.pyfunc.log_model(
                    artifact_path="als_pyfunc_model",
                    python_model=ALSModel(),
                    artifacts=artifacts
                )
        
        # Register the model
        model_version = mlflow.register_model(
            model_uri=f"runs:/{run_id}/als_pyfunc_model",
            name=self.als_model_name
        )
        
        logger.info(f"✅ ALS model registered as version {model_version.version}")
        return model_version.version
    
    def register_lightgbm_model(self, model_path: str, run_id: Optional[str] = None,
                               model_metadata: Optional[Dict] = None) -> str:
        """Register LightGBM model in MLflow registry."""
        logger.info(f"📝 Registering LightGBM model from {model_path}")
        
        # Start MLflow run if not provided
        if run_id is None:
            with mlflow.start_run(experiment_id=self._get_or_create_experiment("lightgbm_ranking")) as run:
                run_id = run.info.run_id
                
                # Load the LightGBM model
                import lightgbm as lgb
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                if 'model' in model_data:
                    lgb_model = model_data['model']
                    
                    # Log the LightGBM model
                    mlflow.lightgbm.log_model(
                        lgb_model,
                        artifact_path="lightgbm_model",
                        registered_model_name=self.lightgbm_model_name
                    )
                    
                    # Log model parameters
                    if hasattr(lgb_model, 'params'):
                        for key, value in lgb_model.params.items():
                            mlflow.log_param(key, value)
                    
                    # Log feature importance
                    if hasattr(lgb_model, 'feature_importance'):
                        importance = lgb_model.feature_importance()
                        feature_names = model_data.get('feature_names', [f"feature_{i}" for i in range(len(importance))])
                        
                        importance_df = pd.DataFrame({
                            'feature': feature_names,
                            'importance': importance
                        }).sort_values('importance', ascending=False)
                        
                        mlflow.log_text(importance_df.to_string(), "feature_importance.txt")
                    
                    # Log model metadata
                    if model_metadata:
                        for key, value in model_metadata.items():
                            mlflow.log_param(f"metadata_{key}", value)
                    
                    # Log model metrics if available
                    if 'metrics' in model_data:
                        for key, value in model_data['metrics'].items():
                            mlflow.log_metric(key, value)
        
        # Get the latest version that was just registered
        latest_versions = self.client.get_latest_versions(
            self.lightgbm_model_name,
            stages=["None"]
        )
        
        if latest_versions:
            model_version = latest_versions[0].version
            logger.info(f"✅ LightGBM model registered as version {model_version}")
            return model_version
        else:
            logger.error("Failed to register LightGBM model")
            return None
    
    def register_similarity_model(self, models_dir: str = "models/phase5",
                                 run_id: Optional[str] = None,
                                 model_metadata: Optional[Dict] = None) -> str:
        """Register similarity engine models and indices."""
        logger.info(f"📝 Registering similarity models from {models_dir}")
        
        models_path = Path(models_dir)
        
        if run_id is None:
            with mlflow.start_run(experiment_id=self._get_or_create_experiment("similarity_engine")) as run:
                run_id = run.info.run_id
                
                # Log similarity matrices and indices
                similarity_files = [
                    "als_similarity_matrix.npy",
                    "copurchase_similarity_matrix.npy", 
                    "hybrid_similarity_matrix.npy",
                    "similarity_index.pkl",
                    "similarity_metadata.pkl"
                ]
                
                for file_name in similarity_files:
                    file_path = models_path / file_name
                    if file_path.exists():
                        mlflow.log_artifact(str(file_path), "similarity_models")
                        logger.info(f"   Logged {file_name}")
                
                # Log metadata
                if model_metadata:
                    for key, value in model_metadata.items():
                        mlflow.log_param(f"metadata_{key}", value)
                
                # Load and log similarity statistics
                metadata_path = models_path / "similarity_metadata.pkl"
                if metadata_path.exists():
                    with open(metadata_path, 'rb') as f:
                        metadata = pickle.load(f)
                    
                    if 'stats' in metadata:
                        stats = metadata['stats']
                        mlflow.log_metric("n_items", stats.get('n_items', 0))
                        
                        if 'similarity_index' in stats:
                            mlflow.log_metric("avg_similar_items", 
                                           stats['similarity_index'].get('avg_similar_items', 0))
                
                # Create a custom PyFunc model for similarity
                class SimilarityModel(mlflow.pyfunc.PythonModel):
                    def load_context(self, context):
                        import pickle
                        import numpy as np
                        
                        # Load similarity index
                        with open(f"{context.artifacts['similarity_models']}/similarity_index.pkl", 'rb') as f:
                            self.similarity_index = pickle.load(f)
                    
                    def predict(self, context, model_input):
                        if isinstance(model_input, pd.DataFrame):
                            item_ids = model_input['item_id'].values
                            top_k = model_input.get('top_k', [10] * len(item_ids))
                        else:
                            item_ids = [model_input]
                            top_k = [10]
                        
                        results = []
                        for item_id, k in zip(item_ids, top_k):
                            similar_items = self.similarity_index.get(item_id, [])[:k]
                            results.append({
                                'item_id': item_id,
                                'similar_items': similar_items
                            })
                        return results
                
                # Log the PyFunc model
                artifacts = {"similarity_models": str(models_path)}
                mlflow.pyfunc.log_model(
                    artifact_path="similarity_pyfunc_model",
                    python_model=SimilarityModel(),
                    artifacts=artifacts
                )
        
        # Register the model
        model_version = mlflow.register_model(
            model_uri=f"runs:/{run_id}/similarity_pyfunc_model",
            name=self.similarity_model_name
        )
        
        logger.info(f"✅ Similarity model registered as version {model_version.version}")
        return model_version.version
    
    def transition_model_stage(self, model_name: str, version: str, 
                              stage: str, archive_existing: bool = True) -> bool:
        """Transition model to a new stage."""
        logger.info(f"🔄 Transitioning {model_name} v{version} to {stage}")
        
        try:
            # Archive existing model in target stage if requested
            if archive_existing and stage == "Production":
                current_prod_versions = self.client.get_latest_versions(
                    model_name, stages=["Production"]
                )
                
                for version_info in current_prod_versions:
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=version_info.version,
                        stage="Archived"
                    )
                    logger.info(f"   Archived previous production version {version_info.version}")
            
            # Transition to new stage
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            
            logger.info(f"✅ Successfully transitioned {model_name} v{version} to {stage}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to transition model: {e}")
            return False
    
    def get_model_version(self, model_name: str, stage: str = "Production") -> Optional[str]:
        """Get the current model version for a given stage."""
        try:
            versions = self.client.get_latest_versions(model_name, stages=[stage])
            if versions:
                return versions[0].version
            return None
        except Exception as e:
            logger.error(f"Failed to get model version: {e}")
            return None
    
    def load_production_model(self, model_name: str):
        """Load the production version of a model."""
        try:
            model_uri = f"models:/{model_name}/Production"
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"✅ Loaded production model: {model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load production model {model_name}: {e}")
            return None
    
    def compare_model_versions(self, model_name: str, version1: str, version2: str) -> Dict:
        """Compare two model versions."""
        logger.info(f"📊 Comparing {model_name} versions {version1} vs {version2}")
        
        comparison = {
            'model_name': model_name,
            'version1': version1,
            'version2': version2,
            'comparison_time': datetime.now().isoformat()
        }
        
        try:
            # Get model version details
            mv1 = self.client.get_model_version(model_name, version1)
            mv2 = self.client.get_model_version(model_name, version2)
            
            comparison['version1_info'] = {
                'creation_time': mv1.creation_timestamp,
                'stage': mv1.current_stage,
                'run_id': mv1.run_id
            }
            
            comparison['version2_info'] = {
                'creation_time': mv2.creation_timestamp,
                'stage': mv2.current_stage,
                'run_id': mv2.run_id
            }
            
            # Get run metrics for comparison
            run1_metrics = self.client.get_run(mv1.run_id).data.metrics
            run2_metrics = self.client.get_run(mv2.run_id).data.metrics
            
            comparison['metrics_comparison'] = {}
            
            # Compare common metrics
            common_metrics = set(run1_metrics.keys()) & set(run2_metrics.keys())
            for metric in common_metrics:
                comparison['metrics_comparison'][metric] = {
                    'version1': run1_metrics[metric],
                    'version2': run2_metrics[metric],
                    'difference': run2_metrics[metric] - run1_metrics[metric]
                }
            
            logger.info(f"✅ Model comparison completed")
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            comparison['error'] = str(e)
            return comparison
    
    def get_model_lineage(self, model_name: str, version: str) -> Dict:
        """Get model lineage and metadata."""
        try:
            mv = self.client.get_model_version(model_name, version)
            run = self.client.get_run(mv.run_id)
            
            lineage = {
                'model_name': model_name,
                'version': version,
                'run_id': mv.run_id,
                'creation_time': mv.creation_timestamp,
                'stage': mv.current_stage,
                'parameters': run.data.params,
                'metrics': run.data.metrics,
                'tags': run.data.tags,
                'artifacts': [f.path for f in self.client.list_artifacts(mv.run_id)]
            }
            
            return lineage
            
        except Exception as e:
            logger.error(f"Failed to get model lineage: {e}")
            return {'error': str(e)}
    
    def _get_or_create_experiment(self, experiment_name: str) -> str:
        """Get or create MLflow experiment."""
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                return experiment.experiment_id
            else:
                return mlflow.create_experiment(experiment_name)
        except Exception as e:
            logger.error(f"Failed to get/create experiment: {e}")
            return "0"  # Default experiment
    
    def list_all_models(self) -> Dict:
        """List all registered models and their versions."""
        models_info = {}
        
        try:
            for model_name in [self.als_model_name, self.lightgbm_model_name, self.similarity_model_name]:
                try:
                    model = self.client.get_registered_model(model_name)
                    versions = []
                    
                    for version in model.latest_versions:
                        versions.append({
                            'version': version.version,
                            'stage': version.current_stage,
                            'creation_time': version.creation_timestamp,
                            'run_id': version.run_id
                        })
                    
                    models_info[model_name] = {
                        'description': model.description,
                        'creation_time': model.creation_timestamp,
                        'versions': versions
                    }
                    
                except Exception as e:
                    models_info[model_name] = {'error': str(e)}
            
            return models_info
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return {'error': str(e)}
