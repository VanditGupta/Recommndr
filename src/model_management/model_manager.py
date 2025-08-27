"""
Unified Model Manager for Phase 6

Orchestrates all model management components:
- MLflow Model Registry integration
- Performance monitoring
- Automated rollback system
- Model deployment coordination
- Health checks and validation
"""

import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
from datetime import datetime

from .mlflow_registry import MLflowModelRegistry
from .model_monitor import ModelPerformanceMonitor, PerformanceAlert
from .rollback_system import AutomatedRollbackSystem, RollbackConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ModelManager:
    """Unified model management system for the recommendation engine."""
    
    def __init__(self, mlflow_tracking_uri: str = "http://localhost:5001"):
        """Initialize the model manager."""
        self.tracking_uri = mlflow_tracking_uri
        
        # Initialize components
        self.registry = MLflowModelRegistry(mlflow_tracking_uri)
        self.monitor = ModelPerformanceMonitor()
        self.rollback_system = AutomatedRollbackSystem(
            self.registry, 
            self.monitor,
            RollbackConfig()
        )
        
        # Model status tracking
        self.model_status = {}
        self.last_health_check = 0
        
        logger.info("🎯 Model Manager initialized with all components")
    
    def start_monitoring(self):
        """Start all monitoring systems."""
        logger.info("🚀 Starting model management monitoring systems...")
        
        # Start performance monitoring
        self.monitor.start_monitoring(check_interval_seconds=30)
        
        # Start rollback monitoring
        self.rollback_system.start_rollback_monitoring()
        
        logger.info("✅ All monitoring systems started")
    
    def stop_monitoring(self):
        """Stop all monitoring systems."""
        logger.info("⏹️ Stopping model management systems...")
        
        self.monitor.stop_monitoring()
        self.rollback_system.stop_rollback_monitoring()
        
        logger.info("✅ All monitoring systems stopped")
    
    def register_current_models(self) -> Dict[str, Any]:
        """Register all current models in the MLflow registry."""
        logger.info("📝 Registering current models in MLflow registry...")
        
        registration_results = {
            'timestamp': time.time(),
            'results': {}
        }
        
        try:
            # Register ALS model
            als_model_path = "models/phase3/als_model.pkl"
            if Path(als_model_path).exists():
                als_version = self.registry.register_als_model(
                    als_model_path,
                    model_metadata={
                        'phase': 'phase3',
                        'model_type': 'collaborative_filtering',
                        'algorithm': 'alternating_least_squares',
                        'registration_time': datetime.now().isoformat()
                    }
                )
                registration_results['results']['als_model'] = {
                    'version': als_version,
                    'status': 'success'
                }
                
                # Promote to staging first
                self.registry.transition_model_stage(
                    self.registry.als_model_name, als_version, "Staging"
                )
                
                logger.info(f"✅ ALS model registered as version {als_version}")
            else:
                registration_results['results']['als_model'] = {
                    'status': 'error',
                    'message': 'Model file not found'
                }
        
        except Exception as e:
            logger.error(f"Failed to register ALS model: {e}")
            registration_results['results']['als_model'] = {
                'status': 'error',
                'message': str(e)
            }
        
        try:
            # Register LightGBM model
            lgb_model_path = "models/phase4/lightgbm_ranker.pkl"
            if Path(lgb_model_path).exists():
                lgb_version = self.registry.register_lightgbm_model(
                    lgb_model_path,
                    model_metadata={
                        'phase': 'phase4',
                        'model_type': 'ranking',
                        'algorithm': 'lightgbm',
                        'registration_time': datetime.now().isoformat()
                    }
                )
                registration_results['results']['lightgbm_model'] = {
                    'version': lgb_version,
                    'status': 'success'
                }
                
                # Promote to staging first
                if lgb_version:
                    self.registry.transition_model_stage(
                        self.registry.lightgbm_model_name, lgb_version, "Staging"
                    )
                
                logger.info(f"✅ LightGBM model registered as version {lgb_version}")
            else:
                registration_results['results']['lightgbm_model'] = {
                    'status': 'error',
                    'message': 'Model file not found'
                }
        
        except Exception as e:
            logger.error(f"Failed to register LightGBM model: {e}")
            registration_results['results']['lightgbm_model'] = {
                'status': 'error',
                'message': str(e)
            }
        
        try:
            # Register Similarity model
            similarity_models_dir = "models/phase5"
            if Path(similarity_models_dir).exists():
                sim_version = self.registry.register_similarity_model(
                    similarity_models_dir,
                    model_metadata={
                        'phase': 'phase5',
                        'model_type': 'similarity',
                        'algorithm': 'hybrid_similarity',
                        'registration_time': datetime.now().isoformat()
                    }
                )
                registration_results['results']['similarity_model'] = {
                    'version': sim_version,
                    'status': 'success'
                }
                
                # Promote to staging first
                self.registry.transition_model_stage(
                    self.registry.similarity_model_name, sim_version, "Staging"
                )
                
                logger.info(f"✅ Similarity model registered as version {sim_version}")
            else:
                registration_results['results']['similarity_model'] = {
                    'status': 'error',
                    'message': 'Model directory not found'
                }
        
        except Exception as e:
            logger.error(f"Failed to register Similarity model: {e}")
            registration_results['results']['similarity_model'] = {
                'status': 'error',
                'message': str(e)
            }
        
        return registration_results
    
    def promote_models_to_production(self) -> Dict[str, Any]:
        """Promote all staging models to production."""
        logger.info("🚀 Promoting staging models to production...")
        
        promotion_results = {
            'timestamp': time.time(),
            'results': {}
        }
        
        model_names = [
            self.registry.als_model_name,
            self.registry.lightgbm_model_name,
            self.registry.similarity_model_name
        ]
        
        for model_name in model_names:
            try:
                # Get staging version
                staging_version = self.registry.get_model_version(model_name, "Staging")
                
                if staging_version:
                    # Promote to production
                    success = self.registry.transition_model_stage(
                        model_name, staging_version, "Production", archive_existing=True
                    )
                    
                    promotion_results['results'][model_name] = {
                        'version': staging_version,
                        'status': 'success' if success else 'failed',
                        'promoted_to': 'Production'
                    }
                    
                    if success:
                        logger.info(f"✅ Promoted {model_name} v{staging_version} to Production")
                    else:
                        logger.error(f"❌ Failed to promote {model_name} v{staging_version}")
                else:
                    promotion_results['results'][model_name] = {
                        'status': 'error',
                        'message': 'No staging version found'
                    }
                    
            except Exception as e:
                logger.error(f"Error promoting {model_name}: {e}")
                promotion_results['results'][model_name] = {
                    'status': 'error',
                    'message': str(e)
                }
        
        return promotion_results
    
    def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        logger.info("🏥 Performing system health check...")
        
        health_check = {
            'timestamp': time.time(),
            'overall_status': 'healthy',
            'components': {}
        }
        
        # Check MLflow registry connectivity
        try:
            models_info = self.registry.list_all_models()
            health_check['components']['mlflow_registry'] = {
                'status': 'healthy',
                'models_registered': len(models_info),
                'details': models_info
            }
        except Exception as e:
            health_check['components']['mlflow_registry'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_check['overall_status'] = 'degraded'
        
        # Check model monitoring
        try:
            current_metrics = self.monitor.get_current_metrics(5)
            health_check['components']['performance_monitoring'] = {
                'status': 'healthy',
                'active': self.monitor.monitoring_active,
                'metrics_collected': len(current_metrics.get('metrics', {}))
            }
        except Exception as e:
            health_check['components']['performance_monitoring'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_check['overall_status'] = 'degraded'
        
        # Check rollback system
        try:
            rollback_stats = self.rollback_system.get_rollback_stats()
            health_check['components']['rollback_system'] = {
                'status': 'healthy',
                'active': rollback_stats['system_active'],
                'total_rollbacks': rollback_stats['total_rollbacks'],
                'success_rate': rollback_stats['success_rate']
            }
        except Exception as e:
            health_check['components']['rollback_system'] = {
                'status': 'unhealthy', 
                'error': str(e)
            }
            health_check['overall_status'] = 'degraded'
        
        # Check production models
        try:
            production_models = {}
            model_names = [
                self.registry.als_model_name,
                self.registry.lightgbm_model_name,
                self.registry.similarity_model_name
            ]
            
            for model_name in model_names:
                prod_version = self.registry.get_model_version(model_name, "Production")
                production_models[model_name] = {
                    'version': prod_version,
                    'status': 'deployed' if prod_version else 'not_deployed'
                }
            
            health_check['components']['production_models'] = {
                'status': 'healthy',
                'models': production_models
            }
        except Exception as e:
            health_check['components']['production_models'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_check['overall_status'] = 'degraded'
        
        # Overall health assessment
        unhealthy_components = [
            comp for comp, status in health_check['components'].items()
            if status.get('status') == 'unhealthy'
        ]
        
        if unhealthy_components:
            health_check['overall_status'] = 'unhealthy'
            health_check['unhealthy_components'] = unhealthy_components
        
        self.last_health_check = time.time()
        logger.info(f"🏥 Health check completed: {health_check['overall_status']}")
        
        return health_check
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            'timestamp': time.time(),
            'system_info': {
                'mlflow_tracking_uri': self.tracking_uri,
                'monitoring_active': self.monitor.monitoring_active,
                'rollback_system_active': self.rollback_system.system_active
            }
        }
        
        # Get model registry status
        try:
            status['model_registry'] = self.registry.list_all_models()
        except Exception as e:
            status['model_registry'] = {'error': str(e)}
        
        # Get performance metrics
        try:
            status['performance_metrics'] = self.monitor.get_current_metrics(60)
        except Exception as e:
            status['performance_metrics'] = {'error': str(e)}
        
        # Get rollback history
        try:
            status['rollback_stats'] = self.rollback_system.get_rollback_stats()
            status['recent_rollbacks'] = [
                {
                    'event_id': event.event_id,
                    'timestamp': event.timestamp,
                    'model_name': event.model_name,
                    'trigger': event.trigger.value,
                    'success': event.rollback_success
                }
                for event in self.rollback_system.get_rollback_history(24)
            ]
        except Exception as e:
            status['rollback_info'] = {'error': str(e)}
        
        return status
    
    def simulate_performance_degradation(self, metric_type: str = "latency") -> Dict[str, Any]:
        """Simulate performance degradation for testing rollback system."""
        logger.info(f"🧪 Simulating {metric_type} degradation for testing...")
        
        simulation_results = {
            'timestamp': time.time(),
            'metric_type': metric_type,
            'simulation_events': []
        }
        
        if metric_type == "latency":
            # Simulate high latency
            for i in range(10):
                latency = 350 + (i * 10)  # Gradually increasing latency
                self.monitor.record_latency(latency, user_id=9999, model_version="test")
                simulation_results['simulation_events'].append({
                    'event': f"Recorded latency: {latency}ms",
                    'timestamp': time.time()
                })
                time.sleep(0.1)
        
        elif metric_type == "error_rate":
            # Simulate high error rate
            for i in range(20):
                self.monitor.record_error("simulated_error", user_id=9999, model_version="test")
                simulation_results['simulation_events'].append({
                    'event': "Recorded error",
                    'timestamp': time.time()
                })
                time.sleep(0.1)
        
        elif metric_type == "ctr":
            # Simulate CTR degradation
            for i in range(10):
                degraded_ctr = 0.01 - (i * 0.001)  # Decreasing CTR
                self.monitor.record_ctr(degraded_ctr, user_id=9999, model_version="test")
                simulation_results['simulation_events'].append({
                    'event': f"Recorded CTR: {degraded_ctr:.3f}",
                    'timestamp': time.time()
                })
                time.sleep(0.1)
        
        # Wait for monitoring to detect the issue
        time.sleep(2)
        
        # Check if alerts were triggered
        recent_alerts = [alert for alert in self.monitor.alerts if alert.timestamp > time.time() - 10]
        simulation_results['alerts_triggered'] = len(recent_alerts)
        simulation_results['alert_details'] = [
            {
                'alert_id': alert.alert_id,
                'metric_name': alert.metric_name,
                'severity': alert.severity,
                'message': alert.message
            }
            for alert in recent_alerts
        ]
        
        logger.info(f"✅ Simulation completed. Triggered {len(recent_alerts)} alerts")
        return simulation_results
    
    def get_model_comparison_report(self, model_name: str, 
                                  version1: str, version2: str) -> Dict[str, Any]:
        """Generate detailed model comparison report."""
        return self.registry.compare_model_versions(model_name, version1, version2)
    
    def backup_model_registry(self, backup_dir: str = "backups/mlflow_registry") -> Dict[str, Any]:
        """Create backup of model registry metadata."""
        logger.info("💾 Creating model registry backup...")
        
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        backup_info = {
            'timestamp': time.time(),
            'backup_dir': str(backup_path),
            'models_backed_up': []
        }
        
        try:
            # Get all model information
            models_info = self.registry.list_all_models()
            
            # Save models metadata
            backup_file = backup_path / f"models_registry_{int(time.time())}.json"
            with open(backup_file, 'w') as f:
                json.dump(models_info, f, indent=2)
            
            backup_info['backup_file'] = str(backup_file)
            backup_info['models_backed_up'] = list(models_info.keys())
            
            logger.info(f"✅ Model registry backup created: {backup_file}")
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            backup_info['error'] = str(e)
        
        return backup_info
