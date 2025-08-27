"""
Model Management API for Phase 6

FastAPI endpoints for model management operations:
- Model registry operations
- Performance monitoring
- Rollback triggers and status
- Health checks and system status
- Feature metadata management
"""

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import time
from datetime import datetime

from .model_manager import ModelManager
from .feast_metadata import FeatureMetadata
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Pydantic models for API requests/responses
class ModelRegistrationRequest(BaseModel):
    """Request model for registering models."""
    register_als: bool = True
    register_lightgbm: bool = True
    register_similarity: bool = True


class ModelPromotionRequest(BaseModel):
    """Request model for promoting models."""
    model_name: str
    target_stage: str = "Production"


class ManualRollbackRequest(BaseModel):
    """Request model for manual rollback."""
    model_name: str
    target_version: Optional[str] = None
    reason: str = "Manual rollback"


class PerformanceMetricRequest(BaseModel):
    """Request model for recording performance metrics."""
    metric_type: str  # 'latency', 'ctr', 'error'
    metric_value: float
    user_id: Optional[int] = None
    model_version: Optional[str] = None


class ModelManagementAPI:
    """FastAPI application for model management."""
    
    def __init__(self, mlflow_tracking_uri: str = "http://localhost:5001"):
        """Initialize model management API."""
        self.app = FastAPI(
            title="Recommndr Model Management API",
            description="Phase 6: Model Registry, Monitoring & Rollback Management",
            version="1.0.0"
        )
        
        # Initialize model manager
        self.model_manager = ModelManager(mlflow_tracking_uri)
        self.feature_metadata = FeatureMetadata()
        
        # Setup routes
        self._setup_routes()
        
        # Start monitoring (will be done in startup event)
        self._monitoring_started = False
        
        logger.info("🎯 Model Management API initialized")
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize systems on startup."""
            try:
                logger.info("🚀 Starting Model Management API...")
                
                # Start monitoring systems
                self.model_manager.start_monitoring()
                self._monitoring_started = True
                
                # Register feature metadata
                self.feature_metadata.register_all_feature_metadata()
                
                logger.info("✅ Model Management API startup completed")
                
            except Exception as e:
                logger.error(f"Failed to start Model Management API: {e}")
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup on shutdown."""
            if self._monitoring_started:
                self.model_manager.stop_monitoring()
            logger.info("👋 Model Management API shutdown completed")
        
        @self.app.get("/")
        async def root():
            """Root endpoint."""
            return {
                "service": "Recommndr Model Management API",
                "version": "1.0.0",
                "phase": 6,
                "description": "Model registry, monitoring & automated rollback system",
                "monitoring_active": self._monitoring_started
            }
        
        @self.app.get("/health")
        async def health_check():
            """Comprehensive system health check."""
            try:
                health_status = self.model_manager.perform_health_check()
                return health_status
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/status")
        async def system_status():
            """Get comprehensive system status."""
            try:
                status = self.model_manager.get_system_status()
                return status
            except Exception as e:
                logger.error(f"Failed to get system status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/models/register")
        async def register_models(request: ModelRegistrationRequest):
            """Register models in MLflow registry."""
            try:
                logger.info("📝 API request: Register models")
                results = self.model_manager.register_current_models()
                return {
                    "status": "completed",
                    "timestamp": time.time(),
                    "registration_results": results
                }
            except Exception as e:
                logger.error(f"Model registration failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/models/promote")
        async def promote_models():
            """Promote staging models to production."""
            try:
                logger.info("🚀 API request: Promote models to production")
                results = self.model_manager.promote_models_to_production()
                return {
                    "status": "completed",
                    "timestamp": time.time(),
                    "promotion_results": results
                }
            except Exception as e:
                logger.error(f"Model promotion failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models/list")
        async def list_models():
            """List all registered models and their versions."""
            try:
                models_info = self.model_manager.registry.list_all_models()
                return {
                    "timestamp": time.time(),
                    "models": models_info
                }
            except Exception as e:
                logger.error(f"Failed to list models: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models/{model_name}/versions")
        async def get_model_versions(model_name: str):
            """Get versions for a specific model."""
            try:
                if model_name not in [
                    self.model_manager.registry.als_model_name,
                    self.model_manager.registry.lightgbm_model_name,
                    self.model_manager.registry.similarity_model_name
                ]:
                    raise HTTPException(status_code=404, detail="Model not found")
                
                # Get model info
                model = self.model_manager.registry.client.get_registered_model(model_name)
                versions = []
                
                for version in model.latest_versions:
                    versions.append({
                        'version': version.version,
                        'stage': version.current_stage,
                        'creation_time': version.creation_timestamp,
                        'run_id': version.run_id
                    })
                
                return {
                    "model_name": model_name,
                    "description": model.description,
                    "versions": versions
                }
                
            except Exception as e:
                logger.error(f"Failed to get model versions: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models/{model_name}/compare/{version1}/{version2}")
        async def compare_model_versions(model_name: str, version1: str, version2: str):
            """Compare two model versions."""
            try:
                comparison = self.model_manager.get_model_comparison_report(
                    model_name, version1, version2
                )
                return comparison
            except Exception as e:
                logger.error(f"Model comparison failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/monitoring/metrics")
        async def get_performance_metrics(
            window_minutes: int = Query(60, ge=1, le=1440, description="Time window in minutes")
        ):
            """Get current performance metrics."""
            try:
                metrics = self.model_manager.monitor.get_current_metrics(window_minutes)
                return metrics
            except Exception as e:
                logger.error(f"Failed to get metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/monitoring/metrics/record")
        async def record_performance_metric(request: PerformanceMetricRequest):
            """Record a performance metric."""
            try:
                if request.metric_type == "latency":
                    self.model_manager.monitor.record_latency(
                        request.metric_value, request.user_id, request.model_version
                    )
                elif request.metric_type == "ctr":
                    self.model_manager.monitor.record_ctr(
                        request.metric_value, request.user_id, request.model_version
                    )
                elif request.metric_type == "error":
                    self.model_manager.monitor.record_error(
                        "api_recorded", request.user_id, request.model_version
                    )
                else:
                    raise HTTPException(status_code=400, detail="Invalid metric type")
                
                return {
                    "status": "recorded",
                    "timestamp": time.time(),
                    "metric_type": request.metric_type,
                    "metric_value": request.metric_value
                }
                
            except Exception as e:
                logger.error(f"Failed to record metric: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/monitoring/alerts")
        async def get_recent_alerts(
            hours: int = Query(24, ge=1, le=168, description="Hours to look back")
        ):
            """Get recent performance alerts."""
            try:
                cutoff_time = time.time() - (hours * 3600)
                recent_alerts = [
                    {
                        'alert_id': alert.alert_id,
                        'timestamp': alert.timestamp,
                        'metric_name': alert.metric_name,
                        'current_value': alert.current_value,
                        'threshold_value': alert.threshold_value,
                        'severity': alert.severity,
                        'message': alert.message,
                        'suggested_action': alert.suggested_action
                    }
                    for alert in self.model_manager.monitor.alerts
                    if alert.timestamp >= cutoff_time
                ]
                
                return {
                    "timestamp": time.time(),
                    "hours_back": hours,
                    "alert_count": len(recent_alerts),
                    "alerts": recent_alerts
                }
                
            except Exception as e:
                logger.error(f"Failed to get alerts: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/rollback/status")
        async def get_rollback_status():
            """Get rollback system status and statistics."""
            try:
                stats = self.model_manager.rollback_system.get_rollback_stats()
                return stats
            except Exception as e:
                logger.error(f"Failed to get rollback status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/rollback/history")
        async def get_rollback_history(
            hours: int = Query(24, ge=1, le=168, description="Hours to look back")
        ):
            """Get rollback history."""
            try:
                history = self.model_manager.rollback_system.get_rollback_history(hours)
                
                rollback_events = []
                for event in history:
                    rollback_events.append({
                        'event_id': event.event_id,
                        'timestamp': event.timestamp,
                        'trigger': event.trigger.value,
                        'model_name': event.model_name,
                        'from_version': event.from_version,
                        'to_version': event.to_version,
                        'rollback_success': event.rollback_success,
                        'rollback_duration_seconds': event.rollback_duration_seconds,
                        'notes': event.notes
                    })
                
                return {
                    "timestamp": time.time(),
                    "hours_back": hours,
                    "rollback_count": len(rollback_events),
                    "rollback_events": rollback_events
                }
                
            except Exception as e:
                logger.error(f"Failed to get rollback history: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/rollback/manual")
        async def trigger_manual_rollback(request: ManualRollbackRequest):
            """Trigger manual rollback for a model."""
            try:
                logger.info(f"🔄 API request: Manual rollback for {request.model_name}")
                
                success = self.model_manager.rollback_system.manual_rollback(
                    request.model_name,
                    request.target_version,
                    request.reason
                )
                
                return {
                    "status": "completed" if success else "failed",
                    "timestamp": time.time(),
                    "model_name": request.model_name,
                    "target_version": request.target_version,
                    "reason": request.reason,
                    "rollback_success": success
                }
                
            except Exception as e:
                logger.error(f"Manual rollback failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/testing/simulate_degradation")
        async def simulate_performance_degradation(
            metric_type: str = Query("latency", regex="^(latency|error_rate|ctr)$"),
            background_tasks: BackgroundTasks = None
        ):
            """Simulate performance degradation for testing rollback system."""
            try:
                logger.info(f"🧪 API request: Simulate {metric_type} degradation")
                
                # Run simulation in background
                def run_simulation():
                    return self.model_manager.simulate_performance_degradation(metric_type)
                
                if background_tasks:
                    background_tasks.add_task(run_simulation)
                    return {
                        "status": "simulation_started",
                        "timestamp": time.time(),
                        "metric_type": metric_type,
                        "message": "Simulation running in background"
                    }
                else:
                    results = run_simulation()
                    return results
                
            except Exception as e:
                logger.error(f"Simulation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/features/metadata")
        async def get_feature_metadata():
            """Get feature metadata summary."""
            try:
                summary = self.feature_metadata.get_feature_summary()
                return {
                    "timestamp": time.time(),
                    "feature_summary": summary,
                    "feature_definitions": self.feature_metadata.feature_definitions
                }
            except Exception as e:
                logger.error(f"Failed to get feature metadata: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/features/documentation")
        async def get_feature_documentation():
            """Get comprehensive feature documentation."""
            try:
                documentation = self.feature_metadata.generate_feature_documentation()
                return {
                    "timestamp": time.time(),
                    "documentation": documentation
                }
            except Exception as e:
                logger.error(f"Failed to generate documentation: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/reports/performance")
        async def get_performance_report(
            hours: int = Query(24, ge=1, le=168, description="Hours to include in report")
        ):
            """Generate comprehensive performance report."""
            try:
                report = self.model_manager.monitor.get_performance_report(hours)
                return report
            except Exception as e:
                logger.error(f"Failed to generate performance report: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/backup/registry")
        async def backup_model_registry():
            """Create backup of model registry."""
            try:
                logger.info("💾 API request: Backup model registry")
                backup_info = self.model_manager.backup_model_registry()
                return backup_info
            except Exception as e:
                logger.error(f"Backup failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application."""
        return self.app


def create_model_management_api(mlflow_tracking_uri: str = "http://localhost:5001") -> FastAPI:
    """Create and return the model management API application."""
    api = ModelManagementAPI(mlflow_tracking_uri)
    return api.get_app()
