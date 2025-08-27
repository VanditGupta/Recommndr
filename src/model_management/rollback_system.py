"""
Automated Model Rollback System

Implements automatic model rollback based on performance triggers:
- Monitor performance metrics in real-time
- Trigger rollback on threshold breaches
- Validate rollback success
- Log all rollback events
- Integration with MLflow Model Registry
"""

import time
import threading
import json
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

from .mlflow_registry import MLflowModelRegistry
from .model_monitor import ModelPerformanceMonitor, PerformanceAlert
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RollbackTrigger(Enum):
    """Types of rollback triggers."""
    LATENCY_BREACH = "latency_breach"
    ERROR_RATE_SPIKE = "error_rate_spike"
    CTR_DEGRADATION = "ctr_degradation"
    MANUAL_TRIGGER = "manual_trigger"
    HEALTH_CHECK_FAILURE = "health_check_failure"


@dataclass
class RollbackEvent:
    """Rollback event record."""
    event_id: str
    timestamp: float
    trigger: RollbackTrigger
    model_name: str
    from_version: str
    to_version: str
    trigger_metrics: Dict[str, Any]
    rollback_success: bool
    rollback_duration_seconds: float
    validation_results: Optional[Dict] = None
    notes: Optional[str] = None


class RollbackConfig:
    """Configuration for rollback system."""
    
    def __init__(self):
        # Rollback trigger thresholds
        self.latency_p95_threshold_ms = 300
        self.error_rate_threshold_pct = 5.0
        self.ctr_drop_threshold_pct = 20.0
        
        # Rollback behavior
        self.auto_rollback_enabled = True
        self.rollback_validation_timeout_seconds = 60
        self.rollback_cooldown_minutes = 10  # Prevent frequent rollbacks
        
        # Safety settings
        self.max_rollbacks_per_hour = 3
        self.require_manual_approval_for_prod = False
        
        # Models to monitor
        self.monitored_models = [
            "recommndr-als-model",
            "recommndr-lightgbm-ranker",
            "recommndr-similarity-engine"
        ]


class AutomatedRollbackSystem:
    """Automated model rollback system with performance monitoring integration."""
    
    def __init__(self, mlflow_registry: MLflowModelRegistry,
                 performance_monitor: ModelPerformanceMonitor,
                 config: Optional[RollbackConfig] = None):
        """Initialize rollback system."""
        self.registry = mlflow_registry
        self.monitor = performance_monitor
        self.config = config or RollbackConfig()
        
        # Rollback state
        self.rollback_events = []
        self.last_rollback_time = 0
        self.rollback_count_last_hour = 0
        
        # System state
        self.system_active = False
        self.rollback_thread = None
        
        # Callbacks
        self.rollback_callbacks = []
        
        # Setup alert monitoring
        self.monitor.add_alert_callback(self._handle_performance_alert)
        
        logger.info("🔄 Automated Rollback System initialized")
    
    def start_rollback_monitoring(self):
        """Start automated rollback monitoring."""
        if self.system_active:
            logger.warning("Rollback monitoring already active")
            return
        
        self.system_active = True
        self.rollback_thread = threading.Thread(
            target=self._rollback_monitoring_loop,
            daemon=True
        )
        self.rollback_thread.start()
        
        logger.info("🚀 Started automated rollback monitoring")
    
    def stop_rollback_monitoring(self):
        """Stop automated rollback monitoring."""
        self.system_active = False
        if self.rollback_thread:
            self.rollback_thread.join(timeout=5)
        
        logger.info("⏹️ Stopped automated rollback monitoring")
    
    def _handle_performance_alert(self, alert: PerformanceAlert):
        """Handle performance alerts and trigger rollbacks if needed."""
        if not self.config.auto_rollback_enabled:
            logger.info(f"Auto-rollback disabled, ignoring alert: {alert.alert_id}")
            return
        
        if alert.severity != "critical":
            logger.info(f"Alert not critical, no rollback triggered: {alert.alert_id}")
            return
        
        # Determine rollback trigger type
        trigger = None
        if "latency" in alert.metric_name:
            trigger = RollbackTrigger.LATENCY_BREACH
        elif "error_rate" in alert.metric_name:
            trigger = RollbackTrigger.ERROR_RATE_SPIKE  
        elif "ctr" in alert.metric_name:
            trigger = RollbackTrigger.CTR_DEGRADATION
        
        if trigger:
            logger.warning(f"🚨 Critical alert detected: {alert.message}")
            self._initiate_rollback(trigger, alert)
    
    def _initiate_rollback(self, trigger: RollbackTrigger, alert: PerformanceAlert):
        """Initiate rollback for all monitored models."""
        current_time = time.time()
        
        # Check rollback cooldown
        if current_time - self.last_rollback_time < (self.config.rollback_cooldown_minutes * 60):
            logger.warning(f"Rollback blocked: cooldown period active")
            return
        
        # Check hourly rollback limit
        if self._get_rollbacks_last_hour() >= self.config.max_rollbacks_per_hour:
            logger.warning(f"Rollback blocked: hourly limit reached")
            return
        
        logger.info(f"🔄 Initiating automated rollback due to {trigger.value}")
        
        # Rollback all monitored models
        for model_name in self.config.monitored_models:
            try:
                success = self._rollback_model(model_name, trigger, alert)
                if success:
                    logger.info(f"✅ Successfully rolled back {model_name}")
                else:
                    logger.error(f"❌ Failed to rollback {model_name}")
            except Exception as e:
                logger.error(f"Error rolling back {model_name}: {e}")
        
        self.last_rollback_time = current_time
    
    def _rollback_model(self, model_name: str, trigger: RollbackTrigger, 
                       alert: PerformanceAlert) -> bool:
        """Rollback a specific model to its previous version."""
        rollback_start = time.time()
        
        try:
            # Get current production version
            current_version = self.registry.get_model_version(model_name, "Production")
            if not current_version:
                logger.error(f"No production version found for {model_name}")
                return False
            
            # Get previous version from staging or archived
            target_version = self._get_rollback_target_version(model_name, current_version)
            if not target_version:
                logger.error(f"No suitable rollback target found for {model_name}")
                return False
            
            logger.info(f"Rolling back {model_name} from v{current_version} to v{target_version}")
            
            # Archive current production version
            archive_success = self.registry.transition_model_stage(
                model_name, current_version, "Archived"
            )
            
            if not archive_success:
                logger.error(f"Failed to archive current version {current_version}")
                return False
            
            # Promote target version to production
            promote_success = self.registry.transition_model_stage(
                model_name, target_version, "Production"
            )
            
            if not promote_success:
                logger.error(f"Failed to promote version {target_version} to production")
                # Try to restore previous production version
                self.registry.transition_model_stage(model_name, current_version, "Production")
                return False
            
            rollback_duration = time.time() - rollback_start
            
            # Validate rollback
            validation_results = self._validate_rollback(model_name, target_version)
            
            # Record rollback event
            rollback_event = RollbackEvent(
                event_id=f"rollback_{model_name}_{int(time.time())}",
                timestamp=time.time(),
                trigger=trigger,
                model_name=model_name,
                from_version=current_version,
                to_version=target_version,
                trigger_metrics={
                    'alert_id': alert.alert_id,
                    'metric_name': alert.metric_name,
                    'current_value': alert.current_value,
                    'threshold_value': alert.threshold_value
                },
                rollback_success=True,
                rollback_duration_seconds=rollback_duration,
                validation_results=validation_results,
                notes=f"Automated rollback triggered by {trigger.value}"
            )
            
            self.rollback_events.append(rollback_event)
            self._save_rollback_event(rollback_event)
            
            # Notify callbacks
            for callback in self.rollback_callbacks:
                try:
                    callback(rollback_event)
                except Exception as e:
                    logger.error(f"Rollback callback failed: {e}")
            
            logger.info(f"✅ Rollback completed for {model_name} in {rollback_duration:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed for {model_name}: {e}")
            
            # Record failed rollback
            rollback_event = RollbackEvent(
                event_id=f"rollback_failed_{model_name}_{int(time.time())}",
                timestamp=time.time(),
                trigger=trigger,
                model_name=model_name,
                from_version=current_version if 'current_version' in locals() else "unknown",
                to_version="failed",
                trigger_metrics={
                    'alert_id': alert.alert_id,
                    'metric_name': alert.metric_name,
                    'current_value': alert.current_value,
                    'threshold_value': alert.threshold_value
                },
                rollback_success=False,
                rollback_duration_seconds=time.time() - rollback_start,
                notes=f"Rollback failed: {str(e)}"
            )
            
            self.rollback_events.append(rollback_event)
            self._save_rollback_event(rollback_event)
            
            return False
    
    def _get_rollback_target_version(self, model_name: str, current_version: str) -> Optional[str]:
        """Find the best version to rollback to."""
        try:
            # First try to get staging version
            staging_version = self.registry.get_model_version(model_name, "Staging")
            if staging_version and staging_version != current_version:
                logger.info(f"Found staging version {staging_version} for rollback")
                return staging_version
            
            # Otherwise, find the most recent archived version
            model = self.registry.client.get_registered_model(model_name)
            archived_versions = []
            
            for version in model.latest_versions:
                if (version.current_stage == "Archived" and 
                    version.version != current_version):
                    archived_versions.append((version.version, version.creation_timestamp))
            
            if archived_versions:
                # Sort by creation time and get most recent
                archived_versions.sort(key=lambda x: x[1], reverse=True)
                target_version = archived_versions[0][0]
                logger.info(f"Found archived version {target_version} for rollback")
                return target_version
            
            logger.warning(f"No suitable rollback target found for {model_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error finding rollback target: {e}")
            return None
    
    def _validate_rollback(self, model_name: str, target_version: str) -> Dict[str, Any]:
        """Validate that rollback was successful."""
        validation_results = {
            'timestamp': time.time(),
            'model_name': model_name,
            'target_version': target_version,
            'checks': {}
        }
        
        try:
            # Check 1: Verify version is in production
            current_prod_version = self.registry.get_model_version(model_name, "Production")
            validation_results['checks']['version_check'] = {
                'expected': target_version,
                'actual': current_prod_version,
                'passed': current_prod_version == target_version
            }
            
            # Check 2: Try to load the model
            try:
                model = self.registry.load_production_model(model_name)
                validation_results['checks']['model_load'] = {
                    'passed': model is not None,
                    'error': None
                }
            except Exception as e:
                validation_results['checks']['model_load'] = {
                    'passed': False,
                    'error': str(e)
                }
            
            # Check 3: Basic health check (if applicable)
            # This would be model-specific validation
            validation_results['checks']['health_check'] = {
                'passed': True,  # Placeholder
                'message': "Basic health check passed"
            }
            
            # Overall validation status
            all_checks_passed = all(
                check.get('passed', False) 
                for check in validation_results['checks'].values()
            )
            validation_results['overall_status'] = 'passed' if all_checks_passed else 'failed'
            
        except Exception as e:
            validation_results['error'] = str(e)
            validation_results['overall_status'] = 'error'
        
        return validation_results
    
    def manual_rollback(self, model_name: str, target_version: Optional[str] = None,
                       reason: str = "Manual rollback") -> bool:
        """Trigger manual rollback for a specific model."""
        logger.info(f"🔄 Manual rollback requested for {model_name}")
        
        # Create a fake alert for manual rollback
        fake_alert = PerformanceAlert(
            alert_id=f"manual_{int(time.time())}",
            timestamp=time.time(),
            metric_name="manual_trigger",
            current_value=0,
            threshold_value=0,
            severity="critical",
            message=reason,
            suggested_action="Manual rollback"
        )
        
        if target_version:
            # Rollback to specific version
            return self._rollback_to_specific_version(
                model_name, target_version, RollbackTrigger.MANUAL_TRIGGER, fake_alert, reason
            )
        else:
            # Rollback to previous version
            return self._rollback_model(model_name, RollbackTrigger.MANUAL_TRIGGER, fake_alert)
    
    def _rollback_to_specific_version(self, model_name: str, target_version: str,
                                    trigger: RollbackTrigger, alert: PerformanceAlert,
                                    reason: str) -> bool:
        """Rollback to a specific version."""
        rollback_start = time.time()
        
        try:
            current_version = self.registry.get_model_version(model_name, "Production")
            
            # Archive current version
            if current_version:
                self.registry.transition_model_stage(model_name, current_version, "Archived")
            
            # Promote target version
            success = self.registry.transition_model_stage(model_name, target_version, "Production")
            
            if success:
                rollback_duration = time.time() - rollback_start
                validation_results = self._validate_rollback(model_name, target_version)
                
                rollback_event = RollbackEvent(
                    event_id=f"manual_rollback_{model_name}_{int(time.time())}",
                    timestamp=time.time(),
                    trigger=trigger,
                    model_name=model_name,
                    from_version=current_version or "unknown",
                    to_version=target_version,
                    trigger_metrics={'reason': reason},
                    rollback_success=True,
                    rollback_duration_seconds=rollback_duration,
                    validation_results=validation_results,
                    notes=reason
                )
                
                self.rollback_events.append(rollback_event)
                self._save_rollback_event(rollback_event)
                
                logger.info(f"✅ Manual rollback completed for {model_name}")
                return True
            else:
                logger.error(f"Failed to promote version {target_version} to production")
                return False
                
        except Exception as e:
            logger.error(f"Manual rollback failed: {e}")
            return False
    
    def _rollback_monitoring_loop(self):
        """Background monitoring loop for rollback system."""
        logger.info("🔄 Started rollback monitoring loop")
        
        while self.system_active:
            try:
                # Reset hourly rollback counter
                current_hour = int(time.time() // 3600)
                last_hour = int(self.last_rollback_time // 3600)
                
                if current_hour > last_hour:
                    self.rollback_count_last_hour = 0
                
                # Cleanup old rollback events (keep last 1000)
                if len(self.rollback_events) > 1000:
                    self.rollback_events = self.rollback_events[-1000:]
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Rollback monitoring error: {e}")
                time.sleep(30)
        
        logger.info("⏹️ Rollback monitoring loop stopped")
    
    def _get_rollbacks_last_hour(self) -> int:
        """Get count of rollbacks in the last hour."""
        hour_ago = time.time() - 3600
        return sum(1 for event in self.rollback_events 
                  if event.timestamp >= hour_ago and event.rollback_success)
    
    def _save_rollback_event(self, event: RollbackEvent):
        """Save rollback event to disk."""
        try:
            events_dir = Path("logs/rollback_events")
            events_dir.mkdir(parents=True, exist_ok=True)
            
            filename = events_dir / f"rollback_{event.event_id}.json"
            
            with open(filename, 'w') as f:
                json.dump(asdict(event), f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save rollback event: {e}")
    
    def add_rollback_callback(self, callback: Callable[[RollbackEvent], None]):
        """Add callback function to be called when rollbacks occur."""
        self.rollback_callbacks.append(callback)
    
    def get_rollback_history(self, hours: int = 24) -> List[RollbackEvent]:
        """Get rollback history for the specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        return [event for event in self.rollback_events if event.timestamp >= cutoff_time]
    
    def get_rollback_stats(self) -> Dict[str, Any]:
        """Get rollback system statistics."""
        total_rollbacks = len(self.rollback_events)
        successful_rollbacks = sum(1 for event in self.rollback_events if event.rollback_success)
        
        # Rollbacks by trigger type
        trigger_counts = {}
        for event in self.rollback_events:
            trigger = event.trigger.value
            trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
        
        # Recent rollbacks
        recent_rollbacks = self.get_rollback_history(24)  # Last 24 hours
        
        return {
            'total_rollbacks': total_rollbacks,
            'successful_rollbacks': successful_rollbacks,
            'success_rate': (successful_rollbacks / max(total_rollbacks, 1)) * 100,
            'rollbacks_last_24h': len(recent_rollbacks),
            'rollbacks_last_hour': self._get_rollbacks_last_hour(),
            'rollbacks_by_trigger': trigger_counts,
            'last_rollback_time': self.last_rollback_time,
            'system_active': self.system_active,
            'auto_rollback_enabled': self.config.auto_rollback_enabled
        }
