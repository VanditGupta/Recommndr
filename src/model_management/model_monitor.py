"""
Model Performance Monitoring System

Tracks model performance metrics and detects degradation:
- Real-time latency monitoring
- CTR and conversion tracking
- Error rate monitoring
- Performance alerting
- Drift detection preparation
"""

import time
import threading
import json
from collections import defaultdict, deque
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
import statistics
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetric:
    """Single performance metric measurement."""
    timestamp: float
    metric_name: str
    metric_value: float
    user_id: Optional[int] = None
    model_version: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class PerformanceAlert:
    """Performance alert when thresholds are breached."""
    alert_id: str
    timestamp: float
    metric_name: str
    current_value: float
    threshold_value: float
    severity: str  # 'warning', 'critical'
    message: str
    suggested_action: str


class MetricBuffer:
    """Thread-safe circular buffer for storing metrics."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def add(self, metric: PerformanceMetric):
        """Add a metric to the buffer."""
        with self.lock:
            self.buffer.append(metric)
    
    def get_recent(self, minutes: int = 5) -> List[PerformanceMetric]:
        """Get metrics from the last N minutes."""
        cutoff_time = time.time() - (minutes * 60)
        
        with self.lock:
            return [m for m in self.buffer if m.timestamp >= cutoff_time]
    
    def get_all(self) -> List[PerformanceMetric]:
        """Get all metrics in buffer."""
        with self.lock:
            return list(self.buffer)


class PerformanceThresholds:
    """Configurable performance thresholds."""
    
    def __init__(self):
        # Latency thresholds (milliseconds)
        self.latency_warning_ms = 200
        self.latency_critical_ms = 300
        
        # Error rate thresholds (percentage)
        self.error_rate_warning_pct = 2.0
        self.error_rate_critical_pct = 5.0
        
        # CTR thresholds (absolute values)
        self.ctr_drop_warning_pct = 10.0  # 10% drop from baseline
        self.ctr_drop_critical_pct = 20.0  # 20% drop from baseline
        
        # Baseline values (updated automatically)
        self.baseline_latency_ms = 100
        self.baseline_ctr = 0.05  # 5% baseline CTR
        self.baseline_error_rate = 0.5  # 0.5% baseline error rate
        
        # Moving window for calculations
        self.window_minutes = 5
        self.baseline_window_hours = 24


class ModelPerformanceMonitor:
    """Real-time model performance monitoring system."""
    
    def __init__(self, save_dir: str = "logs/performance"):
        """Initialize performance monitor."""
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Metric storage
        self.metrics = defaultdict(MetricBuffer)
        self.alerts = deque(maxlen=1000)  # Store last 1000 alerts
        
        # Configuration
        self.thresholds = PerformanceThresholds()
        self.alert_callbacks = []
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        
        logger.info("📊 Model Performance Monitor initialized")
    
    def start_monitoring(self, check_interval_seconds: int = 30):
        """Start background monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(check_interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info(f"🔄 Started performance monitoring (check interval: {check_interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("⏹️ Stopped performance monitoring")
    
    def record_latency(self, latency_ms: float, user_id: Optional[int] = None,
                      model_version: Optional[str] = None):
        """Record model inference latency."""
        metric = PerformanceMetric(
            timestamp=time.time(),
            metric_name="latency_ms",
            metric_value=latency_ms,
            user_id=user_id,
            model_version=model_version
        )
        
        self.metrics["latency_ms"].add(metric)
    
    def record_ctr(self, ctr_value: float, user_id: Optional[int] = None,
                  model_version: Optional[str] = None):
        """Record click-through rate."""
        metric = PerformanceMetric(
            timestamp=time.time(),
            metric_name="ctr",
            metric_value=ctr_value,
            user_id=user_id,
            model_version=model_version
        )
        
        self.metrics["ctr"].add(metric)
    
    def record_error(self, error_type: str = "general", user_id: Optional[int] = None,
                    model_version: Optional[str] = None):
        """Record an error occurrence."""
        metric = PerformanceMetric(
            timestamp=time.time(),
            metric_name="error",
            metric_value=1.0,  # Count of 1
            user_id=user_id,
            model_version=model_version,
            metadata={"error_type": error_type}
        )
        
        self.metrics["error"].add(metric)
    
    def record_recommendation_quality(self, quality_score: float, 
                                    metric_type: str = "general",
                                    user_id: Optional[int] = None,
                                    model_version: Optional[str] = None):
        """Record recommendation quality metrics."""
        metric = PerformanceMetric(
            timestamp=time.time(),
            metric_name=f"quality_{metric_type}",
            metric_value=quality_score,
            user_id=user_id,
            model_version=model_version
        )
        
        self.metrics[f"quality_{metric_type}"].add(metric)
    
    def get_current_metrics(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get current performance metrics summary."""
        metrics_summary = {
            'timestamp': time.time(),
            'window_minutes': window_minutes,
            'metrics': {}
        }
        
        for metric_name, buffer in self.metrics.items():
            recent_metrics = buffer.get_recent(window_minutes)
            
            if not recent_metrics:
                continue
            
            values = [m.metric_value for m in recent_metrics]
            
            if metric_name == "error":
                # Error rate calculation
                total_requests = len(self.metrics.get("latency_ms", MetricBuffer()).get_recent(window_minutes))
                error_count = len(values)
                error_rate = (error_count / max(total_requests, 1)) * 100
                
                metrics_summary['metrics'][metric_name] = {
                    'count': error_count,
                    'rate_percent': error_rate,
                    'total_requests': total_requests
                }
            else:
                # Standard metrics
                metrics_summary['metrics'][metric_name] = {
                    'count': len(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'min': min(values),
                    'max': max(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0
                }
                
                # Percentiles for latency
                if metric_name == "latency_ms" and len(values) > 10:
                    sorted_values = sorted(values)
                    n = len(sorted_values)
                    
                    metrics_summary['metrics'][metric_name].update({
                        'p50': sorted_values[int(0.5 * n)],
                        'p90': sorted_values[int(0.9 * n)],
                        'p95': sorted_values[int(0.95 * n)],
                        'p99': sorted_values[int(0.99 * n)]
                    })
        
        return metrics_summary
    
    def check_performance_alerts(self) -> List[PerformanceAlert]:
        """Check for performance threshold breaches."""
        alerts = []
        current_metrics = self.get_current_metrics(self.thresholds.window_minutes)
        
        # Check latency alerts
        if 'latency_ms' in current_metrics['metrics']:
            latency_stats = current_metrics['metrics']['latency_ms']
            
            if 'p95' in latency_stats:
                p95_latency = latency_stats['p95']
                
                if p95_latency > self.thresholds.latency_critical_ms:
                    alert = PerformanceAlert(
                        alert_id=f"latency_critical_{int(time.time())}",
                        timestamp=time.time(),
                        metric_name="latency_p95",
                        current_value=p95_latency,
                        threshold_value=self.thresholds.latency_critical_ms,
                        severity="critical",
                        message=f"P95 latency ({p95_latency:.1f}ms) exceeds critical threshold ({self.thresholds.latency_critical_ms}ms)",
                        suggested_action="Consider immediate rollback to previous model version"
                    )
                    alerts.append(alert)
                    
                elif p95_latency > self.thresholds.latency_warning_ms:
                    alert = PerformanceAlert(
                        alert_id=f"latency_warning_{int(time.time())}",
                        timestamp=time.time(),
                        metric_name="latency_p95", 
                        current_value=p95_latency,
                        threshold_value=self.thresholds.latency_warning_ms,
                        severity="warning",
                        message=f"P95 latency ({p95_latency:.1f}ms) exceeds warning threshold ({self.thresholds.latency_warning_ms}ms)",
                        suggested_action="Monitor closely and investigate performance issues"
                    )
                    alerts.append(alert)
        
        # Check error rate alerts
        if 'error' in current_metrics['metrics']:
            error_stats = current_metrics['metrics']['error']
            error_rate = error_stats['rate_percent']
            
            if error_rate > self.thresholds.error_rate_critical_pct:
                alert = PerformanceAlert(
                    alert_id=f"error_rate_critical_{int(time.time())}",
                    timestamp=time.time(),
                    metric_name="error_rate",
                    current_value=error_rate,
                    threshold_value=self.thresholds.error_rate_critical_pct,
                    severity="critical",
                    message=f"Error rate ({error_rate:.1f}%) exceeds critical threshold ({self.thresholds.error_rate_critical_pct}%)",
                    suggested_action="Immediate rollback recommended - high error rate detected"
                )
                alerts.append(alert)
                
            elif error_rate > self.thresholds.error_rate_warning_pct:
                alert = PerformanceAlert(
                    alert_id=f"error_rate_warning_{int(time.time())}",
                    timestamp=time.time(),
                    metric_name="error_rate",
                    current_value=error_rate,
                    threshold_value=self.thresholds.error_rate_warning_pct,
                    severity="warning",
                    message=f"Error rate ({error_rate:.1f}%) exceeds warning threshold ({self.thresholds.error_rate_warning_pct}%)",
                    suggested_action="Investigate error causes and monitor closely"
                )
                alerts.append(alert)
        
        # Check CTR degradation
        if 'ctr' in current_metrics['metrics']:
            ctr_stats = current_metrics['metrics']['ctr']
            current_ctr = ctr_stats['mean']
            
            ctr_drop_pct = ((self.thresholds.baseline_ctr - current_ctr) / self.thresholds.baseline_ctr) * 100
            
            if ctr_drop_pct > self.thresholds.ctr_drop_critical_pct:
                alert = PerformanceAlert(
                    alert_id=f"ctr_drop_critical_{int(time.time())}",
                    timestamp=time.time(),
                    metric_name="ctr_drop",
                    current_value=ctr_drop_pct,
                    threshold_value=self.thresholds.ctr_drop_critical_pct,
                    severity="critical",
                    message=f"CTR dropped {ctr_drop_pct:.1f}% from baseline (current: {current_ctr:.3f}, baseline: {self.thresholds.baseline_ctr:.3f})",
                    suggested_action="Model performance severely degraded - rollback recommended"
                )
                alerts.append(alert)
                
            elif ctr_drop_pct > self.thresholds.ctr_drop_warning_pct:
                alert = PerformanceAlert(
                    alert_id=f"ctr_drop_warning_{int(time.time())}",
                    timestamp=time.time(),
                    metric_name="ctr_drop",
                    current_value=ctr_drop_pct,
                    threshold_value=self.thresholds.ctr_drop_warning_pct,
                    severity="warning",
                    message=f"CTR dropped {ctr_drop_pct:.1f}% from baseline (current: {current_ctr:.3f}, baseline: {self.thresholds.baseline_ctr:.3f})",
                    suggested_action="Monitor recommendation quality and user engagement"
                )
                alerts.append(alert)
        
        # Store alerts
        for alert in alerts:
            self.alerts.append(alert)
        
        return alerts
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add callback function to be called when alerts are triggered."""
        self.alert_callbacks.append(callback)
    
    def _monitoring_loop(self, check_interval_seconds: int):
        """Background monitoring loop."""
        logger.info("🔄 Started monitoring loop")
        
        while self.monitoring_active:
            try:
                # Check for alerts
                alerts = self.check_performance_alerts()
                
                # Trigger callbacks for new alerts
                for alert in alerts:
                    logger.warning(f"🚨 ALERT: {alert.message}")
                    
                    for callback in self.alert_callbacks:
                        try:
                            callback(alert)
                        except Exception as e:
                            logger.error(f"Alert callback failed: {e}")
                
                # Save metrics periodically
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    self._save_metrics()
                
                time.sleep(check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(check_interval_seconds)
        
        logger.info("⏹️ Monitoring loop stopped")
    
    def _save_metrics(self):
        """Save current metrics to disk."""
        try:
            current_metrics = self.get_current_metrics(60)  # Last hour
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            filename = self.save_dir / f"metrics_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(current_metrics, f, indent=2)
            
            # Also save recent alerts
            recent_alerts = [asdict(alert) for alert in list(self.alerts)[-100:]]
            alerts_filename = self.save_dir / f"alerts_{timestamp}.json"
            
            with open(alerts_filename, 'w') as f:
                json.dump(recent_alerts, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            'report_timestamp': time.time(),
            'report_period_hours': hours,
            'summary': {},
            'alerts_summary': {},
            'recommendations': []
        }
        
        # Current metrics
        current_metrics = self.get_current_metrics(60)  # Last hour
        report['current_metrics'] = current_metrics
        
        # Alert summary
        cutoff_time = time.time() - (hours * 3600)
        recent_alerts = [alert for alert in self.alerts if alert.timestamp >= cutoff_time]
        
        alert_counts = defaultdict(int)
        for alert in recent_alerts:
            alert_counts[alert.severity] += 1
        
        report['alerts_summary'] = {
            'total_alerts': len(recent_alerts),
            'by_severity': dict(alert_counts),
            'recent_alerts': [asdict(alert) for alert in recent_alerts[-10:]]  # Last 10 alerts
        }
        
        # Performance summary
        if 'latency_ms' in current_metrics['metrics']:
            latency_stats = current_metrics['metrics']['latency_ms']
            if latency_stats['mean'] > self.thresholds.latency_warning_ms:
                report['recommendations'].append("Consider optimizing model inference latency")
        
        if 'error' in current_metrics['metrics']:
            error_stats = current_metrics['metrics']['error']
            if error_stats['rate_percent'] > self.thresholds.error_rate_warning_pct:
                report['recommendations'].append("Investigate and reduce error rate")
        
        if not report['recommendations']:
            report['recommendations'].append("System performance is within acceptable thresholds")
        
        return report
    
    def update_baselines(self):
        """Update baseline metrics from recent performance."""
        logger.info("📊 Updating performance baselines")
        
        # Get metrics from last 24 hours for baseline calculation
        baseline_metrics = self.get_current_metrics(self.thresholds.baseline_window_hours * 60)
        
        if 'latency_ms' in baseline_metrics['metrics']:
            latency_stats = baseline_metrics['metrics']['latency_ms']
            self.thresholds.baseline_latency_ms = latency_stats['mean']
        
        if 'ctr' in baseline_metrics['metrics']:
            ctr_stats = baseline_metrics['metrics']['ctr']
            self.thresholds.baseline_ctr = ctr_stats['mean']
        
        if 'error' in baseline_metrics['metrics']:
            error_stats = baseline_metrics['metrics']['error']
            self.thresholds.baseline_error_rate = error_stats['rate_percent']
        
        logger.info(f"✅ Updated baselines: latency={self.thresholds.baseline_latency_ms:.1f}ms, "
                   f"CTR={self.thresholds.baseline_ctr:.3f}, error_rate={self.thresholds.baseline_error_rate:.2f}%")
