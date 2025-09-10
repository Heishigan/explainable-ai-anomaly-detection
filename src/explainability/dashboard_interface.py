"""
Dashboard interface for explainable AI cybersecurity anomaly detection.
Provides data structures and utilities for realtime dashboard integration.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
import json
import logging
from collections import deque
import threading
import time
from enum import Enum

from .realtime_explainer import RealtimeExplainer, RealtimeExplanationResult
from .explanation_aggregator import AggregatedExplanation

class AlertLevel(Enum):
    """Alert severity levels for dashboard display"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DashboardAlert:
    """Alert structure for dashboard notifications"""
    id: str
    timestamp: str
    level: AlertLevel
    title: str
    message: str
    sample_id: Optional[str] = None
    confidence: Optional[float] = None
    auto_dismiss: bool = False
    dismiss_after_seconds: int = 30

@dataclass
class DashboardMetrics:
    """Real-time metrics for dashboard display"""
    total_samples_processed: int = 0
    attacks_detected: int = 0
    normal_traffic: int = 0
    detection_rate: float = 0.0
    false_positive_rate: float = 0.0
    avg_processing_time_ms: float = 0.0
    samples_per_minute: float = 0.0
    system_load: float = 0.0
    
    # Time-series data for charts
    detection_timeline: List[Dict[str, Any]] = field(default_factory=list)
    attack_type_distribution: Dict[str, int] = field(default_factory=dict)
    confidence_distribution: Dict[str, int] = field(default_factory=dict)

@dataclass
class DashboardDetection:
    """Individual detection result for dashboard display"""
    id: str
    timestamp: str
    prediction: str  # "Attack" or "Normal"
    probability: float
    confidence: float
    risk_level: str
    processing_time_ms: float
    
    # Explanation summary
    top_features: List[Dict[str, Any]]
    explanation_methods: List[str]
    methods_agreement: float
    
    # Source information
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    protocol: Optional[str] = None
    service: Optional[str] = None

class DashboardDataManager:
    """Manages dashboard data, real-time updates, and historical storage"""
    
    def __init__(self, max_detections: int = 1000, max_alerts: int = 100,
                 metrics_window_minutes: int = 60):
        """
        Initialize dashboard data manager
        
        Args:
            max_detections: Maximum detections to keep in memory
            max_alerts: Maximum alerts to keep
            metrics_window_minutes: Time window for metrics calculation
        """
        self.max_detections = max_detections
        self.max_alerts = max_alerts
        self.metrics_window_minutes = metrics_window_minutes
        
        # Thread-safe data storage
        self.lock = threading.RLock()
        
        # Real-time data
        self.detections = deque(maxlen=max_detections)
        self.alerts = deque(maxlen=max_alerts)
        self.metrics = DashboardMetrics()
        
        # Historical data for time-series analysis
        self.detection_history = deque(maxlen=10000)
        self.metrics_history = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        
        # Subscribers for real-time updates
        self.subscribers = []
        self.alert_callbacks = []
        
        # Periodic metrics calculation
        self.last_metrics_update = datetime.now()
        self.metrics_update_interval = 60  # seconds
        
        self.logger = logging.getLogger(__name__)
        
        # Start background metrics updater
        self._start_metrics_updater()
    
    def add_detection(self, explanation_result: RealtimeExplanationResult):
        """Add a new detection result to the dashboard"""
        if not explanation_result.success or not explanation_result.explanation:
            return
        
        explanation = explanation_result.explanation
        
        # Create dashboard detection
        detection = DashboardDetection(
            id=explanation_result.sample_id,
            timestamp=explanation.timestamp,
            prediction="Attack" if float(explanation.prediction) >= 0.5 else "Normal",
            probability=explanation.prediction_proba or 0.0,
            confidence=explanation.confidence_score or 0.0,
            risk_level=explanation.risk_level or "unknown",
            processing_time_ms=explanation_result.processing_time_ms or 0.0,
            top_features=self._extract_top_features(explanation),
            explanation_methods=explanation_result.methods_used or [],
            methods_agreement=explanation.explanation_agreement or 0.0
        )
        
        # Add source information if available
        self._enrich_detection_with_network_info(detection, explanation)
        
        with self.lock:
            self.detections.appendleft(detection)
            self.detection_history.append({
                'timestamp': detection.timestamp,
                'prediction': detection.prediction,
                'probability': detection.probability,
                'confidence': detection.confidence,
                'risk_level': detection.risk_level,
                'processing_time_ms': detection.processing_time_ms
            })
            
            # Update metrics
            self.metrics.total_samples_processed += 1
            if detection.prediction == "Attack":
                self.metrics.attacks_detected += 1
            else:
                self.metrics.normal_traffic += 1
            
            # Update confidence distribution
            confidence_bucket = self._get_confidence_bucket(detection.confidence)
            self.metrics.confidence_distribution[confidence_bucket] = \
                self.metrics.confidence_distribution.get(confidence_bucket, 0) + 1
        
        # Generate alerts for high-risk detections
        self._check_and_generate_alerts(detection)
        
        # Notify subscribers
        self._notify_subscribers('detection_added', detection)
    
    def add_alert(self, alert: DashboardAlert):
        """Add a new alert to the dashboard"""
        with self.lock:
            self.alerts.appendleft(alert)
        
        # Notify alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
        
        self._notify_subscribers('alert_added', alert)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data snapshot"""
        with self.lock:
            return {
                'metrics': asdict(self.metrics),
                'recent_detections': [asdict(det) for det in list(self.detections)[:50]],
                'active_alerts': [asdict(alert) for alert in self.alerts if not alert.auto_dismiss or 
                                self._is_alert_active(alert)],
                'timestamp': datetime.now().isoformat(),
                'system_status': self._get_system_status()
            }
    
    def get_detections_paginated(self, page: int = 1, per_page: int = 50,
                               filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get paginated detection results with optional filtering"""
        with self.lock:
            detections_list = list(self.detections)
        
        # Apply filters
        if filters:
            detections_list = self._filter_detections(detections_list, filters)
        
        # Pagination
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_detections = detections_list[start_idx:end_idx]
        
        return {
            'detections': [asdict(det) for det in page_detections],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total_items': len(detections_list),
                'total_pages': (len(detections_list) + per_page - 1) // per_page
            }
        }
    
    def get_time_series_data(self, hours: int = 24) -> Dict[str, Any]:
        """Get time-series data for dashboard charts with adaptive intervals"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            # Filter historical data by time window
            filtered_history = [
                entry for entry in self.detection_history
                if datetime.fromisoformat(entry['timestamp']) > cutoff_time
            ]
        
        # Adaptive interval calculation based on time range
        if hours <= 1:
            interval_minutes = 2  # 2-minute intervals for 1 hour (30 points)
            max_points = 30
        elif hours <= 6:
            interval_minutes = 15  # 15-minute intervals for 6 hours (24 points)
            max_points = 24
        elif hours <= 24:
            interval_minutes = 30  # 30-minute intervals for 24 hours (48 points)
            max_points = 48
        else:
            interval_minutes = 60  # 1-hour intervals for longer periods
            max_points = min(48, hours)
        
        time_series = self._group_by_time_interval(filtered_history, interval_minutes)
        
        # Additional processing for consistency
        if not time_series:
            time_series = self._generate_empty_timeline(interval_minutes, max_points)
        
        return {
            'detection_timeline': time_series,
            'interval_minutes': interval_minutes,
            'total_samples': len(filtered_history),
            'time_range_hours': hours,
            'data_points': len(time_series)
        }
    
    def get_attack_analytics(self) -> Dict[str, Any]:
        """Get attack analytics and trends"""
        with self.lock:
            recent_attacks = [
                det for det in self.detections 
                if det.prediction == "Attack"
            ][:100]
        
        if not recent_attacks:
            return {'message': 'No recent attacks detected'}
        
        # Risk and confidence analysis
        risk_levels = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        confidence_sum = 0
        
        for attack in recent_attacks:
            # Risk level distribution
            if attack.risk_level in risk_levels:
                risk_levels[attack.risk_level] += 1
            
            confidence_sum += attack.confidence
        
        avg_confidence = confidence_sum / len(recent_attacks)
        
        return {
            'total_recent_attacks': len(recent_attacks),
            'risk_level_distribution': risk_levels,
            'average_confidence': avg_confidence
        }
    
    def subscribe_to_updates(self, callback: Callable[[str, Any], None]):
        """Subscribe to real-time dashboard updates"""
        self.subscribers.append(callback)
    
    def subscribe_to_alerts(self, callback: Callable[[DashboardAlert], None]):
        """Subscribe to alert notifications"""
        self.alert_callbacks.append(callback)
    
    def dismiss_alert(self, alert_id: str):
        """Dismiss a specific alert"""
        with self.lock:
            self.alerts = deque([alert for alert in self.alerts if alert.id != alert_id],
                              maxlen=self.max_alerts)
        
        self._notify_subscribers('alert_dismissed', {'alert_id': alert_id})
    
    def _extract_top_features(self, explanation: AggregatedExplanation) -> List[Dict[str, Any]]:
        """Extract top features from explanation for dashboard display"""
        top_features = []
        
        if explanation.consensus_features:
            for feature in explanation.consensus_features[:10]:
                top_features.append({
                    'name': feature['feature'],
                    'importance': feature['avg_importance'],
                    'agreement': feature['agreement']
                })
        elif explanation.shap_explanation and explanation.shap_explanation.get('top_features'):
            for feature in explanation.shap_explanation['top_features'][:10]:
                top_features.append({
                    'name': feature['name'],
                    'importance': feature['abs_importance'],
                    'value': feature['value']
                })
        
        return top_features
    
    def _enrich_detection_with_network_info(self, detection: DashboardDetection, 
                                          explanation: AggregatedExplanation):
        """Enrich detection with network information if available in features"""
        # This would be implemented based on your specific feature mapping
        # For now, we'll add placeholder logic
        if explanation.shap_explanation and explanation.shap_explanation.get('top_features'):
            features = {f['name']: f['value'] for f in explanation.shap_explanation['top_features']}
            
            # Extract protocol information
            if 'proto' in features:
                proto_mapping = {0: 'ICMP', 1: 'TCP', 2: 'UDP'}
                detection.protocol = proto_mapping.get(features['proto'], 'Unknown')
            
            # Extract service information  
            if 'service' in features:
                detection.service = str(features['service'])
    
    def _check_and_generate_alerts(self, detection: DashboardDetection):
        """Generate alerts based on detection characteristics"""
        if detection.prediction != "Attack":
            return
        
        alert_level = AlertLevel.INFO
        auto_dismiss = True
        dismiss_time = 30
        
        # Determine alert level based on risk and confidence
        if detection.risk_level == "critical":
            alert_level = AlertLevel.CRITICAL
            auto_dismiss = False
        elif detection.risk_level == "high":
            alert_level = AlertLevel.HIGH
            auto_dismiss = False
        elif detection.risk_level == "medium" and detection.confidence > 0.8:
            alert_level = AlertLevel.MEDIUM
            dismiss_time = 60
        
        # Generate alert
        alert = DashboardAlert(
            id=f"alert_{detection.id}",
            timestamp=detection.timestamp,
            level=alert_level,
            title=f"{detection.risk_level.title()} Risk Attack Detected",
            message=f"Detected attack with {detection.confidence:.1%} confidence",
            sample_id=detection.id,
            confidence=detection.confidence,
            auto_dismiss=auto_dismiss,
            dismiss_after_seconds=dismiss_time
        )
        
        self.add_alert(alert)
    
    def _get_confidence_bucket(self, confidence: float) -> str:
        """Categorize confidence into buckets for distribution analysis"""
        if confidence >= 0.9:
            return "very_high"
        elif confidence >= 0.7:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        elif confidence >= 0.3:
            return "low"
        else:
            return "very_low"
    
    def _is_alert_active(self, alert: DashboardAlert) -> bool:
        """Check if auto-dismissible alert is still active"""
        if not alert.auto_dismiss:
            return True
        
        alert_time = datetime.fromisoformat(alert.timestamp)
        elapsed = (datetime.now() - alert_time).total_seconds()
        return elapsed < alert.dismiss_after_seconds
    
    def _filter_detections(self, detections: List[DashboardDetection], 
                         filters: Dict[str, Any]) -> List[DashboardDetection]:
        """Apply filters to detection list"""
        filtered = detections
        
        if 'prediction' in filters:
            filtered = [d for d in filtered if d.prediction == filters['prediction']]
        
        if 'risk_level' in filters:
            filtered = [d for d in filtered if d.risk_level == filters['risk_level']]
        
        
        if 'min_confidence' in filters:
            filtered = [d for d in filtered if d.confidence >= filters['min_confidence']]
        
        if 'time_range' in filters:
            start_time = datetime.fromisoformat(filters['time_range']['start'])
            end_time = datetime.fromisoformat(filters['time_range']['end'])
            filtered = [d for d in filtered 
                       if start_time <= datetime.fromisoformat(d.timestamp) <= end_time]
        
        return filtered
    
    def _group_by_time_interval(self, history: List[Dict], interval_minutes: int) -> List[Dict]:
        """Group historical data by time intervals with continuous timeline"""
        if not history:
            # Return empty timeline with current time buckets
            return self._generate_empty_timeline(interval_minutes)
        
        # Sort by timestamp
        history.sort(key=lambda x: x['timestamp'])
        
        # Create continuous timeline
        now = datetime.now()
        earliest_time = datetime.fromisoformat(history[0]['timestamp']) if history else now
        
        # Round to interval boundaries
        start_time = earliest_time.replace(second=0, microsecond=0)
        start_time = start_time.replace(
            minute=(start_time.minute // interval_minutes) * interval_minutes
        )
        
        end_time = now.replace(second=0, microsecond=0)
        end_time = end_time.replace(
            minute=(end_time.minute // interval_minutes) * interval_minutes
        )
        
        # Generate all time buckets
        time_buckets = {}
        current_time = start_time
        while current_time <= end_time:
            time_buckets[current_time.isoformat()] = {'attacks': 0, 'normal': 0, 'total': 0}
            current_time += timedelta(minutes=interval_minutes)
        
        # Populate buckets with actual data
        for entry in history:
            entry_time = datetime.fromisoformat(entry['timestamp'])
            interval_start = entry_time.replace(second=0, microsecond=0)
            interval_start = interval_start.replace(
                minute=(interval_start.minute // interval_minutes) * interval_minutes
            )
            
            bucket_key = interval_start.isoformat()
            if bucket_key in time_buckets:
                bucket = time_buckets[bucket_key]
                bucket['total'] += 1
                if entry['prediction'] == 'Attack':
                    bucket['attacks'] += 1
                else:
                    bucket['normal'] += 1
        
        # Convert to sorted list
        time_series = []
        for timestamp in sorted(time_buckets.keys()):
            bucket = time_buckets[timestamp]
            time_series.append({
                'timestamp': timestamp,
                **bucket
            })
        
        # Limit to reasonable number of points (max 100)
        if len(time_series) > 100:
            step = len(time_series) // 100
            time_series = time_series[::step]
        
        return time_series
    
    def _generate_empty_timeline(self, interval_minutes: int, num_buckets: int = 20) -> List[Dict]:
        """Generate empty timeline for when no historical data exists"""
        now = datetime.now()
        time_series = []
        
        for i in range(num_buckets):
            bucket_time = now - timedelta(minutes=interval_minutes * (num_buckets - 1 - i))
            bucket_time = bucket_time.replace(second=0, microsecond=0)
            bucket_time = bucket_time.replace(
                minute=(bucket_time.minute // interval_minutes) * interval_minutes
            )
            
            time_series.append({
                'timestamp': bucket_time.isoformat(),
                'attacks': 0,
                'normal': 0,
                'total': 0
            })
        
        return time_series
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get current system status for dashboard"""
        return {
            'status': 'operational',  # Would be determined by actual system health checks
            'uptime_seconds': int(time.time() - self.metrics.total_samples_processed),  # Placeholder
            'memory_usage_mb': 0,  # Would be actual memory usage
            'cpu_usage_percent': 0,  # Would be actual CPU usage
            'queue_size': 0,  # Would be actual queue size
            'active_connections': len(self.subscribers)
        }
    
    def _notify_subscribers(self, event_type: str, data: Any):
        """Notify all subscribers of an event"""
        for callback in self.subscribers:
            try:
                callback(event_type, data)
            except Exception as e:
                self.logger.error(f"Subscriber notification failed: {e}")
    
    def _start_metrics_updater(self):
        """Start background thread for periodic metrics updates"""
        def update_metrics():
            while True:
                try:
                    self._update_periodic_metrics()
                    time.sleep(self.metrics_update_interval)
                except Exception as e:
                    self.logger.error(f"Metrics update failed: {e}")
                    time.sleep(self.metrics_update_interval)
        
        metrics_thread = threading.Thread(target=update_metrics, daemon=True)
        metrics_thread.start()
    
    def _update_periodic_metrics(self):
        """Update periodic metrics calculations"""
        with self.lock:
            # Calculate detection rate
            total = self.metrics.total_samples_processed
            if total > 0:
                self.metrics.detection_rate = self.metrics.attacks_detected / total
            
            # Calculate samples per minute (rolling window)
            current_time = datetime.now()
            window_start = current_time - timedelta(minutes=self.metrics_window_minutes)
            
            recent_samples = [
                entry for entry in self.detection_history
                if datetime.fromisoformat(entry['timestamp']) > window_start
            ]
            
            self.metrics.samples_per_minute = len(recent_samples) / self.metrics_window_minutes
            
            # Update processing time average
            if recent_samples:
                processing_times = [entry['processing_time_ms'] for entry in recent_samples]
                self.metrics.avg_processing_time_ms = sum(processing_times) / len(processing_times)
            
            # Store metrics snapshot
            metrics_snapshot = asdict(self.metrics)
            metrics_snapshot['timestamp'] = current_time.isoformat()
            self.metrics_history.append(metrics_snapshot)
            
            self.last_metrics_update = current_time