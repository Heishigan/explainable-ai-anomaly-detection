"""
Explainable AI module for cybersecurity anomaly detection.

This module provides comprehensive explainability capabilities for network intrusion
detection models, including:

- SHAP (SHapley Additive exPlanations) for model-agnostic explanations
- LIME (Local Interpretable Model-agnostic Explanations) for local explanations
- Explanation aggregation and consensus analysis
- Realtime explanation pipeline for production deployment
- Dashboard interface for visualization and monitoring

Usage:
    from src.explainability import (
        ShapExplainer, LimeExplainer, ExplanationAggregator,
        RealtimeExplainer, DashboardDataManager
    )
"""

from .shap_explainer import ShapExplainer, ShapExplanation
from .lime_explainer import LimeExplainer, LimeExplanation
from .explanation_aggregator import ExplanationAggregator, AggregatedExplanation
from .realtime_explainer import (
    RealtimeExplainer, 
    RealtimeExplanationRequest, 
    RealtimeExplanationResult
)
from .dashboard_interface import (
    DashboardDataManager, 
    DashboardAlert, 
    DashboardMetrics, 
    DashboardDetection,
    AlertLevel
)

__all__ = [
    # SHAP explainer
    'ShapExplainer',
    'ShapExplanation',
    
    # LIME explainer
    'LimeExplainer', 
    'LimeExplanation',
    
    # Explanation aggregation
    'ExplanationAggregator',
    'AggregatedExplanation',
    
    # Realtime processing
    'RealtimeExplainer',
    'RealtimeExplanationRequest',
    'RealtimeExplanationResult',
    
    # Dashboard interface
    'DashboardDataManager',
    'DashboardAlert',
    'DashboardMetrics', 
    'DashboardDetection',
    'AlertLevel'
]

# Version information
__version__ = '1.0.0'
__author__ = 'Explainable AI Team'
__description__ = 'Explainable AI for Cybersecurity Anomaly Detection'