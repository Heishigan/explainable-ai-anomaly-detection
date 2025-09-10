"""
Explanation aggregation and formatting utilities for cybersecurity anomaly detection.
Combines SHAP and LIME explanations, provides consensus analysis, and formats for dashboard display.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from .shap_explainer import ShapExplanation, ShapExplainer
from .lime_explainer import LimeExplanation, LimeExplainer

@dataclass
class AggregatedExplanation:
    """Combined explanation from multiple explainability methods"""
    prediction: float
    prediction_proba: float
    timestamp: str
    sample_id: Optional[str] = None
    
    # Method-specific explanations
    shap_explanation: Optional[Dict[str, Any]] = None
    lime_explanation: Optional[Dict[str, Any]] = None
    
    # Consensus analysis
    consensus_features: Optional[List[Dict[str, Any]]] = None
    confidence_score: Optional[float] = None
    explanation_agreement: Optional[float] = None
    
    # Risk assessment
    risk_level: Optional[str] = None

class ExplanationAggregator:
    """Aggregates and analyzes explanations from multiple XAI methods"""
    
    def __init__(self, feature_names: List[str]):
        """
        Initialize explanation aggregator
        
        Args:
            feature_names: List of feature names in the dataset
        """
        self.feature_names = feature_names
        self.logger = logging.getLogger(__name__)
        
        
        # Risk level thresholds
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'critical': 0.95
        }
    
    
    def aggregate_explanations(self, shap_explanation: Optional[ShapExplanation] = None,
                             lime_explanation: Optional[LimeExplanation] = None,
                             sample_id: Optional[str] = None) -> AggregatedExplanation:
        """
        Aggregate explanations from different XAI methods
        
        Args:
            shap_explanation: SHAP explanation object
            lime_explanation: LIME explanation object
            sample_id: Unique identifier for the sample
            
        Returns:
            AggregatedExplanation object
        """
        if not shap_explanation and not lime_explanation:
            raise ValueError("At least one explanation method must be provided")
        
        # Get base prediction info
        if shap_explanation:
            prediction = shap_explanation.prediction
            prediction_proba = shap_explanation.prediction_proba
        else:
            prediction = lime_explanation.prediction
            prediction_proba = lime_explanation.prediction_proba
        
        # Format individual explanations
        shap_formatted = None
        if shap_explanation:
            shap_formatted = self._format_shap_for_aggregation(shap_explanation)
        
        lime_formatted = None
        if lime_explanation:
            lime_formatted = self._format_lime_for_aggregation(lime_explanation)
        
        # Perform consensus analysis
        consensus_features = self._analyze_consensus(shap_explanation, lime_explanation)
        agreement_score = self._calculate_agreement(shap_explanation, lime_explanation)
        
        # Calculate overall confidence
        confidence = self._calculate_overall_confidence(shap_explanation, lime_explanation)
        
        # Determine risk level
        risk_level = self._determine_risk_level(prediction_proba, confidence)
        
        return AggregatedExplanation(
            prediction=prediction,
            prediction_proba=prediction_proba,
            timestamp=datetime.now().isoformat(),
            sample_id=sample_id,
            shap_explanation=shap_formatted,
            lime_explanation=lime_formatted,
            consensus_features=consensus_features,
            confidence_score=confidence,
            explanation_agreement=agreement_score,
            risk_level=risk_level
        )
    
    def _format_shap_for_aggregation(self, explanation: ShapExplanation) -> Dict[str, Any]:
        """Format SHAP explanation for aggregation"""
        feature_importance = list(zip(
            self.feature_names,
            explanation.feature_values,
            explanation.shap_values
        ))
        
        # Sort by absolute importance
        feature_importance.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return {
            "method": "SHAP",
            "base_value": explanation.base_value,
            "confidence": explanation.confidence_score,
            "top_features": [
                {
                    "name": name,
                    "value": float(value),
                    "importance": float(shap_val),
                    "abs_importance": abs(float(shap_val))
                }
                for name, value, shap_val in feature_importance[:20]
            ]
        }
    
    def _format_lime_for_aggregation(self, explanation: LimeExplanation) -> Dict[str, Any]:
        """Format LIME explanation for aggregation"""
        return {
            "method": "LIME",
            "local_prediction": explanation.local_prediction,
            "model_score": explanation.score,
            "top_features": [
                {
                    "name": name,
                    "importance": float(importance),
                    "abs_importance": abs(float(importance))
                }
                for name, importance in explanation.feature_explanations[:20]
            ]
        }
    
    def _analyze_consensus(self, shap_exp: Optional[ShapExplanation], 
                         lime_exp: Optional[LimeExplanation]) -> List[Dict[str, Any]]:
        """Analyze consensus between different explanation methods"""
        if not shap_exp or not lime_exp:
            return []
        
        # Get feature importance from both methods
        shap_importance = dict(zip(self.feature_names, shap_exp.shap_values))
        lime_importance = dict(lime_exp.feature_explanations)
        
        # Find features explained by both methods
        common_features = set(shap_importance.keys()) & set(lime_importance.keys())
        
        consensus_features = []
        for feature in common_features:
            shap_imp = shap_importance[feature]
            lime_imp = lime_importance[feature]
            
            # Check if both methods agree on direction (positive/negative)
            agreement = (shap_imp > 0 and lime_imp > 0) or (shap_imp < 0 and lime_imp < 0)
            
            # Calculate consensus strength
            consensus_strength = min(abs(shap_imp), abs(lime_imp)) / max(abs(shap_imp), abs(lime_imp))
            if abs(shap_imp) < 1e-6 or abs(lime_imp) < 1e-6:
                consensus_strength = 0.0
            
            if consensus_strength > 0.1:  # Minimum threshold for inclusion
                consensus_features.append({
                    "feature": feature,
                    "shap_importance": float(shap_imp),
                    "lime_importance": float(lime_imp),
                    "agreement": agreement,
                    "consensus_strength": consensus_strength,
                    "avg_importance": (abs(shap_imp) + abs(lime_imp)) / 2
                })
        
        # Sort by average importance
        consensus_features.sort(key=lambda x: x["avg_importance"], reverse=True)
        return consensus_features[:15]
    
    def _calculate_agreement(self, shap_exp: Optional[ShapExplanation], 
                           lime_exp: Optional[LimeExplanation]) -> float:
        """Calculate overall agreement score between explanation methods"""
        if not shap_exp or not lime_exp:
            return 1.0  # Perfect agreement if only one method
        
        # Get top features from both methods
        shap_top = dict(list(zip(self.feature_names, shap_exp.shap_values))[:10])
        lime_top = dict(lime_exp.feature_explanations[:10])
        
        # Calculate correlation of importance scores for common features
        common_features = set(shap_top.keys()) & set(lime_top.keys())
        
        if len(common_features) < 2:
            return 0.5  # Neutral agreement if insufficient overlap
        
        shap_values = [shap_top[f] for f in common_features]
        lime_values = [lime_top[f] for f in common_features]
        
        # Calculate Pearson correlation
        if len(shap_values) > 1:
            correlation = np.corrcoef(shap_values, lime_values)[0, 1]
            if np.isnan(correlation):
                return 0.5
            return max(0.0, correlation)  # Convert to 0-1 scale
        
        return 0.5
    
    def _calculate_overall_confidence(self, shap_exp: Optional[ShapExplanation], 
                                    lime_exp: Optional[LimeExplanation]) -> float:
        """Calculate overall confidence combining different methods"""
        confidences = []
        
        if shap_exp and shap_exp.confidence_score:
            confidences.append(shap_exp.confidence_score)
        
        if lime_exp and lime_exp.prediction_proba:
            # Convert prediction probability to confidence
            lime_confidence = abs(lime_exp.prediction_proba - 0.5) * 2
            confidences.append(lime_confidence)
        
        if not confidences:
            return 0.5
        
        # Weighted average (give more weight if multiple methods agree)
        base_confidence = np.mean(confidences)
        
        # Boost confidence if methods agree
        if len(confidences) > 1:
            agreement = self._calculate_agreement(shap_exp, lime_exp)
            base_confidence = base_confidence * (0.5 + 0.5 * agreement)
        
        return min(1.0, base_confidence)
    
    
    def _determine_risk_level(self, prediction_proba: float, confidence: float) -> str:
        """Determine risk level based on prediction probability and confidence"""
        if not prediction_proba:
            return "unknown"
        
        # Adjust probability by confidence
        adjusted_prob = prediction_proba * confidence
        
        if adjusted_prob >= self.risk_thresholds['critical']:
            return "critical"
        elif adjusted_prob >= self.risk_thresholds['high']:
            return "high"
        elif adjusted_prob >= self.risk_thresholds['medium']:
            return "medium"
        elif adjusted_prob >= self.risk_thresholds['low']:
            return "low"
        else:
            return "minimal"
    
    def format_for_dashboard(self, aggregated_explanation: AggregatedExplanation) -> Dict[str, Any]:
        """Format aggregated explanation for dashboard display"""
        is_attack = float(aggregated_explanation.prediction) >= 0.5
        
        return {
            "type": "attack" if is_attack else "normal",  # Add type field for frontend
            "prediction_summary": {
                "label": "Attack" if is_attack else "Normal",
                "probability": aggregated_explanation.prediction_proba,
                "confidence": aggregated_explanation.confidence_score,
                "risk_level": aggregated_explanation.risk_level,
                "timestamp": aggregated_explanation.timestamp
            },
            "explanation_methods": {
                "shap_available": aggregated_explanation.shap_explanation is not None,
                "lime_available": aggregated_explanation.lime_explanation is not None,
                "methods_agreement": aggregated_explanation.explanation_agreement
            },
            "consensus_analysis": {
                "top_consensus_features": aggregated_explanation.consensus_features[:10] if aggregated_explanation.consensus_features else [],
                "explanation_strength": len(aggregated_explanation.consensus_features or [])
            },
            "individual_explanations": {
                "shap": aggregated_explanation.shap_explanation,
                "lime": aggregated_explanation.lime_explanation
            },
            "risk_assessment": {
                "level": aggregated_explanation.risk_level,
                "description": self._get_risk_description(aggregated_explanation.risk_level)
            }
        }
    
    def _get_risk_description(self, risk_level: str) -> str:
        """Get human-readable risk level description"""
        descriptions = {
            "minimal": "Low threat - Normal network behavior with minimal anomaly indicators",
            "low": "Low risk - Some unusual patterns but likely benign activity",
            "medium": "Medium risk - Suspicious activity that requires monitoring",
            "high": "High risk - Strong indicators of malicious activity, immediate attention needed",
            "critical": "Critical threat - High confidence attack detection, immediate response required",
            "unknown": "Risk level could not be determined"
        }
        return descriptions.get(risk_level, "Unknown risk level")
    
    def export_explanation(self, aggregated_explanation: AggregatedExplanation, 
                         format_type: str = "json") -> Union[str, Dict[str, Any]]:
        """Export explanation in various formats"""
        data = asdict(aggregated_explanation)
        
        if format_type == "json":
            return json.dumps(data, indent=2, default=str)
        elif format_type == "dict":
            return data
        else:
            raise ValueError(f"Unsupported format type: {format_type}")