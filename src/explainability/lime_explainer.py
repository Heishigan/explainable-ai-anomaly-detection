"""
LIME-based explainability module for cybersecurity anomaly detection.
Provides local interpretable explanations for individual network intrusion predictions.
"""

import numpy as np
import pandas as pd
from lime import lime_tabular
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import json

@dataclass
class LimeExplanation:
    """Structure for LIME explanation results"""
    prediction: float
    prediction_proba: float
    feature_explanations: List[Tuple[str, float]]
    feature_values: np.ndarray
    local_prediction: float
    score: float
    attack_category: Optional[str] = None
    intercept: Optional[float] = None

class LimeExplainer:
    """LIME explainer for cybersecurity anomaly detection models"""
    
    def __init__(self, training_data: np.ndarray, feature_names: List[str], 
                 categorical_features: Optional[List[int]] = None):
        """
        Initialize LIME explainer
        
        Args:
            training_data: Training dataset for LIME to understand feature distributions
            feature_names: List of feature names
            categorical_features: Indices of categorical features
        """
        self.training_data = training_data
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []
        self.logger = logging.getLogger(__name__)
        
        # Initialize LIME tabular explainer
        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data,
            feature_names=feature_names,
            categorical_features=self.categorical_features,
            class_names=['Normal', 'Attack'],
            mode='classification',
            discretize_continuous=True,
            random_state=42
        )
        
        # Cybersecurity-specific feature insights
        self.feature_insights = self._define_feature_insights()
    
    def _define_feature_insights(self) -> Dict[str, str]:
        """Define human-readable insights for cybersecurity features"""
        return {
            "dur": "Connection duration - longer durations may indicate data exfiltration",
            "sbytes": "Source bytes - large outbound data could indicate data theft",
            "dbytes": "Destination bytes - large inbound data might be attack payloads",
            "sttl": "Source TTL - abnormal TTL values may indicate spoofed packets",
            "dttl": "Destination TTL - can reveal OS fingerprinting attempts",
            "sload": "Source load - high load might indicate DoS attack source",
            "dload": "Destination load - high load could indicate target under attack",
            "spkts": "Source packets - packet count anomalies suggest scanning",
            "dpkts": "Destination packets - response patterns reveal attack success",
            "swin": "Source TCP window - unusual window sizes indicate evasion",
            "dwin": "Destination TCP window - can reveal target system behavior",
            "proto": "Protocol type - unusual protocols may bypass security controls",
            "service": "Network service - targeting specific services indicates intent",
            "state": "Connection state - incomplete connections suggest scanning",
            "sjit": "Source jitter - timing variations may indicate evasion techniques",
            "djit": "Destination jitter - response timing can reveal system stress",
            "ct_state_ttl": "State-TTL connection count - persistence patterns",
            "ct_srv_src": "Service-source connections - service targeting behavior",
            "ct_srv_dst": "Service-destination connections - attack distribution",
            "is_ftp_login": "FTP login indicator - credential attack attempts",
            "is_sm_ips_ports": "Small IPs/ports indicator - scanning behavior"
        }
    
    def explain_prediction(self, model, sample: np.ndarray, 
                         num_features: int = 20, 
                         num_samples: int = 5000) -> LimeExplanation:
        """
        Generate LIME explanation for a single prediction
        
        Args:
            model: Trained ML model
            sample: Single sample to explain (1D array)
            num_features: Number of features to include in explanation
            num_samples: Number of samples for LIME's local model
            
        Returns:
            LimeExplanation object with explanation results
        """
        if sample.ndim > 1:
            sample = sample.flatten()
        
        try:
            # Get original prediction
            prediction = model.predict([sample])[0]
            prediction_proba = None
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba([sample])[0]
                prediction_proba = proba[1]  # Attack probability
            
            # Generate LIME explanation
            explanation = self.explainer.explain_instance(
                sample, 
                model.predict_proba,
                num_features=num_features,
                num_samples=num_samples,
                labels=[1]  # Explain attack class
            )
            
            # Extract explanation details
            lime_exp = explanation.as_list(label=1)
            local_exp = explanation.local_exp[1] if 1 in explanation.local_exp else []
            
            # Get feature explanations with proper names
            feature_explanations = []
            for feature_desc, importance in lime_exp:
                # Parse feature description to get feature name
                feature_name = self._parse_feature_description(feature_desc)
                feature_explanations.append((feature_name, importance))
            
            # Handle local prediction safely
            local_pred = prediction_proba
            if hasattr(explanation, 'local_pred') and explanation.local_pred:
                if len(explanation.local_pred) > 1:
                    local_pred = explanation.local_pred[1]
                elif len(explanation.local_pred) == 1:
                    local_pred = explanation.local_pred[0]
            
            # Handle intercept safely
            intercept = None
            if hasattr(explanation, 'intercept') and explanation.intercept:
                if isinstance(explanation.intercept, dict):
                    if 1 in explanation.intercept:
                        intercept = explanation.intercept[1]
                    elif 0 in explanation.intercept:
                        intercept = explanation.intercept[0]
                elif hasattr(explanation.intercept, '__len__'):
                    if len(explanation.intercept) > 1:
                        intercept = explanation.intercept[1]
                    elif len(explanation.intercept) == 1:
                        intercept = explanation.intercept[0]
            
            return LimeExplanation(
                prediction=float(prediction),
                prediction_proba=float(prediction_proba) if prediction_proba is not None else None,
                feature_explanations=feature_explanations,
                feature_values=sample,
                local_prediction=float(local_pred) if local_pred is not None else None,
                score=explanation.score if hasattr(explanation, 'score') else 0.0,
                intercept=float(intercept) if intercept is not None else None
            )
            
        except Exception as e:
            self.logger.error(f"Error generating LIME explanation: {e}")
            raise
    
    def _parse_feature_description(self, feature_desc: str) -> str:
        """
        Parse LIME's feature description to extract feature name
        
        Args:
            feature_desc: LIME's feature description (e.g., "feature_name <= 5.0")
            
        Returns:
            Clean feature name
        """
        # LIME descriptions often contain conditions like "feature <= value"
        # Extract just the feature name
        for name in self.feature_names:
            if name in feature_desc:
                return name
        
        # Fallback: try to extract from the beginning of the description
        parts = feature_desc.split()
        if parts:
            return parts[0]
        
        return feature_desc
    
    def explain_batch(self, model, samples: np.ndarray, 
                     max_samples: int = 50) -> List[LimeExplanation]:
        """
        Generate LIME explanations for a batch of predictions
        
        Args:
            model: Trained ML model
            samples: Batch of samples to explain
            max_samples: Maximum samples to process (LIME is computationally expensive)
            
        Returns:
            List of LimeExplanation objects
        """
        if len(samples) > max_samples:
            self.logger.warning(f"Limiting batch size to {max_samples} for LIME efficiency")
            samples = samples[:max_samples]
        
        explanations = []
        for i, sample in enumerate(samples):
            try:
                self.logger.debug(f"Generating LIME explanation {i+1}/{len(samples)}")
                explanation = self.explain_prediction(model, sample)
                explanations.append(explanation)
            except Exception as e:
                self.logger.error(f"Failed to explain sample {i}: {e}")
                continue
        
        return explanations
    
    def get_feature_stability_analysis(self, model, sample: np.ndarray, 
                                     num_trials: int = 10) -> Dict[str, float]:
        """
        Analyze stability of LIME explanations across multiple runs
        
        Args:
            model: Trained ML model
            sample: Sample to analyze
            num_trials: Number of explanation trials
            
        Returns:
            Dictionary with stability metrics for each feature
        """
        feature_importances = {name: [] for name in self.feature_names}
        
        for trial in range(num_trials):
            try:
                explanation = self.explain_prediction(model, sample)
                
                # Initialize all features to 0
                trial_importances = {name: 0.0 for name in self.feature_names}
                
                # Update with explained features
                for feature_name, importance in explanation.feature_explanations:
                    if feature_name in trial_importances:
                        trial_importances[feature_name] = importance
                
                # Store results
                for name, importance in trial_importances.items():
                    feature_importances[name].append(importance)
                    
            except Exception as e:
                self.logger.error(f"Trial {trial} failed: {e}")
                continue
        
        # Calculate stability metrics
        stability_metrics = {}
        for feature, importances in feature_importances.items():
            if importances:
                stability_metrics[feature] = {
                    'mean': np.mean(importances),
                    'std': np.std(importances),
                    'stability_score': 1.0 / (1.0 + np.std(importances))  # Higher = more stable
                }
            else:
                stability_metrics[feature] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'stability_score': 0.0
                }
        
        return stability_metrics
    
    def get_human_readable_explanation(self, explanation: LimeExplanation, 
                                     top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Convert LIME explanation to human-readable format with cybersecurity context
        
        Args:
            explanation: LIME explanation object
            top_n: Number of top features to explain
            
        Returns:
            List of human-readable feature explanations
        """
        # Sort features by importance
        sorted_features = sorted(
            explanation.feature_explanations, 
            key=lambda x: abs(x[1]), 
            reverse=True
        )[:top_n]
        
        readable_explanations = []
        
        for feature_name, importance in sorted_features:
            # Get feature value
            if feature_name in self.feature_names:
                feature_idx = self.feature_names.index(feature_name)
                feature_value = explanation.feature_values[feature_idx]
            else:
                feature_value = "Unknown"
            
            # Get human-readable insight
            insight = self.feature_insights.get(feature_name, f"Feature {feature_name}")
            
            # Determine impact direction
            impact = "increases" if importance > 0 else "decreases"
            
            readable_explanations.append({
                "feature": feature_name,
                "value": float(feature_value) if isinstance(feature_value, (int, float)) else str(feature_value),
                "importance": float(importance),
                "impact": impact,
                "explanation": f"{insight}. Current value {feature_value} {impact} attack probability by {abs(importance):.3f}",
                "insight": insight
            })
        
        return readable_explanations
    
    def format_explanation_for_dashboard(self, explanation: LimeExplanation, 
                                       top_n: int = 10) -> Dict[str, Any]:
        """
        Format LIME explanation for dashboard display
        
        Args:
            explanation: LIME explanation object
            top_n: Number of top features to include
            
        Returns:
            Dictionary formatted for dashboard consumption
        """
        human_readable = self.get_human_readable_explanation(explanation, top_n)
        
        return {
            "prediction": {
                "label": "Attack" if explanation.prediction == 1 else "Normal",
                "probability": explanation.prediction_proba,
                "local_prediction": explanation.local_prediction,
                "confidence": abs(explanation.prediction_proba - 0.5) * 2 if explanation.prediction_proba else None
            },
            "explanation": {
                "method": "LIME",
                "model_score": explanation.score,
                "intercept": explanation.intercept
            },
            "feature_explanations": human_readable,
            "summary": {
                "top_positive_features": [
                    item for item in human_readable[:5] 
                    if item["importance"] > 0
                ],
                "top_negative_features": [
                    item for item in human_readable[:5] 
                    if item["importance"] < 0
                ],
                "explanation_strength": sum(abs(item["importance"]) for item in human_readable)
            }
        }