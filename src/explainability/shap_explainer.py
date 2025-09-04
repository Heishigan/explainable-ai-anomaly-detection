"""
SHAP-based explainability module for cybersecurity anomaly detection.
Provides model-agnostic explanations for network intrusion predictions.
"""

import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import json

@dataclass
class ShapExplanation:
    """Structure for SHAP explanation results"""
    prediction: float
    prediction_proba: float
    shap_values: np.ndarray
    feature_names: List[str]
    feature_values: np.ndarray
    base_value: float
    attack_category: Optional[str] = None
    confidence_score: Optional[float] = None

class ShapExplainer:
    """SHAP explainer for cybersecurity anomaly detection models"""
    
    def __init__(self, model, background_data: np.ndarray, feature_names: List[str]):
        """
        Initialize SHAP explainer
        
        Args:
            model: Trained ML model with predict_proba method
            background_data: Representative background dataset for SHAP
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self.explainer = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize explainer based on model type
        self._initialize_explainer()
        
        # Cybersecurity feature categories for grouping
        self.feature_categories = self._define_feature_categories()
    
    def _initialize_explainer(self):
        """Initialize appropriate SHAP explainer based on model type"""
        try:
            # Try TreeExplainer first (faster for tree-based models)
            if hasattr(self.model, 'feature_importances_'):
                self.explainer = shap.TreeExplainer(self.model)
                self.explainer_type = "tree"
            else:
                # Use KernelExplainer for other models
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba, 
                    self.background_data[:100]  # Sample for efficiency
                )
                self.explainer_type = "kernel"
                
        except Exception as e:
            self.logger.warning(f"Failed to initialize TreeExplainer: {e}")
            # Fallback to KernelExplainer
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, 
                self.background_data[:100]
            )
            self.explainer_type = "kernel"
    
    def _define_feature_categories(self) -> Dict[str, List[str]]:
        """Define cybersecurity-specific feature categories"""
        return {
            "network_flow": [
                "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss",
                "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb"
            ],
            "protocol_info": [
                "proto", "service", "state"
            ],
            "timing_patterns": [
                "sjit", "djit", "sintpkt", "dintpkt", "tcprtt", "synack", "ackdat"
            ],
            "behavioral_indicators": [
                "ct_state_ttl", "ct_flw_http_mthd", "ct_ftp_cmd", "ct_srv_src",
                "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
                "ct_dst_src_ltm"
            ],
            "binary_flags": [
                "is_ftp_login", "is_sm_ips_ports"
            ]
        }
    
    def explain_prediction(self, sample: np.ndarray, 
                         return_probabilities: bool = True) -> ShapExplanation:
        """
        Generate SHAP explanation for a single prediction
        
        Args:
            sample: Single sample to explain (1D array)
            return_probabilities: Whether to include prediction probabilities
            
        Returns:
            ShapExplanation object with explanation results
        """
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)
            
        try:
            # Get prediction
            prediction = self.model.predict(sample)[0]
            prediction_proba = None
            
            if return_probabilities and hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(sample)[0]
                prediction_proba = proba[1]  # Probability of attack class
            
            # Get SHAP values
            if self.explainer_type == "tree":
                shap_values = self.explainer.shap_values(sample)
                if isinstance(shap_values, list):
                    # For binary classification, take attack class explanations
                    shap_values = shap_values[1]
                base_value = self.explainer.expected_value
                if isinstance(base_value, np.ndarray):
                    base_value = base_value[1]
            else:
                shap_values = self.explainer.shap_values(sample)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                base_value = self.explainer.expected_value[1]
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(shap_values, prediction_proba)
            
            return ShapExplanation(
                prediction=float(prediction),
                prediction_proba=float(prediction_proba) if prediction_proba is not None else None,
                shap_values=shap_values.flatten(),
                feature_names=self.feature_names,
                feature_values=sample.flatten(),
                base_value=float(base_value),
                confidence_score=confidence_score
            )
            
        except Exception as e:
            self.logger.error(f"Error generating SHAP explanation: {e}")
            raise
    
    def explain_batch(self, samples: np.ndarray, 
                     max_samples: int = 100) -> List[ShapExplanation]:
        """
        Generate SHAP explanations for a batch of predictions
        
        Args:
            samples: Batch of samples to explain
            max_samples: Maximum number of samples to process (for efficiency)
            
        Returns:
            List of ShapExplanation objects
        """
        if len(samples) > max_samples:
            self.logger.warning(f"Limiting batch size to {max_samples} for efficiency")
            samples = samples[:max_samples]
        
        explanations = []
        for sample in samples:
            try:
                explanation = self.explain_prediction(sample)
                explanations.append(explanation)
            except Exception as e:
                self.logger.error(f"Failed to explain sample: {e}")
                continue
        
        return explanations
    
    def get_feature_importance_summary(self, explanations: List[ShapExplanation]) -> Dict[str, float]:
        """
        Aggregate feature importance across multiple explanations
        
        Args:
            explanations: List of SHAP explanations
            
        Returns:
            Dictionary mapping feature names to average absolute SHAP values
        """
        if not explanations:
            return {}
        
        # Aggregate SHAP values
        all_shap_values = np.array([exp.shap_values for exp in explanations])
        mean_abs_shap = np.mean(np.abs(all_shap_values), axis=0)
        
        return dict(zip(self.feature_names, mean_abs_shap))
    
    def get_category_importance(self, explanation: ShapExplanation) -> Dict[str, float]:
        """
        Group feature importance by cybersecurity categories
        
        Args:
            explanation: Single SHAP explanation
            
        Returns:
            Dictionary mapping categories to aggregated importance scores
        """
        category_importance = {}
        
        for category, features in self.feature_categories.items():
            category_shap = 0.0
            category_count = 0
            
            for feature in features:
                if feature in self.feature_names:
                    idx = self.feature_names.index(feature)
                    category_shap += abs(explanation.shap_values[idx])
                    category_count += 1
            
            if category_count > 0:
                category_importance[category] = category_shap / category_count
            else:
                category_importance[category] = 0.0
        
        return category_importance
    
    def _calculate_confidence_score(self, shap_values: np.ndarray, 
                                  prediction_proba: Optional[float]) -> float:
        """
        Calculate confidence score based on SHAP values and prediction probability
        
        Args:
            shap_values: SHAP values for the prediction
            prediction_proba: Prediction probability
            
        Returns:
            Confidence score between 0 and 1
        """
        if prediction_proba is None:
            # Use SHAP magnitude as confidence proxy
            shap_magnitude = np.sum(np.abs(shap_values))
            return min(1.0, shap_magnitude / 10.0)  # Normalize
        
        # Combine probability confidence with explanation strength
        prob_confidence = abs(prediction_proba - 0.5) * 2  # Distance from uncertainty
        shap_magnitude = np.sum(np.abs(shap_values))
        
        # Weighted combination
        return min(1.0, 0.7 * prob_confidence + 0.3 * min(1.0, shap_magnitude / 5.0))
    
    def format_explanation_for_dashboard(self, explanation: ShapExplanation, 
                                       top_n: int = 10) -> Dict[str, Any]:
        """
        Format explanation for dashboard display
        
        Args:
            explanation: SHAP explanation object
            top_n: Number of top features to include
            
        Returns:
            Dictionary formatted for dashboard consumption
        """
        # Get top contributing features
        feature_contributions = list(zip(
            self.feature_names, 
            explanation.feature_values,
            explanation.shap_values
        ))
        
        # Sort by absolute SHAP value
        feature_contributions.sort(key=lambda x: abs(x[2]), reverse=True)
        top_features = feature_contributions[:top_n]
        
        # Get category importance
        category_importance = self.get_category_importance(explanation)
        
        return {
            "prediction": {
                "label": "Attack" if explanation.prediction == 1 else "Normal",
                "probability": explanation.prediction_proba,
                "confidence": explanation.confidence_score
            },
            "top_features": [
                {
                    "name": name,
                    "value": float(value),
                    "importance": float(shap_val),
                    "impact": "positive" if shap_val > 0 else "negative"
                }
                for name, value, shap_val in top_features
            ],
            "category_importance": {
                category: float(importance) 
                for category, importance in category_importance.items()
            },
            "base_value": explanation.base_value,
            "explanation_metadata": {
                "explainer_type": self.explainer_type,
                "total_features": len(self.feature_names)
            }
        }