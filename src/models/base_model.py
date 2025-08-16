from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import joblib
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import logging


class BaseAnomalyDetector(ABC):
    """
    Abstract base class for anomaly detection models.
    Provides a consistent interface for training, prediction, and evaluation.
    """
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.classes_ = None
        self.logger = logging.getLogger(__name__)
        
        # Model-specific parameters
        self.model_params = kwargs
    
    @abstractmethod
    def _build_model(self) -> Any:
        """Build the underlying model. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model. Must be implemented by subclasses."""
        pass
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'BaseAnomalyDetector':
        """
        Fit the model to training data.
        
        Args:
            X: Feature matrix
            y: Target vector
            **kwargs: Additional fitting parameters
            
        Returns:
            Self for method chaining
        """
        self.logger.info(f"Training {self.name} model with {X.shape[0]} samples, {X.shape[1]} features")
        
        # Store feature names and classes
        self.feature_names = list(X.columns)
        self.classes_ = np.unique(y)
        
        # Build model if not already built
        if self.model is None:
            self.model = self._build_model()
        
        # Fit the model
        self._fit_model(X, y, **kwargs)
        
        self.is_fitted = True
        self.logger.info(f"{self.name} model training completed")
        
        return self
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make binary predictions. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities. Must be implemented by subclasses."""
        pass
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Dictionary of performance metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Get predictions
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='binary'),
            'recall': recall_score(y, y_pred, average='binary'),
            'f1_score': f1_score(y, y_pred, average='binary'),
            'precision_macro': precision_score(y, y_pred, average='macro'),
            'recall_macro': recall_score(y, y_pred, average='macro'),
            'f1_macro': f1_score(y, y_pred, average='macro'),
        }
        
        # Add AUC if probabilities are available
        if y_proba is not None and len(self.classes_) == 2:
            metrics['auc_roc'] = roc_auc_score(y, y_proba[:, 1])
        
        return metrics
    
    def evaluate_detailed(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Detailed evaluation including confusion matrix and classification report.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Dictionary with detailed evaluation results
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        # Basic metrics
        metrics = self.evaluate(X, y)
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Classification report
        class_report = classification_report(y, y_pred, output_dict=True)
        
        # Attack detection specific metrics
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        detailed_metrics = {
            **metrics,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'attack_detection_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,  # Same as recall
            'normal_classification_rate': tn / (tn + fp) if (tn + fp) > 0 else 0  # Same as specificity
        }
        
        return detailed_metrics
    
    def save(self, filepath: str) -> None:
        """
        Save the fitted model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_data = {
            'model': self.model,
            'name': self.name,
            'feature_names': self.feature_names,
            'classes_': self.classes_,
            'is_fitted': self.is_fitted,
            'model_params': self.model_params
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseAnomalyDetector':
        """
        Load a fitted model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded model instance
        """
        model_data = joblib.load(filepath)
        
        # Create instance with saved parameters
        instance = cls(
            name=model_data['name'],
            **model_data['model_params']
        )
        
        # Restore model state
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.classes_ = model_data['classes_']
        instance.is_fitted = model_data['is_fitted']
        
        return instance
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores if available.
        
        Returns:
            Dictionary mapping feature names to importance scores, or None
        """
        if not self.is_fitted or self.feature_names is None:
            return None
        
        # Try to get feature importance from the model
        importance = None
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models, use absolute coefficients
            importance = np.abs(self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_)
        
        if importance is not None:
            return dict(zip(self.feature_names, importance))
        
        return None
    
    def predict_with_confidence(self, X: pd.DataFrame, confidence_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence scores.
        
        Args:
            X: Feature matrix
            confidence_threshold: Threshold for high confidence predictions
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        y_proba = self.predict_proba(X)
        y_pred = self.predict(X)
        
        # Calculate confidence as max probability
        confidence_scores = np.max(y_proba, axis=1)
        
        return y_pred, confidence_scores
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'name': self.name,
            'is_fitted': self.is_fitted,
            'num_features': len(self.feature_names) if self.feature_names else None,
            'num_classes': len(self.classes_) if self.classes_ else None,
            'model_params': self.model_params,
            'model_type': type(self.model).__name__ if self.model else None
        }
    
    def __str__(self) -> str:
        """String representation of the model."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.name} ({status})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self.is_fitted})"