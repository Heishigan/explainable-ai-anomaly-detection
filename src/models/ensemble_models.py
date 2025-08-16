import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import xgboost as xgb
from .base_model import BaseAnomalyDetector


class RandomForestDetector(BaseAnomalyDetector):
    """Random Forest anomaly detector."""
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 random_state: int = 42,
                 **kwargs):
        super().__init__(
            name="RandomForest",
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            **kwargs
        )
    
    def _build_model(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=self.model_params['n_estimators'],
            max_depth=self.model_params['max_depth'],
            min_samples_split=self.model_params['min_samples_split'],
            min_samples_leaf=self.model_params['min_samples_leaf'],
            random_state=self.model_params['random_state'],
            n_jobs=-1,
            class_weight='balanced'
        )
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)


class XGBoostDetector(BaseAnomalyDetector):
    """XGBoost anomaly detector."""
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 random_state: int = 42,
                 **kwargs):
        super().__init__(
            name="XGBoost",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            **kwargs
        )
    
    def _build_model(self) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(
            n_estimators=self.model_params['n_estimators'],
            max_depth=self.model_params['max_depth'],
            learning_rate=self.model_params['learning_rate'],
            subsample=self.model_params['subsample'],
            colsample_bytree=self.model_params['colsample_bytree'],
            random_state=self.model_params['random_state'],
            n_jobs=-1,
            eval_metric='logloss',
            use_label_encoder=False
        )
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        # Handle class imbalance
        scale_pos_weight = (y == 0).sum() / (y == 1).sum()
        self.model.set_params(scale_pos_weight=scale_pos_weight)
        
        self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)


class LogisticRegressionDetector(BaseAnomalyDetector):
    """Logistic Regression anomaly detector."""
    
    def __init__(self,
                 C: float = 1.0,
                 penalty: str = 'l2',
                 max_iter: int = 1000,
                 random_state: int = 42,
                 **kwargs):
        super().__init__(
            name="LogisticRegression",
            C=C,
            penalty=penalty,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs
        )
    
    def _build_model(self) -> LogisticRegression:
        return LogisticRegression(
            C=self.model_params['C'],
            penalty=self.model_params['penalty'],
            max_iter=self.model_params['max_iter'],
            random_state=self.model_params['random_state'],
            class_weight='balanced',
            n_jobs=-1
        )
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)


class MLPDetector(BaseAnomalyDetector):
    """Multi-layer Perceptron (Neural Network) anomaly detector."""
    
    def __init__(self,
                 hidden_layer_sizes: tuple = (100, 50),
                 activation: str = 'relu',
                 alpha: float = 0.0001,
                 learning_rate_init: float = 0.001,
                 max_iter: int = 500,
                 random_state: int = 42,
                 **kwargs):
        super().__init__(
            name="MLP",
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs
        )
    
    def _build_model(self) -> MLPClassifier:
        return MLPClassifier(
            hidden_layer_sizes=self.model_params['hidden_layer_sizes'],
            activation=self.model_params['activation'],
            alpha=self.model_params['alpha'],
            learning_rate_init=self.model_params['learning_rate_init'],
            max_iter=self.model_params['max_iter'],
            random_state=self.model_params['random_state'],
            early_stopping=True,
            validation_fraction=0.1
        )
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)


class SVMDetector(BaseAnomalyDetector):
    """Support Vector Machine anomaly detector."""
    
    def __init__(self,
                 C: float = 1.0,
                 kernel: str = 'rbf',
                 gamma: str = 'scale',
                 probability: bool = True,
                 random_state: int = 42,
                 **kwargs):
        super().__init__(
            name="SVM",
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=probability,
            random_state=random_state,
            **kwargs
        )
    
    def _build_model(self) -> SVC:
        return SVC(
            C=self.model_params['C'],
            kernel=self.model_params['kernel'],
            gamma=self.model_params['gamma'],
            probability=self.model_params['probability'],
            random_state=self.model_params['random_state'],
            class_weight='balanced'
        )
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        # For large datasets, sample for SVM training
        if X.shape[0] > 10000:
            self.logger.warning(f"Large dataset ({X.shape[0]} samples). Sampling 10000 for SVM training.")
            sample_idx = np.random.choice(X.index, size=10000, replace=False)
            X_sample = X.loc[sample_idx]
            y_sample = y.loc[sample_idx]
            self.model.fit(X_sample, y_sample)
        else:
            self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)


class GradientBoostingDetector(BaseAnomalyDetector):
    """Gradient Boosting anomaly detector."""
    
    def __init__(self,
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 max_depth: int = 3,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 random_state: int = 42,
                 **kwargs):
        super().__init__(
            name="GradientBoosting",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            **kwargs
        )
    
    def _build_model(self) -> GradientBoostingClassifier:
        return GradientBoostingClassifier(
            n_estimators=self.model_params['n_estimators'],
            learning_rate=self.model_params['learning_rate'],
            max_depth=self.model_params['max_depth'],
            min_samples_split=self.model_params['min_samples_split'],
            min_samples_leaf=self.model_params['min_samples_leaf'],
            random_state=self.model_params['random_state']
        )
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)