import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
import logging

from .base_model import BaseAnomalyDetector
from .ensemble_models import (
    RandomForestDetector, XGBoostDetector, LogisticRegressionDetector,
    MLPDetector, SVMDetector, GradientBoostingDetector
)


class ModelTrainer:
    """
    Comprehensive model training and evaluation framework for anomaly detection.
    """
    
    def __init__(self, results_dir: str = "results/models"):
        self.results_dir = results_dir
        self.models = {}
        self.results = {}
        self.logger = logging.getLogger(__name__)
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
    
    def get_default_models(self) -> Dict[str, BaseAnomalyDetector]:
        """
        Get a dictionary of default model configurations.
        
        Returns:
            Dictionary mapping model names to model instances
        """
        return {
            'random_forest': RandomForestDetector(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'xgboost': XGBoostDetector(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'logistic_regression': LogisticRegressionDetector(
                C=1.0,
                penalty='l2',
                max_iter=1000,
                random_state=42
            ),
            'mlp': MLPDetector(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=500,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingDetector(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        }
    
    def add_model(self, name: str, model: BaseAnomalyDetector) -> None:
        """Add a model to the trainer."""
        self.models[name] = model
        self.logger.info(f"Added model: {name}")
    
    def train_single_model(self, 
                          model_name: str,
                          X_train: pd.DataFrame,
                          y_train: pd.Series,
                          X_test: pd.DataFrame,
                          y_test: pd.Series,
                          cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train and evaluate a single model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            X_test: Test features  
            y_test: Test labels
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with training results
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        self.logger.info(f"Training {model_name}...")
        
        # Record training start time
        start_time = datetime.now()
        
        # Fit the model
        model.fit(X_train, y_train)
        
        # Calculate training time
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate on training set
        try:
            train_metrics = model.evaluate_detailed(X_train, y_train)
        except Exception as e:
            self.logger.error(f"Error evaluating {model_name} on training set: {str(e)}")
            train_metrics = {'error': str(e)}
        
        # Evaluate on test set
        try:
            test_metrics = model.evaluate_detailed(X_test, y_test)
        except Exception as e:
            self.logger.error(f"Error evaluating {model_name} on test set: {str(e)}")
            test_metrics = {'error': str(e)}
        
        # Cross-validation (skip if cv_folds <= 1)
        if cv_folds > 1:
            cv_scores = self._cross_validate(model, X_train, y_train, cv_folds)
        else:
            cv_scores = {'scores': {}, 'summary': {}}
        
        # Feature importance if available
        feature_importance = model.get_feature_importance()
        
        # Compile results
        results = {
            'model_name': model_name,
            'training_time': training_time,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_scores': cv_scores,
            'feature_importance': feature_importance,
            'model_info': model.get_model_info()
        }
        
        # Store results
        self.results[model_name] = results
        
        self.logger.info(f"Completed training {model_name}. Test F1: {test_metrics['f1_score']:.3f}")
        
        return results
    
    def train_all_models(self,
                        X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_test: pd.DataFrame,
                        y_test: pd.Series,
                        models_to_train: Optional[List[str]] = None,
                        cv_folds: int = 5) -> Dict[str, Dict]:
        """
        Train and evaluate multiple models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            models_to_train: List of model names to train (None for all)
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with all training results
        """
        # Use all models if none specified
        if models_to_train is None:
            models_to_train = list(self.models.keys())
        
        self.logger.info(f"Training {len(models_to_train)} models: {models_to_train}")
        
        # Train each model
        all_results = {}
        for model_name in models_to_train:
            try:
                results = self.train_single_model(
                    model_name, X_train, y_train, X_test, y_test, cv_folds
                )
                all_results[model_name] = results
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        # Save results
        self._save_results(all_results)
        
        return all_results
    
    def _cross_validate(self, 
                       model: BaseAnomalyDetector,
                       X: pd.DataFrame,
                       y: pd.Series,
                       cv_folds: int) -> Dict[str, Any]:
        """Perform cross-validation evaluation."""
        try:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            # Get the underlying sklearn model for cross-validation
            sklearn_model = model.model
            
            cv_scores = {
                'accuracy': cross_val_score(sklearn_model, X, y, cv=cv, scoring='accuracy'),
                'precision': cross_val_score(sklearn_model, X, y, cv=cv, scoring='precision'),
                'recall': cross_val_score(sklearn_model, X, y, cv=cv, scoring='recall'),
                'f1': cross_val_score(sklearn_model, X, y, cv=cv, scoring='f1'),
                'roc_auc': cross_val_score(sklearn_model, X, y, cv=cv, scoring='roc_auc')
            }
            
            # Calculate mean and std for each metric
            cv_summary = {}
            for metric, scores in cv_scores.items():
                cv_summary[f'{metric}_mean'] = np.mean(scores)
                cv_summary[f'{metric}_std'] = np.std(scores)
            
            return {
                'scores': cv_scores,
                'summary': cv_summary
            }
            
        except Exception as e:
            self.logger.warning(f"Cross-validation failed: {str(e)}")
            return {'scores': {}, 'summary': {}}
    
    def compare_models(self, 
                      metric: str = 'f1_score',
                      show_plot: bool = True) -> pd.DataFrame:
        """
        Compare model performance across different metrics.
        
        Args:
            metric: Primary metric for comparison
            show_plot: Whether to show comparison plot
            
        Returns:
            DataFrame with model comparison
        """
        if not self.results:
            raise ValueError("No training results available. Train models first.")
        
        # Compile comparison data
        comparison_data = []
        for model_name, results in self.results.items():
            test_metrics = results['test_metrics']
            cv_metrics = results.get('cv_scores', {}).get('summary', {})
            
            row = {
                'model': model_name,
                'test_accuracy': test_metrics.get('accuracy', 0),
                'test_precision': test_metrics.get('precision', 0),
                'test_recall': test_metrics.get('recall', 0),
                'test_f1_score': test_metrics.get('f1_score', 0),
                'test_auc_roc': test_metrics.get('auc_roc', 0),
                'cv_f1_mean': cv_metrics.get('f1_mean', 0),
                'cv_f1_std': cv_metrics.get('f1_std', 0),
                'training_time': results.get('training_time', 0),
                'false_positive_rate': test_metrics.get('false_positive_rate', 0),
                'false_negative_rate': test_metrics.get('false_negative_rate', 0)
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values(f'test_{metric}', ascending=False)
        
        if show_plot:
            self._plot_model_comparison(comparison_df, metric)
        
        return comparison_df
    
    def _plot_model_comparison(self, comparison_df: pd.DataFrame, metric: str) -> None:
        """Plot model comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # F1 Score comparison
        axes[0, 0].bar(comparison_df['model'], comparison_df['test_f1_score'])
        axes[0, 0].set_title('Test F1 Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylabel('F1 Score')
        
        # AUC-ROC comparison
        axes[0, 1].bar(comparison_df['model'], comparison_df['test_auc_roc'])
        axes[0, 1].set_title('Test AUC-ROC')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].set_ylabel('AUC-ROC')
        
        # Precision vs Recall
        axes[1, 0].scatter(comparison_df['test_recall'], comparison_df['test_precision'])
        for i, model in enumerate(comparison_df['model']):
            axes[1, 0].annotate(model, 
                              (comparison_df['test_recall'].iloc[i], 
                               comparison_df['test_precision'].iloc[i]))
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision vs Recall')
        
        # Training time vs Performance
        axes[1, 1].scatter(comparison_df['training_time'], comparison_df[f'test_{metric}'])
        for i, model in enumerate(comparison_df['model']):
            axes[1, 1].annotate(model,
                              (comparison_df['training_time'].iloc[i],
                               comparison_df[f'test_{metric}'].iloc[i]))
        axes[1, 1].set_xlabel('Training Time (seconds)')
        axes[1, 1].set_ylabel(f'Test {metric.replace("_", " ").title()}')
        axes[1, 1].set_title(f'Training Time vs {metric.replace("_", " ").title()}')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.results_dir, 'model_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Model comparison plot saved to {plot_path}")
    
    def get_best_model(self, metric: str = 'f1_score') -> Tuple[str, BaseAnomalyDetector]:
        """
        Get the best performing model based on a metric.
        
        Args:
            metric: Metric to optimize for
            
        Returns:
            Tuple of (model_name, model_instance)
        """
        # Try to load results from file if not already loaded
        if not self.results:
            self._load_results_from_file()
        
        if not self.results:
            raise ValueError("No training results available. Train models first.")
        
        best_score = -1
        best_model_name = None
        
        for model_name, results in self.results.items():
            score = results['test_metrics'].get(metric, 0)
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        if best_model_name is None:
            raise ValueError("No valid model found with the specified metric.")
        
        # Load the model file if not in memory
        if best_model_name not in self.models:
            model_path = os.path.join(self.results_dir, f'{best_model_name}_model.joblib')
            if os.path.exists(model_path):
                # Get the appropriate model class
                model_class = self._get_model_class(best_model_name)
                if model_class:
                    self.models[best_model_name] = model_class.load(model_path)
                    self.logger.info(f"Loaded model from {model_path}")
                else:
                    # Fallback to direct joblib loading
                    import joblib
                    self.models[best_model_name] = joblib.load(model_path)
                    self.logger.warning(f"Using fallback loading for {best_model_name}")
            else:
                raise ValueError(f"Model file not found for {best_model_name}")
        
        return best_model_name, self.models[best_model_name]
    
    def _get_model_class(self, model_name: str):
        """Get the appropriate model class for a model name."""
        model_class_map = {
            'random_forest': RandomForestDetector,
            'xgboost': XGBoostDetector,
            'logistic_regression': LogisticRegressionDetector,
            'mlp': MLPDetector,
            'gradient_boosting': GradientBoostingDetector,
            'svm': SVMDetector
        }
        return model_class_map.get(model_name)
    
    def _load_results_from_file(self):
        """Load training results from file if available."""
        results_path = os.path.join(self.results_dir, 'training_results.json')
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    self.results = json.load(f)
                self.logger.info(f"Loaded training results from {results_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load results from {results_path}: {e}")
                self.results = {}
    
    def _save_results(self, results: Dict[str, Dict]) -> None:
        """Save training results to disk."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, model_results in results.items():
            serializable_results[model_name] = self._make_serializable(model_results)
        
        # Save results
        results_path = os.path.join(self.results_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        self.logger.info(f"Training results saved to {results_path}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-serializable objects to serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
    
    def save_model(self, model_name: str, filepath: Optional[str] = None) -> str:
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of the model to save
            filepath: Optional custom filepath
            
        Returns:
            Path where the model was saved
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        if not self.models[model_name].is_fitted:
            raise ValueError(f"Model '{model_name}' is not fitted")
        
        if filepath is None:
            filepath = os.path.join(self.results_dir, f'{model_name}_model.joblib')
        
        self.models[model_name].save(filepath)
        return filepath