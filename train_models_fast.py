#!/usr/bin/env python3
"""
Fast training script for Phase 1 ML Foundation enhancement.
Optimized for quick baseline establishment and comprehensive evaluation.
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config.config import get_config
from src.data.preprocessor import NetworkDataPreprocessor
from src.data.feature_engineering import NetworkFeatureEngineer
from src.models.ensemble_models import (
    RandomForestDetector, XGBoostDetector, LogisticRegressionDetector,
    MLPDetector, GradientBoostingDetector
)

def setup_logging():
    """Setup optimized logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training_fast.log'),
            logging.StreamHandler()
        ]
    )

def load_sample_data(sample_size=10000):
    """Load a representative sample of the dataset for faster training."""
    logger = logging.getLogger(__name__)
    
    logger.info("Loading sampled UNSW-NB15 dataset for fast training...")
    
    # Load and sample training data
    train_df = pd.read_csv('data/raw/UNSW_NB15_training-set.csv')
    logger.info(f"Original training data: {train_df.shape}")
    
    # Stratified sampling to maintain class balance
    attack_samples = train_df[train_df['label'] == 1].sample(n=int(sample_size * 0.68), random_state=42)
    normal_samples = train_df[train_df['label'] == 0].sample(n=int(sample_size * 0.32), random_state=42)
    
    train_sample = pd.concat([attack_samples, normal_samples]).sample(frac=1, random_state=42)
    
    # Load full test data for proper evaluation
    test_df = pd.read_csv('data/raw/UNSW_NB15_testing-set.csv')
    
    logger.info(f"Sampled training data: {train_sample.shape}")
    logger.info(f"Full test data: {test_df.shape}")
    logger.info(f"Sample class balance: {train_sample['label'].value_counts().to_dict()}")
    
    return train_sample, test_df

def fast_preprocess_data(train_df, test_df, config):
    """Fast data preprocessing with saved processor if available."""
    logger = logging.getLogger(__name__)
    
    # Check if preprocessor exists
    preprocessor_path = "results/preprocessing/preprocessor.joblib"
    if os.path.exists(preprocessor_path):
        logger.info("Loading existing preprocessor...")
        import joblib
        preprocessor = joblib.load(preprocessor_path)
        
        # Apply preprocessing
        X_train_processed = preprocessor.transform(train_df)
        X_test_processed = preprocessor.transform(test_df)
        
        # Extract labels
        y_train = train_df['label']
        y_test = test_df['label']
        
    else:
        logger.info("Creating new preprocessor...")
        preprocessor = NetworkDataPreprocessor(
            categorical_columns=config.CATEGORICAL_COLUMNS,
            target_column=config.TARGET_COLUMN,
            attack_category_column=config.ATTACK_CATEGORY_COLUMN,
            scaler_type=config.SCALER_TYPE
        )
        
        X_train_processed, y_train, _ = preprocessor.fit_transform(train_df)
        X_test_processed = preprocessor.transform(test_df)
        
        # Extract test labels (they are not processed)
        y_test = test_df['label']
        
        # Save preprocessor
        os.makedirs("results/preprocessing", exist_ok=True)
        import joblib
        joblib.dump(preprocessor, preprocessor_path)
        logger.info(f"Preprocessor saved to {preprocessor_path}")
    
    logger.info(f"Processed shapes - Train: {X_train_processed.shape}, Test: {X_test_processed.shape}")
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

def fast_feature_engineering(X_train, X_test, y_train, config):
    """Fast feature engineering with top features."""
    logger = logging.getLogger(__name__)
    
    feature_engineer = NetworkFeatureEngineer()
    
    # Create derived features
    X_train_featured = feature_engineer.create_derived_features(X_train)
    X_test_featured = feature_engineer.create_derived_features(X_test)
    
    # Quick feature selection (top 30 features for speed)
    X_train_selected = feature_engineer.select_features(
        X_train_featured, y_train, 
        method='mutual_info', 
        k_features=30
    )
    X_test_selected = feature_engineer.transform_new_data(X_test_featured)
    
    logger.info(f"Feature engineering complete - Train: {X_train_selected.shape}, Test: {X_test_selected.shape}")
    logger.info(f"Total selected features: {X_train_selected.shape[1]}")
    
    return X_train_selected, X_test_selected, feature_engineer

def train_baseline_models(X_train, X_test, y_train, y_test):
    """Train baseline models quickly."""
    logger = logging.getLogger(__name__)
    
    # Optimized model configurations for speed
    models = {
        'random_forest': RandomForestDetector(
            n_estimators=50,  # Reduced for speed
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42
        ),
        'xgboost': XGBoostDetector(
            n_estimators=50,  # Reduced for speed
            max_depth=6,
            learning_rate=0.3,
            random_state=42
        ),
        'logistic_regression': LogisticRegressionDetector(
            C=1.0,
            max_iter=500,  # Reduced for speed
            random_state=42
        ),
        'gradient_boosting': GradientBoostingDetector(
            n_estimators=50,  # Reduced for speed
            learning_rate=0.2,
            max_depth=6,
            random_state=42
        )
    }
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        start_time = datetime.now()
        
        # Train model
        model.fit(pd.DataFrame(X_train), pd.Series(y_train))
        
        # Evaluate model
        train_pred = model.predict(pd.DataFrame(X_train))
        test_pred = model.predict(pd.DataFrame(X_test))
        test_proba = model.predict_proba(pd.DataFrame(X_test))[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        test_precision = precision_score(y_test, test_pred)
        test_recall = recall_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred)
        test_auc = roc_auc_score(y_test, test_proba)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Store results
        results[name] = {
            'model': model,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'training_time_seconds': training_time,
            'predictions': test_pred,
            'probabilities': test_proba
        }
        
        logger.info(f"{name} - Test F1: {test_f1:.3f}, AUC: {test_auc:.3f}, Time: {training_time:.1f}s")
    
    return results

def create_comprehensive_report(results, y_test, feature_engineer):
    """Create comprehensive training report."""
    logger = logging.getLogger(__name__)
    
    # Create results directory
    os.makedirs("results/models", exist_ok=True)
    
    # Performance comparison
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'Model': name,
            'Test_Accuracy': result['test_accuracy'],
            'Test_Precision': result['test_precision'],
            'Test_Recall': result['test_recall'],
            'Test_F1': result['test_f1'],
            'Test_AUC': result['test_auc'],
            'Training_Time_Seconds': result['training_time_seconds']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save results
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'dataset_info': {
            'training_samples': len(y_test),  # Note: using sampled size
            'test_samples': len(y_test),
            'features': 30,
            'class_distribution': {
                'normal': int((y_test == 0).sum()),
                'attack': int((y_test == 1).sum())
            }
        },
        'model_performance': comparison_data,
        'feature_importance': {
            'total_selected': 30
        }
    }
    
    # Save JSON results
    with open("results/models/training_results_fast.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save detailed comparison
    comparison_df.to_csv("results/models/model_comparison.csv", index=False)
    
    # Create visualization
    create_performance_plots(comparison_df, results, y_test)
    
    # Print summary
    logger.info("=== TRAINING SUMMARY ===")
    logger.info(f"Best F1 Score: {comparison_df['Test_F1'].max():.3f} ({comparison_df.loc[comparison_df['Test_F1'].idxmax(), 'Model']})")
    logger.info(f"Best AUC Score: {comparison_df['Test_AUC'].max():.3f} ({comparison_df.loc[comparison_df['Test_AUC'].idxmax(), 'Model']})")
    logger.info(f"Fastest Training: {comparison_df['Training_Time_Seconds'].min():.1f}s ({comparison_df.loc[comparison_df['Training_Time_Seconds'].idxmin(), 'Model']})")
    
    return results_summary, comparison_df

def create_performance_plots(comparison_df, results, y_test):
    """Create performance visualization plots."""
    logger = logging.getLogger(__name__)
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('UNSW-NB15 Anomaly Detection - Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. Performance Metrics Bar Chart
    metrics = ['Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1', 'Test_AUC']
    x = np.arange(len(comparison_df))
    width = 0.15
    
    for i, metric in enumerate(metrics):
        axes[0, 0].bar(x + i * width, comparison_df[metric], width, label=metric.replace('Test_', ''))
    
    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Performance Metrics Comparison')
    axes[0, 0].set_xticks(x + width * 2)
    axes[0, 0].set_xticklabels(comparison_df['Model'], rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. F1 vs AUC Scatter
    axes[0, 1].scatter(comparison_df['Test_F1'], comparison_df['Test_AUC'], s=100, alpha=0.7)
    for i, model in enumerate(comparison_df['Model']):
        axes[0, 1].annotate(model, (comparison_df.iloc[i]['Test_F1'], comparison_df.iloc[i]['Test_AUC']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
    axes[0, 1].set_xlabel('F1 Score')
    axes[0, 1].set_ylabel('AUC Score')
    axes[0, 1].set_title('F1 vs AUC Performance')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Training Time Comparison
    axes[0, 2].bar(comparison_df['Model'], comparison_df['Training_Time_Seconds'])
    axes[0, 2].set_xlabel('Models')
    axes[0, 2].set_ylabel('Training Time (seconds)')
    axes[0, 2].set_title('Training Time Comparison')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Confusion Matrix for Best Model
    best_model_name = comparison_df.loc[comparison_df['Test_F1'].idxmax(), 'Model']
    best_predictions = results[best_model_name]['predictions']
    
    cm = confusion_matrix(y_test, best_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title(f'Confusion Matrix - {best_model_name}')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    
    # 5. ROC Curves
    from sklearn.metrics import roc_curve
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
        axes[1, 1].plot(fpr, tpr, label=f"{name} (AUC={result['test_auc']:.3f})")
    
    axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('ROC Curves Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Class Distribution
    class_counts = pd.Series(y_test).value_counts()
    axes[1, 2].pie(class_counts.values, labels=['Normal', 'Attack'], autopct='%1.1f%%', startangle=90)
    axes[1, 2].set_title('Test Set Class Distribution')
    
    plt.tight_layout()
    plt.savefig('results/models/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Performance plots saved to results/models/performance_comparison.png")

def main():
    """Main training execution."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("=== FAST PHASE 1 ML FOUNDATION TRAINING ===")
        start_time = datetime.now()
        
        # Load configuration
        config = get_config('development')
        
        # Load sample data for faster training
        train_df, test_df = load_sample_data(sample_size=10000)
        
        # Preprocess data
        X_train, X_test, y_train, y_test, preprocessor = fast_preprocess_data(train_df, test_df, config)
        
        # Feature engineering
        X_train_final, X_test_final, feature_engineer = fast_feature_engineering(X_train, X_test, y_train, config)
        
        # Train models
        results = train_baseline_models(X_train_final, X_test_final, y_train, y_test)
        
        # Create comprehensive report
        results_summary, comparison_df = create_comprehensive_report(results, y_test, feature_engineer)
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"=== TRAINING COMPLETED in {total_time:.1f} seconds ===")
        
        return results_summary
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()