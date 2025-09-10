#!/usr/bin/env python3
"""
Training script for the anomaly detection models.
This script demonstrates Phase 1 of the real-time explainable AI system.
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
from datetime import datetimeexi

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config.config import get_config
from src.data.preprocessor import NetworkDataPreprocessor
from src.data.feature_engineering import NetworkFeatureEngineer
from src.models.model_trainer import ModelTrainer
from src.models.ensemble_models import (
    RandomForestDetector, XGBoostDetector, LogisticRegressionDetector,
    MLPDetector, GradientBoostingDetector
)


def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ]
    )


def load_data(config) -> tuple:
    """Load training and testing data."""
    logger = logging.getLogger(__name__)
    
    logger.info("Loading UNSW-NB15 dataset...")
    
    # Load training data
    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    logger.info(f"Loaded training data: {train_df.shape}")
    
    # Load testing data
    test_df = pd.read_csv(config.TEST_DATA_PATH)
    logger.info(f"Loaded testing data: {test_df.shape}")
    
    return train_df, test_df


def preprocess_data(train_df: pd.DataFrame, 
                   test_df: pd.DataFrame, 
                   config) -> tuple:
    """Preprocess the data using the preprocessing pipeline."""
    logger = logging.getLogger(__name__)
    
    logger.info("Starting data preprocessing...")
    
    # Initialize preprocessor
    preprocessor = NetworkDataPreprocessor(
        categorical_columns=config.CATEGORICAL_COLUMNS,
        target_column=config.TARGET_COLUMN,
        attack_category_column=config.ATTACK_CATEGORY_COLUMN,
        scaler_type=config.SCALER_TYPE
    )
    
    # Fit and transform training data
    X_train, y_train, y_train_multiclass = preprocessor.fit_transform(train_df)
    logger.info(f"Training features shape: {X_train.shape}")
    logger.info(f"Class distribution: {y_train.value_counts().to_dict()}")
    
    # Transform test data
    X_test = preprocessor.transform(test_df)
    y_test = test_df[config.TARGET_COLUMN]
    
    # Save preprocessor
    config.create_directories()
    preprocessor_path = os.path.join(config.PREPROCESSING_DIR, 'preprocessor.joblib')
    preprocessor.save(preprocessor_path)
    logger.info(f"Preprocessor saved to {preprocessor_path}")
    
    return X_train, y_train, X_test, y_test, preprocessor


def engineer_features(X_train: pd.DataFrame,
                     X_test: pd.DataFrame,
                     y_train: pd.Series,
                     config) -> tuple:
    """Apply feature engineering."""
    logger = logging.getLogger(__name__)
    
    logger.info("Starting feature engineering...")
    
    # Initialize feature engineer
    feature_engineer = NetworkFeatureEngineer()
    
    # Create derived features for training data
    X_train_enhanced = feature_engineer.create_derived_features(X_train)
    X_train_enhanced = feature_engineer.create_statistical_features(X_train_enhanced)
    
    # Apply feature selection if configured
    if config.NUM_SELECTED_FEATURES and config.NUM_SELECTED_FEATURES < X_train_enhanced.shape[1]:
        logger.info(f"Applying feature selection: {X_train_enhanced.shape[1]} -> {config.NUM_SELECTED_FEATURES} features")
        X_train_selected = feature_engineer.select_features(
            X_train_enhanced, 
            y_train, 
            method=config.FEATURE_SELECTION_METHOD,
            k_features=config.NUM_SELECTED_FEATURES
        )
    else:
        X_train_selected = X_train_enhanced
    
    # Transform test data using the same pipeline
    X_test_enhanced = feature_engineer.transform_new_data(X_test)
    
    logger.info(f"Final feature shapes - Train: {X_train_selected.shape}, Test: {X_test_enhanced.shape}")
    
    # Get feature importance scores if available
    importance_scores = feature_engineer.get_feature_importance_scores()
    if importance_scores:
        logger.info("Top 10 most important features:")
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        for feature, score in sorted_features[:10]:
            logger.info(f"  {feature}: {score:.4f}")
    
    return X_train_selected, X_test_enhanced, feature_engineer


def train_models(X_train: pd.DataFrame,
                y_train: pd.Series,
                X_test: pd.DataFrame,
                y_test: pd.Series,
                config,
                models_to_train: list = None) -> ModelTrainer:
    """Train and evaluate models."""
    logger = logging.getLogger(__name__)
    
    logger.info("Starting model training...")
    
    # Initialize model trainer
    trainer = ModelTrainer(results_dir=config.MODELS_DIR)
    
    # Add models with configurations
    if models_to_train is None:
        models_to_train = config.get_all_model_names()
    
    for model_name in models_to_train:
        model_config = config.get_model_config(model_name)
        
        if model_name == 'random_forest':
            model = RandomForestDetector(**model_config)
        elif model_name == 'xgboost':
            model = XGBoostDetector(**model_config)
        elif model_name == 'logistic_regression':
            model = LogisticRegressionDetector(**model_config)
        elif model_name == 'mlp':
            model = MLPDetector(**model_config)
        elif model_name == 'gradient_boosting':
            model = GradientBoostingDetector(**model_config)
        else:
            logger.warning(f"Unknown model: {model_name}")
            continue
        
        trainer.add_model(model_name, model)
    
    # Train all models
    results = trainer.train_all_models(
        X_train, y_train, X_test, y_test,
        models_to_train=models_to_train,
        cv_folds=config.CROSS_VALIDATION_FOLDS
    )
    
    # Compare models
    show_plot = config.__class__.__name__ != 'TestingConfig'  # Don't show plots in testing
    comparison_df = trainer.compare_models(show_plot=show_plot)
    logger.info("Model comparison:")
    logger.info("\n" + comparison_df.to_string())
    
    # Get best model
    best_model_name, best_model = trainer.get_best_model()
    logger.info(f"Best model: {best_model_name}")
    
    # Save best model
    best_model_path = trainer.save_model(best_model_name)
    logger.info(f"Best model saved to: {best_model_path}")
    
    return trainer


def main():
    """Main training pipeline."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train anomaly detection models')
    parser.add_argument('--env', default='development', 
                       choices=['development', 'production', 'testing'],
                       help='Environment configuration')
    parser.add_argument('--models', nargs='+', 
                       help='Specific models to train (default: all)')
    args = parser.parse_args()
    
    # Get configuration
    config = get_config(args.env)
    
    # Setup logging
    setup_logging(config.LOG_LEVEL)
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("UNSW-NB15 Anomaly Detection Model Training")
    logger.info("="*60)
    logger.info(f"Environment: {args.env}")
    logger.info(f"Models to train: {args.models or 'all'}")
    logger.info(f"Start time: {datetime.now()}")
    
    try:
        # Create directories
        config.create_directories()
        
        # Load data
        train_df, test_df = load_data(config)
        
        # Preprocess data
        X_train, y_train, X_test, y_test, preprocessor = preprocess_data(
            train_df, test_df, config
        )
        
        # Feature engineering
        X_train_final, X_test_final, feature_engineer = engineer_features(
            X_train, X_test, y_train, config
        )
        
        # Train models
        trainer = train_models(
            X_train_final, y_train, X_test_final, y_test, 
            config, args.models
        )
        
        logger.info("="*60)
        logger.info("Training completed successfully!")
        logger.info(f"End time: {datetime.now()}")
        logger.info("="*60)
        
        # Print summary
        comparison_df = trainer.compare_models(show_plot=False)
        print("\nModel Performance Summary:")
        print("="*50)
        print(comparison_df[['model', 'test_f1_score', 'test_auc_roc', 'training_time']].to_string(index=False))
        
        best_model_name, _ = trainer.get_best_model()
        print(f"\nBest Model: {best_model_name}")
        print("\nNext Steps:")
        print("1. Review model performance metrics")
        print("2. Proceed to Phase 2: Explainability integration")
        print("3. Set up real-time pipeline components")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()