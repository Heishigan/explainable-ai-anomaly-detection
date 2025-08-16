import os
from typing import Dict, Any, List


class Config:
    """Configuration settings for the anomaly detection system."""
    
    # Data paths
    DATA_DIR = "data"
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    
    TRAIN_DATA_PATH = os.path.join(RAW_DATA_DIR, "UNSW_NB15_training-set.csv")
    TEST_DATA_PATH = os.path.join(RAW_DATA_DIR, "UNSW_NB15_testing-set.csv")
    FEATURES_DATA_PATH = os.path.join(RAW_DATA_DIR, "NUSW-NB15_features.csv")
    
    # Results paths
    RESULTS_DIR = "results"
    MODELS_DIR = os.path.join(RESULTS_DIR, "models")
    PREPROCESSING_DIR = os.path.join(RESULTS_DIR, "preprocessing")
    EXPLANATIONS_DIR = os.path.join(RESULTS_DIR, "explanations")
    
    # Preprocessing configuration
    CATEGORICAL_COLUMNS = ['proto', 'service', 'state']
    TARGET_COLUMN = 'label'
    ATTACK_CATEGORY_COLUMN = 'attack_cat'
    SCALER_TYPE = 'standard'  # 'standard' or 'robust'
    
    # Feature engineering
    FEATURE_SELECTION_METHOD = 'mutual_info'  # 'mutual_info' or 'f_score'
    NUM_SELECTED_FEATURES = 50
    APPLY_PCA = False
    PCA_VARIANCE_THRESHOLD = 0.95
    
    # Model training
    CROSS_VALIDATION_FOLDS = 5
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    # Model configurations
    MODEL_CONFIGS = {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': RANDOM_STATE
        },
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': RANDOM_STATE
        },
        'logistic_regression': {
            'C': 1.0,
            'penalty': 'l2',
            'max_iter': 1000,
            'random_state': RANDOM_STATE
        },
        'mlp': {
            'hidden_layer_sizes': (100, 50, 25),
            'activation': 'relu',
            'alpha': 0.001,
            'learning_rate_init': 0.001,
            'max_iter': 500,
            'random_state': RANDOM_STATE
        },
        'gradient_boosting': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'random_state': RANDOM_STATE
        }
    }
    
    # Real-time processing
    BATCH_SIZE = 1000
    PREDICTION_THRESHOLD = 0.5
    CONFIDENCE_THRESHOLD = 0.8
    
    # Explainability
    SHAP_SAMPLE_SIZE = 1000  # Number of samples for SHAP background
    LIME_NUM_FEATURES = 10   # Number of features to show in LIME explanations
    EXPLANATION_CACHE_SIZE = 1000
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Performance monitoring
    MONITOR_METRICS = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    ALERT_THRESHOLDS = {
        'accuracy': 0.9,
        'precision': 0.9,
        'recall': 0.9,
        'f1_score': 0.9,
        'auc_roc': 0.9
    }
    
    @classmethod
    def create_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.RESULTS_DIR,
            cls.MODELS_DIR,
            cls.PREPROCESSING_DIR,
            cls.EXPLANATIONS_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def get_model_config(cls, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        if model_name not in cls.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(cls.MODEL_CONFIGS.keys())}")
        return cls.MODEL_CONFIGS[model_name].copy()
    
    @classmethod
    def get_all_model_names(cls) -> List[str]:
        """Get list of all configured model names."""
        return list(cls.MODEL_CONFIGS.keys())


# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development environment configuration."""
    LOG_LEVEL = 'DEBUG'
    CROSS_VALIDATION_FOLDS = 3  # Faster for development


class ProductionConfig(Config):
    """Production environment configuration."""
    LOG_LEVEL = 'WARNING'
    CROSS_VALIDATION_FOLDS = 5
    BATCH_SIZE = 5000
    SHAP_SAMPLE_SIZE = 500  # Smaller for faster inference


class TestingConfig(Config):
    """Testing environment configuration."""
    LOG_LEVEL = 'DEBUG'
    CROSS_VALIDATION_FOLDS = 2
    BATCH_SIZE = 100
    NUM_SELECTED_FEATURES = 10  # Smaller for faster tests


# Configuration factory
def get_config(environment: str = 'development') -> Config:
    """
    Get configuration for specified environment.
    
    Args:
        environment: 'development', 'production', or 'testing'
        
    Returns:
        Configuration class instance
    """
    configs = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig
    }
    
    if environment not in configs:
        raise ValueError(f"Unknown environment: {environment}. Available: {list(configs.keys())}")
    
    return configs[environment]