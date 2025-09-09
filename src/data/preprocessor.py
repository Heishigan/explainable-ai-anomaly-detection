import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Any
import joblib
import os
import logging


class NetworkDataPreprocessor:
    """
    Preprocessor for UNSW-NB15 network intrusion dataset.
    Handles categorical encoding, feature scaling, and data preparation.
    """
    
    def __init__(self, 
                 categorical_columns: Optional[List[str]] = None,
                 numerical_columns: Optional[List[str]] = None,
                 target_column: str = 'label',
                 attack_category_column: str = 'attack_cat',
                 scaler_type: str = 'standard'):
        """
        Initialize the preprocessor.
        
        Args:
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names  
            target_column: Name of the binary target column
            attack_category_column: Name of the attack category column
            scaler_type: Type of scaler ('standard' or 'robust')
        """
        self.categorical_columns = categorical_columns or ['proto', 'service', 'state']
        self.numerical_columns = numerical_columns
        self.target_column = target_column
        self.attack_category_column = attack_category_column
        self.scaler_type = scaler_type
        
        # Initialize transformers
        self.label_encoders = {}
        self.scaler = StandardScaler() if scaler_type == 'standard' else RobustScaler()
        self.feature_columns = None
        self.is_fitted = False
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def _identify_columns(self, df: pd.DataFrame) -> None:
        """Automatically identify numerical and categorical columns if not provided."""
        if self.numerical_columns is None:
            # Exclude target and attack category columns
            exclude_cols = [self.target_column, self.attack_category_column, 'id']
            exclude_cols = [col for col in exclude_cols if col in df.columns]
            
            # Get numerical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            self.numerical_columns = [col for col in numerical_cols if col not in exclude_cols]
            
            self.logger.info(f"Auto-identified {len(self.numerical_columns)} numerical columns")
        
        if self.categorical_columns is None:
            # Exclude target and attack category columns and numerical columns
            exclude_cols = [self.target_column, self.attack_category_column, 'id']
            exclude_cols = [col for col in exclude_cols if col in df.columns]
            exclude_cols.extend(self.numerical_columns)
            
            # Get categorical columns
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            self.categorical_columns = [col for col in categorical_cols if col not in exclude_cols]
            
            self.logger.info(f"Auto-identified {len(self.categorical_columns)} categorical columns")
    
    def _handle_categorical_encoding(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables using Label Encoding.
        
        Args:
            df: Input dataframe
            fit: Whether to fit the encoders (True for training data)
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df_encoded = df.copy()
        
        for col in self.categorical_columns:
            if col not in df.columns:
                self.logger.warning(f"Categorical column '{col}' not found in data")
                continue
                
            if fit:
                # Fit encoder on training data
                self.label_encoders[col] = LabelEncoder()
                # Handle missing values by replacing with 'unknown'
                df_encoded[col] = df_encoded[col].fillna('unknown')
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col])
            else:
                # Transform using fitted encoder
                if col in self.label_encoders:
                    df_encoded[col] = df_encoded[col].fillna('unknown')
                    # Handle unseen categories by mapping them to 'unknown' if it exists in training
                    known_classes = set(self.label_encoders[col].classes_)
                    if 'unknown' not in known_classes:
                        # If 'unknown' wasn't in training data, map unseen values to the most frequent class
                        most_frequent = self.label_encoders[col].classes_[0]  # First class is most frequent in LabelEncoder
                        df_encoded[col] = df_encoded[col].apply(
                            lambda x: x if x in known_classes else most_frequent
                        )
                    else:
                        df_encoded[col] = df_encoded[col].apply(
                            lambda x: x if x in known_classes else 'unknown'
                        )
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
        
        return df_encoded
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in numerical columns."""
        df_cleaned = df.copy()
        
        for col in self.numerical_columns:
            if col in df.columns:
                # Fill missing values with median for numerical columns
                if df_cleaned[col].isnull().sum() > 0:
                    median_val = df_cleaned[col].median()
                    df_cleaned[col].fillna(median_val, inplace=True)
                    self.logger.info(f"Filled {df_cleaned[col].isnull().sum()} missing values in '{col}' with median: {median_val}")
        
        return df_cleaned
    
    def _scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df: Input dataframe
            fit: Whether to fit the scaler
            
        Returns:
            DataFrame with scaled features
        """
        df_scaled = df.copy()
        
        if fit:
            df_scaled[self.numerical_columns] = self.scaler.fit_transform(
                df_scaled[self.numerical_columns]
            )
        else:
            df_scaled[self.numerical_columns] = self.scaler.transform(
                df_scaled[self.numerical_columns]
            )
        
        return df_scaled
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Fit the preprocessor and transform the training data.
        
        Args:
            df: Input training dataframe
            
        Returns:
            Tuple of (X_processed, y_binary, y_multiclass)
        """
        self.logger.info("Starting preprocessing fit_transform")
        
        # Identify columns if not provided
        self._identify_columns(df)
        
        # Store feature columns for later use
        self.feature_columns = list(self.categorical_columns) + list(self.numerical_columns)
        
        # Step 1: Handle categorical encoding
        df_processed = self._handle_categorical_encoding(df, fit=True)
        
        # Step 2: Handle missing values
        df_processed = self._handle_missing_values(df_processed)
        
        # Step 3: Scale numerical features
        df_processed = self._scale_features(df_processed, fit=True)
        
        # Extract features and targets
        X = df_processed[self.feature_columns]
        y_binary = df_processed[self.target_column]
        y_multiclass = df_processed[self.attack_category_column] if self.attack_category_column in df.columns else None
        
        self.is_fitted = True
        self.logger.info(f"Preprocessing complete. Feature shape: {X.shape}")
        
        return X, y_binary, y_multiclass
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            df: Input dataframe to transform
            
        Returns:
            Processed feature dataframe
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform. Call fit_transform first.")
        
        # Apply same preprocessing steps
        df_processed = self._handle_categorical_encoding(df, fit=False)
        df_processed = self._handle_missing_values(df_processed)
        df_processed = self._scale_features(df_processed, fit=False)
        
        # Return only feature columns
        return df_processed[self.feature_columns]
    
    def save(self, filepath: str) -> None:
        """Save the fitted preprocessor to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor")
        
        preprocessor_data = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'attack_category_column': self.attack_category_column,
            'scaler_type': self.scaler_type,
            'is_fitted': self.is_fitted
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(preprocessor_data, filepath)
        self.logger.info(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'NetworkDataPreprocessor':
        """Load a fitted preprocessor from disk."""
        preprocessor_data = joblib.load(filepath)
        
        preprocessor = cls(
            categorical_columns=preprocessor_data['categorical_columns'],
            numerical_columns=preprocessor_data['numerical_columns'],
            target_column=preprocessor_data['target_column'],
            attack_category_column=preprocessor_data['attack_category_column'],
            scaler_type=preprocessor_data['scaler_type']
        )
        
        preprocessor.label_encoders = preprocessor_data['label_encoders']
        preprocessor.scaler = preprocessor_data['scaler']
        preprocessor.feature_columns = preprocessor_data['feature_columns']
        preprocessor.is_fitted = preprocessor_data['is_fitted']
        
        return preprocessor
    
    def get_feature_names(self) -> List[str]:
        """Get the names of processed features."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        return self.feature_columns.copy()
    
    def get_categorical_mappings(self) -> Dict[str, Dict]:
        """Get the categorical variable mappings for interpretation."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        mappings = {}
        for col, encoder in self.label_encoders.items():
            mappings[col] = {i: label for i, label in enumerate(encoder.classes_)}
        
        return mappings