import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import logging


class NetworkFeatureEngineer:
    """
    Feature engineering for network intrusion detection.
    Creates derived features and performs feature selection.
    """
    
    def __init__(self):
        self.feature_selector = None
        self.pca = None
        self.selected_features = None
        self.is_fitted = False
        self.logger = logging.getLogger(__name__)
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from existing network flow features.
        
        Args:
            df: Input dataframe with network flow features
            
        Returns:
            DataFrame with additional engineered features
        """
        df_enhanced = df.copy()
        
        # Packet size statistics
        if 'spkts' in df.columns and 'sbytes' in df.columns:
            df_enhanced['avg_src_packet_size'] = df_enhanced['sbytes'] / (df_enhanced['spkts'] + 1e-8)
            
        if 'dpkts' in df.columns and 'dbytes' in df.columns:
            df_enhanced['avg_dst_packet_size'] = df_enhanced['dbytes'] / (df_enhanced['dpkts'] + 1e-8)
        
        # Packet ratios
        if 'spkts' in df.columns and 'dpkts' in df.columns:
            total_packets = df_enhanced['spkts'] + df_enhanced['dpkts']
            df_enhanced['src_packet_ratio'] = df_enhanced['spkts'] / (total_packets + 1e-8)
            df_enhanced['packet_asymmetry'] = abs(df_enhanced['spkts'] - df_enhanced['dpkts']) / (total_packets + 1e-8)
        
        # Byte ratios  
        if 'sbytes' in df.columns and 'dbytes' in df.columns:
            total_bytes = df_enhanced['sbytes'] + df_enhanced['dbytes']
            df_enhanced['src_byte_ratio'] = df_enhanced['sbytes'] / (total_bytes + 1e-8)
            df_enhanced['byte_asymmetry'] = abs(df_enhanced['sbytes'] - df_enhanced['dbytes']) / (total_bytes + 1e-8)
        
        # Rate and throughput features
        if 'dur' in df.columns:
            if 'sbytes' in df.columns:
                df_enhanced['src_throughput'] = df_enhanced['sbytes'] / (df_enhanced['dur'] + 1e-8)
            if 'dbytes' in df.columns:
                df_enhanced['dst_throughput'] = df_enhanced['dbytes'] / (df_enhanced['dur'] + 1e-8)
            if 'spkts' in df.columns:
                df_enhanced['src_packet_rate'] = df_enhanced['spkts'] / (df_enhanced['dur'] + 1e-8)
            if 'dpkts' in df.columns:
                df_enhanced['dst_packet_rate'] = df_enhanced['dpkts'] / (df_enhanced['dur'] + 1e-8)
        
        # Jitter ratios
        if 'sjit' in df.columns and 'djit' in df.columns:
            df_enhanced['jitter_ratio'] = df_enhanced['sjit'] / (df_enhanced['djit'] + 1e-8)
            df_enhanced['total_jitter'] = df_enhanced['sjit'] + df_enhanced['djit']
        
        # TCP window features
        if 'swin' in df.columns and 'dwin' in df.columns:
            df_enhanced['window_ratio'] = df_enhanced['swin'] / (df_enhanced['dwin'] + 1e-8)
            df_enhanced['window_diff'] = abs(df_enhanced['swin'] - df_enhanced['dwin'])
        
        # Connection pattern features
        connection_cols = ['ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm']
        available_conn_cols = [col for col in connection_cols if col in df.columns]
        
        if len(available_conn_cols) >= 2:
            # Service connection ratio
            if 'ct_srv_src' in df.columns and 'ct_srv_dst' in df.columns:
                df_enhanced['srv_connection_balance'] = df_enhanced['ct_srv_src'] / (df_enhanced['ct_srv_dst'] + 1e-8)
            
            # Recent connection activity
            if 'ct_dst_ltm' in df.columns and 'ct_src_ltm' in df.columns:
                df_enhanced['connection_activity'] = df_enhanced['ct_dst_ltm'] + df_enhanced['ct_src_ltm']
                df_enhanced['connection_imbalance'] = abs(df_enhanced['ct_dst_ltm'] - df_enhanced['ct_src_ltm'])
        
        # Load balancing features
        if 'sload' in df.columns and 'dload' in df.columns:
            total_load = df_enhanced['sload'] + df_enhanced['dload']
            df_enhanced['load_ratio'] = df_enhanced['sload'] / (df_enhanced['dload'] + 1e-8)
            df_enhanced['load_imbalance'] = abs(df_enhanced['sload'] - df_enhanced['dload']) / (total_load + 1e-8)
        
        # TCP timing features
        tcp_timing_cols = ['tcprtt', 'synack', 'ackdat']
        available_timing_cols = [col for col in tcp_timing_cols if col in df.columns]
        
        if len(available_timing_cols) >= 2:
            if 'tcprtt' in df.columns and 'synack' in df.columns:
                df_enhanced['connection_efficiency'] = df_enhanced['synack'] / (df_enhanced['tcprtt'] + 1e-8)
        
        # Binary feature combinations
        binary_cols = ['is_ftp_login', 'is_sm_ips_ports']
        available_binary_cols = [col for col in binary_cols if col in df.columns]
        
        if len(available_binary_cols) >= 2:
            df_enhanced['security_flags'] = sum([df_enhanced[col] for col in available_binary_cols])
        
        # TTL features
        if 'sttl' in df.columns and 'dttl' in df.columns:
            df_enhanced['ttl_diff'] = abs(df_enhanced['sttl'] - df_enhanced['dttl'])
            df_enhanced['ttl_ratio'] = df_enhanced['sttl'] / (df_enhanced['dttl'] + 1e-8)
        
        # Loss features
        if 'sloss' in df.columns and 'dloss' in df.columns:
            total_loss = df_enhanced['sloss'] + df_enhanced['dloss']
            df_enhanced['total_loss'] = total_loss
            df_enhanced['loss_imbalance'] = abs(df_enhanced['sloss'] - df_enhanced['dloss']) / (total_loss + 1e-8)
        
        self.logger.info(f"Created {df_enhanced.shape[1] - df.shape[1]} derived features")
        return df_enhanced
    
    def create_statistical_features(self, df: pd.DataFrame, window_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create statistical aggregation features for relevant columns.
        
        Args:
            df: Input dataframe
            window_cols: Columns to compute rolling statistics for
            
        Returns:
            DataFrame with statistical features
        """
        df_stats = df.copy()
        
        if window_cols is None:
            # Default columns for statistical features
            window_cols = ['rate', 'sload', 'dload', 'sinpkt', 'dinpkt', 'sjit', 'djit']
            window_cols = [col for col in window_cols if col in df.columns]
        
        # Add percentile-based features for key metrics
        percentile_cols = ['rate', 'sbytes', 'dbytes', 'dur']
        available_percentile_cols = [col for col in percentile_cols if col in df.columns]
        
        for col in available_percentile_cols:
            # Create binned categorical features based on percentiles
            try:
                df_stats[f'{col}_quartile'] = pd.qcut(df_stats[col], q=4, labels=['low', 'medium', 'high', 'very_high'], duplicates='drop')
            except ValueError:
                # Fallback to regular cut if qcut fails due to duplicate values
                self.logger.warning(f"qcut failed for {col}, using regular cut instead")
                df_stats[f'{col}_quartile'] = pd.cut(df_stats[col], bins=4, labels=['low', 'medium', 'high', 'very_high'], duplicates='drop')
        
        return df_stats
    
    def select_features(self, 
                       X: pd.DataFrame, 
                       y: pd.Series, 
                       method: str = 'mutual_info',
                       k_features: int = 50) -> pd.DataFrame:
        """
        Perform feature selection.
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Selection method ('f_score', 'mutual_info')
            k_features: Number of features to select
            
        Returns:
            DataFrame with selected features
        """
        if method == 'f_score':
            self.feature_selector = SelectKBest(f_classif, k=k_features)
        elif method == 'mutual_info':
            self.feature_selector = SelectKBest(mutual_info_classif, k=k_features)
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        # Only use numerical columns for feature selection (exclude categorical quartile features)
        numerical_columns = X.select_dtypes(include=[np.number]).columns
        categorical_columns = X.select_dtypes(exclude=[np.number]).columns
        
        if len(categorical_columns) > 0:
            self.logger.info(f"Excluding {len(categorical_columns)} categorical columns from feature selection: {list(categorical_columns)}")
            X_numerical = X[numerical_columns]
        else:
            X_numerical = X
        
        # Fit and transform
        X_selected = self.feature_selector.fit_transform(X_numerical, y)
        
        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        self.selected_features = numerical_columns[selected_mask].tolist()
        
        X_selected_df = pd.DataFrame(
            X_selected, 
            columns=self.selected_features,
            index=X.index
        )
        
        self.logger.info(f"Selected {len(self.selected_features)} features using {method}")
        return X_selected_df
    
    def apply_pca(self, 
                  X: pd.DataFrame, 
                  n_components: Optional[int] = None,
                  variance_threshold: float = 0.95) -> pd.DataFrame:
        """
        Apply PCA for dimensionality reduction.
        
        Args:
            X: Feature matrix
            n_components: Number of components (if None, use variance_threshold)
            variance_threshold: Cumulative variance threshold
            
        Returns:
            DataFrame with PCA components
        """
        if n_components is None:
            # Determine components needed for variance threshold
            temp_pca = PCA()
            temp_pca.fit(X)
            cumvar = np.cumsum(temp_pca.explained_variance_ratio_)
            n_components = np.argmax(cumvar >= variance_threshold) + 1
        
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X)
        
        # Create DataFrame with component names
        component_names = [f'PC_{i+1}' for i in range(n_components)]
        X_pca_df = pd.DataFrame(
            X_pca,
            columns=component_names,
            index=X.index
        )
        
        self.logger.info(f"Applied PCA: {X.shape[1]} -> {n_components} components "
                        f"(variance explained: {self.pca.explained_variance_ratio_.sum():.3f})")
        
        return X_pca_df
    
    def get_feature_importance_scores(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores from feature selection."""
        if self.feature_selector is None or self.selected_features is None:
            return None
        
        scores = self.feature_selector.scores_
        selected_indices = self.feature_selector.get_support(indices=True)
        
        return {
            feature: scores[idx] 
            for feature, idx in zip(self.selected_features, selected_indices)
        }
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the same feature engineering to new data.
        
        Args:
            df: New data to transform
            
        Returns:
            Transformed dataframe
        """
        # Apply feature engineering
        df_enhanced = self.create_derived_features(df)
        # Skip statistical features for new data to avoid feature mismatch
        # df_enhanced = self.create_statistical_features(df_enhanced)
        
        # Apply feature selection if fitted
        if self.feature_selector is not None and self.selected_features is not None:
            # Ensure all selected features are present
            available_features = [col for col in self.selected_features if col in df_enhanced.columns]
            if len(available_features) != len(self.selected_features):
                self.logger.warning(f"Some selected features missing in new data: "
                                  f"{set(self.selected_features) - set(available_features)}")
            df_enhanced = df_enhanced[available_features]
        
        return df_enhanced
    
    def get_derived_feature_names(self) -> List[str]:
        """Get names of all derived features that will be created."""
        derived_features = [
            'avg_src_packet_size', 'avg_dst_packet_size', 'src_packet_ratio', 
            'packet_asymmetry', 'src_byte_ratio', 'byte_asymmetry',
            'src_throughput', 'dst_throughput', 'src_packet_rate', 'dst_packet_rate',
            'jitter_ratio', 'total_jitter', 'window_ratio', 'window_diff',
            'srv_connection_balance', 'connection_activity', 'connection_imbalance',
            'load_ratio', 'load_imbalance', 'connection_efficiency', 'security_flags',
            'ttl_diff', 'ttl_ratio', 'total_loss', 'loss_imbalance'
        ]
        return derived_features
    
    def save(self, filepath: str) -> None:
        """
        Save the feature engineer state to disk.
        
        Args:
            filepath: Path to save the feature engineer
        """
        import joblib
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the complete state
        state = {
            'feature_selector': self.feature_selector,
            'pca': self.pca,
            'selected_features': self.selected_features,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(state, filepath)
        self.logger.info(f"Feature engineer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'NetworkFeatureEngineer':
        """
        Load a feature engineer from disk.
        
        Args:
            filepath: Path to the saved feature engineer
            
        Returns:
            Loaded NetworkFeatureEngineer instance
        """
        import joblib
        
        # Create new instance
        feature_engineer = cls()
        
        # Load state
        state = joblib.load(filepath)
        feature_engineer.feature_selector = state['feature_selector']
        feature_engineer.pca = state['pca']
        feature_engineer.selected_features = state['selected_features']
        feature_engineer.is_fitted = state['is_fitted']
        
        feature_engineer.logger.info(f"Feature engineer loaded from {filepath}")
        return feature_engineer