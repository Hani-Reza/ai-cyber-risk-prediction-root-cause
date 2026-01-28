"""
Cybersecurity Data Preprocessing Pipeline
Production-grade feature engineering for ML risk prediction
Author: Principal AI/ML Engineer
Date: 2024
Version: 1.0
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, Dict, Any
import warnings
import pickle
import os
from pathlib import Path
import sys

# Add project root to path for module imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

warnings.filterwarnings('ignore')

class CybersecurityPreprocessor:
    """
    Production-grade preprocessing pipeline for cybersecurity event data.
    Implements enterprise-grade feature engineering with deterministic transformations.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize preprocessor with cybersecurity domain logic.
        
        Args:
            seed: Random seed for deterministic splits
        """
        self.seed = seed
        np.random.seed(seed)
        
        # Define feature types based on cybersecurity domain knowledge
        self.numerical_features = [
            'failed_login_attempts',
            'login_velocity', 
            'ip_reputation_score',
            'device_trust_score',
            'time_anomaly_score'
        ]
        
        self.categorical_features = [
            'privilege_level',
            'system_criticality'
        ]
        
        self.binary_features = [
            'geo_location_change',
            'malware_indicator'
        ]
        
        # Target variable
        self.target_column = 'risk_label'
        
        # Preprocessing artifacts
        self.preprocessor = None
        self.scaler = None
        self.encoder = None
        self.feature_names = None
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and validate raw cybersecurity event data.
        
        Args:
            data_path: Path to raw CSV file
            
        Returns:
            Cleaned DataFrame with validated schema
        """
        print(f"Loading data from: {data_path}")
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Validate schema
        self._validate_data_schema(df)
        
        # Remove event_id (not a feature)
        if 'event_id' in df.columns:
            df = df.drop('event_id', axis=1)
        
        print(f"✓ Loaded {len(df):,} events with {len(df.columns)} features")
        return df
    
    def _validate_data_schema(self, df: pd.DataFrame) -> None:
        """
        Validate data schema and data quality.
        
        Raises:
            ValueError: If data doesn't meet quality standards
        """
        # Check required columns
        required_cols = set(self.numerical_features + self.categorical_features + 
                           self.binary_features + [self.target_column])
        missing_cols = required_cols - set(df.columns)
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            print(f"⚠ Missing values detected:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"  • {col}: {count} missing ({count/len(df)*100:.2f}%)")
            
            # For production, we'd implement imputation strategy
            # For this exercise, we'll drop rows with missing values
            df.dropna(inplace=True)
            print(f"  • Dropped rows with missing values")
        
        # Validate numerical ranges
        range_checks = [
            ('failed_login_attempts', (0, None)),  # Non-negative
            ('login_velocity', (0, None)),         # Non-negative
            ('ip_reputation_score', (0, 100)),     # 0-100 scale
            ('device_trust_score', (0, 100)),      # 0-100 scale
            ('time_anomaly_score', (0, 1)),        # 0-1 probability
        ]
        
        for col, (min_val, max_val) in range_checks:
            if min_val is not None and (df[col] < min_val).any():
                raise ValueError(f"{col} has values below minimum {min_val}")
            if max_val is not None and (df[col] > max_val).any():
                raise ValueError(f"{col} has values above maximum {max_val}")
        
        # Check target distribution
        target_dist = df[self.target_column].value_counts(normalize=True)
        print(f"✓ Target distribution: {target_dist[0]*100:.1f}% benign, "
              f"{target_dist[1]*100:.1f}% risky")
        
        if target_dist[1] < 0.1:  # Less than 10% risky
            print("⚠ Warning: Low prevalence of risky events may affect model performance")
    
    def _create_cybersecurity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific feature engineering for cybersecurity.
        
        Args:
            df: Original DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df_engineered = df.copy()
        
        # 1. COMPOUND RISK INDICATOR
        # Weighted combination of key risk signals
        weights = {
            'failed_login_attempts': 0.3,
            'login_velocity': 0.25,
            'ip_reputation_score': -0.2,  # Negative weight (low score = higher risk)
            'malware_indicator': 0.25
        }
        
        # Normalize features to 0-1 range for weighted combination
        df_engineered['normalized_failed_logins'] = MinMaxScaler().fit_transform(
            df_engineered[['failed_login_attempts']]
        )
        df_engineered['normalized_login_velocity'] = MinMaxScaler().fit_transform(
            df_engineered[['login_velocity']]
        )
        df_engineered['normalized_ip_reputation'] = 1 - MinMaxScaler().fit_transform(
            df_engineered[['ip_reputation_score']]
        )  # Inverse: low reputation = high risk
        
        # Calculate compound risk score
        df_engineered['compound_risk_score'] = (
            weights['failed_login_attempts'] * df_engineered['normalized_failed_logins'] +
            weights['login_velocity'] * df_engineered['normalized_login_velocity'] +
            weights['ip_reputation_score'] * df_engineered['normalized_ip_reputation'] +
            weights['malware_indicator'] * df_engineered['malware_indicator']
        )
        
        # 2. PRIVILEGE-RISK INTERACTION
        # Higher privilege + suspicious activity = higher risk
        privilege_risk_map = {
            'user': 1.0,
            'admin': 2.0,
            'super_admin': 3.0,
            'system': 4.0
        }
        
        df_engineered['privilege_risk_multiplier'] = df_engineered['privilege_level'].map(
            privilege_risk_map
        )
        
        # 3. TIME-SENSITIVE RISK
        # Higher anomaly during non-business hours (UAE time: 9 AM - 5 PM)
        # We'll create a binary feature for non-business hours
        # For simulation, we'll assume time_anomaly_score > 0.7 indicates non-business hours
        df_engineered['non_business_hours_risk'] = (
            df_engineered['time_anomaly_score'] > 0.7
        ).astype(int)
        
        # 4. DEVICE-IP CORRELATION RISK
        # Low device trust + low IP reputation = compounded risk
        df_engineered['device_ip_risk'] = (
            (100 - df_engineered['device_trust_score']) / 100 *
            (100 - df_engineered['ip_reputation_score']) / 100
        )
        
        # 5. LOGIN ATTEMPT PATTERN
        # Rate of failed attempts (failed attempts per login velocity unit)
        df_engineered['failed_per_velocity'] = np.where(
            df_engineered['login_velocity'] > 0,
            df_engineered['failed_login_attempts'] / df_engineered['login_velocity'],
            0
        )
        
        # Clip extreme values
        df_engineered['failed_per_velocity'] = np.clip(
            df_engineered['failed_per_velocity'], 0, 10
        )
        
        print(f"✓ Engineered {len(df_engineered.columns) - len(df.columns)} "
              f"new cybersecurity features")
        
        # Drop intermediate columns
        df_engineered = df_engineered.drop([
            'normalized_failed_logins',
            'normalized_login_velocity',
            'normalized_ip_reputation'
        ], axis=1, errors='ignore')
        
        return df_engineered
    
    def _build_preprocessing_pipeline(self) -> ColumnTransformer:
        """
        Build sklearn preprocessing pipeline for production use.
        
        Returns:
            Fitted ColumnTransformer
        """
        # Define preprocessing steps
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())  # Standardize to mean=0, std=1
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ])
        
        # Binary features don't need transformation
        binary_transformer = 'passthrough'
        
        # Combine all transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features),
                ('bin', binary_transformer, self.binary_features)
            ],
            remainder='drop'  # Drop any columns not explicitly handled
        )
        
        return preprocessor
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Fit preprocessing pipeline and transform data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            X_processed: Transformed feature matrix
            y: Target vector
            feature_names: List of feature names after transformation
        """
        print("=" * 60)
        print("CYBERSECURITY FEATURE ENGINEERING PIPELINE")
        print("=" * 60)
        
        # Step 1: Create domain-specific features
        print("\n1. Creating cybersecurity-specific features...")
        df_engineered = self._create_cybersecurity_features(df)
        
        # Add engineered features to feature lists
        engineered_features = [
            'compound_risk_score',
            'privilege_risk_multiplier',
            'non_business_hours_risk',
            'device_ip_risk',
            'failed_per_velocity'
        ]
        
        self.numerical_features.extend(engineered_features)
        
        # Step 2: Separate features and target
        X = df_engineered.drop(self.target_column, axis=1)
        y = df_engineered[self.target_column].values
        
        print(f"   • Original features: {len(df.columns) - 1}")
        print(f"   • Engineered features: {len(engineered_features)}")
        print(f"   • Total features: {X.shape[1]}")
        
        # Step 3: Build and fit preprocessing pipeline
        print("\n2. Building preprocessing pipeline...")
        self.preprocessor = self._build_preprocessing_pipeline()
        X_processed = self.preprocessor.fit_transform(X)
        
        # Get feature names after one-hot encoding
        self.feature_names = self._get_feature_names(self.preprocessor)
        print(f"   • Features after encoding: {len(self.feature_names)}")
        print(f"   • Data shape: {X_processed.shape}")
        
        # Step 4: Create separate scaler for risk scoring (preserves interpretability)
        print("\n3. Creating interpretable scalers...")
        self._create_interpretable_scalers(df_engineered)
        
        return X_processed, y, self.feature_names
    
    def _get_feature_names(self, preprocessor: ColumnTransformer) -> list:
        """Extract feature names after preprocessing transformations."""
        feature_names = []
        
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'num':
                # Numerical features keep their names
                feature_names.extend(columns)
            elif name == 'cat':
                # One-hot encoded features get expanded names
                encoder = transformer.named_steps['onehot']
                encoded_names = encoder.get_feature_names_out(columns)
                feature_names.extend(encoded_names)
            elif name == 'bin':
                # Binary features keep their names
                feature_names.extend(columns)
        
        return feature_names
    
    def _create_interpretable_scalers(self, df: pd.DataFrame) -> None:
        """
        Create MinMaxScalers for interpretable risk scoring.
        StandardScaler (z-score) is good for ML but hard to interpret.
        MinMaxScaler (0-1) is better for human-understandable risk scores.
        """
        # Create scaler for key risk indicators
        risk_features = [
            'failed_login_attempts',
            'login_velocity',
            'ip_reputation_score',
            'device_trust_score',
            'compound_risk_score'
        ]
        
        # Only include features that exist in the dataframe
        existing_features = [f for f in risk_features if f in df.columns]
        
        self.scaler = MinMaxScaler()
        self.scaler.fit(df[existing_features])
    
    def train_test_split(self, X: np.ndarray, y: np.ndarray, 
                        test_size: float = 0.2) -> Tuple:
        """
        Create deterministic train-test split.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion for test set
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print(f"\n4. Creating train-test split (test_size={test_size})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=self.seed,
            stratify=y  # Preserve class distribution
        )
        
        print(f"   • Training set: {X_train.shape[0]:,} samples")
        print(f"   • Test set: {X_test.shape[0]:,} samples")
        print(f"   • Positive class in train: {y_train.mean():.3f}")
        print(f"   • Positive class in test: {y_test.mean():.3f}")
        
        return X_train, X_test, y_train, y_test
    
    def save_artifacts(self, output_dir: str) -> None:
        """
        Save preprocessing artifacts for production inference.
        
        Args:
            output_dir: Directory to save artifacts
        """
        os.makedirs(output_dir, exist_ok=True)
        
        artifacts = {
            'preprocessor': self.preprocessor,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'binary_features': self.binary_features,
            'seed': self.seed
        }
        
        for name, artifact in artifacts.items():
            if artifact is not None:
                path = os.path.join(output_dir, f"{name}.pkl")
                with open(path, 'wb') as f:
                    pickle.dump(artifact, f)
                print(f"✓ Saved {name} to {path}")
    
    def save_processed_data(self, X_train: np.ndarray, X_test: np.ndarray,
                          y_train: np.ndarray, y_test: np.ndarray,
                          output_dir: str) -> None:
        """
        Save processed data arrays for ML training.
        
        Args:
            X_train, X_test, y_train, y_test: Processed data arrays
            output_dir: Directory to save processed data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as numpy arrays for efficiency
        np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
        
        print(f"✓ Saved processed data to {output_dir}")
        print(f"  • X_train: {X_train.shape}")
        print(f"  • X_test: {X_test.shape}")
        print(f"  • y_train: {y_train.shape}")
        print(f"  • y_test: {y_test.shape}")


def main():
    """Main execution for preprocessing pipeline."""
    # Define paths
    project_root = Path(__file__).parent.parent
    raw_data_path = project_root / "data" / "raw" / "security_events.csv"
    processed_data_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"
    
    # Initialize preprocessor
    preprocessor = CybersecurityPreprocessor(seed=42)
    
    try:
        # Load raw data
        df = preprocessor.load_data(raw_data_path)
        
        # Fit and transform data
        X_processed, y, feature_names = preprocessor.fit_transform(df)
        
        # Split data
        X_train, X_test, y_train, y_test = preprocessor.train_test_split(
            X_processed, y, test_size=0.2
        )
        
        # Save artifacts
        preprocessor.save_artifacts(models_dir)
        preprocessor.save_processed_data(
            X_train, X_test, y_train, y_test, 
            processed_data_dir
        )
        
        # Display feature importance preview
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING SUMMARY")
        print("=" * 60)
        print(f"\nOriginal features processed:")
        print(f"  • Numerical: {len(preprocessor.numerical_features)} features")
        print(f"  • Categorical: {len(preprocessor.categorical_features)} features")
        print(f"  • Binary: {len(preprocessor.binary_features)} features")
        
        print(f"\nEngineered cybersecurity features:")
        engineered = [
            'compound_risk_score',
            'privilege_risk_multiplier', 
            'non_business_hours_risk',
            'device_ip_risk',
            'failed_per_velocity'
        ]
        for feat in engineered:
            print(f"  • {feat}")
        
        print(f"\nTotal features after preprocessing: {len(feature_names)}")
        
        # Show sample of transformed features
        print(f"\nSample of transformed data (first 3 rows, first 5 features):")
        sample_df = pd.DataFrame(
            X_train[:3, :5], 
            columns=feature_names[:5]
        )
        print(sample_df.to_string())
        
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    main()