"""
Cybersecurity SOC Event Generator
Production-grade simulated data for ML risk prediction
Author: Principal AI/ML Engineer
Date: 2024
Version: 1.0
"""

import numpy as np
import pandas as pd
from typing import Dict
import random
import warnings
import os
import sys
from pathlib import Path

# Add project root to path for module imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

warnings.filterwarnings('ignore')

class SecurityEventGenerator:
    """
    Generates realistic SOC/SIEM-style security events with cybersecurity logic.
    All randomness is seeded for deterministic, reproducible results.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize generator with cybersecurity domain logic.
        
        Args:
            seed: Random seed for deterministic generation
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Cybersecurity context parameters
        self.risk_factors = {
            'failed_login_attempts': {
                'benign_range': (0, 3),
                'risky_range': (4, 20),
                'distribution': 'poisson'
            },
            'login_velocity': {
                'benign_range': (0.1, 2.0),  # logins per hour
                'risky_range': (3.0, 50.0),
                'distribution': 'exponential'
            },
            'ip_reputation_score': {
                'benign_range': (70, 100),   # Higher = more trustworthy
                'risky_range': (0, 69),
                'distribution': 'normal'
            },
            'device_trust_score': {
                'benign_range': (80, 100),   # Corporate-managed devices
                'risky_range': (0, 79),      # Personal/unmanaged devices
                'distribution': 'normal'
            }
        }
        
        # UAE-specific context: Enterprise/Government environments
        self.privilege_levels = ['user', 'admin', 'super_admin', 'system']
        self.geo_locations = ['AE-DXB', 'AE-AUH', 'AE-SHJ', 'INTL', 'VPN']
        self.system_criticality_levels = ['low', 'medium', 'high', 'critical']
        
    def _generate_benign_features(self, n_samples: int) -> Dict[str, np.ndarray]:
        """
        Generate features for benign (non-risky) events.
        Based on normal enterprise user behavior patterns.
        """
        features = {}
        
        # 1. Failed Login Attempts - Rare for legitimate users
        # Poisson distribution: most users have 0-1 failed attempts
        features['failed_login_attempts'] = np.random.poisson(0.5, n_samples)
        
        # 2. Login Velocity - Normal work patterns
        # Exponential with low rate for benign users
        features['login_velocity'] = np.random.exponential(0.5, n_samples)
        
        # 3. IP Reputation Score - Mostly high for corporate IPs
        # Normal distribution centered around high trust
        features['ip_reputation_score'] = np.clip(
            np.random.normal(85, 10, n_samples), 0, 100
        )
        
        # 4. Geo Location Change - Mostly within UAE
        # 90% within UAE, 10% international
        geo_changes = np.zeros(n_samples)
        intl_mask = np.random.random(n_samples) < 0.1
        geo_changes[intl_mask] = 1
        features['geo_location_change'] = geo_changes
        
        # 5. Privilege Level - Mostly regular users
        # Weighted distribution: more users, fewer admins
        weights = [0.7, 0.2, 0.08, 0.02]  # user, admin, super_admin, system
        features['privilege_level'] = np.random.choice(
            self.privilege_levels, n_samples, p=weights
        )
        
        # 6. Device Trust Score - Mostly high for corporate devices
        features['device_trust_score'] = np.clip(
            np.random.normal(90, 5, n_samples), 0, 100
        )
        
        # 7. Malware Indicator - Very rare for benign
        # Bernoulli with very low probability
        features['malware_indicator'] = np.random.binomial(1, 0.01, n_samples)
        
        # 8. System Criticality - Mixed but leaning toward lower criticality
        sys_weights = [0.4, 0.3, 0.2, 0.1]  # low, medium, high, critical
        features['system_criticality'] = np.random.choice(
            self.system_criticality_levels, n_samples, p=sys_weights
        )
        
        # 9. Time Anomaly Score - Mostly normal working hours
        # Lower scores during 9-5 UAE time
        features['time_anomaly_score'] = np.clip(
            np.random.exponential(0.3, n_samples), 0, 1
        )
        
        return features
    
    def _generate_risky_features(self, n_samples: int) -> Dict[str, np.ndarray]:
        """
        Generate features for risky events (attack patterns).
        Based on known threat behaviors and intrusion patterns.
        """
        features = {}
        
        # 1. Failed Login Attempts - High for brute force attacks
        # Poisson with higher lambda for attack patterns
        features['failed_login_attempts'] = np.random.poisson(8, n_samples)
        
        # 2. Login Velocity - Very high for credential stuffing
        features['login_velocity'] = np.random.exponential(10, n_samples)
        
        # 3. IP Reputation Score - Often low for malicious IPs
        features['ip_reputation_score'] = np.clip(
            np.random.normal(30, 20, n_samples), 0, 100
        )
        
        # 4. Geo Location Change - Frequent for compromised accounts
        # 70% have geo changes (rapid location hopping)
        geo_changes = np.ones(n_samples)
        stable_mask = np.random.random(n_samples) < 0.3
        geo_changes[stable_mask] = 0
        features['geo_location_change'] = geo_changes
        
        # 5. Privilege Level - Attackers target privileged accounts
        # Weighted toward admin accounts
        weights = [0.3, 0.4, 0.25, 0.05]  # Shifted toward admin/super_admin
        features['privilege_level'] = np.random.choice(
            self.privilege_levels, n_samples, p=weights
        )
        
        # 6. Device Trust Score - Often lower for attacker devices
        features['device_trust_score'] = np.clip(
            np.random.normal(40, 20, n_samples), 0, 100
        )
        
        # 7. Malware Indicator - More common in attacks
        features['malware_indicator'] = np.random.binomial(1, 0.4, n_samples)
        
        # 8. System Criticality - Attackers target critical systems
        sys_weights = [0.1, 0.2, 0.3, 0.4]  # Shifted toward critical
        features['system_criticality'] = np.random.choice(
            self.system_criticality_levels, n_samples, p=sys_weights
        )
        
        # 9. Time Anomaly Score - Attacks often at unusual hours
        features['time_anomaly_score'] = np.clip(
            np.random.beta(2, 5, n_samples), 0, 1
        )
        
        return features
    
    def _add_correlations(self, features: Dict[str, np.ndarray], is_risky: bool) -> Dict[str, np.ndarray]:
        """
        Add realistic correlations between features.
        Real attacks have correlated indicators.
        """
        # Create correlation between failed logins and login velocity
        if is_risky:
            # In attacks, high failed logins correlate with high velocity
            velocity_boost = features['failed_login_attempts'] * 0.5
            features['login_velocity'] += velocity_boost
            
            # Low IP reputation correlates with geo changes
            low_ip_mask = features['ip_reputation_score'] < 30
            features['geo_location_change'][low_ip_mask] = 1
            
            # Malware often on low-trust devices
            malware_mask = features['malware_indicator'] == 1
            if malware_mask.any():
                features['device_trust_score'][malware_mask] = np.clip(
                    features['device_trust_score'][malware_mask] * 0.7, 0, 100
                )
        
        return features
    
    def generate_dataset(self, n_samples: int = 10000, 
                        benign_ratio: float = 0.7) -> pd.DataFrame:
        """
        Generate complete dataset with mixed benign and risky events.
        
        Args:
            n_samples: Total number of samples to generate
            benign_ratio: Proportion of benign samples (0.7 = 70% benign)
            
        Returns:
            DataFrame with features and risk labels
        """
        # Calculate sample counts
        n_benign = int(n_samples * benign_ratio)
        n_risky = n_samples - n_benign
        
        print(f"Generating {n_benign:,} benign and {n_risky:,} risky events...")
        
        # Generate features for each class
        benign_features = self._generate_benign_features(n_benign)
        risky_features = self._generate_risky_features(n_risky)
        
        # Add realistic correlations
        benign_features = self._add_correlations(benign_features, is_risky=False)
        risky_features = self._add_correlations(risky_features, is_risky=True)
        
        # Combine features and create labels
        all_features = {}
        for feature_name in benign_features.keys():
            all_features[feature_name] = np.concatenate([
                benign_features[feature_name], 
                risky_features[feature_name]
            ])
        
        # Create labels: 0 = benign, 1 = risky
        labels = np.concatenate([
            np.zeros(n_benign, dtype=int),
            np.ones(n_risky, dtype=int)
        ])
        
        # Create DataFrame
        df = pd.DataFrame(all_features)
        df['risk_label'] = labels
        
        # Shuffle the dataset while maintaining seed reproducibility
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        # Add unique event ID for traceability
        df['event_id'] = [f"EVENT_{i:06d}" for i in range(len(df))]
        
        # Reorder columns for clarity
        column_order = ['event_id', 'failed_login_attempts', 'login_velocity', 
                       'ip_reputation_score', 'geo_location_change', 'privilege_level',
                       'device_trust_score', 'malware_indicator', 'system_criticality',
                       'time_anomaly_score', 'risk_label']
        df = df[column_order]
        
        return df
    
    def validate_dataset(self, df: pd.DataFrame) -> bool:
        """
        Validate generated dataset for quality and realism.
        
        Returns:
            True if validation passes, raises ValueError otherwise
        """
        # Check expected columns
        required_columns = [
            'event_id', 'failed_login_attempts', 'login_velocity', 
            'ip_reputation_score', 'geo_location_change', 'privilege_level',
            'device_trust_score', 'malware_indicator', 'system_criticality',
            'time_anomaly_score', 'risk_label'
        ]
        
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        # Check data types and ranges
        validations = [
            ('failed_login_attempts', lambda x: (x >= 0).all(), "Negative failed logins"),
            ('login_velocity', lambda x: (x >= 0).all(), "Negative login velocity"),
            ('ip_reputation_score', lambda x: ((x >= 0) & (x <= 100)).all(), 
             "IP reputation outside 0-100"),
            ('device_trust_score', lambda x: ((x >= 0) & (x <= 100)).all(),
             "Device trust outside 0-100"),
            ('time_anomaly_score', lambda x: ((x >= 0) & (x <= 1)).all(),
             "Time anomaly outside 0-1"),
            ('risk_label', lambda x: set(x.unique()) == {0, 1},
             "Invalid risk labels"),
        ]
        
        for col, check, msg in validations:
            if not check(df[col]):
                raise ValueError(f"Validation failed: {msg}")
        
        # Check class distribution
        benign_pct = (df['risk_label'] == 0).mean() * 100
        risky_pct = (df['risk_label'] == 1).mean() * 100
        print(f"✓ Class distribution: {benign_pct:.1f}% benign, {risky_pct:.1f}% risky")
        
        # Check feature correlations with risk (should make sense)
        risky_mean_ip = df.loc[df['risk_label'] == 1, 'ip_reputation_score'].mean()
        benign_mean_ip = df.loc[df['risk_label'] == 0, 'ip_reputation_score'].mean()
        print(f"✓ IP Reputation: Risky={risky_mean_ip:.1f}, Benign={benign_mean_ip:.1f}")
        
        if risky_mean_ip >= benign_mean_ip:
            print("⚠ Warning: Risky events don't have lower IP reputation")
        
        return True


def main():
    """Main execution for data generation."""
    # Define the correct absolute path
    project_root = Path(__file__).parent.parent
    raw_data_path = project_root / "data" / "raw" / "security_events.csv"
    
    # Create data directory if it doesn't exist
    raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator with seed for reproducibility
    print("=" * 60)
    print("CYBERSECURITY EVENT GENERATOR")
    print("=" * 60)
    print("Configuration:")
    print(f"  • Python: 3.12.7")
    print(f"  • Seed: 42 (deterministic)")
    print(f"  • Samples: 10,000")
    print(f"  • Target: 70% benign, 30% risky")
    print(f"  • Output: {raw_data_path}")
    print("-" * 60)
    
    # Generate dataset
    generator = SecurityEventGenerator(seed=42)
    df = generator.generate_dataset(n_samples=10000, benign_ratio=0.7)
    
    # Validate
    if generator.validate_dataset(df):
        # Save to CSV
        df.to_csv(raw_data_path, index=False)
        
        print("-" * 60)
        print("✓ Dataset successfully generated and validated")
        print(f"  • Saved to: {raw_data_path}")
        print(f"  • Size: {len(df):,} rows x {len(df.columns)} columns")
        print(f"  • Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Show sample
        print("\nSample of generated data (first 3 rows):")
        print(df.head(3).to_string())
        
        # Show key statistics
        print("\nKey Statistics:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats_df = df[numeric_cols].describe().round(2)
        print(stats_df.loc[['mean', 'std', 'min', 'max']])
        
        print("=" * 60)
        
        return df


if __name__ == "__main__":
    main()