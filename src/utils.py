"""
Utility functions for cybersecurity risk prediction system.
Shared functionality across all modules.
Author: Principal AI/ML Engineer
Date: 2024
Version: 1.0
"""

import os
import sys
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cybersecurity_risk.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class Config:
    """Configuration management for the cybersecurity system."""
    
    # Project structure
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    REPORTS_DIR = PROJECT_ROOT / "reports"
    SRC_DIR = PROJECT_ROOT / "src"
    
    # Data paths
    RAW_DATA_PATH = DATA_DIR / "raw" / "security_events.csv"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    
    # Model paths
    MODEL_PATH = MODELS_DIR / "risk_classifier.pkl"
    PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
    FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.pkl"
    SCALER_PATH = MODELS_DIR / "scaler.pkl"
    
    # System configuration
    SEED = 42
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.1
    
    # Risk thresholds (SOC-aligned)
    RISK_THRESHOLDS = {
        'LOW': 0.30,
        'MEDIUM': 0.60,
        'HIGH': 0.85,
        'CRITICAL': 0.95
    }
    
    # UAE enterprise context
    UAE_CONTEXT = {
        'business_hours': {'start': 9, 'end': 17},  # 9 AM - 5 PM UAE time
        'critical_systems': ['AD_DC', 'SCADA', 'SWIFT', 'ERP'],
        'compliance_frameworks': ['UAE_IA', 'NESA', 'ISO27001', 'GDPR']
    }
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        directories = [
            cls.DATA_DIR / "raw",
            cls.DATA_DIR / "processed",
            cls.MODELS_DIR,
            cls.REPORTS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")


def setup_environment() -> None:
    """Setup project environment and verify dependencies."""
    logger.info("Setting up cybersecurity risk prediction environment")
    
    # Ensure directories
    Config.ensure_directories()
    
    # Verify Python version
    required_python = (3, 12, 7)
    current_python = sys.version_info
    
    if current_python < required_python:
        raise RuntimeError(
            f"Python {required_python[0]}.{required_python[1]}.{required_python[2]} or higher required. "
            f"Current version: {current_python[0]}.{current_python[1]}.{current_python[2]}"
        )
    
    logger.info(f"✓ Python version: {current_python[0]}.{current_python[1]}.{current_python[2]}")
    logger.info(f"✓ Project root: {Config.PROJECT_ROOT}")
    logger.info("Environment setup complete")


def save_artifact(artifact: Any, filepath: Union[str, Path]) -> None:
    """
    Save an artifact (model, scaler, etc.) to disk.
    
    Args:
        artifact: Object to save
        filepath: Path to save the artifact
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(artifact, f)
    
    size_mb = filepath.stat().st_size / (1024 * 1024)
    logger.info(f"Saved artifact to {filepath} ({size_mb:.2f} MB)")


def load_artifact(filepath: Union[str, Path]) -> Any:
    """
    Load an artifact from disk.
    
    Args:
        filepath: Path to the artifact file
        
    Returns:
        Loaded artifact
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Artifact not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        artifact = pickle.load(f)
    
    logger.info(f"Loaded artifact from {filepath}")
    return artifact


def save_json(data: Dict, filepath: Union[str, Path], indent: int = 2) -> None:
    """
    Save data as JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save JSON file
        indent: JSON indentation level
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    
    logger.info(f"Saved JSON to {filepath}")


def load_json(filepath: Union[str, Path]) -> Dict:
    """
    Load data from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded JSON from {filepath}")
    return data


def generate_event_id(prefix: str = "EVENT") -> str:
    """
    Generate a unique event ID.
    
    Args:
        prefix: Prefix for the event ID
        
    Returns:
        Unique event ID string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = np.random.randint(1000, 9999)
    
    return f"{prefix}_{timestamp}_{random_suffix}"


def validate_features(features: Dict[str, Any]) -> bool:
    """
    Validate feature dictionary for cybersecurity events.
    
    Args:
        features: Dictionary of features to validate
        
    Returns:
        True if features are valid, raises ValueError otherwise
    """
    required_features = [
        'failed_login_attempts',
        'login_velocity',
        'ip_reputation_score',
        'geo_location_change',
        'privilege_level',
        'device_trust_score',
        'malware_indicator',
        'system_criticality',
        'time_anomaly_score'
    ]
    
    # Check required features
    missing_features = set(required_features) - set(features.keys())
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Validate ranges
    validations = [
        ('failed_login_attempts', (0, 50), "Failed login attempts must be between 0-50"),
        ('login_velocity', (0, 100), "Login velocity must be between 0-100"),
        ('ip_reputation_score', (0, 100), "IP reputation must be between 0-100"),
        ('device_trust_score', (0, 100), "Device trust must be between 0-100"),
        ('time_anomaly_score', (0, 1), "Time anomaly must be between 0-1"),
        ('geo_location_change', {0, 1}, "Geo location change must be 0 or 1"),
        ('malware_indicator', {0, 1}, "Malware indicator must be 0 or 1"),
    ]
    
    for feature, valid_range, error_msg in validations:
        value = features[feature]
        
        if isinstance(valid_range, tuple):
            # Numeric range
            if not (valid_range[0] <= value <= valid_range[1]):
                raise ValueError(f"{error_msg}. Got {value}")
        elif isinstance(valid_range, set):
            # Set of valid values
            if value not in valid_range:
                raise ValueError(f"{error_msg}. Got {value}")
    
    # Validate categorical features
    valid_privilege = {'user', 'admin', 'super_admin', 'system'}
    valid_criticality = {'low', 'medium', 'high', 'critical'}
    
    if features['privilege_level'] not in valid_privilege:
        raise ValueError(f"Invalid privilege level: {features['privilege_level']}")
    
    if features['system_criticality'] not in valid_criticality:
        raise ValueError(f"Invalid system criticality: {features['system_criticality']}")
    
    logger.debug(f"Feature validation passed for {len(features)} features")
    return True


def calculate_performance_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary of performance metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['roc_auc'] = 0.5
    
    # Confusion matrix components
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics.update({
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
    })
    
    logger.info(f"Calculated performance metrics: Accuracy={metrics['accuracy']:.3f}, "
                f"Recall={metrics['recall']:.3f}")
    
    return metrics


def format_risk_explanation(risk_level: str, probability: float, 
                           top_features: List[Dict]) -> str:
    """
    Format human-readable risk explanation.
    
    Args:
        risk_level: Risk level (LOW/MEDIUM/HIGH/CRITICAL)
        probability: Risk probability
        top_features: List of top contributing features
        
    Returns:
        Formatted explanation string
    """
    explanations = {
        'LOW': f"Low risk ({probability:.1%} probability). Normal user behavior detected.",
        'MEDIUM': f"Medium risk ({probability:.1%} probability). Suspicious activity requiring review.",
        'HIGH': f"High risk ({probability:.1%} probability). Likely security incident requiring investigation.",
        'CRITICAL': f"Critical risk ({probability:.1%} probability). Immediate response required."
    }
    
    explanation = explanations.get(risk_level, "Risk assessment completed.")
    
    if top_features:
        explanation += "\n\nTop contributing factors:\n"
        for i, feat in enumerate(top_features[:3], 1):
            explanation += f"{i}. {feat.get('feature', 'Unknown')}: {feat.get('explanation', '')}\n"
    
    return explanation


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for logging and debugging.
    
    Returns:
        Dictionary with system information
    """
    import platform
    
    info = {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'platform': platform.platform(),
        'processor': platform.processor(),
        'system': platform.system(),
        'machine': platform.machine(),
        'project_root': str(Config.PROJECT_ROOT),
        'requirements_met': check_requirements()
    }
    
    return info


def check_requirements() -> bool:
    """
    Check if required packages are installed.
    
    Returns:
        True if all requirements are met
    """
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'scipy',
        'streamlit', 'plotly', 'matplotlib'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        logger.warning(f"Missing packages: {missing}")
        return False
    
    return True


def log_operation(operation: str, details: Dict[str, Any], 
                  level: str = 'INFO') -> None:
    """
    Log an operation with structured details.
    
    Args:
        operation: Name of the operation
        details: Dictionary with operation details
        level: Log level (INFO, WARNING, ERROR)
    """
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'operation': operation,
        'details': details
    }
    
    log_message = f"{operation}: {json.dumps(details, default=str)}"
    
    if level == 'INFO':
        logger.info(log_message)
    elif level == 'WARNING':
        logger.warning(log_message)
    elif level == 'ERROR':
        logger.error(log_message)
    
    # Also save to operations log file
    ops_log_path = Config.REPORTS_DIR / "operations_log.jsonl"
    ops_log_path.parent.mkdir(exist_ok=True)
    
    with open(ops_log_path, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


# Initialize environment on import
setup_environment()