"""
Cybersecurity Risk Scoring Engine
Converts ML probabilities to actionable SOC risk levels
Author: Principal AI/ML Engineer
Date: 2024
Version: 1.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Union
import pickle
import json
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import warnings
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

warnings.filterwarnings('ignore')


class RiskLevel(str, Enum):
    """Standardized risk levels for SOC operations."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"  # Reserved for manual escalation


@dataclass
class RiskAssessment:
    """Complete risk assessment for a security event."""
    event_id: str
    probability: float
    risk_level: RiskLevel
    confidence: float
    threshold_used: float
    top_features: List[Dict[str, float]]
    timestamp: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'event_id': self.event_id,
            'probability': round(self.probability, 4),
            'risk_level': self.risk_level.value,
            'confidence': round(self.confidence, 4),
            'threshold_used': round(self.threshold_used, 4),
            'top_features': self.top_features,
            'timestamp': self.timestamp,
            'recommended_action': self.get_recommended_action()
        }
    
    def get_recommended_action(self) -> str:
        """Get SOC action based on risk level."""
        actions = {
            RiskLevel.LOW: "Monitor only - normal user behavior",
            RiskLevel.MEDIUM: "Review log details - potential suspicious activity",
            RiskLevel.HIGH: "Immediate investigation - likely security incident",
            RiskLevel.CRITICAL: "Emergency response - active threat detected"
        }
        return actions.get(self.risk_level, "No action defined")


class RiskScoringEngine:
    """
    Production-grade risk scoring engine for cybersecurity events.
    Converts ML probabilities to actionable SOC risk levels with business logic.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize risk scoring engine with SOC-aligned thresholds.
        
        Args:
            seed: Random seed for deterministic operations
        """
        self.seed = seed
        np.random.seed(seed)
        
        # Load trained model and preprocessing artifacts
        self.models_dir = project_root / "models"
        self._load_artifacts()
        
        # Define required input features (from data generator)
        self.required_features = [
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
        
        # SOC-ALIGNED RISK THRESHOLDS
        # Based on UAE enterprise risk appetite and SOC operational guidelines
        self.risk_thresholds = {
            'LOW': 0.30,    # Below 30% probability - normal behavior
            'MEDIUM': 0.60,  # 30-60% probability - suspicious, needs review
            'HIGH': 0.85,    # 60-85% probability - likely attack
            'CRITICAL': 0.95  # Above 85% - confirmed threat (auto-escalation)
        }
        
        # Confidence score parameters
        self.confidence_params = {
            'high_confidence_range': (0.2, 0.8),  # Probabilities in this range are less confident
            'calibration_factor': 0.9  # Adjust confidence based on model calibration
        }
        
        # Business context multipliers (UAE enterprise specific)
        self.business_context = {
            'privilege_multipliers': {
                'user': 1.0,
                'admin': 1.5,
                'super_admin': 2.0,
                'system': 3.0
            },
            'criticality_multipliers': {
                'low': 1.0,
                'medium': 1.3,
                'high': 1.7,
                'critical': 2.5
            },
            'time_multipliers': {
                'business_hours': 1.0,      # 9 AM - 5 PM UAE time
                'after_hours': 1.2,         # 5 PM - 10 PM
                'off_hours': 1.5,           # 10 PM - 6 AM
                'weekend': 1.8              # Weekends and holidays
            }
        }
        
        # SOC operational parameters
        self.soc_guidelines = {
            'max_daily_alerts': 100,  # Avoid alert fatigue
            'high_risk_escalation_time': '15 minutes',
            'medium_risk_review_time': '4 hours',
            'false_positive_tolerance': 0.20,  # Max 20% false positives
        }
        
        print("=" * 60)
        print("RISK SCORING ENGINE INITIALIZED")
        print("=" * 60)
        print(f"Risk Thresholds: LOW<{self.risk_thresholds['LOW']}, "
              f"MEDIUM<{self.risk_thresholds['MEDIUM']}, "
              f"HIGH<{self.risk_thresholds['HIGH']}, "
              f"CRITICAL>={self.risk_thresholds['CRITICAL']}")
        print(f"Business Context: UAE Enterprise Grade")
        print(f"SOC Guidelines: Max {self.soc_guidelines['max_daily_alerts']} alerts/day")
    
    def _load_artifacts(self) -> None:
        """Load trained model and preprocessing artifacts."""
        try:
            # Load model
            with open(self.models_dir / "risk_classifier.pkl", 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.model_metadata = model_data.get('metadata', {})
            
            # Load preprocessor
            with open(self.models_dir / "preprocessor.pkl", 'rb') as f:
                self.preprocessor = pickle.load(f)
            
            # Load feature names
            with open(self.models_dir / "feature_names.pkl", 'rb') as f:
                self.feature_names = pickle.load(f)
            
            # Load scaler for interpretable scoring
            with open(self.models_dir / "scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            
            print("✓ Loaded trained model and preprocessing artifacts")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Required artifacts not found. Run model_training.py first.\n"
                f"Missing: {e.filename}"
            )
    
    def _create_engineered_features(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create engineered features for inference (same as training).
        
        Args:
            event_data: Raw event features
            
        Returns:
            Event data with engineered features added
        """
        event_with_engineered = event_data.copy()
        
        # 1. COMPOUND RISK INDICATOR
        # Same weights as in preprocessing.py
        weights = {
            'failed_login_attempts': 0.3,
            'login_velocity': 0.25,
            'ip_reputation_score': -0.2,  # Negative weight (low score = higher risk)
            'malware_indicator': 0.25
        }
        
        # Normalize features to 0-1 range for weighted combination
        # Simple min-max normalization using reasonable bounds
        normalized_failed_logins = min(event_data['failed_login_attempts'] / 20, 1.0)
        normalized_login_velocity = min(event_data['login_velocity'] / 50, 1.0)
        normalized_ip_reputation = 1 - (event_data['ip_reputation_score'] / 100)
        
        # Calculate compound risk score
        event_with_engineered['compound_risk_score'] = (
            weights['failed_login_attempts'] * normalized_failed_logins +
            weights['login_velocity'] * normalized_login_velocity +
            weights['ip_reputation_score'] * normalized_ip_reputation +
            weights['malware_indicator'] * event_data['malware_indicator']
        )
        
        # 2. PRIVILEGE-RISK INTERACTION
        privilege_risk_map = {
            'user': 1.0,
            'admin': 2.0,
            'super_admin': 3.0,
            'system': 4.0
        }
        
        event_with_engineered['privilege_risk_multiplier'] = privilege_risk_map.get(
            event_data['privilege_level'], 1.0
        )
        
        # 3. TIME-SENSITIVE RISK
        # Higher anomaly during non-business hours
        event_with_engineered['non_business_hours_risk'] = (
            1 if event_data['time_anomaly_score'] > 0.7 else 0
        )
        
        # 4. DEVICE-IP CORRELATION RISK
        # Low device trust + low IP reputation = compounded risk
        device_trust_normalized = (100 - event_data['device_trust_score']) / 100
        ip_reputation_normalized = (100 - event_data['ip_reputation_score']) / 100
        event_with_engineered['device_ip_risk'] = (
            device_trust_normalized * ip_reputation_normalized
        )
        
        # 5. LOGIN ATTEMPT PATTERN
        # Rate of failed attempts (failed attempts per login velocity unit)
        if event_data['login_velocity'] > 0:
            failed_per_velocity = event_data['failed_login_attempts'] / event_data['login_velocity']
        else:
            failed_per_velocity = 0
        
        # Clip extreme values
        event_with_engineered['failed_per_velocity'] = min(failed_per_velocity, 10)
        
        return event_with_engineered
    
    def _validate_input_features(self, event_data: Dict[str, Any]) -> None:
        """
        Validate that all required features are present.
        
        Args:
            event_data: Input event features
            
        Raises:
            ValueError: If required features are missing
        """
        missing_features = set(self.required_features) - set(event_data.keys())
        
        if missing_features:
            raise ValueError(
                f"Missing required features: {missing_features}. "
                f"Required features: {self.required_features}"
            )
        
        # Validate feature ranges
        validations = [
            ('failed_login_attempts', (0, 50), "Failed login attempts must be ≥ 0"),
            ('login_velocity', (0, 100), "Login velocity must be ≥ 0"),
            ('ip_reputation_score', (0, 100), "IP reputation must be between 0-100"),
            ('device_trust_score', (0, 100), "Device trust must be between 0-100"),
            ('time_anomaly_score', (0, 1), "Time anomaly must be between 0-1"),
            ('geo_location_change', (0, 1), "Geo change must be 0 or 1"),
            ('malware_indicator', (0, 1), "Malware indicator must be 0 or 1"),
        ]
        
        for feature, (min_val, max_val), error_msg in validations:
            if feature in event_data:
                value = event_data[feature]
                if not (min_val <= value <= max_val):
                    raise ValueError(f"{error_msg}. Got {value}")
    
    def _calculate_confidence(self, probability: float) -> float:
        """
        Calculate confidence score for probability prediction.
        
        Higher confidence for probabilities near 0 or 1.
        Lower confidence for probabilities near 0.5.
        
        Args:
            probability: Raw model probability (0-1)
            
        Returns:
            Confidence score (0-1)
        """
        # Distance from 0.5 (uncertainty point)
        distance_from_uncertainty = abs(probability - 0.5)
        
        # Sigmoid transformation to emphasize extremes
        confidence = 2 * distance_from_uncertainty
        
        # Adjust based on model calibration
        confidence *= self.confidence_params['calibration_factor']
        
        # Clip to valid range
        return np.clip(confidence, 0, 1)
    
    def _apply_business_context(self, 
                               raw_probability: float,
                               features: Dict[str, Any]) -> float:
        """
        Adjust probability based on business context.
        
        Args:
            raw_probability: Model prediction (0-1)
            features: Original feature values
            
        Returns:
            Context-adjusted probability
        """
        adjusted_probability = raw_probability
        
        # 1. Apply privilege multiplier
        if 'privilege_level' in features:
            privilege = features['privilege_level']
            multiplier = self.business_context['privilege_multipliers'].get(
                privilege, 1.0
            )
            adjusted_probability = self._apply_multiplier(
                adjusted_probability, multiplier
            )
        
        # 2. Apply system criticality multiplier
        if 'system_criticality' in features:
            criticality = features['system_criticality']
            multiplier = self.business_context['criticality_multipliers'].get(
                criticality, 1.0
            )
            adjusted_probability = self._apply_multiplier(
                adjusted_probability, multiplier
            )
        
        # 3. Apply time-based multiplier (simplified - would use actual timestamps in production)
        # Assuming time_anomaly_score indicates unusual hours
        if 'time_anomaly_score' in features:
            anomaly_score = features['time_anomaly_score']
            if anomaly_score > 0.7:  # Unusual hours
                multiplier = self.business_context['time_multipliers']['off_hours']
                adjusted_probability = self._apply_multiplier(
                    adjusted_probability, multiplier
                )
        
        # Clip to valid probability range
        return np.clip(adjusted_probability, 0, 1)
    
    def _apply_multiplier(self, probability: float, multiplier: float) -> float:
        """
        Apply context multiplier to probability.
        Uses logarithmic scaling to prevent extreme values.
        
        Args:
            probability: Base probability
            multiplier: Context multiplier
            
        Returns:
            Adjusted probability
        """
        if multiplier == 1.0:
            return probability
        
        # Use sigmoid-like transformation for multiplier effect
        if probability < 0.5:
            # For low probabilities, multiplier has less effect
            adjustment = (multiplier - 1) * probability * 0.5
        else:
            # For high probabilities, multiplier has more effect
            adjustment = (multiplier - 1) * probability
        
        adjusted = probability + adjustment
        return min(adjusted, 0.99)  # Cap at 0.99
    
    def _determine_risk_level(self, 
                             probability: float, 
                             confidence: float) -> Tuple[RiskLevel, float]:
        """
        Determine risk level based on probability and confidence.
        
        Args:
            probability: Adjusted probability (0-1)
            confidence: Confidence score (0-1)
            
        Returns:
            RiskLevel and effective threshold used
        """
        # Adjust thresholds based on confidence
        # Lower confidence = more conservative thresholds
        confidence_adjustment = 1.0 - (confidence * 0.2)  # Up to 20% adjustment
        
        adjusted_thresholds = {
            level: threshold * confidence_adjustment
            for level, threshold in self.risk_thresholds.items()
        }
        
        # Determine risk level
        if probability >= adjusted_thresholds['CRITICAL']:
            risk_level = RiskLevel.CRITICAL
            threshold_used = adjusted_thresholds['CRITICAL']
        elif probability >= adjusted_thresholds['HIGH']:
            risk_level = RiskLevel.HIGH
            threshold_used = adjusted_thresholds['HIGH']
        elif probability >= adjusted_thresholds['MEDIUM']:
            risk_level = RiskLevel.MEDIUM
            threshold_used = adjusted_thresholds['MEDIUM']
        else:
            risk_level = RiskLevel.LOW
            threshold_used = adjusted_thresholds['LOW']
        
        return risk_level, threshold_used
    
    def _extract_top_features(self, 
                             features_array: np.ndarray,
                             feature_values: Dict[str, Any],
                             top_n: int = 3) -> List[Dict[str, float]]:
        """
        Extract top contributing features for explainability.
        
        Args:
            features_array: Preprocessed feature array
            feature_values: Original feature values
            top_n: Number of top features to return
            
        Returns:
            List of dictionaries with feature names and contributions
        """
        if not hasattr(self.model, 'feature_importances_'):
            return []
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Match with feature names
        feature_importance_map = {
            name: importance 
            for name, importance in zip(self.feature_names, importances)
        }
        
        # Sort by importance
        sorted_features = sorted(
            feature_importance_map.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top N features with their original values
        top_features = []
        for feature_name, importance in sorted_features[:top_n]:
            # Find original feature value
            original_value = None
            
            # Check if this is an encoded categorical feature
            for orig_feature, value in feature_values.items():
                if orig_feature == feature_name:
                    original_value = value
                    break
                elif feature_name.startswith(f"{orig_feature}_"):
                    # This is a one-hot encoded feature
                    original_value = value
                    break
            
            top_features.append({
                'feature': feature_name,
                'importance': round(float(importance), 4),
                'original_value': original_value
            })
        
        return top_features
    
    def preprocess_event(self, event_data: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess single event for model prediction.
        
        Args:
            event_data: Dictionary with raw event features
            
        Returns:
            Preprocessed feature array
        """
        # Validate input features
        self._validate_input_features(event_data)
        
        # Create engineered features (same as training)
        event_with_engineered = self._create_engineered_features(event_data)
        
        # Convert to DataFrame for preprocessing
        df = pd.DataFrame([event_with_engineered])
        
        # Apply preprocessing pipeline
        processed = self.preprocessor.transform(df)
        
        return processed
    
    def calculate_risk(self, 
                      event_data: Dict[str, Any],
                      event_id: str = None) -> RiskAssessment:
        """
        Calculate complete risk assessment for a security event.
        
        Args:
            event_data: Dictionary with event features
            event_id: Optional event identifier
            
        Returns:
            Complete RiskAssessment object
        """
        # Generate event ID if not provided
        if event_id is None:
            event_id = f"EV_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
        
        try:
            # Step 1: Preprocess event
            processed_features = self.preprocess_event(event_data)
            
            # Step 2: Get raw model probability
            raw_probability = self.model.predict_proba(processed_features)[0, 1]
            
            # Step 3: Calculate confidence
            confidence = self._calculate_confidence(raw_probability)
            
            # Step 4: Apply business context adjustments
            adjusted_probability = self._apply_business_context(
                raw_probability, event_data
            )
            
            # Step 5: Determine risk level
            risk_level, threshold_used = self._determine_risk_level(
                adjusted_probability, confidence
            )
            
            # Step 6: Extract top contributing features
            top_features = self._extract_top_features(
                processed_features, event_data
            )
            
            # Step 7: Create assessment
            assessment = RiskAssessment(
                event_id=event_id,
                probability=adjusted_probability,
                risk_level=risk_level,
                confidence=confidence,
                threshold_used=threshold_used,
                top_features=top_features,
                timestamp=datetime.now().isoformat()
            )
            
            return assessment
            
        except Exception as e:
            raise ValueError(f"Risk calculation failed: {str(e)}")
    
    def batch_assess_risk(self, 
                         events_data: List[Dict[str, Any]]) -> List[RiskAssessment]:
        """
        Calculate risk for multiple events efficiently.
        
        Args:
            events_data: List of event feature dictionaries
            
        Returns:
            List of RiskAssessment objects
        """
        assessments = []
        
        print(f"Processing {len(events_data)} events for risk assessment...")
        
        for i, event_data in enumerate(events_data):
            event_id = event_data.get('event_id', f"BATCH_{i+1:06d}")
            
            try:
                assessment = self.calculate_risk(event_data, event_id)
                assessments.append(assessment)
                
                # Progress indicator
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(events_data)} events...")
                    
            except Exception as e:
                print(f"⚠ Failed to assess event {event_id}: {str(e)}")
                # Create a fallback assessment
                assessments.append(RiskAssessment(
                    event_id=event_id,
                    probability=0.0,
                    risk_level=RiskLevel.LOW,
                    confidence=0.0,
                    threshold_used=self.risk_thresholds['LOW'],
                    top_features=[],
                    timestamp=datetime.now().isoformat()
                ))
        
        return assessments
    
    def generate_risk_report(self, 
                           assessments: List[RiskAssessment]) -> Dict[str, Any]:
        """
        Generate comprehensive risk report for SOC.
        
        Args:
            assessments: List of risk assessments
            
        Returns:
            Dictionary with risk summary and statistics
        """
        if not assessments:
            return {}
        
        # Extract probabilities and risk levels
        probabilities = [a.probability for a in assessments]
        risk_levels = [a.risk_level for a in assessments]
        
        # Count by risk level
        risk_counts = {
            RiskLevel.LOW: 0,
            RiskLevel.MEDIUM: 0,
            RiskLevel.HIGH: 0,
            RiskLevel.CRITICAL: 0
        }
        
        for level in risk_levels:
            risk_counts[level] += 1
        
        # Calculate statistics
        total_events = len(assessments)
        high_risk_pct = (risk_counts[RiskLevel.HIGH] + risk_counts[RiskLevel.CRITICAL]) / total_events * 100
        
        report = {
            'summary': {
                'total_events': total_events,
                'timestamp': datetime.now().isoformat(),
                'model_version': self.model_metadata.get('training_date', 'unknown'),
                'risk_thresholds': self.risk_thresholds
            },
            'statistics': {
                'mean_probability': float(np.mean(probabilities)),
                'std_probability': float(np.std(probabilities)),
                'min_probability': float(np.min(probabilities)),
                'max_probability': float(np.max(probabilities)),
                'high_risk_percentage': round(high_risk_pct, 2)
            },
            'risk_distribution': {
                level.value: {
                    'count': count,
                    'percentage': round(count / total_events * 100, 2)
                }
                for level, count in risk_counts.items()
            },
            'soc_alert_analysis': {
                'estimated_daily_alerts': risk_counts[RiskLevel.HIGH] + risk_counts[RiskLevel.CRITICAL],
                'within_capacity': (risk_counts[RiskLevel.HIGH] + risk_counts[RiskLevel.CRITICAL]) <= self.soc_guidelines['max_daily_alerts'],
                'recommended_actions': self._generate_soc_recommendations(risk_counts)
            }
        }
        
        return report
    
    def _generate_soc_recommendations(self, 
                                    risk_counts: Dict[RiskLevel, int]) -> List[str]:
        """
        Generate SOC operational recommendations.
        
        Args:
            risk_counts: Count of events by risk level
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        total_high_risk = risk_counts[RiskLevel.HIGH] + risk_counts[RiskLevel.CRITICAL]
        
        if total_high_risk == 0:
            recommendations.append("No high-risk events detected. Continue routine monitoring.")
        
        elif total_high_risk > self.soc_guidelines['max_daily_alerts']:
            recommendations.append(
                f"⚠ ALERT FATIGUE RISK: {total_high_risk} high-risk events exceed "
                f"daily capacity of {self.soc_guidelines['max_daily_alerts']}. "
                f"Consider adjusting thresholds or adding analysts."
            )
        
        if risk_counts[RiskLevel.CRITICAL] > 0:
            recommendations.append(
                f"🚨 CRITICAL ALERTS: {risk_counts[RiskLevel.CRITICAL]} events require "
                f"immediate escalation within {self.soc_guidelines['high_risk_escalation_time']}."
            )
        
        if risk_counts[RiskLevel.HIGH] > 10:
            recommendations.append(
                f"🔴 HIGH VOLUME: {risk_counts[RiskLevel.HIGH]} high-risk events detected. "
                f"Consider potential coordinated attack."
            )
        
        # Check for threshold effectiveness
        low_risk_pct = risk_counts[RiskLevel.LOW] / sum(risk_counts.values()) * 100
        if low_risk_pct > 80:
            recommendations.append(
                f"📊 THRESHOLD REVIEW: {low_risk_pct:.1f}% events marked as LOW risk. "
                f"Consider lowering thresholds to catch more suspicious activity."
            )
        
        return recommendations
    
    def save_assessments(self, 
                        assessments: List[RiskAssessment],
                        output_path: str) -> None:
        """
        Save risk assessments to JSON file.
        
        Args:
            assessments: List of risk assessments
            output_path: Path to save JSON file
        """
        # Convert to dictionaries
        assessments_dict = [a.to_dict() for a in assessments]
        
        # Add metadata
        output_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'risk_engine_version': '1.0',
                'model_version': self.model_metadata.get('training_date', 'unknown'),
                'total_assessments': len(assessments)
            },
            'assessments': assessments_dict
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Saved {len(assessments)} assessments to {output_path}")


def main():
    """Demonstrate risk scoring engine with sample data."""
    print("=" * 60)
    print("RISK SCORING ENGINE DEMONSTRATION")
    print("=" * 60)
    
    # Initialize engine
    engine = RiskScoringEngine(seed=42)
    
    # Create sample events for demonstration
    sample_events = [
        {
            'event_id': 'SAMPLE_BENIGN_001',
            'failed_login_attempts': 1,
            'login_velocity': 0.5,
            'ip_reputation_score': 85,
            'geo_location_change': 0,
            'privilege_level': 'user',
            'device_trust_score': 92,
            'malware_indicator': 0,
            'system_criticality': 'low',
            'time_anomaly_score': 0.2
        },
        {
            'event_id': 'SAMPLE_SUSPICIOUS_001',
            'failed_login_attempts': 5,
            'login_velocity': 8.2,
            'ip_reputation_score': 45,
            'geo_location_change': 1,
            'privilege_level': 'admin',
            'device_trust_score': 60,
            'malware_indicator': 0,
            'system_criticality': 'medium',
            'time_anomaly_score': 0.6
        },
        {
            'event_id': 'SAMPLE_HIGH_RISK_001',
            'failed_login_attempts': 12,
            'login_velocity': 25.7,
            'ip_reputation_score': 15,
            'geo_location_change': 1,
            'privilege_level': 'super_admin',
            'device_trust_score': 30,
            'malware_indicator': 1,
            'system_criticality': 'critical',
            'time_anomaly_score': 0.9
        }
    ]
    
    # Process each sample event
    print("\n" + "=" * 60)
    print("INDIVIDUAL EVENT ASSESSMENTS")
    print("=" * 60)
    
    assessments = []
    for event in sample_events:
        assessment = engine.calculate_risk(event)
        assessments.append(assessment)
        
        print(f"\n📊 Event: {event['event_id']}")
        print(f"   Probability: {assessment.probability:.3f}")
        print(f"   Risk Level: {assessment.risk_level.value}")
        print(f"   Confidence: {assessment.confidence:.3f}")
        print(f"   Action: {assessment.get_recommended_action()}")
        
        if assessment.top_features:
            print("   Top Features:")
            for feat in assessment.top_features[:2]:
                print(f"     • {feat['feature']}: importance={feat['importance']:.3f}")
    
    # Generate batch report
    print("\n" + "=" * 60)
    print("BATCH RISK REPORT")
    print("=" * 60)
    
    report = engine.generate_risk_report(assessments)
    
    if report:
        print(f"\n📈 Risk Distribution:")
        for level, stats in report['risk_distribution'].items():
            print(f"   {level}: {stats['count']} events ({stats['percentage']}%)")
        
        print(f"\n📊 Statistics:")
        print(f"   Mean Probability: {report['statistics']['mean_probability']:.3f}")
        print(f"   High Risk %: {report['statistics']['high_risk_percentage']}%")
        
        print(f"\n🎯 SOC Recommendations:")
        for rec in report['soc_alert_analysis']['recommended_actions']:
            print(f"   • {rec}")
    
    # Save sample assessments
    output_dir = project_root / "reports"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "sample_risk_assessments.json"
    
    engine.save_assessments(assessments, output_path)
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE ✓")
    print("=" * 60)
    print(f"\nKey Capabilities Demonstrated:")
    print("1. Individual event risk scoring")
    print("2. Business context integration")
    print("3. Confidence-based threshold adjustment")
    print("4. SOC-aligned risk categorization")
    print("5. Batch processing and reporting")
    print(f"\nSample output saved to: {output_path}")


if __name__ == "__main__":
    main()