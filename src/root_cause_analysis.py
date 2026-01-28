"""
Cybersecurity Root Cause Analysis Engine
Explainable AI for SOC analysts - SHAP-style reasoning without external dependencies
Author: Principal AI/ML Engineer
Date: 2024
Version: 1.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import pickle
import json
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import warnings
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

warnings.filterwarnings('ignore')


class SeverityLevel(str, Enum):
    """Severity levels for root cause findings."""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AttackPattern(str, Enum):
    """Common cybersecurity attack patterns."""
    BRUTE_FORCE = "Brute Force Attack"
    CREDENTIAL_STUFFING = "Credential Stuffing"
    ACCOUNT_TAKEOVER = "Account Takeover"
    PRIVILEGE_ESCALATION = "Privilege Escalation"
    MALWARE_INFECTION = "Malware Infection"
    INSIDER_THREAT = "Insider Threat"
    GEO_EVASION = "Geographic Evasion"
    DEVICE_COMPROMISE = "Device Compromise"


@dataclass
class RootCauseFinding:
    """Individual root cause finding for a security event."""
    feature_name: str
    feature_value: Any
    contribution_score: float  # -1 to +1, where +1 strongly indicates attack
    severity: SeverityLevel
    attack_pattern: Optional[AttackPattern]
    explanation: str
    mitigation_recommendation: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'feature': self.feature_name,
            'value': self.feature_value,
            'contribution_score': round(self.contribution_score, 4),
            'severity': self.severity.value,
            'attack_pattern': self.attack_pattern.value if self.attack_pattern else None,
            'explanation': self.explanation,
            'mitigation': self.mitigation_recommendation
        }


@dataclass
class RootCauseAnalysis:
    """Complete root cause analysis for a security event."""
    event_id: str
    risk_level: str
    overall_probability: float
    top_findings: List[RootCauseFinding]
    most_likely_attack_pattern: Optional[AttackPattern]
    confidence_score: float
    timestamp: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'event_id': self.event_id,
            'risk_level': self.risk_level,
            'probability': round(self.overall_probability, 4),
            'confidence': round(self.confidence_score, 4),
            'most_likely_attack': self.most_likely_attack_pattern.value 
                if self.most_likely_attack_pattern else None,
            'top_findings': [f.to_dict() for f in self.top_findings],
            'summary': self.generate_summary(),
            'timestamp': self.timestamp
        }
    
    def generate_summary(self) -> str:
        """Generate human-readable summary for SOC analysts."""
        if not self.top_findings:
            return "No significant risk factors identified."
        
        # Group findings by attack pattern
        attack_patterns = {}
        for finding in self.top_findings:
            if finding.attack_pattern:
                if finding.attack_pattern not in attack_patterns:
                    attack_patterns[finding.attack_pattern] = []
                attack_patterns[finding.attack_pattern].append(finding)
        
        # Build summary
        summary_parts = []
        
        if self.most_likely_attack_pattern:
            summary_parts.append(
                f"Primary suspicion: {self.most_likely_attack_pattern.value}"
            )
        
        # Add top 3 contributing factors
        top_contributors = sorted(
            self.top_findings, 
            key=lambda x: abs(x.contribution_score), 
            reverse=True
        )[:3]
        
        if top_contributors:
            summary_parts.append("Key contributing factors:")
            for i, finding in enumerate(top_contributors, 1):
                summary_parts.append(
                    f"  {i}. {finding.feature_name}: {finding.explanation}"
                )
        
        # Add mitigation if high severity
        critical_findings = [f for f in self.top_findings if f.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]]
        if critical_findings:
            summary_parts.append("Recommended immediate actions:")
            for finding in critical_findings[:2]:
                summary_parts.append(f"  • {finding.mitigation_recommendation}")
        
        return "\n".join(summary_parts)


class RootCauseAnalyzer:
    """
    Production-grade root cause analysis engine for cybersecurity events.
    Provides SHAP-style explanations without external dependencies.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize root cause analyzer with cybersecurity domain knowledge.
        
        Args:
            seed: Random seed for deterministic operations
        """
        self.seed = seed
        np.random.seed(seed)
        
        # Load model and artifacts
        self.models_dir = project_root / "models"
        self._load_artifacts()
        
        # Cybersecurity domain knowledge base
        self.feature_thresholds = {
            'failed_login_attempts': {
                'low': 3,
                'medium': 8,
                'high': 15,
                'critical': 25,
                'description': 'Number of consecutive failed login attempts'
            },
            'login_velocity': {
                'low': 2.0,
                'medium': 10.0,
                'high': 30.0,
                'critical': 50.0,
                'description': 'Login attempts per hour'
            },
            'ip_reputation_score': {
                'low': 80,      # High reputation
                'medium': 60,
                'high': 40,
                'critical': 20,  # Low reputation
                'description': 'IP reputation score (0-100, higher is better)',
                'inverse': True  # Lower score = higher risk
            },
            'device_trust_score': {
                'low': 80,
                'medium': 60,
                'high': 40,
                'critical': 20,
                'description': 'Device trust score (0-100, higher is better)',
                'inverse': True
            },
            'time_anomaly_score': {
                'low': 0.3,
                'medium': 0.6,
                'high': 0.8,
                'critical': 0.9,
                'description': 'Deviation from normal activity patterns (0-1)'
            },
            'compound_risk_score': {
                'low': 0.3,
                'medium': 0.5,
                'high': 0.7,
                'critical': 0.9,
                'description': 'Weighted combination of multiple risk indicators'
            }
        }
        
        # Attack pattern detection rules
        self.attack_pattern_rules = {
            AttackPattern.BRUTE_FORCE: {
                'required_features': ['failed_login_attempts', 'login_velocity'],
                'thresholds': {
                    'failed_login_attempts': 10,
                    'login_velocity': 15.0
                },
                'description': 'Rapid, repeated login attempts suggesting automated attack'
            },
            AttackPattern.CREDENTIAL_STUFFING: {
                'required_features': ['failed_login_attempts', 'ip_reputation_score'],
                'thresholds': {
                    'failed_login_attempts': 5,
                    'ip_reputation_score': 40
                },
                'description': 'Multiple failed attempts from low-reputation IPs'
            },
            AttackPattern.ACCOUNT_TAKEOVER: {
                'required_features': ['geo_location_change', 'time_anomaly_score'],
                'thresholds': {
                    'geo_location_change': 1,
                    'time_anomaly_score': 0.7
                },
                'description': 'Access from new location at unusual time'
            },
            AttackPattern.PRIVILEGE_ESCALATION: {
                'required_features': ['privilege_level', 'system_criticality'],
                'thresholds': {
                    'privilege_level': ['admin', 'super_admin', 'system'],
                    'system_criticality': ['high', 'critical']
                },
                'description': 'High-privilege access to critical systems'
            },
            AttackPattern.MALWARE_INFECTION: {
                'required_features': ['malware_indicator', 'device_trust_score'],
                'thresholds': {
                    'malware_indicator': 1,
                    'device_trust_score': 50
                },
                'description': 'Malware detected on low-trust device'
            }
        }
        
        # SOC analyst-friendly explanations
        self.explanation_templates = {
            'failed_login_attempts': {
                'low': 'Normal number of failed login attempts',
                'medium': 'Elevated failed login attempts - monitor for patterns',
                'high': 'High number of failed attempts - possible brute force attack',
                'critical': 'Critical level of failed attempts - immediate investigation required'
            },
            'login_velocity': {
                'low': 'Normal login frequency',
                'medium': 'Above average login rate - review for suspicious patterns',
                'high': 'Unusually high login velocity - potential credential stuffing',
                'critical': 'Extremely rapid login attempts - likely automated attack'
            },
            'ip_reputation_score': {
                'low': 'High reputation IP - low risk',
                'medium': 'Moderate reputation IP - standard monitoring',
                'high': 'Low reputation IP - increased scrutiny recommended',
                'critical': 'Very low reputation IP - high probability of malicious origin'
            },
            'device_trust_score': {
                'low': 'High trust device - corporate managed',
                'medium': 'Moderate trust device - review compliance status',
                'high': 'Low trust device - possible personal/unmanaged device',
                'critical': 'Very low trust device - potential compromised endpoint'
            }
        }
        
        # Mitigation recommendations
        self.mitigation_recommendations = {
            'failed_login_attempts': {
                'high': 'Implement account lockout after 10 failed attempts',
                'critical': 'Immediate account lock and manual review required'
            },
            'login_velocity': {
                'high': 'Enable rate limiting for login attempts',
                'critical': 'Block IP temporarily and investigate source'
            },
            'ip_reputation_score': {
                'high': 'Add IP to watchlist for enhanced monitoring',
                'critical': 'Block IP and report to threat intelligence feeds'
            },
            'geo_location_change': {
                'high': 'Require multi-factor authentication for new locations',
                'critical': 'Temporarily block access and verify user identity'
            },
            'privilege_level': {
                'high': 'Review privileged account activity logs',
                'critical': 'Temporarily revoke privileges pending investigation'
            }
        }
        
        print("=" * 60)
        print("ROOT CAUSE ANALYSIS ENGINE INITIALIZED")
        print("=" * 60)
        print(f"Attack Patterns Detected: {len(self.attack_pattern_rules)}")
        print(f"Feature Thresholds: {len(self.feature_thresholds)} cybersecurity indicators")
        print(f"Explanation Templates: {len(self.explanation_templates)} human-readable formats")
    
    def _load_artifacts(self) -> None:
        """Load trained model and feature importance data."""
        try:
            # Load model with feature importance
            with open(self.models_dir / "risk_classifier.pkl", 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.feature_importances = model_data.get('feature_importance', None)
            
            # Load feature names
            with open(self.models_dir / "feature_names.pkl", 'rb') as f:
                self.feature_names = pickle.load(f)
            
            # Load preprocessor for understanding feature transformations
            with open(self.models_dir / "preprocessor.pkl", 'rb') as f:
                self.preprocessor = pickle.load(f)
            
            print("✓ Loaded model and feature importance data")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Required artifacts not found. Run model_training.py first.\n"
                f"Missing: {e.filename}"
            )
    
    def _calculate_feature_contribution(self, 
                                       feature_name: str, 
                                       feature_value: Any,
                                       model_prediction: float) -> float:
        """
        Calculate feature contribution to final prediction.
        Simplified SHAP-style approximation without dependencies.
        
        Args:
            feature_name: Name of the feature
            feature_value: Value of the feature
            model_prediction: Overall model probability
            
        Returns:
            Contribution score (-1 to +1)
        """
        # Get feature importance from model
        feature_idx = None
        for i, name in enumerate(self.feature_names):
            if feature_name in name:  # Handle one-hot encoded features
                feature_idx = i
                break
        
        if feature_idx is None or self.feature_importances is None:
            # Fallback: Use domain knowledge-based contribution
            return self._domain_based_contribution(feature_name, feature_value)
        
        # Base contribution from feature importance
        base_contribution = self.feature_importances[feature_idx]
        
        # Adjust based on feature value deviation from baseline
        deviation_score = self._calculate_deviation_score(feature_name, feature_value)
        
        # Combine importance with deviation
        contribution = base_contribution * deviation_score
        
        # Normalize to -1 to +1 range
        contribution = np.clip(contribution * 10, -1, 1)  # Scale factor based on empirical analysis
        
        return contribution
    
    def _domain_based_contribution(self, 
                                  feature_name: str, 
                                  feature_value: Any) -> float:
        """
        Calculate contribution based on cybersecurity domain knowledge.
        
        Args:
            feature_name: Name of the feature
            feature_value: Value of the feature
            
        Returns:
            Contribution score (-1 to +1)
        """
        # Domain knowledge contributions
        domain_knowledge = {
            'failed_login_attempts': lambda x: min(x / 20, 1.0),  # Max at 20 attempts
            'login_velocity': lambda x: min(x / 40, 1.0),        # Max at 40 logins/hour
            'ip_reputation_score': lambda x: (100 - x) / 100,    # Inverse: low score = high risk
            'device_trust_score': lambda x: (100 - x) / 100,     # Inverse: low trust = high risk
            'malware_indicator': lambda x: 1.0 if x == 1 else -0.1,  # Malware present = high risk
            'geo_location_change': lambda x: 0.8 if x == 1 else -0.2,  # Location change = moderate risk
            'time_anomaly_score': lambda x: x,                   # Direct correlation
            'compound_risk_score': lambda x: x,                  # Direct correlation
            'privilege_level': lambda x: {
                'user': 0.0,
                'admin': 0.4,
                'super_admin': 0.7,
                'system': 0.9
            }.get(x, 0.0),
            'system_criticality': lambda x: {
                'low': 0.0,
                'medium': 0.3,
                'high': 0.6,
                'critical': 0.9
            }.get(x, 0.0)
        }
        
        if feature_name in domain_knowledge:
            contribution = domain_knowledge[feature_name](feature_value)
            return contribution
        else:
            # Check for one-hot encoded features
            for base_feature in ['privilege_level', 'system_criticality']:
                if feature_name.startswith(f"{base_feature}_"):
                    return 0.5  # Moderate contribution for categorical presence
            
        return 0.0  # Neutral contribution for unknown features
    
    def _calculate_deviation_score(self, 
                                  feature_name: str, 
                                  feature_value: Any) -> float:
        """
        Calculate how much a feature value deviates from normal baseline.
        
        Args:
            feature_name: Name of the feature
            feature_value: Value of the feature
            
        Returns:
            Deviation score (0 to 2, where 1 is baseline)
        """
        if feature_name not in self.feature_thresholds:
            return 1.0  # No deviation information
        
        thresholds = self.feature_thresholds[feature_name]
        
        # Get baseline (medium threshold)
        baseline = thresholds.get('medium', 0)
        
        if isinstance(feature_value, (int, float)):
            # Numerical feature
            if feature_value == baseline:
                return 1.0
            elif feature_value > baseline:
                # Positive deviation (higher risk)
                deviation = 1.0 + (feature_value - baseline) / (thresholds.get('critical', baseline * 2) - baseline)
            else:
                # Negative deviation (lower risk)
                deviation = 1.0 - (baseline - feature_value) / baseline
            
            return np.clip(deviation, 0.1, 2.0)
        
        else:
            # Categorical feature
            # For simplicity, return moderate deviation for non-default values
            default_values = {
                'privilege_level': 'user',
                'system_criticality': 'low'
            }
            
            for cat_feature, default_value in default_values.items():
                if cat_feature in feature_name:
                    if str(feature_value).lower() != default_value:
                        return 1.5  # Moderate deviation
                    else:
                        return 0.8  # Slightly below baseline (safer)
            
            return 1.0
    
    def _determine_severity_level(self, 
                                 feature_name: str, 
                                 feature_value: Any) -> SeverityLevel:
        """
        Determine severity level based on feature value and thresholds.
        
        Args:
            feature_name: Name of the feature
            feature_value: Value of the feature
            
        Returns:
            SeverityLevel
        """
        if feature_name not in self.feature_thresholds:
            return SeverityLevel.LOW
        
        thresholds = self.feature_thresholds[feature_name]
        
        # Handle numerical features
        if isinstance(feature_value, (int, float)):
            if 'inverse' in thresholds and thresholds['inverse']:
                # Inverse threshold (lower value = higher risk)
                if feature_value <= thresholds.get('critical', 0):
                    return SeverityLevel.CRITICAL
                elif feature_value <= thresholds.get('high', 0):
                    return SeverityLevel.HIGH
                elif feature_value <= thresholds.get('medium', 0):
                    return SeverityLevel.MODERATE
                else:
                    return SeverityLevel.LOW
            else:
                # Normal threshold (higher value = higher risk)
                if feature_value >= thresholds.get('critical', float('inf')):
                    return SeverityLevel.CRITICAL
                elif feature_value >= thresholds.get('high', float('inf')):
                    return SeverityLevel.HIGH
                elif feature_value >= thresholds.get('medium', float('inf')):
                    return SeverityLevel.MODERATE
                else:
                    return SeverityLevel.LOW
        
        # Handle categorical features
        elif isinstance(feature_value, str):
            high_risk_values = {
                'privilege_level': ['super_admin', 'system'],
                'system_criticality': ['high', 'critical']
            }
            
            for cat_feature, risky_values in high_risk_values.items():
                if cat_feature in feature_name:
                    if feature_value in risky_values:
                        return SeverityLevel.HIGH
                    elif feature_value == 'admin':
                        return SeverityLevel.MODERATE
                    else:
                        return SeverityLevel.LOW
        
        # Handle binary features
        elif feature_value == 1:  # Presence of risk indicator
            if feature_name in ['malware_indicator', 'geo_location_change']:
                return SeverityLevel.MODERATE
            else:
                return SeverityLevel.LOW
        
        return SeverityLevel.LOW
    
    def _detect_attack_pattern(self, 
                              feature_values: Dict[str, Any]) -> Optional[AttackPattern]:
        """
        Detect most likely attack pattern based on feature values.
        
        Args:
            feature_values: Dictionary of feature names and values
            
        Returns:
            Most likely AttackPattern or None
        """
        pattern_scores = {}
        
        for pattern, rules in self.attack_pattern_rules.items():
            score = 0
            requirements_met = 0
            
            for feature in rules['required_features']:
                if feature in feature_values:
                    value = feature_values[feature]
                    threshold = rules['thresholds'].get(feature)
                    
                    if isinstance(threshold, list):
                        # Categorical threshold
                        if value in threshold:
                            requirements_met += 1
                            score += 1.0
                    elif isinstance(threshold, (int, float)):
                        # Numerical threshold
                        if ('inverse' in self.feature_thresholds.get(feature, {}) and 
                            self.feature_thresholds[feature].get('inverse')):
                            # Inverse comparison (lower value = higher risk)
                            if value <= threshold:
                                requirements_met += 1
                                score += (threshold - value) / threshold
                        else:
                            # Normal comparison (higher value = higher risk)
                            if value >= threshold:
                                requirements_met += 1
                                score += (value - threshold) / (threshold * 2)
            
            # Only consider patterns where all requirements are met
            if requirements_met == len(rules['required_features']):
                pattern_scores[pattern] = score
        
        if pattern_scores:
            # Return pattern with highest score
            return max(pattern_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    def _generate_explanation(self, 
                             feature_name: str, 
                             feature_value: Any,
                             severity: SeverityLevel) -> str:
        """
        Generate human-readable explanation for a feature contribution.
        
        Args:
            feature_name: Name of the feature
            feature_value: Value of the feature
            severity: Severity level
            
        Returns:
            Human-readable explanation
        """
        # Try to use template first
        base_feature = feature_name.split('_')[0] if '_' in feature_name else feature_name
        
        if base_feature in self.explanation_templates:
            templates = self.explanation_templates[base_feature]
            severity_key = severity.value.lower()
            
            if severity_key in templates:
                return templates[severity_key]
        
        # Generate dynamic explanation
        if isinstance(feature_value, (int, float)):
            if feature_name in self.feature_thresholds:
                desc = self.feature_thresholds[feature_name].get('description', feature_name)
                
                if 'inverse' in self.feature_thresholds[feature_name]:
                    # Inverse feature (lower value = higher risk)
                    if severity == SeverityLevel.CRITICAL:
                        return f"Very low {desc} ({feature_value}) indicates high probability of malicious activity"
                    elif severity == SeverityLevel.HIGH:
                        return f"Low {desc} ({feature_value}) suggests potential security concern"
                    else:
                        return f"Normal {desc} ({feature_value})"
                else:
                    # Normal feature (higher value = higher risk)
                    if severity == SeverityLevel.CRITICAL:
                        return f"Extremely high {desc} ({feature_value}) indicates active attack"
                    elif severity == SeverityLevel.HIGH:
                        return f"High {desc} ({feature_value}) suggests suspicious activity"
                    else:
                        return f"Normal {desc} ({feature_value})"
        
        elif isinstance(feature_value, str):
            if 'privilege' in feature_name.lower():
                if severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
                    return f"High-privilege account ({feature_value}) accessing sensitive resources"
                else:
                    return f"Standard user account ({feature_value})"
            
            elif 'criticality' in feature_name.lower():
                if severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
                    return f"Access to critical system ({feature_value}) requires enhanced scrutiny"
                else:
                    return f"Access to {feature_value} priority system"
        
        # Default explanation
        return f"Feature '{feature_name}' with value '{feature_value}' indicates {severity.value.lower()} risk"
    
    def _generate_mitigation(self, 
                            feature_name: str, 
                            severity: SeverityLevel) -> str:
        """
        Generate mitigation recommendation for a feature.
        
        Args:
            feature_name: Name of the feature
            severity: Severity level
            
        Returns:
            Mitigation recommendation
        """
        base_feature = feature_name.split('_')[0] if '_' in feature_name else feature_name
        
        if base_feature in self.mitigation_recommendations:
            mitigations = self.mitigation_recommendations[base_feature]
            severity_key = severity.value.lower()
            
            if severity_key in mitigations:
                return mitigations[severity_key]
        
        # Default mitigations based on severity
        if severity == SeverityLevel.CRITICAL:
            return "Immediate investigation and containment required"
        elif severity == SeverityLevel.HIGH:
            return "Investigate within next hour and implement controls"
        elif severity == SeverityLevel.MODERATE:
            return "Review during next scheduled security audit"
        else:
            return "Monitor for patterns and escalate if repeated"
    
    def analyze_root_cause(self, 
                          event_data: Dict[str, Any],
                          model_prediction: float,
                          risk_level: str,
                          event_id: str = None) -> RootCauseAnalysis:
        """
        Perform comprehensive root cause analysis for a security event.
        
        Args:
            event_data: Dictionary with event features
            model_prediction: Model probability (0-1)
            risk_level: Risk level from scoring engine
            event_id: Optional event identifier
            
        Returns:
            Complete RootCauseAnalysis object
        """
        if event_id is None:
            event_id = f"RCA_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n🔍 Analyzing root cause for event: {event_id}")
        print(f"   Risk Level: {risk_level}, Probability: {model_prediction:.3f}")
        
        # Step 1: Calculate contributions for all features
        findings = []
        
        for feature_name, feature_value in event_data.items():
            # Skip event_id if present
            if feature_name == 'event_id':
                continue
            
            # Calculate contribution
            contribution = self._calculate_feature_contribution(
                feature_name, feature_value, model_prediction
            )
            
            # Only include features with significant contribution
            if abs(contribution) > 0.1:  # 10% contribution threshold
                # Determine severity
                severity = self._determine_severity_level(feature_name, feature_value)
                
                # Generate explanation
                explanation = self._generate_explanation(feature_name, feature_value, severity)
                
                # Generate mitigation
                mitigation = self._generate_mitigation(feature_name, severity)
                
                # Detect attack pattern for this feature
                attack_pattern = None
                if severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
                    # Check if this feature triggers any attack pattern
                    for pattern, rules in self.attack_pattern_rules.items():
                        if feature_name in rules['required_features']:
                            attack_pattern = pattern
                            break
                
                # Create finding
                finding = RootCauseFinding(
                    feature_name=feature_name,
                    feature_value=feature_value,
                    contribution_score=contribution,
                    severity=severity,
                    attack_pattern=attack_pattern,
                    explanation=explanation,
                    mitigation_recommendation=mitigation
                )
                
                findings.append(finding)
        
        # Step 2: Detect overall attack pattern
        attack_pattern = self._detect_attack_pattern(event_data)
        
        # Step 3: Sort findings by absolute contribution (most impactful first)
        findings.sort(key=lambda x: abs(x.contribution_score), reverse=True)
        
        # Step 4: Calculate confidence score
        confidence = self._calculate_confidence_score(findings, model_prediction)
        
        # Step 5: Create analysis
        analysis = RootCauseAnalysis(
            event_id=event_id,
            risk_level=risk_level,
            overall_probability=model_prediction,
            top_findings=findings[:5],  # Top 5 findings only
            most_likely_attack_pattern=attack_pattern,
            confidence_score=confidence,
            timestamp=datetime.now().isoformat()
        )
        
        return analysis
    
    def _calculate_confidence_score(self, 
                                   findings: List[RootCauseFinding],
                                   model_prediction: float) -> float:
        """
        Calculate confidence score for root cause analysis.
        
        Args:
            findings: List of root cause findings
            model_prediction: Model probability
            
        Returns:
            Confidence score (0-1)
        """
        if not findings:
            return 0.0
        
        # Base confidence from model prediction clarity
        prediction_clarity = 1 - 2 * abs(model_prediction - 0.5)
        
        # Confidence from findings consistency
        positive_contributions = sum(1 for f in findings if f.contribution_score > 0)
        total_findings = len(findings)
        
        if total_findings > 0:
            consistency = positive_contributions / total_findings
            # Prefer either mostly positive or mostly negative contributions
            consistency_score = max(consistency, 1 - consistency)
        else:
            consistency_score = 0.5
        
        # Confidence from severity levels
        high_severity_count = sum(1 for f in findings if f.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL])
        severity_score = min(high_severity_count / 3, 1.0)  # Cap at 3 high severity findings
        
        # Combine scores
        confidence = (prediction_clarity * 0.4 + 
                     consistency_score * 0.3 + 
                     severity_score * 0.3)
        
        return np.clip(confidence, 0, 1)
    
    def batch_analyze(self, 
                     events_data: List[Dict[str, Any]],
                     predictions: List[float],
                     risk_levels: List[str]) -> List[RootCauseAnalysis]:
        """
        Perform root cause analysis for multiple events.
        
        Args:
            events_data: List of event feature dictionaries
            predictions: List of model predictions
            risk_levels: List of risk levels
            
        Returns:
            List of RootCauseAnalysis objects
        """
        analyses = []
        
        print(f"Performing root cause analysis for {len(events_data)} events...")
        
        for i, (event_data, prediction, risk_level) in enumerate(zip(events_data, predictions, risk_levels)):
            event_id = event_data.get('event_id', f"BATCH_RCA_{i+1:06d}")
            
            try:
                analysis = self.analyze_root_cause(
                    event_data, prediction, risk_level, event_id
                )
                analyses.append(analysis)
                
                # Progress indicator
                if (i + 1) % 50 == 0:
                    print(f"  Analyzed {i + 1}/{len(events_data)} events...")
                    
            except Exception as e:
                print(f"⚠ Failed to analyze event {event_id}: {str(e)}")
        
        return analyses
    
    def generate_analysis_report(self, 
                                analyses: List[RootCauseAnalysis],
                                output_path: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive root cause analysis report.
        
        Args:
            analyses: List of root cause analyses
            output_path: Optional path to save JSON report
            
        Returns:
            Dictionary with analysis summary
        """
        if not analyses:
            return {}
        
        # Extract statistics
        total_events = len(analyses)
        high_risk_events = sum(1 for a in analyses if a.risk_level in ['HIGH', 'CRITICAL'])
        
        # Count attack patterns
        attack_pattern_counts = {}
        for analysis in analyses:
            if analysis.most_likely_attack_pattern:
                pattern = analysis.most_likely_attack_pattern
                attack_pattern_counts[pattern] = attack_pattern_counts.get(pattern, 0) + 1
        
        # Most common findings
        all_findings = []
        for analysis in analyses:
            all_findings.extend(analysis.top_findings)
        
        # Count feature occurrences in top findings
        feature_counts = {}
        for finding in all_findings:
            feature_counts[finding.feature_name] = feature_counts.get(finding.feature_name, 0) + 1
        
        # Top 5 most common risk indicators
        top_indicators = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Create report
        report = {
            'summary': {
                'total_events_analyzed': total_events,
                'high_risk_events': high_risk_events,
                'high_risk_percentage': round(high_risk_events / total_events * 100, 2),
                'timestamp': datetime.now().isoformat(),
                'analysis_engine_version': '1.0'
            },
            'attack_pattern_distribution': {
                pattern.value: count for pattern, count in attack_pattern_counts.items()
            },
            'top_risk_indicators': [
                {'feature': feature, 'count': count, 'percentage': round(count / total_events * 100, 2)}
                for feature, count in top_indicators
            ],
            'detailed_analyses': [analysis.to_dict() for analysis in analyses]
        }
        
        # Save to file if path provided
        if output_path:
            output_dir = Path(output_path).parent
            output_dir.mkdir(exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"✓ Root cause analysis report saved to: {output_path}")
        
        return report


def main():
    """Demonstrate root cause analysis engine with sample data."""
    print("=" * 60)
    print("ROOT CAUSE ANALYSIS ENGINE DEMONSTRATION")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = RootCauseAnalyzer(seed=42)
    
    # Create sample events for demonstration
    sample_events = [
        {
            'event_id': 'RCA_BENIGN_001',
            'failed_login_attempts': 1,
            'login_velocity': 0.5,
            'ip_reputation_score': 85,
            'geo_location_change': 0,
            'privilege_level': 'user',
            'device_trust_score': 92,
            'malware_indicator': 0,
            'system_criticality': 'low',
            'time_anomaly_score': 0.2,
            'compound_risk_score': 0.15
        },
        {
            'event_id': 'RCA_SUSPICIOUS_001',
            'failed_login_attempts': 8,
            'login_velocity': 12.5,
            'ip_reputation_score': 35,
            'geo_location_change': 1,
            'privilege_level': 'admin',
            'device_trust_score': 45,
            'malware_indicator': 0,
            'system_criticality': 'medium',
            'time_anomaly_score': 0.75,
            'compound_risk_score': 0.68
        },
        {
            'event_id': 'RCA_HIGH_RISK_001',
            'failed_login_attempts': 18,
            'login_velocity': 42.3,
            'ip_reputation_score': 12,
            'geo_location_change': 1,
            'privilege_level': 'super_admin',
            'device_trust_score': 22,
            'malware_indicator': 1,
            'system_criticality': 'critical',
            'time_anomaly_score': 0.95,
            'compound_risk_score': 0.92
        }
    ]
    
    sample_predictions = [0.12, 0.67, 0.89]
    sample_risk_levels = ['LOW', 'HIGH', 'CRITICAL']
    
    print("\n" + "=" * 60)
    print("INDIVIDUAL EVENT ANALYSIS")
    print("=" * 60)
    
    analyses = []
    for event_data, prediction, risk_level in zip(sample_events, sample_predictions, sample_risk_levels):
        analysis = analyzer.analyze_root_cause(
            event_data, prediction, risk_level, event_data['event_id']
        )
        analyses.append(analysis)
        
        print(f"\n📋 Event: {event_data['event_id']}")
        print(f"   Risk Level: {risk_level}, Probability: {prediction:.3f}")
        
        if analysis.most_likely_attack_pattern:
            print(f"   🎯 Most Likely Attack: {analysis.most_likely_attack_pattern.value}")
        
        print(f"   🔍 Top Findings:")
        for i, finding in enumerate(analysis.top_findings[:3], 1):
            print(f"      {i}. {finding.feature_name}: {finding.explanation}")
            print(f"         Severity: {finding.severity.value}, Contribution: {finding.contribution_score:.3f}")
        
        print(f"   📝 Summary: {analysis.generate_summary()[:100]}...")
    
    # Generate comprehensive report
    print("\n" + "=" * 60)
    print("COMPREHENSIVE ANALYSIS REPORT")
    print("=" * 60)
    
    report = analyzer.generate_analysis_report(
        analyses,
        output_path=project_root / "reports" / "root_cause_analysis.json"
    )
    
    if report:
        print(f"\n📊 Summary Statistics:")
        print(f"   Total Events Analyzed: {report['summary']['total_events_analyzed']}")
        print(f"   High Risk Events: {report['summary']['high_risk_events']} "
              f"({report['summary']['high_risk_percentage']}%)")
        
        if 'attack_pattern_distribution' in report and report['attack_pattern_distribution']:
            print(f"\n🎯 Attack Pattern Distribution:")
            for pattern, count in report['attack_pattern_distribution'].items():
                print(f"   • {pattern}: {count} events")
        
        if 'top_risk_indicators' in report:
            print(f"\n📈 Top Risk Indicators:")
            for indicator in report['top_risk_indicators'][:3]:
                print(f"   • {indicator['feature']}: {indicator['count']} events "
                      f"({indicator['percentage']}%)")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE ✓")
    print("=" * 60)
    print(f"\nKey Capabilities Demonstrated:")
    print("1. Feature contribution analysis (SHAP-style)")
    print("2. Attack pattern detection")
    print("3. Human-readable explanations")
    print("4. Severity-based prioritization")
    print("5. Mitigation recommendations")
    print(f"\nReport saved to: reports/root_cause_analysis.json")


if __name__ == "__main__":
    main()