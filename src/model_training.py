"""
Cybersecurity Risk Prediction - Model Training
Production-grade ML model training with rigorous validation
Author: Principal AI/ML Engineer
Date: 2024
Version: 1.0
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pickle
import json
import warnings
from typing import Tuple, Dict, Any
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for CLI
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

warnings.filterwarnings('ignore')

class CybersecurityModelTrainer:
    """
    Production-grade ML model trainer for cybersecurity risk prediction.
    Implements rigorous validation, model selection, and artifact management.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize model trainer with enterprise-grade configuration.
        
        Args:
            seed: Random seed for deterministic training
        """
        self.seed = seed
        np.random.seed(seed)
        
        # Model selection criteria (SOC-aligned)
        self.model_selection_criteria = {
            'logistic_regression': {
                'pros': ['Interpretable', 'Fast inference', 'Good for linear patterns'],
                'cons': ['Limited to linear relationships', 'May underfit complex patterns'],
                'use_case': 'Baseline model, regulatory compliance'
            },
            'random_forest': {
                'pros': ['Handles non-linear patterns', 'Feature importance', 'Robust to outliers'],
                'cons': ['Less interpretable', 'Slower inference', 'Can overfit'],
                'use_case': 'Production deployment, complex threat patterns'
            }
        }
        
        # Chosen model: Random Forest (justified below)
        self.selected_model_name = 'random_forest'
        self.model = None
        self.feature_importance = None
        self.metrics = {}
        self.cv_scores = {}
        
        # Create output directories
        self.models_dir = project_root / "models"
        self.reports_dir = project_root / "reports"
        self.models_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load preprocessed data from Phase 3.
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        processed_dir = project_root / "data" / "processed"
        
        print("Loading preprocessed data...")
        X_train = np.load(processed_dir / "X_train.npy")
        X_test = np.load(processed_dir / "X_test.npy")
        y_train = np.load(processed_dir / "y_train.npy")
        y_test = np.load(processed_dir / "y_test.npy")
        
        print(f"✓ Training data: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
        print(f"✓ Test data: {X_test.shape[0]:,} samples")
        print(f"✓ Class distribution - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")
        
        return X_train, X_test, y_train, y_test
    
    def _load_feature_names(self) -> list:
        """Load feature names from preprocessing artifacts."""
        with open(self.models_dir / "feature_names.pkl", 'rb') as f:
            feature_names = pickle.load(f)
        return feature_names
    
    def _justify_model_selection(self) -> None:
        """
        Provide detailed justification for model selection.
        Aligns with cybersecurity domain requirements.
        """
        print("\n" + "=" * 60)
        print("MODEL SELECTION JUSTIFICATION")
        print("=" * 60)
        
        justification = """
        SELECTED MODEL: RANDOM FOREST CLASSIFIER
        
        Domain Requirements Analysis:
        1. NON-LINEAR PATTERNS: Cyber attacks often have complex, non-linear patterns
           (e.g., failed_logins × login_velocity × ip_reputation interactions)
        
        2. FEATURE IMPORTANCE: SOC analysts need to understand which indicators 
           contribute most to risk decisions for root cause analysis
        
        3. ROBUSTNESS: Must handle:
           - Imbalanced data (30% attacks)
           - Correlated features (common in security events)
           - Outliers (extreme values in attack patterns)
        
        4. UAE ENTERPRISE CONSTRAINTS:
           - Model must be deterministic for audit trails
           - Feature importance required for regulatory compliance
           - Inference speed: ~100ms per event for real-time SOC
        
        Why Not Logistic Regression?
        - Cyber threats rarely follow linear decision boundaries
        - Feature interactions are crucial (privilege × anomaly_score × geo_change)
        - Would require extensive feature engineering for non-linearities
        
        Why Not Deep Learning?
        - Limited interpretability violates regulatory requirements
        - 10K samples insufficient for deep learning
        - Longer training/inference times not justified
        - Hard to explain to non-technical stakeholders
        
        Random Forest Advantages for Cybersecurity:
        • Built-in feature importance (explainability)
        • Handles mixed data types naturally
        • Robust to feature scaling (already done)
        • Provides probability estimates for risk scoring
        • Ensemble nature reduces overfitting risk
        """
        print(justification)
        
        # Display model comparison
        print("\nMODEL COMPARISON TABLE:")
        print("-" * 80)
        print(f"{'Model':<25} {'Best For':<25} {'Limitations':<30}")
        print("-" * 80)
        for model_name, info in self.model_selection_criteria.items():
            print(f"{model_name:<25} {info['use_case']:<25} {info['cons'][0]:<30}")
        print("=" * 60)
    
    def _initialize_model(self) -> Any:
        """
        Initialize Random Forest with cybersecurity-optimized hyperparameters.
        
        Hyperparameter Tuning Rationale:
        - n_estimators=100: Balance between performance and training time
        - max_depth=10: Prevent overfitting while capturing patterns
        - min_samples_split=5: Require sufficient samples for splits
        - class_weight='balanced': Handle imbalanced classes (70/30)
        - random_state=seed: Deterministic training
        """
        print("\nInitializing Random Forest Classifier...")
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced',  # Handle class imbalance
            n_jobs=-1,  # Use all CPU cores
            random_state=self.seed,
            verbose=0
        )
        
        print("✓ Hyperparameters configured for cybersecurity domain:")
        print(f"  • n_estimators: {model.n_estimators} (ensemble size)")
        print(f"  • max_depth: {model.max_depth} (prevents overfitting)")
        print(f"  • class_weight: {model.class_weight} (handles 30% attack prevalence)")
        print(f"  • random_state: {model.random_state} (deterministic training)")
        
        return model
    
    def perform_cross_validation(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """
        Perform rigorous cross-validation to assess model stability.
        
        Args:
            X_train, y_train: Training data
            
        Returns:
            Dictionary with CV scores
        """
        print("\n" + "=" * 60)
        print("CROSS-VALIDATION (5-Fold Stratified)")
        print("=" * 60)
        
        # Initialize model for CV
        cv_model = self._initialize_model()
        
        # Define metrics to compute
        scoring_metrics = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',  # Most important for cybersecurity (detect attacks)
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        }
        
        cv_results = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        
        for metric_name, scoring in scoring_metrics.items():
            scores = cross_val_score(
                cv_model, X_train, y_train,
                cv=cv, scoring=scoring, n_jobs=-1
            )
            cv_results[metric_name] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores.tolist()
            }
            
            print(f"{metric_name.upper():<12} Mean: {np.mean(scores):.4f} (±{np.std(scores):.4f})")
        
        # Store for reporting
        self.cv_scores = cv_results
        
        # Determine if model is stable (low variance across folds)
        recall_std = cv_results['recall']['std']
        if recall_std > 0.05:
            print(f"⚠ Warning: High variance in recall ({recall_std:.4f}) - model may be unstable")
        else:
            print("✓ Model shows stable performance across folds")
        
        return cv_results
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """
        Train the selected model on full training data.
        
        Args:
            X_train, y_train: Training data
            
        Returns:
            Trained model
        """
        print("\n" + "=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)
        
        # Initialize and train model
        self.model = self._initialize_model()
        
        print("Training Random Forest...")
        self.model.fit(X_train, y_train)
        
        # Extract feature importance
        self.feature_importance = self.model.feature_importances_
        
        print("✓ Model training completed")
        print(f"  • Number of trees: {len(self.model.estimators_)}")
        print(f"  • Total training samples: {X_train.shape[0]:,}")
        print(f"  • Feature importance computed")
        
        return self.model
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Comprehensive model evaluation on test set.
        
        Args:
            X_test, y_test: Test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("\n" + "=" * 60)
        print("MODEL EVALUATION ON TEST SET")
        print("=" * 60)
        
        # Generate predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]  # Probability of class 1 (risky)
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_metrics = {
            'true_negatives': cm[0, 0],
            'false_positives': cm[0, 1],
            'false_negatives': cm[1, 0],
            'true_positives': cm[1, 1]
        }
        
        # Calculate rates
        total = cm.sum()
        metrics.update({
            'false_positive_rate': cm_metrics['false_positives'] / (cm_metrics['false_positives'] + cm_metrics['true_negatives']),
            'false_negative_rate': cm_metrics['false_negatives'] / (cm_metrics['false_negatives'] + cm_metrics['true_positives']),
            'confusion_matrix': cm.tolist()
        })
        
        # Store metrics
        self.metrics = metrics
        
        # Display results
        print("\nPERFORMANCE METRICS:")
        print("-" * 50)
        for metric_name, value in metrics.items():
            if metric_name != 'confusion_matrix':
                print(f"{metric_name.upper():<20} {value:.4f}")
        
        print("\nCONFUSION MATRIX (Absolute):")
        print(f"                  Predicted")
        print(f"                 Benign  Risky")
        print(f"Actual Benign    {cm[0,0]:>6}  {cm[0,1]:>6}")
        print(f"       Risky     {cm[1,0]:>6}  {cm[1,1]:>6}")
        
        print("\nCONFUSION MATRIX (Percentage):")
        cm_percent = cm / cm.sum() * 100
        print(f"                  Predicted")
        print(f"                 Benign   Risky")
        print(f"Actual Benign    {cm_percent[0,0]:>6.1f}%  {cm_percent[0,1]:>6.1f}%")
        print(f"       Risky     {cm_percent[1,0]:>6.1f}%  {cm_percent[1,1]:>6.1f}%")
        
        # Cybersecurity-specific interpretation
        print("\n" + "=" * 60)
        print("CYBERSECURITY INTERPRETATION")
        print("=" * 60)
        
        # Most important: Recall (detecting actual attacks)
        if metrics['recall'] > 0.85:
            print("✓ EXCELLENT: Model detects >85% of actual attacks")
        elif metrics['recall'] > 0.70:
            print("✓ GOOD: Model detects >70% of actual attacks")
        else:
            print("⚠ NEEDS IMPROVEMENT: Attack detection rate is low")
        
        # False positive rate (SOC alert fatigue)
        if metrics['false_positive_rate'] < 0.10:
            print("✓ EXCELLENT: Low false alarm rate (<10%)")
        elif metrics['false_positive_rate'] < 0.20:
            print("✓ ACCEPTABLE: Moderate false alarm rate")
        else:
            print("⚠ CONCERN: High false alarm rate may cause alert fatigue")
        
        # ROC-AUC (overall discrimination)
        if metrics['roc_auc'] > 0.90:
            print("✓ EXCELLENT: Strong discrimination between classes")
        elif metrics['roc_auc'] > 0.80:
            print("✓ GOOD: Reasonable discrimination capability")
        
        return metrics
    
    def analyze_feature_importance(self) -> pd.DataFrame:
        """
        Analyze and visualize feature importance for explainability.
        
        Returns:
            DataFrame with feature importance sorted descending
        """
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)
        
        # Load feature names
        feature_names = self._load_feature_names()
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        # Normalize to percentage
        importance_df['importance_pct'] = importance_df['importance'] * 100
        
        # Display top features
        print("\nTOP 10 MOST IMPORTANT FEATURES:")
        print("-" * 60)
        for i, row in importance_df.head(10).iterrows():
            print(f"{row['feature']:<30} {row['importance_pct']:>6.2f}%")
        
        # Cybersecurity interpretation of top features
        print("\n" + "-" * 60)
        print("DOMAIN INTERPRETATION OF TOP FEATURES:")
        print("-" * 60)
        
        top_features = importance_df.head(5)['feature'].tolist()
        interpretations = {
            'compound_risk_score': 'Weighted combination of multiple risk indicators',
            'login_velocity': 'Rapid login attempts indicate credential stuffing',
            'failed_login_attempts': 'Direct indicator of brute force attacks',
            'ip_reputation_score': 'Low reputation IPs often associated with threats',
            'device_trust_score': 'Untrusted devices pose higher risk',
            'failed_per_velocity': 'Failed attempts per login rate - strong attack pattern',
            'privilege_level_admin': 'Admin accounts are high-value targets',
            'time_anomaly_score': 'Unusual timing suggests compromised accounts'
        }
        
        for feature in top_features:
            if feature in interpretations:
                print(f"• {feature}: {interpretations[feature]}")
            else:
                # Handle encoded categorical features
                if 'privilege_level_' in feature:
                    level = feature.replace('privilege_level_', '')
                    print(f"• {feature}: {level.upper()} accounts are higher risk targets")
        
        return importance_df
    
    def generate_visualizations(self, X_test: np.ndarray, y_test: np.ndarray, 
                              importance_df: pd.DataFrame) -> None:
        """
        Generate and save evaluation visualizations.
        
        Args:
            X_test, y_test: Test data
            importance_df: Feature importance DataFrame
        """
        print("\nGenerating evaluation visualizations...")
        
        # 1. Feature Importance Plot
        plt.figure(figsize=(12, 8))
        top_10 = importance_df.head(10)
        plt.barh(range(len(top_10)), top_10['importance_pct'])
        plt.yticks(range(len(top_10)), top_10['feature'])
        plt.xlabel('Importance (%)')
        plt.title('Top 10 Feature Importance for Cybersecurity Risk Prediction')
        plt.gca().invert_yaxis()  # Most important at top
        plt.tight_layout()
        importance_path = self.reports_dir / "feature_importance.png"
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confusion Matrix Heatmap
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, self.model.predict(X_test))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Benign', 'Risky'],
                   yticklabels=['Benign', 'Risky'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        cm_path = self.reports_dir / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. ROC Curve
        plt.figure(figsize=(8, 6))
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'Random Forest (AUC = {self.metrics["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Cybersecurity Risk Prediction')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        roc_path = self.reports_dir / "roc_curve.png"
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualizations saved to {self.reports_dir}")
    
    def save_model(self) -> str:
        """
        Save trained model and metadata.
        
        Returns:
            Path to saved model
        """
        model_path = self.models_dir / "risk_classifier.pkl"
        
        # Create model metadata
        metadata = {
            'model_name': 'RandomForestClassifier',
            'model_params': self.model.get_params(),
            'training_date': datetime.now().isoformat(),
            'seed': self.seed,
            'metrics': self.metrics,
            'cv_scores': self.cv_scores,
            'feature_count': self.model.n_features_in_,
            'classes': self.model.classes_.tolist()
        }
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'metadata': metadata,
                'feature_importance': self.feature_importance
            }, f)
        
        # Save metadata separately for easy reading
        metadata_path = self.models_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Model saved to: {model_path}")
        print(f"✓ Metadata saved to: {metadata_path}")
        
        # Display model size
        model_size = os.path.getsize(model_path) / 1024  # KB
        print(f"✓ Model size: {model_size:.2f} KB")
        
        return str(model_path)
    
    def generate_training_report(self) -> None:
        """Generate comprehensive training report."""
        report_path = self.reports_dir / "training_report.md"
        
        report_content = f"""
# Cybersecurity Risk Prediction - Model Training Report

## Model Information
- **Model Type**: Random Forest Classifier
- **Training Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Random Seed**: {self.seed}
- **Feature Count**: {self.model.n_features_in_}

## Performance Metrics
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Accuracy | {self.metrics.get('accuracy', 0):.4f} | Overall correct predictions |
| Precision | {self.metrics.get('precision', 0):.4f} | Correct risky predictions |
| **Recall** | **{self.metrics.get('recall', 0):.4f}** | **Attack detection rate** |
| F1-Score | {self.metrics.get('f1_score', 0):.4f} | Balance of precision/recall |
| ROC-AUC | {self.metrics.get('roc_auc', 0):.4f} | Discrimination capability |

## Cross-Validation Stability
Accuracy: {self.cv_scores.get('accuracy', {}).get('mean', 0):.4f} (±{self.cv_scores.get('accuracy', {}).get('std', 0):.4f})
Precision: {self.cv_scores.get('precision', {}).get('mean', 0):.4f} (±{self.cv_scores.get('precision', {}).get('std', 0):.4f})
Recall: {self.cv_scores.get('recall', {}).get('mean', 0):.4f} (±{self.cv_scores.get('recall', {}).get('std', 0):.4f})
F1-Score: {self.cv_scores.get('f1', {}).get('mean', 0):.4f} (±{self.cv_scores.get('f1', {}).get('std', 0):.4f})


## Top 5 Features
{self.analyze_feature_importance().head(5).to_string()}

## SOC Deployment Readiness
- **Attack Detection Rate**: {self.metrics.get('recall', 0)*100:.1f}% of attacks detected
- **False Alarm Rate**: {self.metrics.get('false_positive_rate', 0)*100:.1f}% false positives
- **Model Stability**: {'Stable' if self.cv_scores.get('recall', {}).get('std', 0) < 0.05 else 'Needs attention'}

## Next Steps for Production
1. Deploy model via `app.py` CLI for batch processing
2. Integrate with Streamlit dashboard for visualization
3. Monitor false positive rate in production
4. Retrain quarterly with new attack patterns
"""

        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"✓ Training report saved to: {report_path}")


def main():
    """Main execution for model training."""
    print("=" * 60)
    print("CYBERSECURITY RISK PREDICTION - MODEL TRAINING")
    print("=" * 60)
    
    # Initialize trainer
    trainer = CybersecurityModelTrainer(seed=42)
    
    try:
        # 1. Load data
        X_train, X_test, y_train, y_test = trainer.load_data()
        
        # 2. Justify model selection
        trainer._justify_model_selection()
        
        # 3. Perform cross-validation
        cv_results = trainer.perform_cross_validation(X_train, y_train)
        
        # 4. Train model on full training set
        model = trainer.train_model(X_train, y_train)
        
        # 5. Evaluate on test set
        metrics = trainer.evaluate_model(X_test, y_test)
        
        # 6. Analyze feature importance
        importance_df = trainer.analyze_feature_importance()
        
        # 7. Generate visualizations
        trainer.generate_visualizations(X_test, y_test, importance_df)
        
        # 8. Save model and artifacts
        model_path = trainer.save_model()
        
        # 9. Generate training report
        trainer.generate_training_report()
        
        print("\n" + "=" * 60)
        print("MODEL TRAINING COMPLETE ✓")
        print("=" * 60)
        
        # Final summary
        print("\nDEPLOYMENT SUMMARY:")
        print(f"• Model: Random Forest (100 trees)")
        print(f"• Test Accuracy: {metrics['accuracy']:.3f}")
        print(f"• Attack Detection Rate (Recall): {metrics['recall']:.3f}")
        print(f"• False Alarm Rate: {metrics['false_positive_rate']:.3f}")
        print(f"• Model saved: {model_path}")
        print(f"• Reports saved: {trainer.reports_dir}")
        
    except Exception as e:
        print(f"\n❌ Model training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()