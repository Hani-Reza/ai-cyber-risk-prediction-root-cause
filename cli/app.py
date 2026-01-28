"""
Cybersecurity Risk Prediction CLI Application
Production-grade orchestration layer for SOC operations
Author: Principal AI/ML Engineer
Date: 2024
Version: 1.0
"""

import sys
import os
from pathlib import Path
import argparse
import json
import csv
from typing import List, Dict, Any, Optional
from datetime import datetime
import warnings

# Add project root to path for module imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import core modules
from src.risk_scoring import RiskScoringEngine, RiskAssessment
from src.root_cause_analysis import RootCauseAnalyzer, RootCauseAnalysis
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class CybersecurityCLI:
    """
    Command-line interface for cybersecurity risk prediction system.
    SOC analyst workflow orchestration with batch processing capabilities.
    """
    
    def __init__(self):
        """Initialize CLI application with all required engines."""
        self.project_root = project_root
        self.models_dir = project_root / "models"
        self.reports_dir = project_root / "reports"
        self.data_dir = project_root / "data"
        
        # Create necessary directories
        self.reports_dir.mkdir(exist_ok=True)
        
        # Initialize engines
        print("=" * 60)
        print("CYBERSECURITY RISK PREDICTION SYSTEM")
        print("=" * 60)
        print("Initializing engines...")
        
        try:
            self.risk_engine = RiskScoringEngine(seed=42)
            print("✓ Risk Scoring Engine initialized")
            
            self.root_cause_analyzer = RootCauseAnalyzer(seed=42)
            print("✓ Root Cause Analyzer initialized")
            
            print("-" * 60)
            
        except Exception as e:
            print(f"❌ Failed to initialize engines: {e}")
            print("\nPlease ensure you have:")
            print("1. Run data_generator.py")
            print("2. Run preprocessing.py") 
            print("3. Run model_training.py")
            sys.exit(1)
    
    def _complete_event_data(self, partial_event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete partial event data with default values for missing features.
        
        Args:
            partial_event: Partial event data
            
        Returns:
            Complete event data with all required features
        """
        # Default values for a normal user
        default_event = {
            'geo_location_change': 0,
            'privilege_level': 'user',
            'device_trust_score': 75,
            'malware_indicator': 0,
            'system_criticality': 'medium',
            'time_anomaly_score': 0.3
        }
        
        # Update with provided values
        complete_event = default_event.copy()
        complete_event.update(partial_event)
        
        # Add event_id if missing
        if 'event_id' not in complete_event:
            complete_event['event_id'] = f"CLI_{datetime.now().strftime('%H%M%S')}"
        
        return complete_event
    
    def validate_csv_file(self, file_path: Path) -> bool:
        """
        Validate CSV file structure for processing.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            True if valid, False otherwise
        """
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return False
        
        try:
            # Try to read the CSV
            df = pd.read_csv(file_path)
            
            # Check required columns
            required_columns = [
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
            
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                print(f"❌ Missing required columns: {missing_columns}")
                return False
            
            # Check for event_id column, add if missing
            if 'event_id' not in df.columns:
                print("⚠ No event_id column found, generating IDs...")
                df['event_id'] = [f"CLI_{i:06d}" for i in range(len(df))]
            
            print(f"✓ CSV validation passed: {len(df)} rows, {len(df.columns)} columns")
            return True
            
        except Exception as e:
            print(f"❌ CSV validation failed: {e}")
            return False
    
    def process_single_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process single security event through full pipeline.
        
        Args:
            event_data: Dictionary with event features
            
        Returns:
            Dictionary with complete analysis
        """
        try:
            # Complete event data if partial
            if len(event_data) < 9:  # Less than all 9 required features
                print("⚠ Partial event data provided, completing with defaults...")
                event_data = self._complete_event_data(event_data)
            
            # Step 1: Risk Scoring
            risk_assessment = self.risk_engine.calculate_risk(event_data)
            
            # Step 2: Root Cause Analysis
            root_cause_analysis = self.root_cause_analyzer.analyze_root_cause(
                event_data=event_data,
                model_prediction=risk_assessment.probability,
                risk_level=risk_assessment.risk_level.value,
                event_id=risk_assessment.event_id
            )
            
            # Combine results
            result = {
                'risk_assessment': risk_assessment.to_dict(),
                'root_cause_analysis': root_cause_analysis.to_dict(),
                'processing_timestamp': datetime.now().isoformat(),
                'event_data_used': event_data
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Failed to process event: {e}")
            return {
                'error': str(e),
                'event_data': event_data,
                'processing_timestamp': datetime.now().isoformat()
            }
    
    def process_csv_file(self, input_path: Path, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Process CSV file with batch of security events.
        
        Args:
            input_path: Path to input CSV file
            output_path: Optional path for output JSON file
            
        Returns:
            Dictionary with batch processing results
        """
        print(f"\n📂 Processing file: {input_path}")
        
        # Validate file
        if not self.validate_csv_file(input_path):
            return {'status': 'failed', 'error': 'CSV validation failed'}
        
        # Read CSV
        try:
            df = pd.read_csv(input_path)
            print(f"📊 Loaded {len(df)} events from CSV")
            
            # Process each event
            results = []
            successful = 0
            failed = 0
            
            for idx, row in df.iterrows():
                event_data = row.to_dict()
                
                # Show progress
                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1}/{len(df)} events...")
                
                result = self.process_single_event(event_data)
                
                if 'error' in result:
                    failed += 1
                else:
                    successful += 1
                    results.append(result)
            
            # Generate batch report
            batch_report = self._generate_batch_report(results, successful, failed)
            
            # Save results if output path specified
            if output_path:
                self._save_batch_results(results, batch_report, output_path)
            
            return batch_report
            
        except Exception as e:
            print(f"❌ Batch processing failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _generate_batch_report(self, results: List[Dict], successful: int, failed: int) -> Dict[str, Any]:
        """
        Generate comprehensive batch processing report.
        
        Args:
            results: List of processing results
            successful: Number of successful processes
            failed: Number of failed processes
            
        Returns:
            Dictionary with batch report
        """
        if not results:
            return {
                'status': 'failed',
                'message': 'No successful processes'
            }
        
        # Extract metrics
        risk_levels = []
        probabilities = []
        attack_patterns = []
        
        for result in results:
            risk_assessment = result['risk_assessment']
            root_cause = result['root_cause_analysis']
            
            risk_levels.append(risk_assessment['risk_level'])
            probabilities.append(risk_assessment['probability'])
            
            if root_cause['most_likely_attack']:
                attack_patterns.append(root_cause['most_likely_attack'])
        
        # Calculate statistics
        total = successful + failed
        success_rate = (successful / total * 100) if total > 0 else 0
        
        risk_distribution = {}
        for level in risk_levels:
            risk_distribution[level] = risk_distribution.get(level, 0) + 1
        
        attack_pattern_distribution = {}
        for pattern in attack_patterns:
            attack_pattern_distribution[pattern] = attack_pattern_distribution.get(pattern, 0) + 1
        
        report = {
            'summary': {
                'total_events': total,
                'successful_processing': successful,
                'failed_processing': failed,
                'success_rate': round(success_rate, 2),
                'processing_timestamp': datetime.now().isoformat(),
                'system_version': '1.0'
            },
            'risk_analysis': {
                'mean_probability': round(float(np.mean(probabilities)), 4),
                'std_probability': round(float(np.std(probabilities)), 4),
                'high_risk_count': sum(1 for level in risk_levels if level in ['HIGH', 'CRITICAL']),
                'risk_distribution': risk_distribution
            },
            'threat_analysis': {
                'unique_attack_patterns': len(set(attack_patterns)),
                'attack_pattern_distribution': attack_pattern_distribution,
                'most_common_attack': max(attack_pattern_distribution.items(), key=lambda x: x[1])[0] 
                    if attack_pattern_distribution else None
            },
            'soc_recommendations': self._generate_soc_recommendations(risk_distribution, successful)
        }
        
        return report
    
    def _generate_soc_recommendations(self, risk_distribution: Dict, total_events: int) -> List[str]:
        """
        Generate SOC operational recommendations based on batch results.
        
        Args:
            risk_distribution: Distribution of risk levels
            total_events: Total number of events processed
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Calculate high risk percentage
        high_risk_count = risk_distribution.get('HIGH', 0) + risk_distribution.get('CRITICAL', 0)
        high_risk_pct = (high_risk_count / total_events * 100) if total_events > 0 else 0
        
        # Generate recommendations
        if high_risk_pct > 50:
            recommendations.append(
                f"🚨 HIGH ALERT VOLUME: {high_risk_pct:.1f}% events are HIGH/CRITICAL risk. "
                f"Consider activating incident response team."
            )
        elif high_risk_pct > 20:
            recommendations.append(
                f"⚠ ELEVATED RISK: {high_risk_pct:.1f}% events require investigation. "
                f"Ensure SOC has sufficient staffing."
            )
        else:
            recommendations.append(
                f"✅ NORMAL OPERATIONS: Only {high_risk_pct:.1f}% high-risk events. "
                f"Continue routine monitoring."
            )
        
        # Check for critical events
        critical_count = risk_distribution.get('CRITICAL', 0)
        if critical_count > 0:
            recommendations.append(
                f"🔴 CRITICAL EVENTS: {critical_count} events require immediate attention. "
                f"Escalate according to incident response plan."
            )
        
        # Check distribution
        low_risk_pct = (risk_distribution.get('LOW', 0) / total_events * 100) if total_events > 0 else 0
        if low_risk_pct > 80:
            recommendations.append(
                f"📊 THRESHOLD REVIEW: {low_risk_pct:.1f}% events marked LOW risk. "
                f"Consider adjusting sensitivity thresholds."
            )
        
        return recommendations
    
    def _save_batch_results(self, results: List[Dict], report: Dict, output_path: Path) -> None:
        """
        Save batch processing results to JSON file.
        
        Args:
            results: List of processing results
            report: Batch report
            output_path: Path to save output
        """
        try:
            output_data = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'system': 'Cybersecurity Risk Prediction CLI',
                    'version': '1.0'
                },
                'batch_report': report,
                'detailed_results': results[:100]  # Only save first 100 for readability
            }
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"✓ Results saved to: {output_path}")
            print(f"  • {len(results)} events processed")
            print(f"  • First 100 detailed analyses saved")
            
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
    
    def generate_sample_event(self, event_type: str = 'random') -> Dict[str, Any]:
        """
        Generate a sample security event for testing.
        
        Args:
            event_type: 'benign', 'suspicious', 'malicious', or 'random'
            
        Returns:
            Dictionary with sample event data
        """
        samples = {
            'benign': {
                'failed_login_attempts': 1,
                'login_velocity': 0.8,
                'ip_reputation_score': 90,
                'geo_location_change': 0,
                'privilege_level': 'user',
                'device_trust_score': 95,
                'malware_indicator': 0,
                'system_criticality': 'low',
                'time_anomaly_score': 0.1
            },
            'suspicious': {
                'failed_login_attempts': 6,
                'login_velocity': 15.2,
                'ip_reputation_score': 35,
                'geo_location_change': 1,
                'privilege_level': 'admin',
                'device_trust_score': 55,
                'malware_indicator': 0,
                'system_criticality': 'medium',
                'time_anomaly_score': 0.7
            },
            'malicious': {
                'failed_login_attempts': 22,
                'login_velocity': 48.7,
                'ip_reputation_score': 10,
                'geo_location_change': 1,
                'privilege_level': 'super_admin',
                'device_trust_score': 25,
                'malware_indicator': 1,
                'system_criticality': 'critical',
                'time_anomaly_score': 0.95
            }
        }
        
        if event_type == 'random':
            import random
            event_type = random.choice(['benign', 'suspicious', 'malicious'])
        
        sample = samples[event_type].copy()
        sample['event_id'] = f"SAMPLE_{event_type.upper()}_{datetime.now().strftime('%H%M%S')}"
        
        return sample
    
    def real_time_monitoring(self, interval: int = 5, count: int = 10) -> None:
        """
        Simulate real-time security event monitoring.
        
        Args:
            interval: Seconds between simulated events
            count: Number of events to simulate
        """
        import time
        
        print(f"\n🔄 Starting real-time monitoring simulation")
        print(f"   Events: {count}, Interval: {interval}s")
        print("-" * 60)
        
        for i in range(count):
            # Generate random event
            event_types = ['benign', 'benign', 'benign', 'suspicious', 'malicious']  # Weighted
            event_type = event_types[i % len(event_types)] if i < len(event_types) else 'random'
            
            event = self.generate_sample_event(event_type)
            event['event_id'] = f"RT_{i+1:04d}_{datetime.now().strftime('%H%M%S')}"
            
            print(f"\n📡 Event {i+1}/{count}: {event['event_id']}")
            print(f"   Type: {event_type.upper()}")
            
            # Process event
            result = self.process_single_event(event)
            
            if 'risk_assessment' in result:
                risk = result['risk_assessment']
                print(f"   ⚡ Risk Level: {risk['risk_level']}")
                print(f"   📊 Probability: {risk['probability']:.3f}")
                print(f"   🎯 Action: {risk['recommended_action'][:50]}...")
            
            if i < count - 1:
                print(f"   ⏳ Waiting {interval} seconds...")
                time.sleep(interval)
        
        print("\n" + "=" * 60)
        print("REAL-TIME MONITORING COMPLETE ✓")
        print("=" * 60)
    
    def system_status(self) -> Dict[str, Any]:
        """
        Check system status and component health.
        
        Returns:
            Dictionary with system status
        """
        print("\n" + "=" * 60)
        print("SYSTEM STATUS CHECK")
        print("=" * 60)
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'overall': 'HEALTHY'
        }
        
        # Check model files
        model_files = [
            ('risk_classifier.pkl', 'Trained ML Model'),
            ('preprocessor.pkl', 'Feature Preprocessor'),
            ('feature_names.pkl', 'Feature Names'),
            ('scaler.pkl', 'Scaler')
        ]
        
        print("\n📁 Model Artifacts:")
        for filename, description in model_files:
            filepath = self.models_dir / filename
            if filepath.exists():
                size_kb = filepath.stat().st_size / 1024
                status['components'][filename] = {
                    'status': 'OK',
                    'size_kb': round(size_kb, 2),
                    'description': description
                }
                print(f"  ✓ {description}: {size_kb:.1f} KB")
            else:
                status['components'][filename] = {
                    'status': 'MISSING',
                    'description': description
                }
                print(f"  ❌ {description}: File not found")
                status['overall'] = 'DEGRADED'
        
        # Check data directories
        print("\n📂 Data Directories:")
        data_dirs = ['raw', 'processed']
        for dirname in data_dirs:
            dirpath = self.data_dir / dirname
            if dirpath.exists():
                file_count = len(list(dirpath.glob('*')))
                print(f"  ✓ {dirname}/: {file_count} files")
            else:
                print(f"  ❌ {dirname}/: Directory not found")
        
        # Check report directory
        print(f"\n📊 Report Directory: {self.reports_dir}")
        report_files = list(self.reports_dir.glob('*'))
        print(f"  ✓ Contains {len(report_files)} report files")
        
        # System information
        print("\n💻 System Information:")
        print(f"  Python: {sys.version.split()[0]}")
        print(f"  Platform: {sys.platform}")
        
        return status


def main():
    """Main CLI entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Cybersecurity Risk Prediction CLI - SOC Analyst Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a CSV file
  python app.py process --input data/raw/security_events.csv --output reports/batch_analysis.json
  
  # Analyze single event (simplified - just key metrics)
  python app.py analyze --failed-logins 15 --login-velocity 25.7 --ip-reputation 20
  
  # Real-time monitoring simulation
  python app.py monitor --count 20 --interval 2
  
  # Check system status
  python app.py status
  
  # Generate sample CSV
  python app.py sample --count 100 --output data/sample_events.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Process CSV command
    process_parser = subparsers.add_parser('process', help='Process CSV file with security events')
    process_parser.add_argument('--input', type=str, required=True, 
                               help='Input CSV file path')
    process_parser.add_argument('--output', type=str, 
                               help='Output JSON file path (optional)')
    
    # Analyze single event command (simplified version)
    analyze_parser = subparsers.add_parser('analyze', help='Analyze single security event')
    analyze_parser.add_argument('--failed-logins', type=int, default=1,
                               help='Number of failed login attempts')
    analyze_parser.add_argument('--login-velocity', type=float, default=1.0,
                               help='Login attempts per hour')
    analyze_parser.add_argument('--ip-reputation', type=float, default=80.0,
                               help='IP reputation score (0-100, higher is better)')
    analyze_parser.add_argument('--geo-change', type=int, choices=[0, 1], default=0,
                               help='Geographic location change (0=no, 1=yes)')
    analyze_parser.add_argument('--privilege', type=str, 
                               choices=['user', 'admin', 'super_admin', 'system'], 
                               default='user',
                               help='User privilege level')
    analyze_parser.add_argument('--device-trust', type=float, default=80.0,
                               help='Device trust score (0-100, higher is better)')
    analyze_parser.add_argument('--malware', type=int, choices=[0, 1], default=0,
                               help='Malware indicator (0=no, 1=yes)')
    analyze_parser.add_argument('--criticality', type=str,
                               choices=['low', 'medium', 'high', 'critical'],
                               default='medium',
                               help='System criticality level')
    analyze_parser.add_argument('--time-anomaly', type=float, default=0.3,
                               help='Time anomaly score (0-1)')
    
    # Real-time monitoring command
    monitor_parser = subparsers.add_parser('monitor', help='Simulate real-time monitoring')
    monitor_parser.add_argument('--count', type=int, default=10,
                               help='Number of events to simulate')
    monitor_parser.add_argument('--interval', type=int, default=5,
                               help='Seconds between events')
    
    # System status command
    subparsers.add_parser('status', help='Check system status and health')
    
    # Generate sample data command
    sample_parser = subparsers.add_parser('sample', help='Generate sample CSV file')
    sample_parser.add_argument('--count', type=int, default=50,
                              help='Number of sample events to generate')
    sample_parser.add_argument('--output', type=str, required=True,
                              help='Output CSV file path')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize CLI
    cli = CybersecurityCLI()
    
    # Execute command
    if args.command == 'process':
        input_path = Path(args.input)
        output_path = Path(args.output) if args.output else None
        
        if not output_path:
            # Generate default output path
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = cli.reports_dir / f"batch_analysis_{timestamp}.json"
        
        result = cli.process_csv_file(input_path, output_path)
        
        # Print summary
        if 'summary' in result:
            summary = result['summary']
            print(f"\n" + "=" * 60)
            print("BATCH PROCESSING COMPLETE")
            print("=" * 60)
            print(f"Total Events: {summary.get('total_events', 0)}")
            print(f"Successful: {summary.get('successful_processing', 0)}")
            print(f"Failed: {summary.get('failed_processing', 0)}")
            print(f"Success Rate: {summary.get('success_rate', 0)}%")
            
            if 'risk_analysis' in result:
                risk = result['risk_analysis']
                print(f"\nRisk Analysis:")
                print(f"  Mean Probability: {risk.get('mean_probability', 0):.3f}")
                print(f"  High Risk Events: {risk.get('high_risk_count', 0)}")
            
            if 'soc_recommendations' in result:
                print(f"\nSOC Recommendations:")
                for rec in result['soc_recommendations']:
                    print(f"  • {rec}")
    
    elif args.command == 'analyze':
        # Build event data from arguments
        event_data = {
            'failed_login_attempts': args.failed_logins,
            'login_velocity': args.login_velocity,
            'ip_reputation_score': args.ip_reputation,
            'geo_location_change': args.geo_change,
            'privilege_level': args.privilege,
            'device_trust_score': args.device_trust,
            'malware_indicator': args.malware,
            'system_criticality': args.criticality,
            'time_anomaly_score': args.time_anomaly
        }
        
        print("\n" + "=" * 60)
        print("EVENT ANALYSIS")
        print("=" * 60)
        print("Event Data:")
        for key, value in event_data.items():
            print(f"  {key}: {value}")
        
        result = cli.process_single_event(event_data)
        
        if 'risk_assessment' in result:
            risk = result['risk_assessment']
            root_cause = result['root_cause_analysis']
            
            print(f"\n" + "=" * 60)
            print("ANALYSIS RESULTS")
            print("=" * 60)
            print(f"Event ID: {risk['event_id']}")
            print(f"Risk Level: {risk['risk_level']}")
            print(f"Probability: {risk['probability']:.3f}")
            print(f"Confidence: {risk['confidence']:.3f}")
            print(f"\nRecommended Action: {risk['recommended_action']}")
            
            if root_cause['most_likely_attack']:
                print(f"\n🎯 Most Likely Attack Pattern: {root_cause['most_likely_attack']}")
            
            if root_cause['top_findings']:
                print(f"\n🔍 Top Contributing Factors:")
                for i, finding in enumerate(root_cause['top_findings'][:3], 1):
                    print(f"  {i}. {finding['feature']}")
                    print(f"     Value: {finding['value']}")
                    print(f"     Explanation: {finding['explanation']}")
                    print(f"     Contribution: {finding['contribution_score']:.3f}")
                    print(f"     Severity: {finding['severity']}")
            
            print(f"\n📝 Summary:\n{root_cause['summary']}")
        
        elif 'error' in result:
            print(f"\n❌ Analysis failed: {result['error']}")
    
    elif args.command == 'monitor':
        cli.real_time_monitoring(interval=args.interval, count=args.count)
    
    elif args.command == 'status':
        status = cli.system_status()
        print(f"\nOverall Status: {status['overall']}")
    
    elif args.command == 'sample':
        # Generate sample CSV
        samples = []
        event_types = ['benign', 'suspicious', 'malicious']
        
        for i in range(args.count):
            # Mix of event types
            event_type = event_types[i % len(event_types)]
            event = cli.generate_sample_event(event_type)
            event['event_id'] = f"SAMPLE_{i+1:04d}"
            samples.append(event)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(samples)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        print(f"✓ Generated {len(samples)} sample events")
        print(f"✓ Saved to: {output_path}")
        print(f"\nSample of generated data (first 3 rows):")
        print(df.head(3).to_string())
    
    else:
        # No command provided, show help
        parser.print_help()
        
        # Show quick system status
        print("\n" + "=" * 60)
        print("QUICK START")
        print("=" * 60)
        print("To get started, try:")
        print("  1. python app.py status           # Check system health")
        print("  2. python app.py monitor          # Simulate real-time monitoring")
        print("  3. python app.py analyze --failed-logins 15 --login-velocity 25.7 --ip-reputation 20")
        print("  4. python app.py process --input data/raw/security_events.csv")
        print("\nFor more details: python app.py --help")


if __name__ == "__main__":
    main()