#!/usr/bin/env python3
"""
Real-Time Explainable AI Cybersecurity Demonstration Pipeline

This script demonstrates the complete explainable AI pipeline for network intrusion detection
using the UNSW-NB15 dataset. It simulates real-time data streaming without requiring external
infrastructure like Kafka, showcasing predictions, explanations, and dashboard integration.

Features:
- Real-time data streaming simulation
- Live predictions with explanations
- Interactive console dashboard
- Multiple demo modes (fast, extended, interactive)
- Performance metrics and monitoring
- Export capabilities for analysis

Author: Explainable AI Anomaly Detection System
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict
import argparse
from collections import deque, defaultdict
import joblib
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
from src.config.config import get_config
from src.data.preprocessor import NetworkDataPreprocessor
from src.data.feature_engineering import NetworkFeatureEngineer
from src.models.model_trainer import ModelTrainer
from src.explainability.realtime_explainer import (
    RealtimeExplainer, RealtimeExplanationResult
)
from src.explainability.dashboard_interface import (
    DashboardDataManager, DashboardAlert, AlertLevel
)

# ANSI color codes for console output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    # Risk level colors
    CRITICAL = '\033[95m'  # Magenta
    HIGH = '\033[91m'      # Red
    MEDIUM = '\033[93m'    # Yellow
    LOW = '\033[92m'       # Green
    NORMAL = '\033[96m'    # Cyan

class RealtimeDemonstrator:
    """Main demonstration class for the real-time explainable AI pipeline"""
    
    def __init__(self, config_env: str = 'development'):
        """Initialize the demonstrator with configuration"""
        self.config = get_config(config_env)
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.model = None
        self.preprocessor = None
        self.feature_engineer = None
        self.realtime_explainer = None
        self.dashboard_manager = None
        
        # Data
        self.test_data = None
        self.original_test_data = None
        self.feature_names = None
        
        # Demo state
        self.demo_running = False
        self.demo_stats = {
            'start_time': None,
            'samples_processed': 0,
            'attacks_detected': 0,
            'normal_detected': 0,
            'total_processing_time': 0.0,
            'prediction_history': deque(maxlen=1000),
            'attack_types_detected': defaultdict(int),
            'confidence_scores': deque(maxlen=100)
        }
        
        # Configuration
        self.demo_config = {
            'sample_interval': 1.0,  # seconds between samples
            'batch_size': 1,
            'max_samples': 100,
            'show_explanations': True,
            'show_feature_details': False,
            'console_width': 80,
            'export_results': True
        }
        
        # Attack type mapping
        self.attack_mapping = {
            'Normal': 'Normal Traffic',
            'Generic': 'Generic Attack',
            'Exploits': 'Exploit Attack', 
            'Fuzzers': 'Fuzzing Attack',
            'DoS': 'Denial of Service',
            'Reconnaissance': 'Network Reconnaissance',
            'Analysis': 'Network Analysis',
            'Backdoor': 'Backdoor Attack',
            'Shellcode': 'Shellcode Injection',
            'Worms': 'Worm Attack'
        }

    def initialize_system(self) -> bool:
        """Initialize all system components"""
        print(f"{Colors.HEADER}🔧 Initializing Explainable AI System...{Colors.ENDC}")
        
        try:
            # Load models and preprocessors
            if not self._load_trained_components():
                return False
            
            # Load test data
            if not self._load_test_data():
                return False
                
            # Initialize explainer
            if not self._initialize_explainer():
                return False
                
            # Initialize dashboard
            self._initialize_dashboard()
            
            print(f"{Colors.OKGREEN}✅ System initialization complete!{Colors.ENDC}\n")
            return True
            
        except Exception as e:
            print(f"{Colors.FAIL}❌ System initialization failed: {e}{Colors.ENDC}")
            return False

    def _load_trained_components(self) -> bool:
        """Load trained models and preprocessors"""
        print("📂 Loading trained models and preprocessors...")
        
        # Check if models exist
        models_dir = Path(self.config.MODELS_DIR)
        preprocessing_dir = Path(self.config.PREPROCESSING_DIR)
        
        if not models_dir.exists():
            print(f"{Colors.FAIL}❌ Models directory not found: {models_dir}{Colors.ENDC}")
            print("Please run 'python train_models.py' first to train models.")
            return False
            
        # Load preprocessor
        preprocessor_path = preprocessing_dir / 'preprocessor.joblib'
        if preprocessor_path.exists():
            self.preprocessor = joblib.load(preprocessor_path)
            print(f"{Colors.OKGREEN}✓{Colors.ENDC} Loaded preprocessor")
        else:
            print(f"{Colors.WARNING}⚠ Preprocessor not found, will create new one{Colors.ENDC}")
            
        # Load best model
        trainer = ModelTrainer(results_dir=self.config.MODELS_DIR)
        try:
            best_model_name, self.model = trainer.get_best_model()
            if self.model is None:
                # Try loading any available model
                model_files = list(models_dir.glob('*.joblib'))
                if model_files:
                    model_path = model_files[0]
                    self.model = joblib.load(model_path)
                    best_model_name = model_path.stem
                    print(f"{Colors.OKGREEN}✓{Colors.ENDC} Loaded model: {best_model_name}")
                else:
                    print(f"{Colors.FAIL}❌ No trained models found{Colors.ENDC}")
                    return False
            else:
                print(f"{Colors.OKGREEN}✓{Colors.ENDC} Loaded best model: {best_model_name}")
        except Exception as e:
            print(f"{Colors.WARNING}⚠ Could not load best model: {e}{Colors.ENDC}")
            return False
            
        return True

    def _load_test_data(self) -> bool:
        """Load and preprocess test data"""
        print("📊 Loading UNSW-NB15 test data...")
        
        try:
            # Load raw test data
            if not os.path.exists(self.config.TEST_DATA_PATH):
                print(f"{Colors.FAIL}❌ Test data not found: {self.config.TEST_DATA_PATH}{Colors.ENDC}")
                return False
                
            self.original_test_data = pd.read_csv(self.config.TEST_DATA_PATH)
            print(f"{Colors.OKGREEN}✓{Colors.ENDC} Loaded {len(self.original_test_data):,} test samples")
            
            # If we have a preprocessor, use it; otherwise create new preprocessing
            if self.preprocessor:
                # Transform test data using existing preprocessor
                self.test_data = self.preprocessor.transform(self.original_test_data)
                self.feature_names = self.preprocessor.feature_names_
                print(f"{Colors.OKGREEN}✓{Colors.ENDC} Preprocessed data: {self.test_data.shape}")
            else:
                # Create new preprocessing pipeline
                print("🔄 Creating new preprocessing pipeline...")
                self.preprocessor = NetworkDataPreprocessor(
                    categorical_columns=self.config.CATEGORICAL_COLUMNS,
                    target_column=self.config.TARGET_COLUMN,
                    attack_category_column=self.config.ATTACK_CATEGORY_COLUMN,
                    scaler_type=self.config.SCALER_TYPE
                )
                
                # We need training data to fit the preprocessor
                if os.path.exists(self.config.TRAIN_DATA_PATH):
                    train_df = pd.read_csv(self.config.TRAIN_DATA_PATH)
                    self.test_data, _, _ = self.preprocessor.fit_transform(train_df)
                    self.test_data = self.preprocessor.transform(self.original_test_data)
                    self.feature_names = self.preprocessor.feature_names_
                else:
                    print(f"{Colors.FAIL}❌ Training data needed for preprocessing{Colors.ENDC}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"{Colors.FAIL}❌ Failed to load test data: {e}{Colors.ENDC}")
            return False

    def _initialize_explainer(self) -> bool:
        """Initialize the real-time explainer"""
        print("🔍 Initializing real-time explainer...")
        
        try:
            # Use a subset of data for background (SHAP requirement)
            background_data = self.test_data[:100].values
            
            self.realtime_explainer = RealtimeExplainer(
                model=self.model,
                training_data=background_data,
                feature_names=list(self.feature_names),
                max_workers=2,  # Reduced for demo stability
                queue_size=500
            )
            
            # Configure for demonstration
            self.realtime_explainer.update_configuration({
                'enable_shap': True,
                'enable_lime': False,  # Disabled for performance
                'fast_mode': True,
                'cache_explanations': True,
                'shap_background_size': 50  # Smaller for faster processing
            })
            
            # Start processing
            self.realtime_explainer.start_processing()
            
            # Wait a moment for initialization
            time.sleep(2)
            
            print(f"{Colors.OKGREEN}✓{Colors.ENDC} Real-time explainer ready")
            return True
            
        except Exception as e:
            print(f"{Colors.FAIL}❌ Failed to initialize explainer: {e}{Colors.ENDC}")
            return False

    def _initialize_dashboard(self):
        """Initialize dashboard manager"""
        print("📊 Initializing dashboard manager...")
        
        self.dashboard_manager = DashboardDataManager(
            max_detections=500,
            max_alerts=50,
            metrics_window_minutes=10
        )
        
        # Subscribe to alerts for console display
        def alert_callback(alert):
            self._display_alert(alert)
        
        self.dashboard_manager.subscribe_to_alerts(alert_callback)
        print(f"{Colors.OKGREEN}✓{Colors.ENDC} Dashboard manager ready")

    def print_system_info(self):
        """Print system information and capabilities"""
        print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}🛡️  EXPLAINABLE AI CYBERSECURITY DEMONSTRATION{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
        print()
        
        # Model information
        model_name = type(self.model).__name__ if self.model else "Unknown"
        print(f"{Colors.OKBLUE}🤖 Model:{Colors.ENDC} {model_name}")
        print(f"{Colors.OKBLUE}📊 Features:{Colors.ENDC} {len(self.feature_names)} network features")
        print(f"{Colors.OKBLUE}🔍 Explainer:{Colors.ENDC} SHAP-based real-time explanations")
        print(f"{Colors.OKBLUE}📈 Data:{Colors.ENDC} UNSW-NB15 dataset ({len(self.test_data):,} samples)")
        print()
        
        # Attack categories
        if self.config.ATTACK_CATEGORY_COLUMN in self.original_test_data.columns:
            attack_dist = self.original_test_data[self.config.ATTACK_CATEGORY_COLUMN].value_counts()
            print(f"{Colors.OKBLUE}🎯 Attack Types Available:{Colors.ENDC}")
            for attack_type, count in attack_dist.head(5).items():
                display_name = self.attack_mapping.get(attack_type, attack_type)
                print(f"   • {display_name}: {count:,} samples")
            if len(attack_dist) > 5:
                print(f"   • ... and {len(attack_dist) - 5} more types")
            print()

    def run_demo_mode(self, mode: str = 'fast'):
        """Run demonstration in specified mode"""
        print(f"{Colors.HEADER}🚀 Starting {mode.title()} Demo Mode{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
        
        # Configure demo based on mode
        if mode == 'fast':
            self.demo_config.update({
                'sample_interval': 0.5,
                'max_samples': 20,
                'show_feature_details': False
            })
            print(f"{Colors.OKCYAN}⚡ Fast demo: 20 samples, 0.5s intervals{Colors.ENDC}")
            
        elif mode == 'extended':
            self.demo_config.update({
                'sample_interval': 1.0,
                'max_samples': 50,
                'show_feature_details': True
            })
            print(f"{Colors.OKCYAN}📈 Extended demo: 50 samples, 1s intervals{Colors.ENDC}")
            
        elif mode == 'interactive':
            return self._run_interactive_mode()
            
        print()
        input(f"{Colors.WARNING}Press Enter to start the demonstration...{Colors.ENDC}")
        print("\n" + f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
        
        # Start streaming simulation
        self._simulate_realtime_stream()

    def _simulate_realtime_stream(self):
        """Simulate real-time data streaming with live predictions"""
        self.demo_running = True
        self.demo_stats['start_time'] = datetime.now()
        
        # Print header
        self._print_stream_header()
        
        # Select random samples for diverse demonstration
        sample_indices = np.random.choice(
            len(self.test_data), 
            size=min(self.demo_config['max_samples'], len(self.test_data)),
            replace=False
        )
        
        try:
            for i, sample_idx in enumerate(sample_indices):
                if not self.demo_running:
                    break
                    
                # Get sample and metadata
                sample = self.test_data.iloc[sample_idx].values
                original_sample = self.original_test_data.iloc[sample_idx]
                
                # Get ground truth
                actual_label = original_sample[self.config.TARGET_COLUMN]
                actual_attack_type = original_sample.get(self.config.ATTACK_CATEGORY_COLUMN, 'Unknown')
                
                # Process sample
                self._process_sample_realtime(
                    sample=sample,
                    sample_id=f"stream_{i:04d}",
                    actual_label=actual_label,
                    actual_attack_type=actual_attack_type,
                    sample_index=i + 1
                )
                
                # Update stats
                self.demo_stats['samples_processed'] = i + 1
                
                # Show progress periodically
                if (i + 1) % 10 == 0:
                    self._print_interim_stats()
                
                # Wait for next sample
                if i < len(sample_indices) - 1:  # Don't wait after last sample
                    time.sleep(self.demo_config['sample_interval'])
                    
        except KeyboardInterrupt:
            print(f"\n{Colors.WARNING}🛑 Demo interrupted by user{Colors.ENDC}")
        
        finally:
            self.demo_running = False
            self._print_final_summary()

    def _process_sample_realtime(self, sample: np.ndarray, sample_id: str, 
                               actual_label: int, actual_attack_type: str, sample_index: int):
        """Process a single sample in real-time"""
        
        # Get explanation
        start_time = time.time()
        result = self.realtime_explainer.explain_sample_sync(
            sample=sample,
            sample_id=sample_id,
            methods=["shap"],
            timeout=5.0
        )
        processing_time = (time.time() - start_time) * 1000
        
        if result.success and result.explanation:
            explanation = result.explanation
            
            # Extract prediction info
            prediction = explanation.prediction
            confidence = explanation.confidence_score or 0.0
            predicted_attack_type = explanation.predicted_attack_type or 'Unknown'
            risk_level = explanation.risk_level or 'unknown'
            
            # Add to dashboard
            self.dashboard_manager.add_detection(result)
            
            # Update stats
            self.demo_stats['total_processing_time'] += processing_time
            self.demo_stats['confidence_scores'].append(confidence)
            
            if prediction == 1:
                self.demo_stats['attacks_detected'] += 1
                self.demo_stats['attack_types_detected'][predicted_attack_type] += 1
            else:
                self.demo_stats['normal_detected'] += 1
            
            # Store prediction history
            self.demo_stats['prediction_history'].append({
                'timestamp': datetime.now().isoformat(),
                'prediction': prediction,
                'actual': actual_label,
                'confidence': confidence,
                'processing_time_ms': processing_time,
                'risk_level': risk_level
            })
            
            # Display result
            self._display_prediction_result(
                sample_index=sample_index,
                prediction=prediction,
                actual_label=actual_label,
                confidence=confidence,
                risk_level=risk_level,
                predicted_attack_type=predicted_attack_type,
                actual_attack_type=actual_attack_type,
                processing_time=processing_time,
                explanation=explanation if self.demo_config['show_explanations'] else None
            )
            
        else:
            # Handle failed explanation
            print(f"{Colors.FAIL}❌ Sample {sample_index:3d}: Explanation failed - {result.error}{Colors.ENDC}")

    def _display_prediction_result(self, sample_index: int, prediction: int, actual_label: int,
                                 confidence: float, risk_level: str, predicted_attack_type: str,
                                 actual_attack_type: str, processing_time: float, explanation=None):
        """Display a single prediction result with formatting"""
        
        # Determine colors
        pred_color = Colors.FAIL if prediction == 1 else Colors.OKGREEN
        correct = "✓" if prediction == actual_label else "✗"
        correct_color = Colors.OKGREEN if prediction == actual_label else Colors.FAIL
        
        # Risk level coloring
        risk_colors = {
            'critical': Colors.CRITICAL,
            'high': Colors.HIGH, 
            'medium': Colors.MEDIUM,
            'low': Colors.LOW,
            'normal': Colors.NORMAL,
            'unknown': Colors.ENDC
        }
        risk_color = risk_colors.get(risk_level.lower(), Colors.ENDC)
        
        # Format prediction
        pred_text = "🚨 ATTACK" if prediction == 1 else "✅ NORMAL"
        actual_text = "ATTACK" if actual_label == 1 else "NORMAL"
        
        # Display main prediction
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{Colors.OKBLUE}[{timestamp}]{Colors.ENDC} "
              f"Sample {sample_index:3d}: "
              f"{pred_color}{pred_text}{Colors.ENDC} "
              f"({confidence:.1%}) "
              f"{correct_color}{correct}{Colors.ENDC} "
              f"vs {actual_text} "
              f"{risk_color}[{risk_level.upper()}]{Colors.ENDC} "
              f"({processing_time:.1f}ms)")
        
        # Show attack types if relevant
        if prediction == 1 or actual_label == 1:
            predicted_display = self.attack_mapping.get(predicted_attack_type, predicted_attack_type)
            actual_display = self.attack_mapping.get(actual_attack_type, actual_attack_type)
            print(f"         Attack Type: {predicted_display} vs {actual_display}")
        
        # Show top features if enabled
        if explanation and self.demo_config['show_feature_details']:
            self._display_feature_explanation(explanation)
        
        print()  # Extra spacing

    def _display_feature_explanation(self, explanation):
        """Display feature importance explanation"""
        if hasattr(explanation, 'consensus_features') and explanation.consensus_features:
            features = explanation.consensus_features[:3]  # Top 3
            print(f"         {Colors.OKCYAN}Top Features:{Colors.ENDC}")
            for feature in features:
                importance = feature.get('avg_importance', 0)
                feature_name = feature.get('feature', 'unknown')
                direction = "↑" if importance > 0 else "↓"
                print(f"           {direction} {feature_name}: {abs(importance):.3f}")

    def _display_alert(self, alert):
        """Display dashboard alert"""
        level_colors = {
            AlertLevel.CRITICAL: Colors.CRITICAL,
            AlertLevel.HIGH: Colors.HIGH,
            AlertLevel.MEDIUM: Colors.WARNING,
            AlertLevel.LOW: Colors.OKGREEN,
            AlertLevel.INFO: Colors.OKCYAN
        }
        
        color = level_colors.get(alert.level, Colors.ENDC)
        print(f"{color}🚨 ALERT [{alert.level.value.upper()}]: {alert.title}{Colors.ENDC}")
        print(f"    {alert.message}")
        print()

    def _print_stream_header(self):
        """Print streaming header"""
        print(f"{Colors.HEADER}📡 REAL-TIME NETWORK MONITORING{Colors.ENDC}")
        print(f"{Colors.HEADER}{'-'*80}{Colors.ENDC}")
        print(f"Timestamp    Sample  Prediction     Accuracy  Risk      Processing")
        print(f"{Colors.HEADER}{'-'*80}{Colors.ENDC}")

    def _print_interim_stats(self):
        """Print interim statistics"""
        elapsed = (datetime.now() - self.demo_stats['start_time']).total_seconds()
        throughput = self.demo_stats['samples_processed'] / elapsed if elapsed > 0 else 0
        
        print(f"{Colors.OKCYAN}📊 Interim Stats: "
              f"{self.demo_stats['samples_processed']} samples, "
              f"{self.demo_stats['attacks_detected']} attacks detected, "
              f"{throughput:.1f} samples/sec{Colors.ENDC}")
        print()

    def _print_final_summary(self):
        """Print final demonstration summary"""
        elapsed = (datetime.now() - self.demo_stats['start_time']).total_seconds()
        
        print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}📊 DEMONSTRATION SUMMARY{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
        
        # Basic stats
        total_samples = self.demo_stats['samples_processed']
        attacks_detected = self.demo_stats['attacks_detected']
        normal_detected = self.demo_stats['normal_detected']
        
        print(f"{Colors.OKBLUE}🎯 Samples Processed:{Colors.ENDC} {total_samples:,}")
        print(f"{Colors.OKBLUE}🚨 Attacks Detected:{Colors.ENDC} {attacks_detected:,}")
        print(f"{Colors.OKBLUE}✅ Normal Traffic:{Colors.ENDC} {normal_detected:,}")
        print(f"{Colors.OKBLUE}⏱️  Total Runtime:{Colors.ENDC} {elapsed:.1f} seconds")
        
        if total_samples > 0:
            avg_processing_time = self.demo_stats['total_processing_time'] / total_samples
            throughput = total_samples / elapsed if elapsed > 0 else 0
            detection_rate = attacks_detected / total_samples
            
            print(f"{Colors.OKBLUE}⚡ Avg Processing Time:{Colors.ENDC} {avg_processing_time:.1f}ms per sample")
            print(f"{Colors.OKBLUE}📈 Throughput:{Colors.ENDC} {throughput:.1f} samples/second")
            print(f"{Colors.OKBLUE}🔍 Attack Detection Rate:{Colors.ENDC} {detection_rate:.1%}")
        
        # Confidence statistics
        if self.demo_stats['confidence_scores']:
            confidences = list(self.demo_stats['confidence_scores'])
            avg_confidence = np.mean(confidences)
            print(f"{Colors.OKBLUE}🎯 Average Confidence:{Colors.ENDC} {avg_confidence:.1%}")
        
        # Attack types detected
        if self.demo_stats['attack_types_detected']:
            print(f"\n{Colors.OKBLUE}🎯 Attack Types Detected:{Colors.ENDC}")
            for attack_type, count in self.demo_stats['attack_types_detected'].items():
                display_name = self.attack_mapping.get(attack_type, attack_type)
                print(f"   • {display_name}: {count}")
        
        # System performance
        explainer_stats = self.realtime_explainer.get_performance_stats()
        print(f"\n{Colors.OKBLUE}🔧 System Performance:{Colors.ENDC}")
        print(f"   • Cache Hit Rate: {explainer_stats['cache_size']} cached explanations")
        print(f"   • Success Rate: {explainer_stats['successful_explanations']}/{explainer_stats['total_requests']}")
        
        # Export results if configured
        if self.demo_config['export_results']:
            self._export_demo_results()
        
        print(f"\n{Colors.OKGREEN}✅ Demonstration completed successfully!{Colors.ENDC}")

    def _run_interactive_mode(self):
        """Run interactive demonstration mode"""
        print(f"{Colors.HEADER}🎮 Interactive Demo Mode{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
        print()
        print("Interactive commands:")
        print("  'sample [N]' - Process N random samples (default: 1)")
        print("  'attack'     - Process a random attack sample")  
        print("  'normal'     - Process a random normal sample")
        print("  'batch [N]'  - Process N samples quickly")
        print("  'stats'      - Show current statistics")
        print("  'config'     - Show configuration")
        print("  'quit'       - Exit interactive mode")
        print()
        
        self.demo_stats['start_time'] = datetime.now()
        
        while True:
            try:
                command = input(f"{Colors.OKCYAN}demo> {Colors.ENDC}").strip().lower()
                
                if command in ['quit', 'exit', 'q']:
                    break
                elif command.startswith('sample'):
                    parts = command.split()
                    n = int(parts[1]) if len(parts) > 1 else 1
                    self._process_random_samples(n)
                elif command == 'attack':
                    self._process_specific_sample_type(target_label=1)
                elif command == 'normal':
                    self._process_specific_sample_type(target_label=0)
                elif command.startswith('batch'):
                    parts = command.split()
                    n = int(parts[1]) if len(parts) > 1 else 5
                    self._process_batch_samples(n)
                elif command == 'stats':
                    self._show_interactive_stats()
                elif command == 'config':
                    self._show_configuration()
                elif command == 'help':
                    self._show_interactive_help()
                else:
                    print(f"{Colors.WARNING}Unknown command. Type 'help' for available commands.{Colors.ENDC}")
                    
            except KeyboardInterrupt:
                print(f"\n{Colors.WARNING}Exiting interactive mode...{Colors.ENDC}")
                break
            except Exception as e:
                print(f"{Colors.FAIL}Error: {e}{Colors.ENDC}")
        
        self._print_final_summary()

    def _process_random_samples(self, n: int):
        """Process N random samples"""
        indices = np.random.choice(len(self.test_data), size=n, replace=False)
        
        for i, idx in enumerate(indices):
            sample = self.test_data.iloc[idx].values
            original = self.original_test_data.iloc[idx]
            
            self._process_sample_realtime(
                sample=sample,
                sample_id=f"interactive_{self.demo_stats['samples_processed']:04d}",
                actual_label=original[self.config.TARGET_COLUMN],
                actual_attack_type=original.get(self.config.ATTACK_CATEGORY_COLUMN, 'Unknown'),
                sample_index=self.demo_stats['samples_processed'] + 1
            )
            
            self.demo_stats['samples_processed'] += 1

    def _process_specific_sample_type(self, target_label: int):
        """Process a sample of specific type (attack/normal)"""
        # Find samples with target label
        mask = self.original_test_data[self.config.TARGET_COLUMN] == target_label
        matching_indices = self.original_test_data[mask].index.tolist()
        
        if not matching_indices:
            print(f"{Colors.WARNING}No samples found with label {target_label}{Colors.ENDC}")
            return
        
        # Select random matching sample
        idx = np.random.choice(matching_indices)
        sample = self.test_data.iloc[idx].values
        original = self.original_test_data.iloc[idx]
        
        self._process_sample_realtime(
            sample=sample,
            sample_id=f"targeted_{self.demo_stats['samples_processed']:04d}",
            actual_label=original[self.config.TARGET_COLUMN],
            actual_attack_type=original.get(self.config.ATTACK_CATEGORY_COLUMN, 'Unknown'),
            sample_index=self.demo_stats['samples_processed'] + 1
        )
        
        self.demo_stats['samples_processed'] += 1

    def _process_batch_samples(self, n: int):
        """Process N samples in batch mode"""
        print(f"{Colors.OKCYAN}Processing {n} samples in batch mode...{Colors.ENDC}")
        
        indices = np.random.choice(len(self.test_data), size=n, replace=False)
        samples = self.test_data.iloc[indices].values
        
        # Use batch processing
        results = self.realtime_explainer.explain_batch_realtime(samples, f"batch_{int(time.time())}")
        
        # Display results
        for i, (result, idx) in enumerate(zip(results, indices)):
            if result.success:
                original = self.original_test_data.iloc[idx]
                prediction = result.explanation.prediction if result.explanation else 0
                confidence = result.explanation.confidence_score if result.explanation else 0.0
                
                pred_text = "ATTACK" if prediction == 1 else "NORMAL"
                actual_label = original[self.config.TARGET_COLUMN]
                actual_text = "ATTACK" if actual_label == 1 else "NORMAL"
                correct = "✓" if prediction == actual_label else "✗"
                
                print(f"  {i+1:2d}. {pred_text} ({confidence:.1%}) {correct} vs {actual_text} "
                      f"({result.processing_time_ms:.1f}ms)")
        
        self.demo_stats['samples_processed'] += len(results)
        print(f"{Colors.OKGREEN}Batch processing complete!{Colors.ENDC}")

    def _show_interactive_stats(self):
        """Show current statistics in interactive mode"""
        elapsed = (datetime.now() - self.demo_stats['start_time']).total_seconds()
        
        print(f"\n{Colors.OKBLUE}📊 Current Statistics:{Colors.ENDC}")
        print(f"  Samples Processed: {self.demo_stats['samples_processed']}")
        print(f"  Attacks Detected: {self.demo_stats['attacks_detected']}")
        print(f"  Normal Detected: {self.demo_stats['normal_detected']}")
        print(f"  Runtime: {elapsed:.1f} seconds")
        
        if self.demo_stats['samples_processed'] > 0:
            avg_time = self.demo_stats['total_processing_time'] / self.demo_stats['samples_processed']
            print(f"  Avg Processing Time: {avg_time:.1f}ms")
        print()

    def _show_configuration(self):
        """Show current configuration"""
        print(f"\n{Colors.OKBLUE}🔧 Current Configuration:{Colors.ENDC}")
        for key, value in self.demo_config.items():
            print(f"  {key}: {value}")
        print()

    def _export_demo_results(self):
        """Export demonstration results to JSON"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = f"demo_results_{timestamp}.json"
            
            # Prepare export data
            export_data = {
                'demo_info': {
                    'timestamp': datetime.now().isoformat(),
                    'duration_seconds': (datetime.now() - self.demo_stats['start_time']).total_seconds(),
                    'configuration': self.demo_config
                },
                'statistics': dict(self.demo_stats),
                'prediction_history': list(self.demo_stats['prediction_history']),
                'dashboard_data': self.dashboard_manager.get_dashboard_data(),
                'system_performance': self.realtime_explainer.get_performance_stats()
            }
            
            # Convert numpy types to JSON serializable
            export_data = self._convert_numpy_types(export_data)
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"{Colors.OKGREEN}📄 Results exported to: {export_path}{Colors.ENDC}")
            
        except Exception as e:
            print(f"{Colors.WARNING}⚠ Export failed: {e}{Colors.ENDC}")

    def _convert_numpy_types(self, obj):
        """Convert numpy types to JSON serializable types"""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, defaultdict):
            return dict(obj)
        elif isinstance(obj, deque):
            return list(obj)
        else:
            return obj

    def cleanup(self):
        """Cleanup resources"""
        if self.realtime_explainer:
            self.realtime_explainer.stop_processing()
        print(f"{Colors.OKGREEN}🧹 Cleanup complete{Colors.ENDC}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Real-Time Explainable AI Cybersecurity Demo')
    parser.add_argument('--mode', choices=['fast', 'extended', 'interactive'], 
                       default='fast', help='Demo mode')
    parser.add_argument('--env', choices=['development', 'production', 'testing'],
                       default='development', help='Environment configuration')
    parser.add_argument('--samples', type=int, help='Maximum samples to process')
    parser.add_argument('--interval', type=float, help='Interval between samples (seconds)')
    parser.add_argument('--no-export', action='store_true', help='Disable result export')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce log noise for demo
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize demonstrator
    demo = RealtimeDemonstrator(args.env)
    
    try:
        # Initialize system
        if not demo.initialize_system():
            return 1
        
        # Apply command line overrides
        if args.samples:
            demo.demo_config['max_samples'] = args.samples
        if args.interval:
            demo.demo_config['sample_interval'] = args.interval
        if args.no_export:
            demo.demo_config['export_results'] = False
        
        # Show system info
        demo.print_system_info()
        
        # Run demonstration
        demo.run_demo_mode(args.mode)
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}🛑 Demonstration interrupted{Colors.ENDC}")
        return 1
    except Exception as e:
        print(f"\n{Colors.FAIL}❌ Demo failed: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        demo.cleanup()

if __name__ == "__main__":
    exit(main())