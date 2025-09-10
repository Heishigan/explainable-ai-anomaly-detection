#!/usr/bin/env python3
"""
Quick startup script for the Web Dashboard

This script provides a simple way to start the web dashboard server
with proper error handling and user-friendly messaging.

Usage:
    python run_web_dashboard.py
    python run_web_dashboard.py --port 8080
    python run_web_dashboard.py --host 0.0.0.0 --port 8000
"""

import sys
import os
import subprocess
import time
from pathlib import Path

def check_requirements():
    """Check if all required dependencies are installed"""
    print("[INFO] Checking requirements...")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'websockets',
        'jinja2'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"[ERROR] Missing required packages: {', '.join(missing_packages)}")
        print("[INFO] Please install them with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    print("[OK] All requirements satisfied")
    return True

def check_system_readiness():
    """Check if the AI system is ready (models trained, data available)"""
    print("[INFO] Checking system readiness...")
    
    # Check for trained models
    models_dir = Path("results/models")
    if not models_dir.exists() or not any(models_dir.glob("*.joblib")):
        print("[ERROR] No trained models found")
        print("[INFO] Please run training first:")
        print("   python train_models.py")
        return False
    
    # Check for preprocessor
    preprocessing_dir = Path("results/preprocessing")
    if not preprocessing_dir.exists() or not (preprocessing_dir / "preprocessor.joblib").exists():
        print("[ERROR] Preprocessor not found")
        print("[INFO] Please run training first:")
        print("   python train_models.py")
        return False
    
    # Check for test data
    test_data_path = Path("data/raw/UNSW_NB15_testing-set.csv")
    if not test_data_path.exists():
        print("[ERROR] Test data not found")
        print("[INFO] Please ensure UNSW-NB15 dataset is available")
        return False
    
    print("[OK] System ready for web dashboard")
    return True

def start_web_dashboard(host="127.0.0.1", port=8000, reload=False):
    """Start the web dashboard server"""
    print(f"[START] Starting Web Dashboard on {host}:{port}")
    print(f"[URL] Dashboard URL: http://{host}:{port}")
    print("[INFO] Real-time Explainable AI Cybersecurity Dashboard")
    print()
    print("[FEATURES] Available features:")
    print("   * Real-time network anomaly detection")
    print("   * SHAP-based explanations")
    print("   * Interactive visualizations")
    print("   * Live WebSocket updates")
    print("   * Attack analytics and alerts")
    print()
    print("[WARNING] To stop the server, press Ctrl+C")
    print("="*60)
    
    try:
        # Import and run the dashboard
        from web_dashboard import main as run_dashboard
        
        # Prepare arguments
        sys.argv = [
            'web_dashboard.py',
            '--host', host,
            '--port', str(port)
        ]
        
        if reload:
            sys.argv.extend(['--reload'])
        
        # Start the server
        run_dashboard()
        
    except KeyboardInterrupt:
        print("\n[STOP] Web dashboard stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Error starting web dashboard: {e}")
        print("\n[HELP] Troubleshooting tips:")
        print("   1. Make sure all requirements are installed")
        print("   2. Ensure models are trained (python train_models.py)")
        print("   3. Check if port is already in use")
        print("   4. Try a different port: --port 8080")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Start the Explainable AI Web Dashboard',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_web_dashboard.py                    # Start on localhost:8000
  python run_web_dashboard.py --port 8080       # Start on localhost:8080
  python run_web_dashboard.py --host 0.0.0.0    # Allow external connections
        """
    )
    
    parser.add_argument('--host', default='127.0.0.1', 
                       help='Host to bind to (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to bind to (default: 8000)')
    parser.add_argument('--reload', action='store_true',
                       help='Enable auto-reload for development')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip system readiness checks')
    
    args = parser.parse_args()
    
    print("[LAUNCHER] Explainable AI Web Dashboard Launcher")
    print("="*60)
    
    # Check requirements
    if not check_requirements():
        return 1
    
    # Check system readiness
    if not args.skip_checks and not check_system_readiness():
        print("\n[INFO] You can skip these checks with --skip-checks if needed")
        return 1
    
    print("\n[SUCCESS] All checks passed! Starting web dashboard...")
    print()
    
    # Start the dashboard
    start_web_dashboard(
        host=args.host,
        port=args.port,
        reload=args.reload
    )
    
    return 0

if __name__ == "__main__":
    exit(main())