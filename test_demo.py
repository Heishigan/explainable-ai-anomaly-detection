#!/usr/bin/env python3
"""
Quick test script for the demo pipeline to verify functionality
"""

import subprocess
import sys
import os

def test_demo_help():
    """Test demo help functionality"""
    print("Testing demo help...")
    try:
        result = subprocess.run([
            sys.executable, "demo_realtime_pipeline.py", "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and "Real-Time Explainable AI" in result.stdout:
            print("PASS: Demo help test passed")
            return True
        else:
            print("FAIL: Demo help test failed")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
    except Exception as e:
        print(f"ERROR: Demo help test error: {e}")
        return False

def test_demo_initialization():
    """Test demo initialization (dry run)"""
    print("Testing demo initialization...")
    try:
        # This would test initialization but exit before running samples
        # We can't easily do this without modifying the demo script
        # For now, we'll just check if the script imports work
        
        # Test imports
        import demo_realtime_pipeline
        print("PASS: Demo imports test passed")
        return True
        
    except ImportError as e:
        print(f"FAIL: Demo imports test failed: {e}")
        return False
    except Exception as e:
        print(f"ERROR: Demo initialization test error: {e}")
        return False

def check_requirements():
    """Check if required files exist"""
    print("Checking requirements...")
    
    required_files = [
        "demo_realtime_pipeline.py",
        "src/config/config.py",
        "src/explainability/realtime_explainer.py",
        "data/raw/UNSW_NB15_testing-set.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("FAIL: Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    else:
        print("PASS: All required files present")
        return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("DEMO PIPELINE TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Requirements Check", check_requirements),
        ("Demo Help", test_demo_help),
        ("Demo Imports", test_demo_initialization)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        if test_func():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"TEST SUMMARY: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("All tests passed! Demo should work correctly.")
        print("\nTo run the demo:")
        print("  python demo_realtime_pipeline.py --mode fast")
    else:
        print("Some tests failed. Please check the issues above.")
    
    print("=" * 60)
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)