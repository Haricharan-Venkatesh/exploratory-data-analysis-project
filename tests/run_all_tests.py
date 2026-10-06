"""
Master Test Runner for Chess AI Detection System
================================================
Discovers and executes all unit and integration tests:
- tests/test_model.py
- tests/test_inference_features.py
- tests/test_edge_cases.py
- tests/test_api.py
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def run_tests():
    print("==================================================")
    print("RUNNING COMPLETE CHESS AI DETECTION TEST SUITE")
    print("==================================================")
    
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=os.path.join(PROJECT_ROOT, "tests"), pattern="test_*.py")
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n==================================================")
    print("TEST SUITE SUMMARY")
    print("==================================================")
    print(f"Total Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100:.1f}%")
    
    if result.wasSuccessful():
        print("[STATUS: ALL TESTS PASSED SUCCESSFULLY!]")
        sys.exit(0)
    else:
        print("[STATUS: FAILURES OR ERRORS DETECTED]")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
