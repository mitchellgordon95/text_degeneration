#!/usr/bin/env python3
"""Run all model tests with clean output."""

import subprocess
import sys
import os

def run_test(script_name, num_samples=1):
    """Run a test script and capture output."""
    print(f"\n{'='*70}")
    print(f"Running {script_name}")
    print('='*70)

    # Run the test with warnings suppressed
    env = os.environ.copy()
    env['PYTHONWARNINGS'] = 'ignore'

    result = subprocess.run(
        [sys.executable, script_name, str(num_samples)],
        env=env,
        capture_output=True,
        text=True
    )

    # Print stdout (the actual test output)
    print(result.stdout)

    # Check if it failed
    if result.returncode != 0:
        print(f"❌ {script_name} failed with exit code {result.returncode}")
        if result.stderr:
            print("Error output:", result.stderr)
        return False

    return True

def main():
    """Run all model tests."""
    print("\n" + "="*70)
    print("RUNNING ALL MODEL TESTS")
    print("="*70)

    tests = [
        ("test_gpt2.py", 1),
        ("test_openai.py", 1),
        ("test_claude.py", 1),
    ]

    all_passed = True
    for test_script, num_samples in tests:
        if os.path.exists(test_script):
            passed = run_test(test_script, num_samples)
            all_passed = all_passed and passed
        else:
            print(f"⚠️  {test_script} not found")

    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*70)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())