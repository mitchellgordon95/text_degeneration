#!/usr/bin/env python3
"""
Validate experiment configurations against model capabilities.

This script checks if all model/method combinations in experiments.yaml
are valid according to the capabilities defined in models.yaml.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config, validate_experiment_config

def main():
    """Validate all experiment configurations."""
    print("Loading configurations...")

    try:
        experiments_config = load_config("config/experiments.yaml")
        models_config = load_config("config/models.yaml")
    except Exception as e:
        print(f"Error loading configurations: {e}")
        return 1

    print(f"Found {len(experiments_config['experiments'])} experiments")
    print(f"Found {len(models_config['models'])} models")

    all_errors = []

    for exp_name, exp_config in experiments_config['experiments'].items():
        print(f"\nValidating experiment: {exp_name}")

        errors = validate_experiment_config(exp_config, models_config['models'])

        if errors:
            print(f"  ❌ {len(errors)} errors found:")
            for error in errors:
                print(f"    - {error}")
            all_errors.extend(errors)
        else:
            print(f"  ✅ No errors found")

    if all_errors:
        print(f"\n❌ Total validation errors: {len(all_errors)}")
        print("\nConfiguration needs to be fixed before running experiments!")
        return 1
    else:
        print(f"\n✅ All experiments are valid!")
        print("Configuration is ready for academic-grade experiments.")
        return 0

if __name__ == "__main__":
    sys.exit(main())