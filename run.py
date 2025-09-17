#!/usr/bin/env python3
"""
Main runner for text degeneration experiments.
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import warnings
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import UnifiedModel
from src.experiments import DegenerationExperiment
from src.utils import load_prompts, load_config, CostTracker
from src.utils.data_loader import load_prompt_set


def setup_environment():
    """Set up environment and check for API keys."""
    # Try to load from .env file if it exists
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Check for API keys
    api_keys = {
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Anthropic"
    }

    available_apis = []
    for key, name in api_keys.items():
        if os.environ.get(key):
            available_apis.append(name)
        else:
            print(f"⚠️  {name} API key not found (set {key})")

    if not available_apis:
        print("\n❌ No API keys found. Please set environment variables:")
        print("   export OPENAI_API_KEY='your-key-here'")
        print("   export ANTHROPIC_API_KEY='your-key-here'")

    return available_apis


def load_models(model_names, models_config):
    """Load specified models."""
    models = {}

    for model_name in model_names:
        print(f"Loading {model_name}...")
        try:
            # Get model config if available
            model_config = models_config.get(model_name, {})

            # Create model
            model = UnifiedModel.create(model_name, **model_config)
            models[model_name] = model
            print(f"✓ {model_name} loaded")

        except Exception as e:
            print(f"✗ Failed to load {model_name}: {e}")
            if "API key" in str(e):
                print("  (Check your API keys)")

    return models


def run_degeneration_experiment(config, models, prompts):
    """Run the degeneration (repetition) experiment."""
    exp_config = config["experiments"]["degeneration"]

    # Filter models to only those requested
    experiment_models = {}
    for model_name in exp_config["models"]:
        if model_name in models:
            experiment_models[model_name] = models[model_name]
        else:
            print(f"⚠️  Model {model_name} not available, skipping")

    if not experiment_models:
        print("❌ No models available for experiment")
        return None

    # Create and run experiment
    experiment = DegenerationExperiment(exp_config, prompts)
    results = experiment.run(experiment_models)

    return experiment


def main():
    parser = argparse.ArgumentParser(description="Run text degeneration experiments")
    parser.add_argument(
        "--experiment",
        choices=["degeneration", "perplexity", "tail", "task", "beam", "all"],
        default="degeneration",
        help="Which experiment to run"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Override models from config (e.g., gpt2-large gpt-4)"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Override methods from config (e.g., greedy beam_10 nucleus_0.95)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Override number of samples"
    )
    parser.add_argument(
        "--config",
        default="config/experiments.yaml",
        help="Path to experiments config"
    )
    parser.add_argument(
        "--prompts-config",
        default="config/prompts.yaml",
        help="Path to prompts config"
    )
    parser.add_argument(
        "--models-config",
        default="config/models.yaml",
        help="Path to models config"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without executing"
    )
    parser.add_argument(
        "--cost-limit",
        type=float,
        default=50.0,
        help="Maximum cost in dollars"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("TEXT DEGENERATION EXPERIMENTS")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Set up environment
    available_apis = setup_environment()

    # Load configurations
    print("\nLoading configurations...")
    config = load_config(args.config)
    models_config = load_config(args.models_config)

    # Load prompts
    print("Loading prompts...")
    prompts = load_prompts(args.prompts_config)
    print(f"Loaded {len(prompts)} prompts")

    # Get experiment config
    exp_name = args.experiment
    if exp_name not in config["experiments"]:
        print(f"❌ Unknown experiment: {exp_name}")
        return 1

    exp_config = config["experiments"][exp_name]

    # Override config with command line args
    if args.models:
        exp_config["models"] = args.models
    if args.methods:
        exp_config["methods"] = args.methods
    if args.num_samples:
        exp_config["num_samples"] = args.num_samples

    # Dry run - just print what would be done
    if args.dry_run:
        print("\n🔍 DRY RUN - Would execute:")
        print(f"Experiment: {exp_name}")
        print(f"Models: {exp_config['models']}")
        print(f"Methods: {exp_config.get('methods', ['default'])}")
        print(f"Samples: {exp_config.get('num_samples', len(prompts))}")
        print(f"Max length: {exp_config.get('max_length', 256)}")
        return 0

    # Load models
    print("\nLoading models...")
    models = load_models(exp_config["models"], models_config)

    if not models:
        print("❌ No models could be loaded")
        return 1

    # Set up cost tracking
    cost_tracker = CostTracker(warn_at=args.cost_limit * 0.5, stop_at=args.cost_limit)

    # Run experiment
    print(f"\n🚀 Running {exp_name} experiment...")

    try:
        if exp_name == "degeneration":
            experiment = run_degeneration_experiment(config, models, prompts)

            if experiment:
                # Print results summary
                print("\n" + "="*60)
                print("RESULTS SUMMARY")
                print("="*60)

                # Analyze results
                df = experiment.analyze_results()
                print("\nRepetition Rates by Model and Method:")
                print(df.to_string(index=False))

                # Compare to Holtzman if applicable
                comparison = experiment.compare_to_holtzman()
                if comparison:
                    print("\n\nComparison to Holtzman et al. 2019:")
                    for model, methods in comparison.items():
                        print(f"\n{model}:")
                        for method, stats in methods.items():
                            print(f"  {method:15} Our: {stats['our_rate']:.2f}% | Holtzman: {stats['holtzman_rate']:.2f}% | Diff: {stats['difference']:+.2f}%")

                # Cost summary
                print(f"\nTotal cost: ${experiment.total_cost:.2f}")
                print(f"Total tokens: {experiment.total_tokens:,}")

        else:
            print(f"❌ Experiment {exp_name} not yet implemented")
            return 1

    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
        return 1

    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n✅ Experiment complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())