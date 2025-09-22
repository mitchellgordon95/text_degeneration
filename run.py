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
from src.utils import load_prompts, load_config
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


def filter_available_models(model_names, models_config):
    """Filter models to only those that have configurations."""
    available_models = []

    for model_name in model_names:
        if model_name in models_config:
            available_models.append(model_name)
        else:
            print(f"⚠️  Model {model_name} not found in config, skipping")

    return available_models


def run_degeneration_experiment(config, model_names, models_config, prompts, human_continuations, experiment_name="degeneration_local"):
    """Run the degeneration (repetition) experiment."""
    exp_config = config["experiments"][experiment_name]

    # Filter to requested models that are in config
    requested_models = exp_config["models"]
    available_models = filter_available_models(requested_models, models_config)

    if not available_models:
        print("❌ No models available for experiment")
        return None

    # Create and run experiment with human continuations for perplexity gap
    experiment = DegenerationExperiment(exp_config, prompts, human_continuations=human_continuations)
    results = experiment.run(available_models, models_config)

    return experiment


def main():
    parser = argparse.ArgumentParser(description="Run text degeneration experiments")
    parser.add_argument(
        "--experiment",
        choices=["degeneration_local", "degeneration_openai", "degeneration_anthropic"],
        default="degeneration_local",
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
    models_config = load_config(args.models_config)["models"]

    # Load prompts and human continuations
    print("Loading prompts and human continuations...")
    prompts = load_prompts(args.prompts_config)
    print(f"Loaded {len(prompts)} prompts")

    # Load human continuations for perplexity gap calculation
    import yaml
    with open(args.prompts_config, 'r') as f:
        prompts_data = yaml.safe_load(f)
    human_continuations = prompts_data.get('human_continuations', [])
    if human_continuations:
        print(f"Loaded {len(human_continuations)} human continuations for perplexity gap")

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

    # Run experiment
    print(f"\n🚀 Running {exp_name} experiment...")

    try:
        if exp_name in ["degeneration_local", "degeneration_openai", "degeneration_anthropic"]:
            experiment = run_degeneration_experiment(config, exp_config["models"], models_config, prompts, human_continuations, exp_name)

            if experiment:
                # Print results summary
                print("\n" + "="*60)
                print("RESULTS SUMMARY")
                print("="*60)

                # Analyze results
                df = experiment.analyze_results()

                # Format output like Holtzman's Table 1
                print("\n" + "="*80)
                print("RESULTS (Format: Holtzman et al. 2019 Table 1)")
                print("="*80)

                # Group by model and display each model's results
                for model_name in df['model'].unique():
                    model_df = df[df['model'] == model_name]
                    print(f"\nModel: {model_name}")
                    print("-" * 80)
                    print(f"{'Method':<20} {'Self-BLEU4':<12} {'Repetition%':<12} {'Perplexity':<12} {'Human PPL':<12} {'Overconf':<10}")
                    print("-" * 80)

                    for _, row in model_df.iterrows():
                        self_bleu = f"{row['self_bleu']:.2f}" if row['self_bleu'] is not None and row['self_bleu'] >= 0 else "N/A"
                        perplexity = f"{row['perplexity']:.2f}" if row['perplexity'] != float('inf') else "N/A"
                        human_ppl = f"{row['human_ppl']:.2f}" if row['human_ppl'] != float('inf') else "N/A"
                        overconf = f"{row['overconf_ratio']:.2f}x" if row['overconf_ratio'] != float('inf') else "N/A"

                        print(f"{row['method']:<20} {self_bleu:<12} {row['repetition_rate']:>11.2f}% {perplexity:<12} {human_ppl:<12} {overconf:<10}")

                print("\n" + "="*80)
                print("BASELINE COMPARISON (Holtzman et al. 2019 - GPT-2 Large)")
                print("="*80)
                print(f"{'Method':<20} {'Self-BLEU4':<12} {'Repetition%':<12} {'Perplexity':<12}")
                print("-" * 80)
                print(f"{'Human':<20} {'0.31':<12} {'0.28%':<12} {'12.38':<12}")
                print(f"{'Greedy':<20} {'0.50':<12} {'73.66%':<12} {'1.50':<12}")
                print(f"{'Beam-16':<20} {'0.44':<12} {'28.94%':<12} {'1.48':<12}")
                print(f"{'Top-k-40':<20} {'0.39':<12} {'0.78%':<12} {'6.88':<12}")
                print(f"{'Nucleus-0.95':<20} {'0.32':<12} {'0.36%':<12} {'13.13':<12}")

                # Compare to Holtzman if applicable
                comparison = experiment.compare_to_holtzman()
                if comparison:
                    print("\n\nComparison to Holtzman et al. 2019:")
                    for model, methods in comparison.items():
                        print(f"\n{model}:")
                        for method, stats in methods.items():
                            print(f"  {method:15} Our: {stats['our_rate']:.2f}% | Holtzman: {stats['holtzman_rate']:.2f}% | Diff: {stats['difference']:+.2f}%")

                # Token summary
                print(f"\nTotal tokens: {experiment.total_tokens:,}")

        else:
            print(f"❌ Unknown experiment: {exp_name}")
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