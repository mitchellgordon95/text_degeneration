"""Base class for all experiments."""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import traceback


class BaseExperiment:
    """Base class for experiments with checkpointing and result management."""

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        prompts: List[str],
        output_dir: str = "outputs"
    ):
        self.name = name
        self.config = config
        self.prompts = prompts
        self.output_dir = Path(output_dir)

        # Create output directories
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.raw_dir = self.output_dir / "raw" / self.name
        self.metrics_dir = self.output_dir / "metrics" / self.name
        self.raw_dir.mkdir(exist_ok=True, parents=True)
        self.metrics_dir.mkdir(exist_ok=True, parents=True)

        # Checkpoint file for resuming
        self.checkpoint_file = self.output_dir / f"{self.name}_checkpoint.pkl"

        # Track costs
        self.total_cost = 0.0
        self.total_tokens = 0

        # Results storage
        self.results = {}

    def run(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run experiment for all models and methods.

        Args:
            models: Dictionary of model_name -> model instance

        Returns:
            Results dictionary
        """
        # Load checkpoint if exists
        self.results = self.load_checkpoint()

        print(f"\n{'='*60}")
        print(f"Running {self.name} Experiment")
        print(f"Models: {list(models.keys())}")
        print(f"Methods: {self.config.get('methods', ['default'])}")
        print(f"Samples: {len(self.prompts)}")
        print(f"{'='*60}\n")

        for model_name, model in models.items():
            print(f"\n[{model_name}]")

            if model_name not in self.results:
                self.results[model_name] = {}

            methods = self.config.get("methods", ["greedy"])

            for method in tqdm(methods, desc=f"  Methods", leave=True):
                if method in self.results[model_name]:
                    print(f"  Skipping {method} (already completed)")
                    continue

                # Generate outputs - let errors propagate
                print(f"  Generating with {method}...")
                outputs = self.generate_outputs(model, method)

                # Save raw outputs immediately
                self.save_raw_outputs(model_name, method, outputs)

                # Compute metrics
                print(f"  Computing metrics...")
                metrics = self.compute_metrics(outputs, model, method)

                # Store results
                self.results[model_name][method] = {
                    "outputs": outputs,
                    "metrics": metrics,
                    "timestamp": datetime.now().isoformat()
                }

                # Update costs
                if hasattr(model, 'total_cost'):
                    self.total_cost += model.total_cost
                if hasattr(model, 'total_tokens'):
                    self.total_tokens += model.total_tokens

                # Save checkpoint after successful completion
                self.save_checkpoint()

                # Save metrics
                self.save_metrics(model_name, method, metrics)

                # Print summary
                self.print_metrics_summary(model_name, method, metrics)

        # Save final results
        self.save_final_results()

        print(f"\n{'='*60}")
        print(f"Experiment Complete!")
        print(f"Total cost: ${self.total_cost:.2f}")
        print(f"Total tokens: {self.total_tokens:,}")
        print(f"{'='*60}\n")

        return self.results

    def generate_outputs(self, model, method: str) -> List[str]:
        """
        Generate outputs for this experiment.
        To be implemented by subclasses.
        """
        raise NotImplementedError

    def compute_metrics(self, outputs: List[str], model, method: str) -> Dict[str, float]:
        """
        Compute metrics for outputs.
        To be implemented by subclasses.
        """
        raise NotImplementedError

    def print_metrics_summary(self, model_name: str, method: str, metrics: Dict[str, float]):
        """Print a summary of metrics."""
        print(f"\n  {model_name} - {method}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")

    def save_checkpoint(self):
        """Save checkpoint for resuming."""
        checkpoint = {
            "results": self.results,
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "timestamp": datetime.now().isoformat()
        }
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self) -> Dict:
        """Load checkpoint if exists."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                    self.total_cost = checkpoint.get("total_cost", 0)
                    self.total_tokens = checkpoint.get("total_tokens", 0)
                    print(f"Loaded checkpoint from {checkpoint.get('timestamp', 'unknown')}")
                    return checkpoint.get("results", {})
            except Exception as e:
                print(f"Could not load checkpoint: {e}")
        return {}

    def save_raw_outputs(self, model_name: str, method: str, outputs: List[str]):
        """Save raw generated outputs."""
        output_file = self.raw_dir / f"{model_name}_{method}_outputs.json"
        with open(output_file, 'w') as f:
            json.dump({
                "model": model_name,
                "method": method,
                "outputs": outputs,
                "prompts": self.prompts,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)

    def save_metrics(self, model_name: str, method: str, metrics: Dict[str, float]):
        """Save computed metrics."""
        metrics_file = self.metrics_dir / f"{model_name}_{method}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump({
                "model": model_name,
                "method": method,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)

    def save_final_results(self):
        """Save final aggregated results."""
        # Save as JSON
        results_file = self.output_dir / f"{self.name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        # Create summary DataFrame
        summary_data = []
        for model_name, model_results in self.results.items():
            for method, method_results in model_results.items():
                row = {
                    "model": model_name,
                    "method": method,
                    **method_results.get("metrics", {})
                }
                summary_data.append(row)

        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_file = self.output_dir / f"{self.name}_summary.csv"
            df.to_csv(csv_file, index=False)
            print(f"\nSummary saved to {csv_file}")

            # Print summary table
            print("\nResults Summary:")
            print(df.to_string(index=False))