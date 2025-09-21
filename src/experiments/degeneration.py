"""Degeneration experiment - testing repetition rates across models and methods."""

from typing import List, Dict, Any
import numpy as np
import pandas as pd
from tqdm import tqdm

from .base_experiment import BaseExperiment
from ..metrics import (
    measure_repetition_rate,
    measure_ngram_repetition,
    compute_self_bleu,
    distinct_n_grams,
    vocabulary_diversity
)


class DegenerationExperiment(BaseExperiment):
    """
    Test for text degeneration (repetition) across different decoding methods.
    Replicates the analysis from Holtzman et al. 2019.
    """

    def __init__(self, config: Dict[str, Any], prompts: List[str], output_dir: str = "outputs"):
        super().__init__("degeneration", config, prompts, output_dir)

        # Experiment-specific config
        self.max_length = config.get("max_length", 256)
        self.n_gram_size = config.get("n_gram_size", 4)  # Default to 4-grams as in paper

    def generate_outputs(self, model, method: str) -> List[str]:
        """Generate outputs with specified method."""
        outputs = []

        # Use subset of prompts if specified
        num_samples = self.config.get("num_samples", len(self.prompts))
        prompts_to_use = self.prompts[:num_samples]

        for prompt in tqdm(prompts_to_use, desc=f"    Generating", leave=False):
            # Generate text using method-specific interface
            output = self._call_method(model, method, prompt, self.max_length)
            outputs.append(output)

        return outputs

    def _call_method(self, model, method: str, prompt: str, max_length: int) -> str:
        """Call the appropriate method-specific generation function."""
        if method == "greedy":
            return model.generate_greedy(prompt, max_length)
        elif method.startswith("beam_"):
            beam_size = int(method.split("_")[1])
            return model.generate_beam(prompt, beam_size, max_length)
        elif method.startswith("nucleus_"):
            top_p = float(method.split("_")[1])
            return model.generate_nucleus(prompt, top_p, max_length)
        elif method.startswith("top_k_"):
            top_k = int(method.split("_")[2])  # top_k_50 -> get the "50" part
            return model.generate_top_k(prompt, top_k, max_length)
        elif method == "temperature" or method.startswith("temperature_"):
            if "_" in method:
                temperature = float(method.split("_")[1])
            else:
                temperature = 1.0  # Default temperature
            return model.generate_temperature(prompt, temperature, max_length)
        else:
            raise ValueError(f"Unknown method: {method}")

    def compute_metrics(self, outputs: List[str], model=None, method: str = None) -> Dict[str, float]:
        """Compute degeneration metrics."""
        # Filter out empty outputs
        valid_outputs = [o for o in outputs if o.strip()]

        if not valid_outputs:
            return {
                "repetition_rate": 0.0,
                "texts_with_repetition": 0.0,
                "self_bleu": 0.0,
                "distinct_1": 0.0,
                "distinct_2": 0.0,
                "distinct_3": 0.0,
                "vocabulary_size": 0,
                "avg_length": 0.0
            }

        # Primary metric from Holtzman et al.
        repetition_rate = measure_repetition_rate(valid_outputs, n=self.n_gram_size)

        # Detailed repetition analysis
        rep_details = measure_ngram_repetition(valid_outputs, n=self.n_gram_size)

        # Diversity metrics
        self_bleu = compute_self_bleu(valid_outputs, n=4)
        distinct_1 = distinct_n_grams(valid_outputs, n=1)
        distinct_2 = distinct_n_grams(valid_outputs, n=2)
        distinct_3 = distinct_n_grams(valid_outputs, n=3)

        # Vocabulary diversity
        vocab_stats = vocabulary_diversity(valid_outputs)

        # Average length
        lengths = [len(o.split()) for o in valid_outputs]
        avg_length = np.mean(lengths) if lengths else 0

        return {
            # Primary metrics (matching Holtzman)
            "repetition_rate": repetition_rate,
            "self_bleu": self_bleu if self_bleu >= 0 else None,

            # Additional repetition details
            "texts_with_repetition": rep_details["texts_with_repetition"],
            "max_repetition_in_text": rep_details["max_repetition_in_text"],

            # Diversity metrics
            "distinct_1": distinct_1,
            "distinct_2": distinct_2,
            "distinct_3": distinct_3,

            # Vocabulary metrics
            "type_token_ratio": vocab_stats["type_token_ratio"],
            "vocabulary_size": vocab_stats["vocabulary_size"],

            # Basic stats
            "avg_length": avg_length,
            "num_outputs": len(valid_outputs)
        }


    def analyze_results(self):
        """Create analysis DataFrame comparing methods."""
        import pandas as pd

        data = []
        for model_name, model_results in self.results.items():
            for method, method_results in model_results.items():
                metrics = method_results.get("metrics", {})
                data.append({
                    "model": model_name,
                    "method": method,
                    "repetition_rate": metrics.get("repetition_rate", 0),
                    "self_bleu": metrics.get("self_bleu", 0),
                    "distinct_2": metrics.get("distinct_2", 0),
                    "vocabulary_size": metrics.get("vocabulary_size", 0)
                })

        df = pd.DataFrame(data)

        # Sort by model and repetition rate
        df = df.sort_values(["model", "repetition_rate"])

        return df

    def compare_to_holtzman(self) -> Dict[str, Any]:
        """Compare results to Holtzman et al. 2019 findings."""
        holtzman_results = {
            "gpt2-large": {
                "greedy": {"repetition_rate": 20.0},  # Approximate
                "beam_10": {"repetition_rate": 28.94},
                "nucleus_0.95": {"repetition_rate": 0.36}
            }
        }

        comparison = {}
        for model_name, model_results in self.results.items():
            if "gpt2" in model_name.lower():
                comparison[model_name] = {}
                for method, method_results in model_results.items():
                    our_rate = method_results["metrics"]["repetition_rate"]

                    # Find matching Holtzman result
                    holtzman_key = None
                    if method == "greedy":
                        holtzman_key = "greedy"
                    elif "beam" in method and "10" in method:
                        holtzman_key = "beam_10"
                    elif "nucleus" in method and "0.95" in method:
                        holtzman_key = "nucleus_0.95"

                    if holtzman_key and holtzman_key in holtzman_results.get("gpt2-large", {}):
                        holtzman_rate = holtzman_results["gpt2-large"][holtzman_key]["repetition_rate"]
                        comparison[model_name][method] = {
                            "our_rate": our_rate,
                            "holtzman_rate": holtzman_rate,
                            "difference": our_rate - holtzman_rate
                        }

        return comparison