"""Simple experiment for testing text generation with core metrics."""

from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm

from .base_experiment import BaseExperiment
from ..metrics.core import (
    measure_repetition_rate,
    compute_self_bleu,
    compute_perplexity,
    compute_zipf_coefficient
)


class DegenerationExperiment(BaseExperiment):
    """
    Test text generation quality across different decoding methods.
    Measures repetition, diversity, perplexity, and Zipf coefficient.
    """

    def __init__(self, config: Dict[str, Any], prompts: List[str], output_dir: str = "outputs"):
        super().__init__("degeneration", config, prompts, output_dir)
        self.max_length = config.get("max_length", 256)

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
        """Compute core metrics: repetition, self-BLEU, perplexity, and Zipf coefficient."""
        # Filter out empty outputs
        valid_outputs = [o for o in outputs if o.strip()]

        if not valid_outputs:
            return {
                "repetition_rate": 0.0,
                "self_bleu": 0.0,
                "perplexity": float('inf'),
                "zipf_coefficient": 0.0,
                "num_outputs": 0
            }

        # Core metrics
        repetition_rate = measure_repetition_rate(valid_outputs, n=4)
        self_bleu = compute_self_bleu(valid_outputs, n=4)
        zipf_coef = compute_zipf_coefficient(valid_outputs)

        # Perplexity (if model supports it)
        perplexity = float('inf')
        if model and hasattr(model, 'compute_perplexity'):
            try:
                perplexity = compute_perplexity(model, valid_outputs)
            except:
                pass  # Some models may not support perplexity

        return {
            "repetition_rate": repetition_rate,
            "self_bleu": self_bleu if self_bleu >= 0 else None,
            "perplexity": perplexity,
            "zipf_coefficient": zipf_coef,
            "num_outputs": len(valid_outputs)
        }

    def analyze_results(self):
        """Create simple analysis summary."""
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
                    "perplexity": metrics.get("perplexity", float('inf')),
                    "zipf_coefficient": metrics.get("zipf_coefficient", 0)
                })

        df = pd.DataFrame(data)

        # Sort by model and repetition rate
        df = df.sort_values(["model", "repetition_rate"])

        return df