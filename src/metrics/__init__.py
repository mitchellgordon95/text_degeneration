"""
Metrics for evaluating text generation quality.

Organized into:
- core: Metrics from Holtzman et al. 2019 paper
- extended: Additional metrics beyond the original paper
"""

# Import core metrics from Holtzman et al. 2019
from .core import (
    measure_repetition_rate,
    compute_perplexity,
    compute_perplexity_gap,
    compute_self_bleu,
    compute_zipf_coefficient
)

# Import extended metrics
# (Currently none implemented - see extended/README.md for planned metrics)

__all__ = [
    # Core metrics (Holtzman et al. 2019)
    'measure_repetition_rate',
    'compute_perplexity',
    'compute_perplexity_gap',
    'compute_self_bleu',
    'compute_zipf_coefficient'
]