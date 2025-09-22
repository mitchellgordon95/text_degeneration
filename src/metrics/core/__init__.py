"""
Core metrics from Holtzman et al. 2019 - "The Curious Case of Neural Text Degeneration"

These are the primary metrics used in the paper to demonstrate the degeneration problem
and evaluate different decoding methods. HUSE (Human Unified with Statistical Evaluation)
is not included as it requires human evaluation.
"""

from .repetition import measure_repetition_rate
from .perplexity import compute_perplexity, compute_perplexity_gap
from .self_bleu import compute_self_bleu
from .zipf import compute_zipf_coefficient

__all__ = [
    'measure_repetition_rate',
    'compute_perplexity',
    'compute_perplexity_gap',
    'compute_self_bleu',
    'compute_zipf_coefficient'
]