"""Zipf coefficient metric for measuring word frequency distribution."""

from typing import List
import numpy as np
from collections import Counter
from scipy import stats


def compute_zipf_coefficient(texts: List[str]) -> float:
    """
    Compute Zipf coefficient from word frequency distribution.

    Zipf's law states that word frequency follows a power law distribution.
    The coefficient (alpha) indicates how steep the distribution is:
    - Higher alpha (>1): Few words dominate (less diverse)
    - Lower alpha (~1): More uniform distribution (more diverse)
    - Human text typically has alpha ~1.0-1.2

    Args:
        texts: List of generated texts

    Returns:
        Zipf coefficient (alpha), or 0 if cannot be computed
    """
    # Collect all words
    all_words = []
    for text in texts:
        all_words.extend(text.lower().split())

    if len(all_words) < 10:  # Need minimum words for meaningful analysis
        return 0.0

    # Count word frequencies
    word_counts = Counter(all_words)

    # Get frequencies sorted in descending order
    frequencies = sorted(word_counts.values(), reverse=True)

    # Create rank array (1, 2, 3, ...)
    ranks = np.arange(1, len(frequencies) + 1)

    # Take log of both ranks and frequencies for linear regression
    # Zipf's law: frequency = C / rank^alpha
    # log(frequency) = log(C) - alpha * log(rank)
    log_ranks = np.log(ranks)
    log_freqs = np.log(frequencies)

    # Perform linear regression to find slope (which is -alpha)
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_ranks, log_freqs)
        # Return absolute value of slope as the Zipf coefficient
        return abs(slope)
    except:
        return 0.0