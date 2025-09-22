"""Self-BLEU metric for measuring diversity in generated text."""

from typing import List
import numpy as np
import warnings

# Try to import BLEU scorer
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    warnings.warn("NLTK not available. Self-BLEU calculation will be disabled.")


def compute_self_bleu(texts: List[str], n: int = 4) -> float:
    """
    Compute Self-BLEU score to measure diversity.
    Lower Self-BLEU indicates more diverse texts.

    Args:
        texts: List of generated texts
        n: Maximum n-gram size for BLEU

    Returns:
        Average Self-BLEU score (0-1), or -1 if NLTK not available
    """
    if not BLEU_AVAILABLE:
        return -1.0  # Indicate not available

    if len(texts) < 2:
        return 0.0

    smoothing = SmoothingFunction().method1
    scores = []

    # For each text, use all others as references
    for i, hypothesis in enumerate(texts):
        references = [texts[j].split() for j in range(len(texts)) if j != i]
        hypothesis_tokens = hypothesis.split()

        if not hypothesis_tokens or not references:
            continue

        # Calculate BLEU with n-gram weights
        weights = tuple([1.0 / n] * n)
        score = sentence_bleu(
            references,
            hypothesis_tokens,
            weights=weights,
            smoothing_function=smoothing
        )
        scores.append(score)

    return np.mean(scores) if scores else 0.0