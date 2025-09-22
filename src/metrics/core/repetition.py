"""Simple repetition metric for generated text."""

from typing import List, Tuple
from collections import Counter


def measure_repetition_rate(texts: List[str], n: int = 4) -> float:
    """
    Measure the percentage of text that consists of repeated n-grams.

    Args:
        texts: List of generated texts
        n: N-gram size (default 4 as in Holtzman et al. 2019)

    Returns:
        Repetition rate as percentage (0-100)
    """
    total_ngrams = 0
    repeated_ngrams = 0

    for text in texts:
        ngrams = _get_ngrams(text, n)
        counter = Counter(ngrams)

        total_ngrams += len(ngrams)
        # Count how many ngrams appear more than once
        for ngram, count in counter.items():
            if count > 1:
                # All occurrences after the first are repetitions
                repeated_ngrams += (count - 1)

    if total_ngrams == 0:
        return 0.0

    return (repeated_ngrams / total_ngrams) * 100


def _get_ngrams(text: str, n: int) -> List[Tuple[str, ...]]:
    """
    Extract n-grams from text.

    Args:
        text: Input text
        n: N-gram size

    Returns:
        List of n-gram tuples
    """
    # Tokenize (simple whitespace tokenization)
    tokens = text.split()

    if len(tokens) < n:
        return []

    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams.append(ngram)

    return ngrams