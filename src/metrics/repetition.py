"""Metrics for measuring repetition in generated text."""

from typing import List, Tuple, Dict
from collections import Counter
import numpy as np


def measure_repetition_rate(texts: List[str], n: int = 4) -> float:
    """
    Measure the percentage of text that consists of repeated n-grams.
    This is the primary metric from Holtzman et al. 2019.

    Args:
        texts: List of generated texts
        n: N-gram size (default 4 as in paper)

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


def measure_ngram_repetition(
    texts: List[str],
    n: int = 4,
    exclude_punctuation: bool = True
) -> Dict[str, float]:
    """
    Comprehensive repetition analysis.

    Args:
        texts: List of generated texts
        n: N-gram size
        exclude_punctuation: Whether to exclude punctuation-only ngrams

    Returns:
        Dictionary with various repetition metrics
    """
    results = {
        "repetition_rate": 0.0,
        "texts_with_repetition": 0.0,
        "max_repetition_in_text": 0.0,
        "avg_repetitions_per_text": 0.0
    }

    texts_with_rep = 0
    all_repetitions = []
    max_rep = 0

    for text in texts:
        ngrams = _get_ngrams(text, n, exclude_punctuation)
        if not ngrams:
            continue

        counter = Counter(ngrams)

        # Check if this text has any repetition
        has_repetition = any(count > 1 for count in counter.values())
        if has_repetition:
            texts_with_rep += 1

        # Calculate repetitions in this text
        text_repetitions = sum(count - 1 for count in counter.values() if count > 1)
        all_repetitions.append(text_repetitions / max(len(ngrams), 1))

        # Track maximum repetition
        if counter:
            max_rep = max(max_rep, max(counter.values()))

    if texts:
        results["repetition_rate"] = measure_repetition_rate(texts, n)
        results["texts_with_repetition"] = (texts_with_rep / len(texts)) * 100
        results["max_repetition_in_text"] = max_rep
        results["avg_repetitions_per_text"] = np.mean(all_repetitions) * 100 if all_repetitions else 0

    return results


def count_repeated_ngrams(text: str, n_values: List[int] = [1, 2, 3, 4]) -> Dict[int, int]:
    """
    Count repeated n-grams for multiple n values.

    Args:
        text: Single text to analyze
        n_values: List of n-gram sizes to check

    Returns:
        Dictionary mapping n to number of repeated n-grams
    """
    results = {}

    for n in n_values:
        ngrams = _get_ngrams(text, n)
        counter = Counter(ngrams)
        # Count ngrams that appear more than once
        repeated = sum(1 for count in counter.values() if count > 1)
        results[n] = repeated

    return results


def _get_ngrams(text: str, n: int, exclude_punctuation: bool = False) -> List[Tuple[str, ...]]:
    """
    Extract n-grams from text.

    Args:
        text: Input text
        n: N-gram size
        exclude_punctuation: Whether to exclude punctuation-only ngrams

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

        # Optionally exclude punctuation-only ngrams
        if exclude_punctuation:
            # Check if all tokens are punctuation
            if all(_is_punctuation(token) for token in ngram):
                continue

        ngrams.append(ngram)

    return ngrams


def _is_punctuation(token: str) -> bool:
    """Check if a token is only punctuation."""
    import string
    return all(c in string.punctuation for c in token)


def find_repetitive_sequences(text: str, min_length: int = 10) -> List[str]:
    """
    Find the actual repetitive sequences in text.
    Useful for debugging and understanding what's being repeated.

    Args:
        text: Text to analyze
        min_length: Minimum character length of sequence to report

    Returns:
        List of repeated sequences
    """
    words = text.split()
    repeated_sequences = []

    # Check all possible sequence lengths
    for seq_len in range(2, len(words) // 2 + 1):
        for start in range(len(words) - seq_len * 2 + 1):
            sequence = words[start:start + seq_len]
            sequence_str = " ".join(sequence)

            if len(sequence_str) < min_length:
                continue

            # Check if this sequence repeats immediately after
            next_sequence = words[start + seq_len:start + seq_len * 2]
            if sequence == next_sequence:
                repeated_sequences.append(sequence_str)

    # Remove duplicates and sort by length
    repeated_sequences = list(set(repeated_sequences))
    repeated_sequences.sort(key=len, reverse=True)

    return repeated_sequences