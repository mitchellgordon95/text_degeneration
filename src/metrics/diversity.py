"""Metrics for measuring diversity in generated text."""

from typing import List, Dict, Set
import numpy as np
from collections import Counter
import warnings

# Try to import BLEU scorer
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    warnings.warn("NLTK not available. Self-BLEU calculation will be disabled.")


def distinct_n_grams(texts: List[str], n: int = 1) -> float:
    """
    Calculate distinct n-grams ratio (distinct-n metric).
    Higher values indicate more diversity.

    Args:
        texts: List of generated texts
        n: N-gram size (1 for unigrams, 2 for bigrams, etc.)

    Returns:
        Ratio of unique n-grams to total n-grams (0-1)
    """
    all_ngrams = []
    for text in texts:
        tokens = text.split()
        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)

    if not all_ngrams:
        return 0.0

    unique_ngrams = set(all_ngrams)
    return len(unique_ngrams) / len(all_ngrams)


def compute_self_bleu(texts: List[str], n: int = 4) -> float:
    """
    Compute Self-BLEU score to measure diversity.
    Lower Self-BLEU indicates more diverse texts.

    Args:
        texts: List of generated texts
        n: Maximum n-gram size for BLEU

    Returns:
        Average Self-BLEU score (0-1)
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


def compute_entropy(texts: List[str], level: str = "word") -> float:
    """
    Compute entropy of the text distribution.
    Higher entropy indicates more diversity.

    Args:
        texts: List of generated texts
        level: "word" or "char" for word-level or character-level entropy

    Returns:
        Entropy value
    """
    if level == "word":
        all_tokens = []
        for text in texts:
            all_tokens.extend(text.split())
    elif level == "char":
        all_tokens = list("".join(texts))
    else:
        raise ValueError(f"Unknown level: {level}")

    if not all_tokens:
        return 0.0

    # Count frequencies
    counter = Counter(all_tokens)
    total = sum(counter.values())

    # Calculate entropy
    entropy = 0.0
    for count in counter.values():
        if count > 0:
            prob = count / total
            entropy -= prob * np.log2(prob)

    return entropy


def vocabulary_diversity(texts: List[str]) -> Dict[str, float]:
    """
    Comprehensive vocabulary diversity metrics.

    Args:
        texts: List of generated texts

    Returns:
        Dictionary with various diversity metrics
    """
    all_words = []
    text_vocabularies = []

    for text in texts:
        words = text.lower().split()
        all_words.extend(words)
        text_vocabularies.append(set(words))

    if not all_words:
        return {
            "type_token_ratio": 0.0,
            "mean_segmental_ttr": 0.0,
            "vocabulary_size": 0,
            "hapax_legomena_ratio": 0.0
        }

    # Type-Token Ratio (TTR)
    unique_words = set(all_words)
    ttr = len(unique_words) / len(all_words)

    # Mean Segmental TTR (MSTTR) - TTR calculated on segments
    segment_size = 100
    segment_ttrs = []
    for i in range(0, len(all_words) - segment_size + 1, segment_size):
        segment = all_words[i:i + segment_size]
        segment_ttr = len(set(segment)) / len(segment)
        segment_ttrs.append(segment_ttr)
    msttr = np.mean(segment_ttrs) if segment_ttrs else ttr

    # Hapax legomena (words that appear only once)
    word_counts = Counter(all_words)
    hapax = sum(1 for count in word_counts.values() if count == 1)
    hapax_ratio = hapax / len(unique_words) if unique_words else 0

    return {
        "type_token_ratio": ttr,
        "mean_segmental_ttr": msttr,
        "vocabulary_size": len(unique_words),
        "hapax_legomena_ratio": hapax_ratio
    }


def pairwise_similarity(texts: List[str]) -> float:
    """
    Calculate average pairwise similarity between texts using Jaccard index.
    Lower values indicate more diversity.

    Args:
        texts: List of generated texts

    Returns:
        Average Jaccard similarity (0-1)
    """
    if len(texts) < 2:
        return 0.0

    similarities = []

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            set1 = set(texts[i].lower().split())
            set2 = set(texts[j].lower().split())

            if not set1 and not set2:
                similarity = 1.0
            elif not set1 or not set2:
                similarity = 0.0
            else:
                intersection = set1.intersection(set2)
                union = set1.union(set2)
                similarity = len(intersection) / len(union)

            similarities.append(similarity)

    return np.mean(similarities) if similarities else 0.0