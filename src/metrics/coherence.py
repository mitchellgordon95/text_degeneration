"""Metrics for measuring coherence and semantic consistency."""

from typing import List, Dict, Optional, Tuple
import numpy as np
import warnings

# Try to import sentence transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    _semantic_model = None
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    warnings.warn("sentence-transformers not available. Semantic coherence will use basic metrics.")


def measure_coherence(
    text: str,
    method: str = "semantic",
    window_size: int = 3
) -> float:
    """
    Measure coherence of a text.

    Args:
        text: Text to analyze
        method: "semantic" for embedding-based, "lexical" for word overlap
        window_size: Number of sentences to consider for local coherence

    Returns:
        Coherence score (0-1, higher is more coherent)
    """
    sentences = _split_into_sentences(text)

    if len(sentences) < 2:
        return 1.0  # Single sentence is perfectly coherent

    if method == "semantic" and SENTENCE_TRANSFORMERS_AVAILABLE:
        return _semantic_coherence(sentences, window_size)
    else:
        return _lexical_coherence(sentences, window_size)


def _semantic_coherence(sentences: List[str], window_size: int = 3) -> float:
    """Calculate coherence using semantic embeddings."""
    global _semantic_model

    # Load model on first use
    if _semantic_model is None:
        _semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Get embeddings
    embeddings = _semantic_model.encode(sentences)

    # Calculate similarity between adjacent sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        # Cosine similarity
        sim = np.dot(embeddings[i], embeddings[i + 1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
        )
        similarities.append(sim)

    # Also calculate similarity within windows
    window_similarities = []
    for i in range(len(sentences) - window_size + 1):
        window_embeds = embeddings[i:i + window_size]
        # Average similarity within window
        window_sim = 0
        count = 0
        for j in range(len(window_embeds)):
            for k in range(j + 1, len(window_embeds)):
                sim = np.dot(window_embeds[j], window_embeds[k]) / (
                    np.linalg.norm(window_embeds[j]) * np.linalg.norm(window_embeds[k])
                )
                window_sim += sim
                count += 1
        if count > 0:
            window_similarities.append(window_sim / count)

    # Combine metrics
    adjacent_score = np.mean(similarities) if similarities else 0
    window_score = np.mean(window_similarities) if window_similarities else 0

    # Weight adjacent similarity more heavily
    return 0.7 * adjacent_score + 0.3 * window_score


def _lexical_coherence(sentences: List[str], window_size: int = 3) -> float:
    """Calculate coherence using word overlap."""
    coherence_scores = []

    for i in range(len(sentences) - 1):
        words1 = set(sentences[i].lower().split())
        words2 = set(sentences[i + 1].lower().split())

        # Remove stopwords (basic list)
        stopwords = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'could', 'to', 'of', 'in', 'for', 'with', 'it', 'this', 'that', 'these', 'those'}
        words1 = words1 - stopwords
        words2 = words2 - stopwords

        if not words1 or not words2:
            coherence_scores.append(0.5)  # Neutral score
            continue

        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        similarity = intersection / union if union > 0 else 0
        coherence_scores.append(similarity)

    return np.mean(coherence_scores) if coherence_scores else 0


def measure_semantic_similarity(text1: str, text2: str) -> float:
    """
    Measure semantic similarity between two texts.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score (0-1)
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        # Fallback to lexical similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0

    global _semantic_model
    if _semantic_model is None:
        _semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = _semantic_model.encode([text1, text2])
    similarity = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )
    return float(similarity)


def topic_consistency(texts: List[str]) -> float:
    """
    Measure how consistent the topics are across multiple texts.

    Args:
        texts: List of texts

    Returns:
        Consistency score (0-1)
    """
    if len(texts) < 2:
        return 1.0

    if SENTENCE_TRANSFORMERS_AVAILABLE:
        global _semantic_model
        if _semantic_model is None:
            _semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Get embeddings for all texts
        embeddings = _semantic_model.encode(texts)

        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(sim)

        return np.mean(similarities) if similarities else 0
    else:
        # Fallback to vocabulary overlap
        all_vocabs = [set(text.lower().split()) for text in texts]
        similarities = []

        for i in range(len(all_vocabs)):
            for j in range(i + 1, len(all_vocabs)):
                if not all_vocabs[i] or not all_vocabs[j]:
                    continue
                intersection = len(all_vocabs[i].intersection(all_vocabs[j]))
                union = len(all_vocabs[i].union(all_vocabs[j]))
                similarities.append(intersection / union if union > 0 else 0)

        return np.mean(similarities) if similarities else 0


def semantic_drift(prompt: str, continuation: str) -> float:
    """
    Measure how much the continuation drifts from the prompt semantically.
    Lower values indicate less drift (better).

    Args:
        prompt: Original prompt
        continuation: Generated continuation

    Returns:
        Drift score (0-1, lower is better)
    """
    similarity = measure_semantic_similarity(prompt, continuation)
    # Convert similarity to drift (inverse)
    return 1.0 - similarity


def recovery_ability(
    model,
    prompt: str,
    bad_token: str,
    continuation_length: int = 100
) -> float:
    """
    Test if model can recover from a bad token choice.

    Args:
        model: Model to test
        prompt: Original prompt
        bad_token: Forced bad token
        continuation_length: How much text to generate

    Returns:
        Recovery score (0-1, higher means better recovery)
    """
    # Generate continuation after bad token
    bad_start = prompt + " " + bad_token
    continuation = model.generate(
        bad_start,
        method="nucleus",
        max_length=continuation_length
    )

    # Split continuation in half
    words = continuation.split()
    mid = len(words) // 2
    first_half = " ".join(words[:mid])
    second_half = " ".join(words[mid:])

    # Measure coherence of each half
    first_coherence = measure_coherence(bad_start + " " + first_half)
    second_coherence = measure_coherence(prompt + " " + second_half)  # Compare to original prompt

    # Recovery is good if second half is more coherent than first
    recovery_score = max(0, second_coherence - first_coherence)

    # Also check if second half is semantically closer to original prompt
    first_drift = semantic_drift(prompt, first_half)
    second_drift = semantic_drift(prompt, second_half)

    drift_improvement = max(0, first_drift - second_drift)

    # Combine both metrics
    return (recovery_score + drift_improvement) / 2


def _split_into_sentences(text: str) -> List[str]:
    """Simple sentence splitter."""
    # Basic splitting on sentence-ending punctuation
    import re
    sentences = re.split(r'[.!?]+', text)
    # Remove empty sentences and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences