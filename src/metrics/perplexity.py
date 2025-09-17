"""Metrics for measuring perplexity and model confidence."""

from typing import List, Dict, Optional
import numpy as np


def compute_perplexity(
    model,
    texts: List[str]
) -> float:
    """
    Compute perplexity of texts under a model.

    Args:
        model: Model with compute_perplexity method
        texts: List of texts to evaluate

    Returns:
        Average perplexity
    """
    if not texts:
        return float('inf')

    # Use model's built-in perplexity computation
    return model.compute_perplexity(texts)


def compute_perplexity_gap(
    model,
    generated_texts: List[str],
    human_texts: List[str]
) -> Dict[str, float]:
    """
    Compute the perplexity gap between generated and human texts.
    This is a key metric from Holtzman et al. 2019.

    Args:
        model: Model to compute perplexity
        generated_texts: Model-generated texts
        human_texts: Human-written texts

    Returns:
        Dictionary with perplexities and gap metrics
    """
    gen_ppl = compute_perplexity(model, generated_texts)
    human_ppl = compute_perplexity(model, human_texts)

    # Calculate overconfidence ratio (human_ppl / gen_ppl)
    # Higher ratio means model is more overconfident in its own generations
    if gen_ppl > 0:
        overconfidence_ratio = human_ppl / gen_ppl
    else:
        overconfidence_ratio = float('inf')

    return {
        "generated_perplexity": gen_ppl,
        "human_perplexity": human_ppl,
        "perplexity_gap": human_ppl - gen_ppl,
        "overconfidence_ratio": overconfidence_ratio
    }


def compute_token_level_perplexity(
    model,
    text: str,
    return_per_token: bool = False
) -> float:
    """
    Compute perplexity at the token level.

    Args:
        model: Model with get_logprobs method
        text: Text to evaluate
        return_per_token: If True, return per-token perplexities

    Returns:
        Average perplexity or list of per-token perplexities
    """
    # Split into prompt and continuation
    tokens = text.split()
    if len(tokens) < 2:
        return float('inf')

    # Use first token as prompt, rest as continuation
    prompt = tokens[0]
    continuation = " ".join(tokens[1:])

    try:
        # Get log probabilities for each token
        log_probs = model.get_logprobs(prompt, continuation)

        if not log_probs:
            return float('inf')

        # Convert log probs to perplexities
        token_perplexities = [np.exp(-lp) for lp in log_probs]

        if return_per_token:
            return token_perplexities
        else:
            # Return geometric mean (equivalent to exp of average log prob)
            return np.exp(-np.mean(log_probs))

    except (AttributeError, NotImplementedError):
        # Model doesn't support get_logprobs
        return -1.0


def analyze_probability_distribution(
    model,
    prompts: List[str],
    temperature: float = 1.0
) -> Dict[str, float]:
    """
    Analyze properties of the probability distribution.

    Args:
        model: Model with get_token_probabilities method
        prompts: List of prompts to analyze
        temperature: Temperature for distribution analysis

    Returns:
        Dictionary with distribution statistics
    """
    entropies = []
    top_probs = []
    nucleus_sizes = []  # How many tokens to reach 95% probability mass

    for prompt in prompts:
        probs_dict = model.get_token_probabilities(prompt)

        if not probs_dict:
            continue

        # Convert to numpy array and sort
        probs = np.array(list(probs_dict.values()))
        probs = np.sort(probs)[::-1]  # Sort descending

        # Calculate entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        entropies.append(entropy)

        # Top probability
        top_probs.append(probs[0] if len(probs) > 0 else 0)

        # Nucleus size (tokens needed for 95% mass)
        cumsum = np.cumsum(probs)
        nucleus_idx = np.argmax(cumsum >= 0.95)
        nucleus_sizes.append(nucleus_idx + 1)

    return {
        "mean_entropy": np.mean(entropies) if entropies else 0,
        "std_entropy": np.std(entropies) if entropies else 0,
        "mean_top_prob": np.mean(top_probs) if top_probs else 0,
        "mean_nucleus_size": np.mean(nucleus_sizes) if nucleus_sizes else 0
    }


def confidence_calibration_analysis(
    model,
    texts_with_labels: List[tuple]
) -> Dict[str, float]:
    """
    Analyze how well model confidence aligns with actual quality.

    Args:
        model: Model to analyze
        texts_with_labels: List of (text, quality_score) tuples

    Returns:
        Calibration metrics
    """
    if not texts_with_labels:
        return {"calibration_error": 0.0, "correlation": 0.0}

    perplexities = []
    quality_scores = []

    for text, quality_score in texts_with_labels:
        ppl = compute_perplexity(model, [text])
        if ppl > 0:  # Valid perplexity
            perplexities.append(ppl)
            quality_scores.append(quality_score)

    if not perplexities:
        return {"calibration_error": 0.0, "correlation": 0.0}

    # Convert perplexity to confidence (lower perplexity = higher confidence)
    confidences = [1.0 / ppl for ppl in perplexities]

    # Normalize both to [0, 1]
    confidences = np.array(confidences)
    confidences = (confidences - confidences.min()) / (confidences.max() - confidences.min() + 1e-10)

    quality_scores = np.array(quality_scores)
    if quality_scores.max() > 1.0:
        quality_scores = quality_scores / quality_scores.max()

    # Calculate calibration error (mean absolute difference)
    calibration_error = np.mean(np.abs(confidences - quality_scores))

    # Calculate correlation
    correlation = np.corrcoef(confidences, quality_scores)[0, 1]

    return {
        "calibration_error": calibration_error,
        "confidence_quality_correlation": correlation
    }