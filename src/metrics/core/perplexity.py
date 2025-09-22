"""Perplexity gap metric for measuring model overconfidence."""

from typing import List, Dict


def compute_perplexity(model, texts: List[str]) -> float:
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