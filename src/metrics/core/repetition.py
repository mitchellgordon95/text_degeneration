"""Repetition metric following Holtzman et al. 2019 definition."""

from typing import List


def measure_repetition_rate(texts: List[str], max_tokens: int = 200) -> float:
    """
    Measure the percentage of texts that contain phrase repetition at the end.

    Following Holtzman et al. 2019: "A phrase (minimum length 2) is considered
    a repetition when it repeats at least three times at the end of the generation."

    Args:
        texts: List of generated texts
        max_tokens: Maximum tokens to consider from each text (default 200 as in paper)

    Returns:
        Repetition rate as percentage (0-100) - percentage of texts with end repetition
    """
    if not texts:
        return 0.0

    repetitive_texts = 0

    for text in texts:
        if _has_end_repetition(text, max_tokens):
            repetitive_texts += 1

    return (repetitive_texts / len(texts)) * 100


def _has_end_repetition(text: str, max_tokens: int) -> bool:
    """
    Check if text has phrase repetition at the end.

    Args:
        text: Input text
        max_tokens: Maximum tokens to consider

    Returns:
        True if text has repetitive phrases at the end
    """
    # Normalize text - lowercase and basic punctuation handling
    normalized_text = text.lower()
    # Simple punctuation removal (keep it basic for now)
    import string
    for punct in string.punctuation:
        normalized_text = normalized_text.replace(punct, ' ')

    tokens = normalized_text.split()

    # Limit to first max_tokens as in the paper
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

    if len(tokens) < 6:  # Need at least 6 tokens for 3 repetitions of 2-token phrase
        return False

    # Check for phrases of length 2 to min(10, len(tokens)//3)
    # Following Holtzman: minimum phrase length is 2 tokens
    max_phrase_len = min(10, len(tokens) // 3)

    for phrase_len in range(2, max_phrase_len + 1):
        if _check_phrase_repetition_at_end(tokens, phrase_len):
            return True

    return False


def _check_phrase_repetition_at_end(tokens: List[str], phrase_len: int) -> bool:
    """
    Check if a phrase of given length repeats at least 3 times at the end.

    Args:
        tokens: List of tokens
        phrase_len: Length of phrase to check

    Returns:
        True if phrase repeats at least 3 times at the end
    """
    if len(tokens) < phrase_len * 3:
        return False

    # Check if the last phrase_len * 3 tokens form a repetitive pattern
    # Start from positions that could contain 3+ repetitions ending at the text end
    min_start = len(tokens) - phrase_len * 3

    for start_pos in range(min_start, len(tokens) - phrase_len * 2):
        phrase = tokens[start_pos:start_pos + phrase_len]

        # Check how many consecutive repetitions of this phrase occur
        repetitions = 1
        pos = start_pos + phrase_len

        while pos + phrase_len <= len(tokens):
            if tokens[pos:pos + phrase_len] == phrase:
                repetitions += 1
                pos += phrase_len
            else:
                break

        # Must have at least 3 repetitions AND extend to near the end (within 2 tokens)
        if repetitions >= 3 and pos >= len(tokens) - 2:
            return True

    return False