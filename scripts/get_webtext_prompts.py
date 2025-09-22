#!/usr/bin/env python3
"""
Script to download and extract prompts AND human continuations from WebText-style data.
For computing perplexity gap, we need both the prompt and the human continuation.
"""

import json
import random
import requests
from pathlib import Path
from typing import List, Dict, Tuple
import re


def get_prompts_from_huggingface():
    """
    Get WebText-style prompts from HuggingFace datasets.
    Using the validation split of a WebText-like dataset.
    """
    print("Fetching WebText-style data from HuggingFace...")

    # We can use several options:
    # 1. OpenWebText (recreation of WebText)
    # 2. WikiText-103 (high quality but different style)
    # 3. BookCorpus
    # 4. C4 (Colossal Clean Crawled Corpus)

    # For now, let's use a simple approach with WikiText since it's easily accessible
    # and has good quality text similar to WebText

    try:
        # Try to use datasets library if available
        from datasets import load_dataset

        print("Loading dataset from HuggingFace...")
        # Load WikiText-103 which has similar quality to WebText
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")

        texts = []
        for item in dataset:
            text = item['text'].strip()
            # Filter for substantial paragraphs (need at least 256 tokens for prompt + continuation)
            if len(text.split()) > 256 and not text.startswith('='):  # Skip headers
                texts.append(text)

        print(f"Found {len(texts)} text passages with sufficient length")
        return texts

    except ImportError:
        print("ERROR: datasets library not installed. Please run: pip install datasets")
        return []




def extract_prompts_and_continuations(
    texts: List[str],
    num_samples: int = 200,
    min_prompt_tokens: int = 10,
    max_prompt_tokens: int = 40,
    total_tokens: int = 256
) -> Tuple[List[str], List[str]]:
    """
    Extract prompts and their human continuations from text passages.

    Args:
        texts: List of text passages
        num_samples: Number of prompt-continuation pairs to extract
        min_prompt_tokens: Minimum prompt length in tokens
        max_prompt_tokens: Maximum prompt length in tokens
        total_tokens: Total length of prompt + continuation

    Returns:
        Tuple of (prompts, continuations)
    """
    prompts = []
    continuations = []

    for text in texts:
        if len(prompts) >= num_samples:
            break

        # Clean text
        text = text.strip()
        if not text:
            continue

        # Split into words (simple tokenization)
        words = text.split()

        # Need enough tokens for prompt + continuation
        if len(words) < total_tokens:
            continue

        # Determine prompt length (random between min and max)
        prompt_length = random.randint(min_prompt_tokens, max_prompt_tokens)

        # Extract prompt and continuation
        prompt = " ".join(words[:prompt_length])

        # Continuation is the next tokens up to total_tokens
        continuation_length = total_tokens - prompt_length
        continuation = " ".join(words[prompt_length:prompt_length + continuation_length])

        prompts.append(prompt)
        continuations.append(continuation)

    return prompts[:num_samples], continuations[:num_samples]


def save_prompts_to_yaml(
    prompts: List[str],
    continuations: List[str],
    output_path: str = "config/prompts.yaml"
):
    """
    Save prompts and human continuations to YAML configuration file.
    """
    import yaml

    # Create prompt-continuation pairs for easier access
    prompt_continuation_pairs = [
        {"prompt": p, "continuation": c}
        for p, c in zip(prompts, continuations)
    ]

    config = {
        "# Note": "WebText-style prompts and human continuations for perplexity gap calculation",
        "# Source": "Similar to prompts used in Holtzman et al. 2019",
        "# Prompt length": "10-40 tokens as in the original paper",
        "# Total length": "256 tokens (prompt + continuation)",

        "webtext_prompts": prompts[:200],  # Keep 200 prompts

        "# Human continuations for perplexity gap": "",
        "human_continuations": continuations[:200],  # Matching continuations

        "# Prompt-continuation pairs": "",
        "prompt_continuation_pairs": prompt_continuation_pairs[:200],

        "# For experiments": "",
        "default_prompts": prompts[:20],  # Quick testing

        "# Task-specific prompts remain the same": "",
        "creative_prompts": [
            "Once upon a time in a land far away",
            "The old mansion stood at the end of the street",
            "She opened the mysterious letter and gasped",
        ],

        "factual_prompts": [
            "The capital of France is",
            "Water boils at a temperature of",
            "The speed of light in vacuum is approximately",
        ],

        "prompt_sets": {
            "degeneration": "webtext_prompts",
            "perplexity": "webtext_prompts",
            "tail_analysis": "webtext_prompts",
        }
    }

    # Write to file
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"Saved {len(prompts)} prompts and continuations to {output_path}")


def main():
    """Main function to get and process WebText prompts with continuations."""
    random.seed(42)  # For reproducibility

    # Get text data
    texts = get_prompts_from_huggingface()

    if not texts or len(texts) == 0:
        print("ERROR: No texts found. Make sure datasets library is installed.")
        return

    # Extract prompts and continuations
    prompts, continuations = extract_prompts_and_continuations(
        texts,
        num_samples=min(200, len(texts)),  # Get up to 200 samples
        min_prompt_tokens=10,     # Minimum 10 tokens for prompt
        max_prompt_tokens=40,     # Maximum 40 tokens for prompt
        total_tokens=256          # Total 256 tokens (matching experiment max_length)
    )

    print(f"\nExtracted {len(prompts)} prompt-continuation pairs")
    print("\nSample prompt-continuation pairs:")
    for i in range(min(3, len(prompts))):
        prompt_tokens = len(prompts[i].split())
        cont_tokens = len(continuations[i].split())
        print(f"\n{i+1}. Prompt [{prompt_tokens} tokens]: {prompts[i][:80]}...")
        print(f"   Continuation [{cont_tokens} tokens]: {continuations[i][:80]}...")
        print(f"   Total tokens: {prompt_tokens + cont_tokens}")

    # Save to YAML
    save_prompts_to_yaml(prompts, continuations)

    print("\n✓ Prompts and continuations ready for experiments!")
    print("   Use human_continuations for perplexity gap calculation")


if __name__ == "__main__":
    main()