#!/usr/bin/env python3
"""
Script to download and extract prompts from WebText-style data.
We'll use OpenWebText or similar publicly available datasets.
"""

import json
import random
import requests
from pathlib import Path
from typing import List, Dict
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
            # Filter for substantial paragraphs
            if len(text) > 100 and not text.startswith('='):  # Skip headers
                texts.append(text)

        print(f"Found {len(texts)} text passages")
        return texts

    except ImportError:
        print("datasets library not installed, using alternative method...")
        return get_sample_webtext_style_prompts()


def get_sample_webtext_style_prompts():
    """
    If we can't access datasets, provide high-quality WebText-style prompts.
    These are designed to match the style and quality of WebText.
    """
    # These are longer, more realistic prompts similar to WebText
    prompts = [
        "The development of quantum computing has accelerated rapidly in recent years, with major technology companies investing billions of dollars in research. These powerful machines promise to revolutionize fields from cryptography to drug discovery",

        "Climate scientists have been tracking the unprecedented rate of ice melt in Antarctica, where massive glaciers are collapsing into the ocean at alarming speeds. The West Antarctic Ice Sheet alone contains enough frozen water",

        "The rise of artificial intelligence in healthcare has begun transforming how doctors diagnose and treat patients. Machine learning algorithms can now detect certain cancers earlier than human radiologists",

        "In the depths of the Amazon rainforest, researchers have discovered a complex network of ancient settlements connected by road systems. These pre-Columbian civilizations were far more sophisticated",

        "The global supply chain crisis that emerged during the pandemic has forced companies to reconsider their manufacturing strategies. Many corporations are now moving production closer to home",

        "Neuroscientists studying the human brain have made remarkable discoveries about how memories are formed and stored. Using advanced imaging techniques, they can now observe neural pathways",

        "The James Webb Space Telescope has captured images of galaxies that formed just a few hundred million years after the Big Bang. These observations are challenging our understanding",

        "Renewable energy sources have become increasingly cost-competitive with fossil fuels, leading to a dramatic shift in global energy markets. Solar and wind power installations",

        "The discovery of CRISPR gene-editing technology has opened unprecedented possibilities for treating genetic diseases. Scientists can now precisely modify DNA sequences",

        "Archaeological excavations in Turkey have uncovered what may be the world's oldest known temple complex, dating back nearly 12,000 years. The site at Göbekli Tepe predates",

        "The rapid evolution of electric vehicle technology has pushed traditional automakers to completely reimagine their business models. Battery costs have fallen by more than 80 percent",

        "Marine biologists exploring the deep ocean have discovered ecosystems thriving in complete darkness around hydrothermal vents. These unique environments host species",

        "The mathematics behind modern cryptography relies on problems that are easy to verify but computationally difficult to solve. As quantum computers become more powerful",

        "Linguists studying endangered languages estimate that half of the world's 7,000 languages will disappear within the next century. Each language that vanishes takes with it",

        "The human microbiome, consisting of trillions of bacteria living in and on our bodies, plays a crucial role in our health. Recent research has shown connections between gut bacteria",

        "Astronomers have detected mysterious radio signals from distant galaxies that repeat in predictable patterns. These fast radio bursts release as much energy in milliseconds",

        "The development of mRNA vaccines during the COVID-19 pandemic has revolutionized our approach to preventing infectious diseases. This technology, decades in the making",

        "Urban planners are increasingly turning to data science and artificial intelligence to design more efficient and livable cities. Smart traffic systems can now predict",

        "The discovery of gravitational waves has opened an entirely new window for observing the universe. These ripples in spacetime, first predicted by Einstein",

        "Coral reefs around the world are experiencing unprecedented bleaching events due to rising ocean temperatures. Scientists estimate that 90 percent of reefs could disappear",
    ]

    return prompts


def extract_prompts(texts: List[str], num_prompts: int = 200, min_tokens: int = 10, max_tokens: int = 40) -> List[str]:
    """
    Extract prompts of appropriate length from text passages.

    Args:
        texts: List of text passages
        num_prompts: Number of prompts to extract
        min_tokens: Minimum prompt length in tokens
        max_tokens: Maximum prompt length in tokens
    """
    prompts = []

    for text in texts:
        if len(prompts) >= num_prompts:
            break

        # Clean text
        text = text.strip()
        if not text:
            continue

        # Split into words (simple tokenization)
        words = text.split()

        if len(words) < min_tokens:
            continue

        # Take between min_tokens and max_tokens from the beginning
        prompt_length = min(random.randint(min_tokens, max_tokens), len(words))
        prompt = " ".join(words[:prompt_length])

        # Clean up the prompt
        # Remove incomplete sentences if possible
        if prompt_length < len(words):
            # Try to end at a reasonable point (not mid-word)
            prompt = prompt.rsplit(' ', 1)[0] if ' ' in prompt else prompt

        prompts.append(prompt)

    # Shuffle for variety
    random.shuffle(prompts)

    return prompts[:num_prompts]


def save_prompts_to_yaml(prompts: List[str], output_path: str = "config/prompts.yaml"):
    """
    Save prompts to YAML configuration file.
    """
    import yaml

    config = {
        "# Note": "WebText-style prompts extracted from high-quality text sources",
        "# Source": "Similar to prompts used in Holtzman et al. 2019",
        "# Prompt length": "10-40 tokens as in the original paper",

        "webtext_prompts": prompts[:200],  # Keep 200 prompts

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

    print(f"Saved {len(prompts)} prompts to {output_path}")


def main():
    """Main function to get and process WebText prompts."""
    random.seed(42)  # For reproducibility

    # Get text data
    texts = get_prompts_from_huggingface()

    if not texts:
        print("Using fallback WebText-style prompts...")
        texts = get_sample_webtext_style_prompts()

    # Extract prompts of appropriate length
    prompts = extract_prompts(
        texts,
        num_prompts=200,  # Get 200 prompts
        min_tokens=10,     # Minimum 10 tokens
        max_tokens=40      # Maximum 40 tokens (as in Holtzman)
    )

    print(f"\nExtracted {len(prompts)} prompts")
    print("\nSample prompts:")
    for i, prompt in enumerate(prompts[:5], 1):
        tokens = len(prompt.split())
        print(f"\n{i}. [{tokens} tokens] {prompt[:100]}...")

    # Save to YAML
    save_prompts_to_yaml(prompts)

    print("\n✓ Prompts ready for experiments!")


if __name__ == "__main__":
    main()