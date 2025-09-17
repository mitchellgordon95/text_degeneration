"""Utilities for loading prompts and configurations."""

import yaml
from pathlib import Path
from typing import List, Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_prompts(prompts_config: str) -> List[str]:
    """
    Load prompts from configuration file.

    Args:
        prompts_config: Path to prompts YAML file

    Returns:
        List of prompt strings
    """
    config = load_config(prompts_config)

    # Return default prompts
    return config.get("default_prompts", [])


def load_prompt_set(prompts_config: str, set_name: str) -> List[str]:
    """
    Load a specific set of prompts.

    Args:
        prompts_config: Path to prompts YAML file
        set_name: Name of the prompt set

    Returns:
        List of prompt strings
    """
    config = load_config(prompts_config)

    # Check if it's a direct prompt list
    if set_name in config:
        return config[set_name]

    # Check if it's a reference to another set
    prompt_sets = config.get("prompt_sets", {})
    if set_name in prompt_sets:
        actual_set = prompt_sets[set_name]
        if actual_set in config:
            return config[actual_set]

    # Default to empty list
    return []


def load_human_texts(path: str = None, num_texts: int = 500) -> List[str]:
    """
    Load human-written texts for comparison.
    For now, returns empty list - would load from dataset.
    """
    # TODO: Implement loading from actual human text dataset
    # Could use WikiText, OpenWebText, etc.
    return []