"""Unified model factory with strict type checking."""

from typing import Dict, Any
from .base import BaseModel
from .openai_model import OpenAIModel
from .anthropic_model import AnthropicModel
from .vllm_model import VLLMModel


class UnifiedModel:
    """
    Factory for creating model instances with appropriate wrappers.

    Automatically selects the correct model class based on model name/config.
    """

    # No longer needed - we require explicit type in config

    @staticmethod
    def create(model_name: str, **kwargs) -> BaseModel:
        """
        Create a model instance based on name or configuration.

        Args:
            model_name: Name of the model
            **kwargs: Additional configuration from models.yaml

        Returns:
            Appropriate BaseModel subclass instance

        Raises:
            ValueError: If model type cannot be determined
        """
        # Model type MUST be explicitly specified in config
        model_type = kwargs.get("type")

        if not model_type:
            raise ValueError(
                f"Model type not specified for '{model_name}'. "
                f"You must specify 'type' in models.yaml. "
                f"Supported types: openai, anthropic, vllm"
            )

        # Route to appropriate model class based on type
        if model_type == "openai":
            return OpenAIModel(model_name, **kwargs)
        elif model_type == "anthropic":
            return AnthropicModel(model_name, **kwargs)
        elif model_type == "vllm":
            model_id = kwargs.get("model_id", model_name)
            # Remove model_id from kwargs to avoid duplicate parameter
            vllm_kwargs = {k: v for k, v in kwargs.items() if k != "model_id"}
            return VLLMModel(model_name, model_id, **vllm_kwargs)
        else:
            raise ValueError(
                f"Unknown model type '{model_type}' for model '{model_name}'. "
                f"Supported types: openai, anthropic, vllm"
            )

    @staticmethod
    def get_model_capabilities(model_name: str, **kwargs) -> Dict[str, Any]:
        """
        Get capabilities of a model without loading it.

        Useful for checking what a model can do before running experiments.

        Args:
            model_name: Name of the model
            **kwargs: Additional configuration

        Returns:
            Dictionary of model capabilities
        """
        # Model type must be explicit
        model_type = kwargs.get("type")

        if not model_type:
            return {
                "name": model_name,
                "type": "unknown",
                "error": f"Model type not specified. Must set 'type' in models.yaml"
            }

        # Return capabilities based on type
        if model_type == "openai":
            return {
                "name": model_name,
                "type": "openai",
                "beam_search": False,
                "logprobs": "limited (top-5 only)",
                "perplexity": "approximate",
                "tail_analysis": False,
                "supported_methods": [
                    "greedy", "temperature", "nucleus_*", "top_k_*"
                ],
                "limitations": [
                    "No beam search",
                    "Only top-5 logprobs",
                    "Approximate perplexity only",
                    "Cannot analyze full distribution"
                ]
            }

        elif model_type == "anthropic":
            return {
                "name": model_name,
                "type": "anthropic",
                "beam_search": False,
                "logprobs": False,
                "perplexity": False,
                "tail_analysis": False,
                "supported_methods": [
                    "greedy", "temperature", "nucleus_*"
                ],
                "limitations": [
                    "No beam search",
                    "No logprobs at all",
                    "Cannot compute perplexity",
                    "No top_k sampling",
                    "Cannot analyze distribution"
                ]
            }

        elif model_type == "vllm":
            return {
                "name": model_name,
                "type": "vllm",
                "beam_search": True,
                "logprobs": "limited (top-20)",
                "perplexity": "approximate",
                "tail_analysis": "limited",
                "supported_methods": [
                    "greedy", "beam_*", "temperature", "nucleus_*", "top_k_*",
                    "pure_sampling"
                ],
                "limitations": [
                    "Requires local GPU/CPU resources",
                    "Limited to top-20 logprobs",
                    "Requires vLLM installation"
                ]
            }

        else:
            return {
                "name": model_name,
                "type": model_type,
                "error": f"Unknown model type '{model_type}'. Supported: openai, anthropic, vllm"
            }