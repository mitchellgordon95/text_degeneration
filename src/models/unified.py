"""Unified model factory with strict type checking."""

from typing import Dict, Any
from .base import BaseModel
from .openai_model import OpenAIModel
from .anthropic_model import AnthropicModel
from .huggingface_model import HuggingFaceModel
from .vllm_model import VLLMModel


class UnifiedModel:
    """
    Factory for creating model instances with appropriate wrappers.

    Automatically selects the correct model class based on model name/config.
    """

    # Model type mappings
    OPENAI_MODELS = {
        "gpt-3.5-turbo-instruct",
        "gpt-4",
        "gpt-4-turbo",
        "gpt-5",
        "text-davinci-003",
        "text-davinci-002"
    }

    ANTHROPIC_MODELS = {
        "claude-3-5-sonnet-20241022",
        "claude-3-opus",
        "claude-3-opus-20240229",
        "claude-3-haiku",
        "claude-3-haiku-20240307",
        "claude-4-opus",
        "claude-opus-4-1-20250805"
    }

    HUGGINGFACE_MODELS = {
        "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
        "llama3-8b", "llama3-70b",
        "qwen2.5-7b", "qwen2.5-72b",
        "mistral-7b", "mistral-small-3-24b",
        "mixtral-8x7b",
        "deepseek-7b", "deepseek-67b"
    }

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
        # Check if model type is explicitly specified in kwargs
        model_type = kwargs.get("type")
        engine = kwargs.get("engine", "huggingface")  # Default engine for HF models

        if model_type:
            # Use explicit type from config
            if model_type == "openai":
                return OpenAIModel(model_name, **kwargs)
            elif model_type == "anthropic":
                return AnthropicModel(model_name, **kwargs)
            elif model_type == "huggingface":
                # Check if vLLM engine is requested
                if engine == "vllm":
                    model_id = kwargs.get("model_id", model_name)
                    # Remove model_id from kwargs to avoid duplicate parameter
                    vllm_kwargs = {k: v for k, v in kwargs.items() if k != "model_id"}
                    return VLLMModel(model_name, model_id, **vllm_kwargs)
                else:
                    return HuggingFaceModel(model_name, **kwargs)
            else:
                raise ValueError(
                    f"Unknown model type '{model_type}' for model '{model_name}'. "
                    f"Supported types: openai, anthropic, huggingface"
                )

        # Infer type from model name
        if model_name in UnifiedModel.OPENAI_MODELS:
            return OpenAIModel(model_name, **kwargs)

        elif model_name in UnifiedModel.ANTHROPIC_MODELS:
            return AnthropicModel(model_name, **kwargs)

        elif model_name in UnifiedModel.HUGGINGFACE_MODELS:
            # Check if vLLM engine is requested for inferred HuggingFace models
            if engine == "vllm":
                model_id = kwargs.get("model_id", model_name)
                vllm_kwargs = {k: v for k, v in kwargs.items() if k != "model_id"}
                return VLLMModel(model_name, model_id, **vllm_kwargs)
            else:
                return HuggingFaceModel(model_name, **kwargs)

        # Check for patterns in model name
        elif "gpt" in model_name.lower() and not model_name.startswith("gpt2"):
            print(f"Inferring OpenAI type for model '{model_name}'")
            return OpenAIModel(model_name, **kwargs)

        elif "claude" in model_name.lower():
            print(f"Inferring Anthropic type for model '{model_name}'")
            return AnthropicModel(model_name, **kwargs)

        else:
            # Default to HuggingFace for unknown models
            print(
                f"Unknown model '{model_name}', assuming HuggingFace model. "
                f"Specify 'type' in config to avoid ambiguity."
            )
            # Check if vLLM engine is requested for default case
            if engine == "vllm":
                model_id = kwargs.get("model_id", model_name)
                vllm_kwargs = {k: v for k, v in kwargs.items() if k != "model_id"}
                return VLLMModel(model_name, model_id, **vllm_kwargs)
            else:
                return HuggingFaceModel(model_name, **kwargs)

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
        # Determine model type
        model_type = kwargs.get("type")

        if not model_type:
            if model_name in UnifiedModel.OPENAI_MODELS:
                model_type = "openai"
            elif model_name in UnifiedModel.ANTHROPIC_MODELS:
                model_type = "anthropic"
            elif model_name in UnifiedModel.HUGGINGFACE_MODELS:
                model_type = "huggingface"
            else:
                model_type = "unknown"

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

        elif model_type == "huggingface":
            return {
                "name": model_name,
                "type": "huggingface",
                "beam_search": True,
                "logprobs": "full",
                "perplexity": "exact",
                "tail_analysis": True,
                "supported_methods": [
                    "greedy", "beam_*", "temperature", "nucleus_*", "top_k_*",
                    "contrastive"
                ],
                "limitations": [
                    "Requires local GPU/CPU resources",
                    "May be slower than API models",
                    "Memory requirements vary by model size"
                ]
            }

        else:
            return {
                "name": model_name,
                "type": "unknown",
                "error": f"Cannot determine capabilities for unknown model type"
            }