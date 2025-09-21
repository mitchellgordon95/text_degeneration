"""Anthropic API model wrapper with strict limitations."""

import os
from typing import List, Dict, Optional, Any, Tuple
import anthropic

from .base import BaseModel, UnsupportedMethodError


class AnthropicModel(BaseModel):
    """
    Anthropic API (Claude) model wrapper.

    Severe limitations:
    - No native beam search support
    - NO logprobs available at all
    - Cannot compute perplexity
    - Cannot analyze probability distribution
    """

    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)

        # Get API key
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize client
        self.client = anthropic.Anthropic(api_key=self.api_key)

        # Model-specific settings
        self.model_id = kwargs.get("model_id", model_name)


    def _generate_impl(
        self,
        prompt: str,
        method: str,
        max_length: int,
        **kwargs
    ) -> str:
        """
        Generate text using Anthropic API with strict parameter enforcement.

        NO SILENT FALLBACKS: Unsupported methods will raise errors.
        """
        # Extract parameters from kwargs
        base_method = kwargs.get("base_method", method)
        temperature = kwargs.get("temperature", 1.0)
        top_p = kwargs.get("top_p", 0.95)
        top_k = kwargs.get("top_k")
        num_beams = kwargs.get("num_beams")

        # Fail fast for unsupported methods - NO SILENT FALLBACKS
        if base_method == "beam" or num_beams and num_beams > 1:
            raise UnsupportedMethodError(
                f"Anthropic API does not support beam search. "
                f"Model: {self.model_name}, Method: {method}, "
                f"Base method: {base_method}, Beams: {num_beams}"
            )

        if base_method == "top_k" or top_k is not None:
            raise UnsupportedMethodError(
                f"Anthropic API does not support top_k sampling. "
                f"Model: {self.model_name}, Method: {method}, "
                f"Base method: {base_method}, top_k: {top_k}. "
                f"Use nucleus sampling instead."
            )

        # Only supported methods reach here
        if base_method not in ["greedy", "temperature", "nucleus"]:
            raise UnsupportedMethodError(
                f"Anthropic model {self.model_name} does not support method {base_method}. "
                f"Supported: greedy, temperature, nucleus"
            )

        # Make API call
        try:
            # Anthropic uses messages API
            response = self.client.messages.create(
                model=self.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_length,
                temperature=temperature,
                top_p=top_p
            )

            # Extract text
            text = response.content[0].text if response.content else ""

            # Track usage (approximate)
            # Anthropic doesn't provide exact token counts in the same way
            approx_tokens = len(prompt.split()) + len(text.split())
            self.total_tokens += approx_tokens

            return text

        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")

    def _get_token_probabilities_impl(
        self,
        prompt: str,
        next_token: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Anthropic does NOT support getting token probabilities.

        Raises:
            UnsupportedMethodError: Always
        """
        raise UnsupportedMethodError(
            f"Anthropic models do not provide token probabilities. "
            f"Model: {self.model_name}"
        )

    def _compute_perplexity_impl(self, texts: List[str]) -> float:
        """
        Anthropic cannot compute perplexity (no logprobs).

        Raises:
            UnsupportedMethodError: Always
        """
        raise UnsupportedMethodError(
            f"Anthropic models cannot compute perplexity "
            f"(no access to token probabilities). Model: {self.model_name}"
        )

    def _analyze_tail_distribution_impl(
        self,
        prompt: str,
        percentile_ranges: List[Tuple[float, float]]
    ) -> Dict[str, Any]:
        """
        Anthropic cannot analyze probability distributions.

        Raises:
            UnsupportedMethodError: Always
        """
        raise UnsupportedMethodError(
            f"Anthropic models cannot analyze probability distributions "
            f"(no access to token probabilities). Model: {self.model_name}"
        )