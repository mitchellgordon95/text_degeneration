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
        self.cost_per_1k = kwargs.get("cost_per_1k", 0.003)

    @property
    def supported_methods(self) -> List[str]:
        """Anthropic supports only basic sampling methods."""
        return [
            "greedy",
            "temperature",
            "nucleus_0.9", "nucleus_0.95", "nucleus_0.99"
            # Note: No top_k support in Anthropic API
        ]

    @property
    def supports_logprobs(self) -> bool:
        """Anthropic does NOT provide any logprobs."""
        return False

    @property
    def supports_full_logprobs(self) -> bool:
        """Anthropic does NOT provide any probabilities."""
        return False

    def _generate_impl(
        self,
        prompt: str,
        method: str,
        max_length: int,
        temperature: float,
        top_p: float,
        top_k: int,
        num_beams: Optional[int],
        **kwargs
    ) -> str:
        """
        Generate text using Anthropic API.

        Note: Many methods will raise errors as they're not supported.
        """
        # Fail fast for unsupported methods
        if method.startswith("beam"):
            raise UnsupportedMethodError(
                f"Anthropic API does not support beam search. "
                f"Model: {self.model_name}, Method: {method}"
            )

        if method.startswith("top_k"):
            raise UnsupportedMethodError(
                f"Anthropic API does not support top_k sampling. "
                f"Model: {self.model_name}, Method: {method}"
            )

        # Configure parameters based on method
        if method == "greedy":
            temperature = 0.0
            top_p = 1.0
        elif method == "temperature":
            temperature = temperature
            top_p = 1.0
        elif method.startswith("nucleus"):
            temperature = temperature
            top_p = top_p
        else:
            raise ValueError(f"Unknown method: {method}")

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
            self.total_cost += (approx_tokens / 1000) * self.cost_per_1k

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