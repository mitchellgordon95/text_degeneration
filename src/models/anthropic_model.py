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


    def generate_greedy(self, prompt: str, max_length: int = 256) -> str:
        """Generate text using greedy decoding (temperature=0)."""
        try:
            response = self.client.messages.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_length,
                temperature=0.0,
                top_p=1.0
            )

            text = response.content[0].text if response.content else ""

            # Track usage (approximate)
            approx_tokens = len(prompt.split()) + len(text.split())
            self.total_tokens += approx_tokens

            return text

        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")

    def generate_beam(self, prompt: str, beam_size: int, max_length: int = 256) -> str:
        """Generate text using beam search decoding (not supported by Anthropic)."""
        raise UnsupportedMethodError(
            f"Anthropic API does not support beam search. "
            f"Model: {self.model_name}, requested beam_size: {beam_size}"
        )

    def generate_nucleus(self, prompt: str, top_p: float, max_length: int = 256) -> str:
        """Generate text using nucleus (top-p) sampling."""
        try:
            response = self.client.messages.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_length,
                temperature=1.0,
                top_p=top_p
            )

            text = response.content[0].text if response.content else ""

            # Track usage (approximate)
            approx_tokens = len(prompt.split()) + len(text.split())
            self.total_tokens += approx_tokens

            return text

        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")

    def generate_top_k(self, prompt: str, top_k: int, max_length: int = 256) -> str:
        """Generate text using top-k sampling (not supported by Anthropic)."""
        raise UnsupportedMethodError(
            f"Anthropic API does not support top_k sampling. "
            f"Model: {self.model_name}, requested top_k: {top_k}. "
            f"Use nucleus sampling instead."
        )

    def generate_temperature(self, prompt: str, temperature: float, max_length: int = 256) -> str:
        """Generate text using temperature sampling."""
        try:
            response = self.client.messages.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_length,
                temperature=temperature,
                top_p=1.0
            )

            text = response.content[0].text if response.content else ""

            # Track usage (approximate)
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