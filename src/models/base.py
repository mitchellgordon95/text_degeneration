"""Base model class with strict method compatibility checking."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

# Import our capability system
from ..utils.capabilities import CapabilityManager, UnsupportedMethodError


class BaseModel(ABC):
    """
    Abstract base class for all models.

    Enforces strict method compatibility using YAML-based capabilities.
    No silent fallbacks allowed.
    """

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.total_tokens = 0

        # Initialize capability manager for this model
        self._capability_manager = CapabilityManager()

    @property
    def supported_methods(self) -> List[str]:
        """
        Return list of supported decoding methods from YAML config.
        """
        return self._capability_manager.get_supported_methods(self.model_name)

    @property
    def supports_logprobs(self) -> bool:
        """Whether this model can return log probabilities."""
        return self._capability_manager.supports_logprobs(self.model_name)

    @property
    def supports_full_logprobs(self) -> bool:
        """Whether this model can return probabilities for all vocabulary tokens."""
        return self._capability_manager.supports_full_logprobs(self.model_name)

    def validate_method(self, method: str) -> None:
        """
        Check if method is supported. Raises exception if not.

        Args:
            method: Decoding method name (e.g., "greedy", "beam_10", "nucleus_0.95")

        Raises:
            UnsupportedMethodError: If method is not supported
        """
        self._capability_manager.validate_method(self.model_name, method)

    def can_use_method(self, method: str) -> bool:
        """
        Check if this model supports a given decoding method.

        Args:
            method: Method name to check

        Returns:
            True if method is supported, False otherwise
        """
        return self._capability_manager.supports_method(self.model_name, method)

    @abstractmethod
    def generate_greedy(self, prompt: str, max_length: int = 256) -> str:
        """
        Generate text using greedy decoding (deterministic).

        Args:
            prompt: Input prompt
            max_length: Maximum tokens to generate

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def generate_beam(self, prompt: str, beam_size: int, max_length: int = 256) -> str:
        """
        Generate text using beam search decoding.

        Args:
            prompt: Input prompt
            beam_size: Number of beams for beam search
            max_length: Maximum tokens to generate

        Returns:
            Generated text

        Raises:
            UnsupportedMethodError: If model doesn't support beam search
        """
        pass

    @abstractmethod
    def generate_nucleus(self, prompt: str, top_p: float, max_length: int = 256) -> str:
        """
        Generate text using nucleus (top-p) sampling.

        Args:
            prompt: Input prompt
            top_p: Cumulative probability threshold for nucleus sampling
            max_length: Maximum tokens to generate

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def generate_top_k(self, prompt: str, top_k: int, max_length: int = 256) -> str:
        """
        Generate text using top-k sampling.

        Args:
            prompt: Input prompt
            top_k: Number of top tokens to consider
            max_length: Maximum tokens to generate

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def generate_temperature(self, prompt: str, temperature: float, max_length: int = 256) -> str:
        """
        Generate text using temperature sampling.

        Args:
            prompt: Input prompt
            temperature: Temperature for sampling (higher = more random)
            max_length: Maximum tokens to generate

        Returns:
            Generated text
        """
        pass

    def get_token_probabilities(
        self,
        prompt: str,
        next_token: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Get probability distribution over next tokens.

        Args:
            prompt: Input prompt
            next_token: If provided, return only probability for this token

        Returns:
            Dictionary mapping tokens to probabilities

        Raises:
            UnsupportedMethodError: If model doesn't support logprobs
        """
        if not self.supports_logprobs:
            raise UnsupportedMethodError(
                f"Model {self.model_name} does not support getting token probabilities"
            )

        return self._get_token_probabilities_impl(prompt, next_token)

    @abstractmethod
    def _get_token_probabilities_impl(
        self,
        prompt: str,
        next_token: Optional[str] = None
    ) -> Dict[str, float]:
        """Implementation of getting token probabilities."""
        pass

    def compute_perplexity(self, texts: List[str]) -> float:
        """
        Compute perplexity of texts under this model.

        Args:
            texts: List of texts to evaluate

        Returns:
            Perplexity score

        Raises:
            UnsupportedMethodError: If model doesn't support perplexity calculation
        """
        if not self.supports_logprobs:
            raise UnsupportedMethodError(
                f"Model {self.model_name} cannot compute perplexity (no logprob access)"
            )

        return self._compute_perplexity_impl(texts)

    @abstractmethod
    def _compute_perplexity_impl(self, texts: List[str]) -> float:
        """Implementation of perplexity calculation."""
        pass

    def analyze_tail_distribution(
        self,
        prompt: str,
        percentile_ranges: List[Tuple[float, float]]
    ) -> Dict[str, Any]:
        """
        Analyze token probability distribution tail.

        Args:
            prompt: Input prompt
            percentile_ranges: List of (min, max) percentile ranges to analyze

        Returns:
            Analysis results for each range

        Raises:
            UnsupportedMethodError: If model doesn't support full distribution access
        """
        if not self.supports_full_logprobs:
            raise UnsupportedMethodError(
                f"Model {self.model_name} cannot analyze tail distribution "
                f"(no full vocabulary access)"
            )

        return self._analyze_tail_distribution_impl(prompt, percentile_ranges)

    @abstractmethod
    def _analyze_tail_distribution_impl(
        self,
        prompt: str,
        percentile_ranges: List[Tuple[float, float]]
    ) -> Dict[str, Any]:
        """Implementation of tail distribution analysis."""
        pass