"""Base model class with strict method compatibility checking."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple
import numpy as np


class UnsupportedMethodError(Exception):
    """Raised when attempting to use an unsupported decoding method."""
    pass


class BaseModel(ABC):
    """
    Abstract base class for all models.

    Enforces strict method compatibility - no silent fallbacks.
    """

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.total_tokens = 0
        self.total_cost = 0.0

    @property
    @abstractmethod
    def supported_methods(self) -> List[str]:
        """
        Return list of supported decoding methods.

        This must be explicitly defined by each model class.
        """
        pass

    @property
    @abstractmethod
    def supports_logprobs(self) -> bool:
        """Whether this model can return log probabilities."""
        pass

    @property
    @abstractmethod
    def supports_full_logprobs(self) -> bool:
        """Whether this model can return probabilities for all vocabulary tokens."""
        pass

    def validate_method(self, method: str) -> None:
        """
        Check if method is supported. Raises exception if not.

        Args:
            method: Decoding method name (e.g., "greedy", "beam_10", "nucleus_0.95")

        Raises:
            UnsupportedMethodError: If method is not supported
        """
        if not self.can_use_method(method):
            raise UnsupportedMethodError(
                f"Model {self.model_name} does not support method '{method}'. "
                f"Supported methods: {', '.join(self.supported_methods)}"
            )

    def can_use_method(self, method: str) -> bool:
        """
        Check if this model supports a given decoding method.

        Args:
            method: Method name to check

        Returns:
            True if method is supported, False otherwise
        """
        # Handle parameterized methods (e.g., "beam_10" -> "beam")
        base_method = method.split('_')[0] if '_' in method else method

        # Check both full method name and base method name
        return method in self.supported_methods or base_method in self.supported_methods

    def generate(
        self,
        prompt: str,
        method: str = "greedy",
        max_length: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50,
        num_beams: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate text using specified decoding method.

        Args:
            prompt: Input prompt
            method: Decoding method (must be in supported_methods)
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            num_beams: Beam search width (extracted from method if needed)
            **kwargs: Additional method-specific parameters

        Returns:
            Generated text

        Raises:
            UnsupportedMethodError: If method is not supported
        """
        # Strict validation - no silent fallbacks
        self.validate_method(method)

        # Extract beam size from method name if applicable
        if method.startswith("beam_") and num_beams is None:
            try:
                num_beams = int(method.split("_")[1])
            except (IndexError, ValueError):
                raise ValueError(f"Invalid beam search method format: {method}")

        # Extract nucleus/top_p value from method name if applicable
        if method.startswith("nucleus_"):
            try:
                top_p = float(method.split("_")[1])
            except (IndexError, ValueError):
                raise ValueError(f"Invalid nucleus sampling method format: {method}")

        # Extract top_k value from method name if applicable
        if method.startswith("top_k_"):
            try:
                top_k = int(method.split("_")[1])
            except (IndexError, ValueError):
                raise ValueError(f"Invalid top-k sampling method format: {method}")

        # Delegate to implementation
        return self._generate_impl(
            prompt=prompt,
            method=method,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs
        )

    @abstractmethod
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
        Actual implementation of text generation.

        To be implemented by subclasses.
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