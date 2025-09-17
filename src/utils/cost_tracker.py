"""Track API costs across experiments."""

from typing import Dict
from datetime import datetime


class CostTracker:
    """Track costs and tokens for API calls."""

    def __init__(self, warn_at: float = 10.0, stop_at: float = 50.0):
        self.costs = {}
        self.tokens = {}
        self.warn_at = warn_at
        self.stop_at = stop_at
        self.warned = False

    def add_cost(self, model_name: str, cost: float, tokens: int):
        """Add cost and tokens for a model."""
        if model_name not in self.costs:
            self.costs[model_name] = 0.0
            self.tokens[model_name] = 0

        self.costs[model_name] += cost
        self.tokens[model_name] += tokens

        # Check thresholds
        total = self.total_cost()
        if total > self.stop_at:
            raise RuntimeError(f"Cost limit exceeded: ${total:.2f} > ${self.stop_at}")
        elif total > self.warn_at and not self.warned:
            print(f"\n⚠️  Warning: Total cost ${total:.2f} exceeds warning threshold ${self.warn_at}")
            self.warned = True

    def total_cost(self) -> float:
        """Get total cost across all models."""
        return sum(self.costs.values())

    def total_tokens(self) -> int:
        """Get total tokens across all models."""
        return sum(self.tokens.values())

    def get_summary(self) -> Dict[str, Any]:
        """Get cost summary."""
        return {
            "total_cost": self.total_cost(),
            "total_tokens": self.total_tokens(),
            "by_model": {
                model: {
                    "cost": self.costs.get(model, 0),
                    "tokens": self.tokens.get(model, 0)
                }
                for model in self.costs
            },
            "timestamp": datetime.now().isoformat()
        }

    def print_summary(self):
        """Print cost summary."""
        print("\n" + "="*50)
        print("COST SUMMARY")
        print("="*50)
        print(f"Total Cost: ${self.total_cost():.2f}")
        print(f"Total Tokens: {self.total_tokens():,}")
        print("\nBy Model:")
        for model, cost in self.costs.items():
            tokens = self.tokens.get(model, 0)
            print(f"  {model:30} ${cost:6.2f} ({tokens:,} tokens)")
        print("="*50)