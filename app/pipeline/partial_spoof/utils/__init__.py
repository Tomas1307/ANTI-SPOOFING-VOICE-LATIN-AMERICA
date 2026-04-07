"""
Utility functions for Partial Spoof Pipeline.
"""
from app.pipeline.partial_spoof.utils.crossfade import apply_crossfade
from app.pipeline.partial_spoof.utils.strategy_factory import create_attack_strategy

__all__ = [
    "apply_crossfade",
    "create_attack_strategy",
]
