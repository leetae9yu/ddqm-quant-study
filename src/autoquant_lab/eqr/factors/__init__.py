"""DDQM2-style factor-return research track for autoquant-lab."""

from .definitions import FactorDefinition, all_factor_definitions, implemented_factor_definitions
from .scores import build_factor_scores
from .returns import build_factor_long_short_returns
from .ddqm2 import train_factor_return_models, build_factor_allocations, backtest_factor_allocations

__all__ = [
    "FactorDefinition",
    "all_factor_definitions",
    "implemented_factor_definitions",
    "build_factor_scores",
    "build_factor_long_short_returns",
    "train_factor_return_models",
    "build_factor_allocations",
    "backtest_factor_allocations",
]
