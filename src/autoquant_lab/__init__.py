"""autoquant_lab package."""

from .config import Config as _Config, load_config as _load_config

Config = _Config
load_config = _load_config

__all__ = ["Config", "load_config"]
