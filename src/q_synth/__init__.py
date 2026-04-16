"""Q synthesizer package for long-horizon task prompt generation."""

from .config import load_config
from .synthesizer import QSynthesizer

__all__ = ["load_config", "QSynthesizer"]
