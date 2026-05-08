"""High-level training and evaluation entrypoints."""

from .eval_pair import evaluate_model_pair_test_only
from .train_vowel import train_one_vowel

__all__ = ["train_one_vowel", "evaluate_model_pair_test_only"]
