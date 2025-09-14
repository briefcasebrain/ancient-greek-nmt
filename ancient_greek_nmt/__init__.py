"""
Ancient Greek Neural Machine Translation Library

A comprehensive library for training and deploying neural machine translation models
between Ancient Greek and modern languages, with a focus on English translation.

This library provides:
- Advanced text preprocessing for Ancient Greek with diacritics handling
- State-of-the-art transformer models (mBART, NLLB, Marian)
- Training utilities with automatic mixed precision and distributed training
- Comprehensive evaluation metrics (BLEU, chrF, METEOR)
- Visualization tools for model analysis and interpretation

Author: Research Team
Version: 1.0.0
"""

from ancient_greek_nmt.core.translator import Translator
from ancient_greek_nmt.preprocessing.normalizer import GreekNormalizer, EnglishNormalizer
from ancient_greek_nmt.training.trainer import NMTTrainer
from ancient_greek_nmt.evaluation.metrics import evaluate_translation

__version__ = "1.0.0"
__all__ = [
    "Translator",
    "GreekNormalizer",
    "EnglishNormalizer",
    "NMTTrainer",
    "evaluate_translation"
]