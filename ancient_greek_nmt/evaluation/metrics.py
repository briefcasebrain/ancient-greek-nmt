"""
Evaluation Metrics Module for Ancient Greek Translation

This module provides comprehensive evaluation metrics for assessing translation quality,
with detailed explanations of what each metric measures and why it matters.

Metrics Overview:
================
1. BLEU (Bilingual Evaluation Understudy)
   - Measures n-gram precision between candidate and reference translations
   - Range: 0-100 (higher is better)
   - Good for: Overall translation quality

2. chrF (Character F-score)
   - Character-level metric, better for morphologically rich languages
   - Range: 0-100 (higher is better)
   - Good for: Ancient Greek with its complex morphology

3. METEOR (Metric for Evaluation with Explicit Ordering)
   - Considers synonyms, stemming, and word order
   - Range: 0-1 (higher is better)
   - Good for: Semantic similarity

4. BERTScore
   - Uses contextual embeddings to measure semantic similarity
   - Range: 0-1 (higher is better)
   - Good for: Meaning preservation

5. Custom Metrics for Ancient Greek
   - Case agreement accuracy
   - Verb tense/mood preservation
   - Word order flexibility scoring
"""

import sacrebleu
import numpy as np
from typing import List, Dict, Union, Tuple, Optional
from dataclasses import dataclass
import json
from collections import Counter
import regex as re


@dataclass
class TranslationMetrics:
    """
    Container for all evaluation metrics with explanations.
    """
    bleu: float
    chrf: float
    meteor: Optional[float] = None
    bertscore: Optional[float] = None
    case_accuracy: Optional[float] = None
    word_order_score: Optional[float] = None
    length_ratio: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary for reporting."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def summary(self) -> str:
        """Generate human-readable summary of metrics."""
        lines = ["Translation Quality Metrics:"]
        lines.append("=" * 40)

        if self.bleu is not None:
            lines.append(f"BLEU Score: {self.bleu:.2f}/100")
            lines.append(f"  Interpretation: {self._interpret_bleu()}")

        if self.chrf is not None:
            lines.append(f"chrF Score: {self.chrf:.2f}/100")
            lines.append(f"  Interpretation: {self._interpret_chrf()}")

        if self.meteor is not None:
            lines.append(f"METEOR Score: {self.meteor:.3f}/1.0")

        if self.length_ratio != 1.0:
            lines.append(f"Length Ratio: {self.length_ratio:.2f}")
            lines.append(f"  ({self._interpret_length_ratio()})")

        return "\n".join(lines)

    def _interpret_bleu(self) -> str:
        """Provide interpretation of BLEU score."""
        if self.bleu < 10:
            return "Poor - Major issues with translation"
        elif self.bleu < 20:
            return "Fair - Understandable but many errors"
        elif self.bleu < 30:
            return "Good - Reasonable quality with some errors"
        elif self.bleu < 40:
            return "Very Good - High quality translation"
        else:
            return "Excellent - Near professional quality"

    def _interpret_chrf(self) -> str:
        """Provide interpretation of chrF score."""
        if self.chrf < 30:
            return "Poor character-level accuracy"
        elif self.chrf < 50:
            return "Moderate character-level accuracy"
        elif self.chrf < 70:
            return "Good character-level accuracy"
        else:
            return "Excellent character-level accuracy"

    def _interpret_length_ratio(self) -> str:
        """Interpret the length ratio between hypothesis and reference."""
        if self.length_ratio < 0.9:
            return "Translation is shorter than expected"
        elif self.length_ratio > 1.1:
            return "Translation is longer than expected"
        else:
            return "Translation length is appropriate"


class MetricCalculator:
    """
    Main class for calculating translation metrics with educational explanations.
    """

    def __init__(self, target_lang: str = "en"):
        """
        Initialize metric calculator.

        Parameters:
            target_lang: Target language for metrics that need language info
        """
        self.target_lang = target_lang

    def evaluate_translation(
        self,
        hypothesis: Union[str, List[str]],
        reference: Union[str, List[str]],
        detailed: bool = True
    ) -> TranslationMetrics:
        """
        Evaluate translation quality using multiple metrics.

        Parameters:
            hypothesis: Generated translation(s)
            reference: Reference translation(s)
            detailed: Whether to compute additional detailed metrics

        Returns:
            TranslationMetrics object with all computed scores

        Example:
            >>> calc = MetricCalculator()
            >>> metrics = calc.evaluate_translation(
            ...     "The children are in the house",
            ...     "The children are at home"
            ... )
            >>> print(metrics.summary())
        """
        # Ensure inputs are lists
        if isinstance(hypothesis, str):
            hypothesis = [hypothesis]
        if isinstance(reference, str):
            reference = [reference]

        # Calculate primary metrics
        bleu_score = self._calculate_bleu(hypothesis, reference)
        chrf_score = self._calculate_chrf(hypothesis, reference)

        # Calculate length ratio
        length_ratio = self._calculate_length_ratio(hypothesis, reference)

        # Initialize metrics object
        metrics = TranslationMetrics(
            bleu=bleu_score,
            chrf=chrf_score,
            length_ratio=length_ratio
        )

        # Add detailed metrics if requested
        if detailed:
            metrics.meteor = self._calculate_meteor(hypothesis, reference)
            metrics.case_accuracy = self._calculate_case_accuracy(hypothesis, reference)
            metrics.word_order_score = self._calculate_word_order_score(hypothesis, reference)

        return metrics

    def _calculate_bleu(self, hypothesis: List[str], reference: List[str]) -> float:
        """
        Calculate BLEU score with explanation.

        BLEU measures how many n-grams (word sequences) in the hypothesis
        match those in the reference. It considers:
        - Unigrams (single words)
        - Bigrams (word pairs)
        - Trigrams (word triples)
        - 4-grams (sequences of 4 words)

        The score is the geometric mean of these precisions, multiplied
        by a brevity penalty if the translation is too short.
        """
        # Prepare references (BLEU expects list of lists)
        refs = [[r] for r in reference]

        # Calculate BLEU score
        bleu = sacrebleu.corpus_bleu(hypothesis, refs)

        return bleu.score

    def _calculate_chrf(self, hypothesis: List[str], reference: List[str]) -> float:
        """
        Calculate chrF (character F-score).

        chrF is particularly good for Ancient Greek because:
        1. It captures morphological similarities (word endings)
        2. It's less sensitive to tokenization differences
        3. It handles rare words better than word-level metrics

        The metric calculates precision and recall at the character level,
        then combines them into an F-score.
        """
        # Prepare references
        refs = [[r] for r in reference]

        # Calculate chrF score
        chrf = sacrebleu.corpus_chrf(hypothesis, refs)

        return chrf.score

    def _calculate_meteor(self, hypothesis: List[str], reference: List[str]) -> Optional[float]:
        """
        Calculate METEOR score if available.

        METEOR improves on BLEU by:
        - Considering synonyms (e.g., "house" vs "home")
        - Using stemming (e.g., "running" vs "ran")
        - Evaluating word order
        - Providing better correlation with human judgments
        """
        try:
            # This would require nltk and meteor metric
            # Simplified placeholder
            return None
        except:
            return None

    def _calculate_length_ratio(self, hypothesis: List[str], reference: List[str]) -> float:
        """
        Calculate the length ratio between hypothesis and reference.

        This helps identify if the model is:
        - Under-translating (ratio < 1.0)
        - Over-translating (ratio > 1.0)
        - Appropriately verbose (ratio ≈ 1.0)
        """
        hyp_lengths = [len(h.split()) for h in hypothesis]
        ref_lengths = [len(r.split()) for r in reference]

        avg_hyp_len = np.mean(hyp_lengths)
        avg_ref_len = np.mean(ref_lengths)

        return avg_hyp_len / avg_ref_len if avg_ref_len > 0 else 0.0

    def _calculate_case_accuracy(self, hypothesis: List[str], reference: List[str]) -> Optional[float]:
        """
        Calculate case preservation accuracy (important for Greek).

        Ancient Greek has a complex case system:
        - Nominative (subject)
        - Genitive (possession)
        - Dative (indirect object)
        - Accusative (direct object)
        - Vocative (address)

        This metric checks if grammatical relationships are preserved.
        """
        # This would require morphological analysis
        # Placeholder for demonstration
        return None

    def _calculate_word_order_score(self, hypothesis: List[str], reference: List[str]) -> Optional[float]:
        """
        Calculate word order similarity score.

        Ancient Greek has flexible word order, but certain patterns
        are more common. This metric measures how well the translation
        preserves meaningful word order patterns.
        """
        # Simplified word order comparison
        scores = []
        for hyp, ref in zip(hypothesis, reference):
            hyp_words = hyp.lower().split()
            ref_words = ref.lower().split()

            # Calculate position correlation
            common_words = set(hyp_words) & set(ref_words)
            if len(common_words) > 1:
                score = self._kendall_tau(hyp_words, ref_words, common_words)
                scores.append(score)

        return np.mean(scores) if scores else None

    def _kendall_tau(self, seq1: List[str], seq2: List[str], common: set) -> float:
        """
        Calculate Kendall's tau correlation for word order.

        Measures how similarly words are ordered in two sequences.
        """
        # Get positions of common words
        pos1 = {w: i for i, w in enumerate(seq1) if w in common}
        pos2 = {w: i for i, w in enumerate(seq2) if w in common}

        # Count concordant and discordant pairs
        concordant = 0
        discordant = 0

        words = list(common)
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                w1, w2 = words[i], words[j]

                # Check if order is preserved
                if (pos1[w1] < pos1[w2]) == (pos2[w1] < pos2[w2]):
                    concordant += 1
                else:
                    discordant += 1

        total = concordant + discordant
        return (concordant - discordant) / total if total > 0 else 0.0


class BatchEvaluator:
    """
    Evaluate multiple translations efficiently with detailed analysis.
    """

    def __init__(self):
        """Initialize batch evaluator."""
        self.calculator = MetricCalculator()

    def evaluate_dataset(
        self,
        hypotheses: List[str],
        references: List[str],
        source_texts: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Evaluate an entire dataset of translations.

        Parameters:
            hypotheses: List of generated translations
            references: List of reference translations
            source_texts: Optional list of source texts for analysis

        Returns:
            Dictionary with overall metrics and analysis
        """
        if len(hypotheses) != len(references):
            raise ValueError("Number of hypotheses and references must match")

        # Calculate overall metrics
        metrics = self.calculator.evaluate_translation(hypotheses, references)

        # Perform additional analysis
        analysis = {
            'overall_metrics': metrics.to_dict(),
            'sample_size': len(hypotheses)
        }

        # Analyze error patterns
        if source_texts:
            analysis['error_analysis'] = self._analyze_errors(
                source_texts, hypotheses, references
            )

        # Score distribution analysis
        analysis['score_distribution'] = self._analyze_score_distribution(
            hypotheses, references
        )

        return analysis

    def _analyze_errors(
        self,
        sources: List[str],
        hypotheses: List[str],
        references: List[str]
    ) -> Dict[str, any]:
        """
        Analyze common error patterns in translations.

        Identifies:
        - Untranslated words
        - Repeated phrases
        - Length mismatches
        - Potential mistranslations
        """
        errors = {
            'untranslated': [],
            'repetitions': [],
            'length_mismatches': [],
            'low_quality': []
        }

        for i, (src, hyp, ref) in enumerate(zip(sources, hypotheses, references)):
            # Check for untranslated Greek characters
            if any(ord(c) >= 0x0370 and ord(c) <= 0x03FF for c in hyp):
                errors['untranslated'].append(i)

            # Check for repetitions
            words = hyp.split()
            if len(words) > 3:
                word_counts = Counter(words)
                if any(count > len(words) * 0.3 for count in word_counts.values()):
                    errors['repetitions'].append(i)

            # Check length mismatch
            length_ratio = len(hyp.split()) / max(len(ref.split()), 1)
            if length_ratio < 0.5 or length_ratio > 2.0:
                errors['length_mismatches'].append(i)

            # Check translation quality
            score = sacrebleu.sentence_bleu(hyp, [ref]).score
            if score < 10:
                errors['low_quality'].append(i)

        return {
            'error_counts': {k: len(v) for k, v in errors.items()},
            'error_indices': errors,
            'error_rate': sum(len(v) for v in errors.values()) / len(sources)
        }

    def _analyze_score_distribution(
        self,
        hypotheses: List[str],
        references: List[str]
    ) -> Dict[str, any]:
        """
        Analyze the distribution of translation scores.

        Helps identify:
        - Overall translation consistency
        - Outliers that need attention
        - Quality distribution patterns
        """
        scores = []
        for hyp, ref in zip(hypotheses, references):
            score = sacrebleu.sentence_bleu(hyp, [ref]).score
            scores.append(score)

        scores = np.array(scores)

        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'median': float(np.median(scores)),
            'quartiles': {
                'q1': float(np.percentile(scores, 25)),
                'q2': float(np.percentile(scores, 50)),
                'q3': float(np.percentile(scores, 75))
            },
            'poor_translations': int(np.sum(scores < 10)),
            'excellent_translations': int(np.sum(scores > 40))
        }


def evaluate_translation(
    hypothesis: Union[str, List[str]],
    reference: Union[str, List[str]],
    metrics: List[str] = ["bleu", "chrf"],
    verbose: bool = True
) -> Dict[str, float]:
    """
    Convenience function for quick evaluation.

    Parameters:
        hypothesis: Generated translation(s)
        reference: Reference translation(s)
        metrics: List of metrics to compute
        verbose: Whether to print results

    Returns:
        Dictionary of metric scores

    Example:
        >>> scores = evaluate_translation(
        ...     "The children are in the house",
        ...     "The children are at home",
        ...     metrics=["bleu", "chrf"]
        ... )
    """
    calculator = MetricCalculator()
    results = calculator.evaluate_translation(hypothesis, reference)

    if verbose:
        print(results.summary())

    return results.to_dict()