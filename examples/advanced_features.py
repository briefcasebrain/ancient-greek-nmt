#!/usr/bin/env python3
"""
Advanced Features Example

This script demonstrates advanced features including:
- Different decoding strategies
- Attention visualization
- Confidence scoring
- Model comparison
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

from ancient_greek_nmt.core.translator import Translator
from ancient_greek_nmt.preprocessing.normalizer import GreekNormalizer
from ancient_greek_nmt.utils.visualizer import AttentionVisualizer
from ancient_greek_nmt.evaluation.metrics import MetricCalculator


def demonstrate_decoding_strategies(translator: Translator, text: str):
    """Show different decoding strategies and their effects."""
    print("\nDECODING STRATEGIES COMPARISON")
    print("=" * 70)
    print(f"Input text: {text}\n")

    strategies = [
        {
            "name": "Greedy Decoding",
            "params": {"num_beams": 1, "do_sample": False}
        },
        {
            "name": "Beam Search (beam=3)",
            "params": {"num_beams": 3, "do_sample": False}
        },
        {
            "name": "Beam Search (beam=5)",
            "params": {"num_beams": 5, "do_sample": False}
        },
        {
            "name": "Sampling (temp=0.8)",
            "params": {"do_sample": True, "temperature": 0.8, "num_beams": 1}
        },
        {
            "name": "Top-p Sampling (p=0.9)",
            "params": {"do_sample": True, "top_p": 0.9, "num_beams": 1}
        },
        {
            "name": "Beam + Sampling",
            "params": {"num_beams": 3, "do_sample": True, "temperature": 0.7}
        }
    ]

    for strategy in strategies:
        result = translator.translate(
            text,
            return_scores=True,
            **strategy["params"]
        )

        print(f"{strategy['name']:25} → {result['translation']}")
        if 'score' in result:
            print(f"{'':25}   (confidence: {result['score']:.3f})")


def visualize_attention(translator: Translator, text: str):
    """Visualize attention patterns for a translation."""
    print("\nATTENTION VISUALIZATION")
    print("=" * 70)

    # Get translation with attention weights
    result = translator.translate(
        text,
        return_attention=True,
        return_scores=True
    )

    print(f"Source: {text}")
    print(f"Translation: {result['translation']}")
    print(f"Confidence: {result.get('score', 'N/A')}")

    if 'attention' in result:
        # Create attention heatmap
        visualizer = AttentionVisualizer()
        visualizer.plot_attention_heatmap(
            attention_weights=result['attention'],
            source_tokens=result.get('source_tokens', text.split()),
            target_tokens=result.get('target_tokens', result['translation'].split()),
            title="Translation Attention Patterns"
        )
    else:
        print("Note: Attention weights not available for this model")


def compare_models(text: str):
    """Compare translations from different models."""
    print("\nMODEL COMPARISON")
    print("=" * 70)
    print(f"Input text: {text}\n")

    models = [
        {
            "name": "mBART-base",
            "model_id": "facebook/mbart-large-50-many-to-many-mmt",
            "src_lang": "grc",
            "tgt_lang": "en"
        },
        {
            "name": "NLLB-600M",
            "model_id": "facebook/nllb-200-distilled-600M",
            "src_lang": "grc_Grek",
            "tgt_lang": "eng_Latn"
        }
    ]

    calculator = MetricCalculator()
    reference = "The children are in the house"  # Known good translation

    results = []

    for model_info in models:
        try:
            print(f"Loading {model_info['name']}...")
            translator = Translator(model_name=model_info['model_id'])

            # Translate
            result = translator.translate(
                text,
                src_lang=model_info['src_lang'],
                tgt_lang=model_info['tgt_lang'],
                return_scores=True
            )

            # Evaluate if reference available
            metrics = None
            if reference:
                metrics = calculator.evaluate_translation(
                    hypothesis=result['translation'],
                    reference=reference
                )

            results.append({
                "model": model_info['name'],
                "translation": result['translation'],
                "confidence": result.get('score', 'N/A'),
                "bleu": metrics.bleu if metrics else None,
                "chrf": metrics.chrf if metrics else None
            })

            print(f"{model_info['name']:15} → {result['translation']}")
            if metrics:
                print(f"{'':15}   BLEU: {metrics.bleu:.1f}, chrF: {metrics.chrf:.1f}")

        except Exception as e:
            print(f"Error with {model_info['name']}: {e}")

    return results


def confidence_analysis(translator: Translator, texts: List[str]):
    """Analyze translation confidence across multiple texts."""
    print("\nCONFIDENCE ANALYSIS")
    print("=" * 70)

    confidences = []
    translations = []

    for text in texts:
        result = translator.translate(text, return_scores=True)
        if 'score' in result:
            confidences.append(result['score'])
            translations.append({
                'text': text,
                'translation': result['translation'],
                'confidence': result['score']
            })

    if confidences:
        # Statistics
        print(f"Average confidence: {np.mean(confidences):.3f}")
        print(f"Min confidence: {np.min(confidences):.3f}")
        print(f"Max confidence: {np.max(confidences):.3f}")
        print(f"Std deviation: {np.std(confidences):.3f}")

        # Show low confidence translations
        print("\nLow confidence translations (< 0.5):")
        for item in translations:
            if item['confidence'] < 0.5:
                print(f"  {item['text'][:30]:30} → {item['translation'][:30]:30} "
                      f"(conf: {item['confidence']:.3f})")

        # Plot confidence distribution
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Translation Confidence Distribution')
        plt.axvline(np.mean(confidences), color='red', linestyle='--',
                   label=f'Mean: {np.mean(confidences):.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


def contextual_translation(translator: Translator):
    """Demonstrate contextual translation with additional information."""
    print("\nCONTEXTUAL TRANSLATION")
    print("=" * 70)

    examples = [
        {
            "text": "λόγος",
            "context": "In philosophical context",
            "expected": "reason/rational principle"
        },
        {
            "text": "λόγος",
            "context": "In grammatical context",
            "expected": "word/speech"
        },
        {
            "text": "λόγος",
            "context": "In mathematical context",
            "expected": "ratio/proportion"
        }
    ]

    for example in examples:
        # Note: Context handling depends on model capabilities
        # This is a demonstration of how it could work
        translation = translator.translate(example['text'])

        print(f"Text: {example['text']}")
        print(f"Context: {example['context']}")
        print(f"Translation: {translation}")
        print(f"Expected: {example['expected']}")
        print("-" * 40)


def main():
    # Initialize components
    print("Initializing translator...")
    translator = Translator(model_name="facebook/mbart-large-50-many-to-many-mmt")
    normalizer = GreekNormalizer()

    # Test text
    test_text = normalizer.normalize("οἱ παῖδες ἐν τῇ οἰκίᾳ εἰσίν")

    # Demo 1: Decoding strategies
    demonstrate_decoding_strategies(translator, test_text)

    # Demo 2: Attention visualization
    print("\n" + "=" * 70)
    visualize_attention(translator, test_text)

    # Demo 3: Model comparison
    print("\n" + "=" * 70)
    compare_models(test_text)

    # Demo 4: Confidence analysis
    print("\n" + "=" * 70)
    test_texts = [
        normalizer.normalize("γνῶθι σεαυτόν"),
        normalizer.normalize("μηδὲν ἄγαν"),
        normalizer.normalize("πάντα ῥεῖ"),
        normalizer.normalize("ἓν οἶδα ὅτι οὐδὲν οἶδα"),
        normalizer.normalize("ὁ βίος βραχύς")
    ]
    confidence_analysis(translator, test_texts)

    # Demo 5: Contextual translation
    print("\n" + "=" * 70)
    contextual_translation(translator)

    print("\n" + "=" * 70)
    print("Advanced features demonstration complete!")


if __name__ == "__main__":
    main()