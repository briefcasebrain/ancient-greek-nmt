#!/usr/bin/env python3
"""
Batch Processing Example

This script demonstrates how to efficiently translate multiple texts at once,
process files, and handle large datasets.
"""

import sys
sys.path.append('..')

import json
import time
from pathlib import Path
from typing import List, Dict

from ancient_greek_nmt.core.translator import Translator
from ancient_greek_nmt.preprocessing.normalizer import GreekNormalizer
from ancient_greek_nmt.evaluation.metrics import MetricCalculator


def process_file(input_file: str, output_file: str, translator: Translator, normalizer: GreekNormalizer):
    """Process a file containing Greek texts."""
    print(f"\nProcessing file: {input_file}")

    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Normalize texts
    normalized_texts = [normalizer.normalize(line.strip()) for line in lines if line.strip()]

    # Translate in batches
    print(f"Translating {len(normalized_texts)} texts...")
    start_time = time.time()

    translations = translator.translate_batch(
        normalized_texts,
        batch_size=8,
        num_beams=5
    )

    elapsed_time = time.time() - start_time
    print(f"Translation completed in {elapsed_time:.2f} seconds")
    print(f"Average time per text: {elapsed_time/len(normalized_texts):.3f} seconds")

    # Save translations
    with open(output_file, 'w', encoding='utf-8') as f:
        for original, translation in zip(lines, translations):
            f.write(f"{original.strip()}\t{translation}\n")

    print(f"Results saved to: {output_file}")
    return translations


def process_parallel_corpus(corpus_file: str, translator: Translator):
    """Process a parallel corpus and evaluate translation quality."""
    print(f"\nProcessing parallel corpus: {corpus_file}")

    # Load corpus
    data = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data.append(item)

    print(f"Loaded {len(data)} parallel texts")

    # Prepare normalizer
    normalizer = GreekNormalizer()

    # Translate and evaluate
    calculator = MetricCalculator()
    results = []
    total_bleu = 0
    total_chrf = 0

    print("\nTranslating and evaluating...")
    for i, item in enumerate(data):
        # Normalize and translate
        normalized = normalizer.normalize(item['greek'])
        translation = translator.translate(normalized)

        # Evaluate
        metrics = calculator.evaluate_translation(
            hypothesis=translation,
            reference=item['english']
        )

        results.append({
            'greek': item['greek'],
            'reference': item['english'],
            'translation': translation,
            'bleu': metrics.bleu,
            'chrf': metrics.chrf
        })

        total_bleu += metrics.bleu
        total_chrf += metrics.chrf

        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(data)} texts")

    # Calculate averages
    avg_bleu = total_bleu / len(data)
    avg_chrf = total_chrf / len(data)

    print(f"\nEvaluation Results:")
    print(f"  Average BLEU: {avg_bleu:.2f}")
    print(f"  Average chrF: {avg_chrf:.2f}")

    return results


def batch_comparison_demo(translator: Translator):
    """Compare different batch sizes for performance."""
    print("\nBatch Size Performance Comparison")
    print("=" * 50)

    # Generate test data
    test_texts = [
        "οἱ ἄνθρωποι τοὺς θεοὺς τιμῶσιν",
        "ἡ σοφία ἐστὶν ἀρετή",
        "οἱ παῖδες μανθάνουσι γράμματα",
        "ὁ χρόνος πάντα δείκνυσι"
    ] * 10  # 40 texts total

    normalizer = GreekNormalizer()
    normalized = [normalizer.normalize(t) for t in test_texts]

    batch_sizes = [1, 4, 8, 16]

    for batch_size in batch_sizes:
        start_time = time.time()

        translations = translator.translate_batch(
            normalized,
            batch_size=batch_size
        )

        elapsed = time.time() - start_time

        print(f"Batch size {batch_size:2d}: {elapsed:.2f}s "
              f"({elapsed/len(test_texts):.3f}s per text)")


def main():
    # Initialize translator
    print("Initializing translator...")
    translator = Translator(model_name="facebook/mbart-large-50-many-to-many-mmt")
    normalizer = GreekNormalizer()

    # Demo 1: Process a list of texts
    print("\n" + "=" * 70)
    print("DEMO 1: Batch Translation of Multiple Texts")
    print("=" * 70)

    sample_texts = [
        "ἀρχὴ ἥμισυ παντός",
        "ἄνθρωπος μέτρον ἁπάντων",
        "νοῦς ὁρᾷ καὶ νοῦς ἀκούει",
        "εὖ πράττειν καὶ εὐδαιμονεῖν",
        "μηδὲν ἄγαν"
    ]

    normalized_texts = [normalizer.normalize(text) for text in sample_texts]
    translations = translator.translate_batch(normalized_texts, batch_size=4)

    for original, translation in zip(sample_texts, translations):
        print(f"{original:30} → {translation}")

    # Demo 2: Batch size comparison
    print("\n" + "=" * 70)
    print("DEMO 2: Batch Size Performance Analysis")
    print("=" * 70)
    batch_comparison_demo(translator)

    # Demo 3: File processing (if example files exist)
    input_file = Path("sample_greek_texts.txt")
    if input_file.exists():
        output_file = "translations_output.txt"
        process_file(str(input_file), output_file, translator, normalizer)
    else:
        print("\n" + "=" * 70)
        print("DEMO 3: File Processing")
        print("=" * 70)
        print("Creating sample file for demonstration...")

        # Create sample file
        sample_content = "\n".join([
            "γνῶθι σεαυτόν",
            "μηδὲν ἄγαν",
            "ἐγγύα πάρα δ' ἄτα",
            "σοφὸς ὁ γινώσκων ἑαυτόν",
            "χαλεπὰ τὰ καλά"
        ])

        with open("sample_greek_texts.txt", "w", encoding="utf-8") as f:
            f.write(sample_content)

        process_file("sample_greek_texts.txt", "translations_output.txt", translator, normalizer)

    # Demo 4: Parallel corpus evaluation (if exists)
    corpus_file = Path("parallel_corpus.jsonl")
    if corpus_file.exists():
        results = process_parallel_corpus(str(corpus_file), translator)

        # Save detailed results
        with open("evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    else:
        print("\n" + "=" * 70)
        print("DEMO 4: Parallel Corpus Evaluation")
        print("=" * 70)
        print("No parallel corpus found. Create 'parallel_corpus.jsonl' with format:")
        print('{"greek": "Greek text", "english": "English translation"}')

    print("\n" + "=" * 70)
    print("Batch processing demonstration complete!")


if __name__ == "__main__":
    main()