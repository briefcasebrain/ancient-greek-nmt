#!/usr/bin/env python3
"""
Text Preprocessing Example

This script demonstrates various text preprocessing techniques for
Ancient Greek text, including normalization, cleaning, and handling
of special characters.
"""

import sys
sys.path.append('..')

import unicodedata
from typing import List, Dict

from ancient_greek_nmt.preprocessing.normalizer import GreekNormalizer, EnglishNormalizer


def demonstrate_normalization_options():
    """Show different normalization options and their effects."""
    print("NORMALIZATION OPTIONS DEMONSTRATION")
    print("=" * 70)

    # Test text with various features
    test_text = "Τὸν Ἄνδρα ὁρῶ· οὗτός ἐστιν ὁ Φίλος."

    print(f"Original text: {test_text}")
    print(f"Length: {len(test_text)} characters\n")

    # Different normalization configurations
    configs = [
        {
            "name": "Preserve Everything",
            "keep_diacritics": True,
            "lowercase": False,
            "normalize_sigma": False,
            "remove_punctuation": False
        },
        {
            "name": "Lowercase Only",
            "keep_diacritics": True,
            "lowercase": True,
            "normalize_sigma": False,
            "remove_punctuation": False
        },
        {
            "name": "Normalize Sigma",
            "keep_diacritics": True,
            "lowercase": True,
            "normalize_sigma": True,
            "remove_punctuation": False
        },
        {
            "name": "Remove Diacritics",
            "keep_diacritics": False,
            "lowercase": True,
            "normalize_sigma": True,
            "remove_punctuation": False
        },
        {
            "name": "Remove Punctuation",
            "keep_diacritics": True,
            "lowercase": True,
            "normalize_sigma": True,
            "remove_punctuation": True
        },
        {
            "name": "Full Normalization",
            "keep_diacritics": False,
            "lowercase": True,
            "normalize_sigma": True,
            "remove_punctuation": True
        }
    ]

    for config in configs:
        normalizer = GreekNormalizer(
            keep_diacritics=config["keep_diacritics"],
            lowercase=config["lowercase"],
            normalize_sigma=config["normalize_sigma"],
            remove_punctuation=config["remove_punctuation"]
        )

        normalized = normalizer.normalize(test_text)
        print(f"{config['name']:20} → {normalized}")

    print()


def analyze_unicode_characters(text: str):
    """Analyze Unicode characteristics of Greek text."""
    print("UNICODE CHARACTER ANALYSIS")
    print("=" * 70)
    print(f"Text: {text}\n")

    for i, char in enumerate(text):
        code_point = ord(char)
        name = unicodedata.name(char, "UNNAMED")
        category = unicodedata.category(char)
        combining = unicodedata.combining(char)

        print(f"Position {i:2}: '{char}' U+{code_point:04X}")
        print(f"  Name: {name}")
        print(f"  Category: {category}")
        if combining:
            print(f"  Combining class: {combining}")
        print()


def demonstrate_step_by_step_normalization():
    """Show step-by-step normalization process."""
    print("STEP-BY-STEP NORMALIZATION")
    print("=" * 70)

    text = "Ὁ Ἄνθρωπος φύσει πολιτικὸν ζῷον"
    normalizer = GreekNormalizer(keep_diacritics=False, lowercase=True)

    print(f"Original: {text}\n")

    # Get explanation of each step
    explanation = normalizer.explain_normalization(text)

    for step, result in explanation.items():
        if step != 'original':
            print(f"{step:25} → {result}")

    print()


def handle_mixed_text():
    """Handle texts with mixed Greek and Latin characters."""
    print("MIXED TEXT HANDLING")
    print("=" * 70)

    mixed_texts = [
        "The word λόγος (logos) means 'word' or 'reason'",
        "Aristotle wrote: ὁ ἄνθρωπος φύσει πολιτικὸν ζῷον",
        "ISBN: 978-0-19-953940-6 contains οἱ Ἕλληνες texts"
    ]

    normalizer = GreekNormalizer()

    for text in mixed_texts:
        # Extract and normalize only Greek parts
        normalized = normalizer.normalize(text)
        print(f"Original: {text}")
        print(f"Normalized: {normalized}")
        print()


def process_inscriptional_text():
    """Process epigraphic/inscriptional conventions."""
    print("INSCRIPTIONAL TEXT PROCESSING")
    print("=" * 70)

    # Common inscriptional conventions
    inscriptions = [
        "ΚΑΙΣΑΡ ΣΕΒΑΣΤΟΣ",  # All capitals (common in inscriptions)
        "[Κ]ΑΙΣΑΡ ΣΕΒ[ΑΣΤΟΣ]",  # Brackets indicate restored text
        "ΚΑΙΣΑΡ · ΣΕΒΑΣΤΟΣ",  # Interpuncts
        "ΚΑΙΣΑΡ | ΣΕΒΑΣΤΟΣ"  # Line breaks marked with |
    ]

    normalizer = GreekNormalizer(lowercase=True)

    for inscription in inscriptions:
        # Clean inscriptional conventions
        cleaned = inscription.replace("[", "").replace("]", "")
        cleaned = cleaned.replace("·", " ").replace("|", " ")

        normalized = normalizer.normalize(cleaned)

        print(f"Inscription: {inscription}")
        print(f"Cleaned: {cleaned}")
        print(f"Normalized: {normalized}")
        print()


def handle_dialectal_variations():
    """Handle different Greek dialects and variations."""
    print("DIALECTAL VARIATIONS")
    print("=" * 70)

    dialects = [
        {"text": "τὸν ἄνδρα", "dialect": "Attic", "note": "Standard form"},
        {"text": "τὸν ἄνδρᾱ", "dialect": "Doric", "note": "Alpha for eta"},
        {"text": "τὼ ἄνδρε", "dialect": "Dual", "note": "Dual number"},
        {"text": "τοῖν ἀνδροῖν", "dialect": "Dual genitive", "note": "Dual genitive/dative"}
    ]

    normalizer = GreekNormalizer()

    for item in dialects:
        normalized = normalizer.normalize(item["text"])
        print(f"{item['dialect']:15} {item['text']:15} → {normalized:15} ({item['note']})")

    print()


def clean_ocr_errors():
    """Clean common OCR errors in Greek text."""
    print("OCR ERROR CORRECTION")
    print("=" * 70)

    # Common OCR mistakes
    ocr_errors = [
        ("οἱ παῖδες", "οι παιδες", "Missing diacritics"),
        ("τὸν ἄνδρα", "τον άνδρα", "Incomplete diacritics"),
        ("ὁρῶ", "ορω", "Missing all diacritics"),
        ("εἰμί", "ειμι", "Missing iota subscript")
    ]

    normalizer = GreekNormalizer(keep_diacritics=False)

    print("Normalizing to handle OCR inconsistencies:\n")
    for correct, ocr, error_type in ocr_errors:
        norm_correct = normalizer.normalize(correct)
        norm_ocr = normalizer.normalize(ocr)

        print(f"Correct: {correct:15} → {norm_correct}")
        print(f"OCR:     {ocr:15} → {norm_ocr}")
        print(f"Match: {norm_correct == norm_ocr} ({error_type})")
        print()


def parallel_normalization():
    """Normalize parallel Greek-English texts."""
    print("PARALLEL TEXT NORMALIZATION")
    print("=" * 70)

    parallel_texts = [
        {
            "greek": "Τὸν ἄνδρα ὁρῶ.",
            "english": "I see the man."
        },
        {
            "greek": "Οἱ παῖδες ἐν τῇ οἰκίᾳ εἰσίν.",
            "english": "The children are in the house."
        },
        {
            "greek": "Γνῶθι σεαυτόν!",
            "english": "Know thyself!"
        }
    ]

    greek_norm = GreekNormalizer(lowercase=True, keep_diacritics=True)
    english_norm = EnglishNormalizer(lowercase=True, expand_contractions=True)

    for pair in parallel_texts:
        norm_greek = greek_norm.normalize(pair["greek"])
        norm_english = english_norm.normalize(pair["english"])

        print(f"Greek Original:  {pair['greek']}")
        print(f"Greek Normalized: {norm_greek}")
        print(f"English Original: {pair['english']}")
        print(f"English Normalized: {norm_english}")
        print()


def main():
    print("\n" + "=" * 70)
    print("TEXT PREPROCESSING EXAMPLES")
    print("=" * 70 + "\n")

    # Demo 1: Normalization options
    demonstrate_normalization_options()

    # Demo 2: Unicode analysis
    print("\n" + "=" * 70)
    analyze_unicode_characters("Ἄνθρωπος")

    # Demo 3: Step-by-step normalization
    print("\n" + "=" * 70)
    demonstrate_step_by_step_normalization()

    # Demo 4: Mixed text handling
    print("\n" + "=" * 70)
    handle_mixed_text()

    # Demo 5: Inscriptional text
    print("\n" + "=" * 70)
    process_inscriptional_text()

    # Demo 6: Dialectal variations
    print("\n" + "=" * 70)
    handle_dialectal_variations()

    # Demo 7: OCR error correction
    print("\n" + "=" * 70)
    clean_ocr_errors()

    # Demo 8: Parallel text normalization
    print("\n" + "=" * 70)
    parallel_normalization()

    print("=" * 70)
    print("Text preprocessing demonstration complete!")


if __name__ == "__main__":
    main()