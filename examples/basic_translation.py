#!/usr/bin/env python3
"""
Basic Translation Example

This script demonstrates the simplest way to translate Ancient Greek text to English
using the ancient_greek_nmt library.
"""

import sys
sys.path.append('..')

from ancient_greek_nmt.core.translator import Translator
from ancient_greek_nmt.preprocessing.normalizer import GreekNormalizer


def main():
    # Initialize the translator
    print("Initializing translator...")
    translator = Translator(model_name="facebook/mbart-large-50-many-to-many-mmt")

    # Initialize text normalizer
    normalizer = GreekNormalizer(
        keep_diacritics=True,
        lowercase=True,
        normalize_sigma=True
    )

    # Example texts
    texts = [
        "οἱ παῖδες ἐν τῇ οἰκίᾳ εἰσίν",
        "γνῶθι σεαυτόν",
        "πάντα ῥεῖ καὶ οὐδὲν μένει",
        "ἓν οἶδα ὅτι οὐδὲν οἶδα",
        "ὁ βίος βραχύς, ἡ δὲ τέχνη μακρή"
    ]

    sources = [
        "Basic Grammar",
        "Delphic Maxim",
        "Heraclitus",
        "Socrates",
        "Hippocrates"
    ]

    print("\nTranslating Ancient Greek texts...\n")
    print("=" * 70)

    for text, source in zip(texts, sources):
        # Normalize the Greek text
        normalized = normalizer.normalize(text)

        # Translate
        translation = translator.translate(normalized)

        # Display results
        print(f"Source: {source}")
        print(f"Greek:  {text}")
        print(f"English: {translation}")
        print("-" * 70)

    # Interactive mode
    print("\nEnter your own Greek text (or 'quit' to exit):")
    while True:
        user_input = input("\nGreek text: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        if user_input:
            try:
                # Normalize and translate
                normalized = normalizer.normalize(user_input)
                translation = translator.translate(normalized)
                print(f"Translation: {translation}")
            except Exception as e:
                print(f"Error: {e}")

    print("\nThank you for using the Ancient Greek translator!")


if __name__ == "__main__":
    main()