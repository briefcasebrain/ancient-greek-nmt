"""
Text Normalization Module for Ancient Greek and English

This module handles the complex task of normalizing Ancient Greek text, which involves:
- Unicode normalization for consistent character representation
- Diacritics handling (accents, breathings, iota subscripts)
- Final sigma normalization
- Punctuation standardization
- Whitespace normalization

Ancient Greek Text Challenges:
1. Diacritics: Greek text contains various diacritical marks:
   - Acute accent (΄): indicates stressed syllable
   - Grave accent (`): appears on final syllables
   - Circumflex (῀): indicates long vowel with falling tone
   - Breathing marks: rough (ἁ) and smooth (ἀ)
   - Iota subscript (ᾳ): long alpha/eta/omega with iota

2. Sigma variations: Greek has two forms of lowercase sigma:
   - σ: used in beginning/middle of words
   - ς: used at end of words (final sigma)

3. Unicode complications: Same visual character can have multiple encodings
   - Precomposed vs decomposed forms
   - Need consistent normalization (NFC/NFD)
"""

import unicodedata
import regex as re
from typing import Optional, List, Dict, Tuple
try:
    from ftfy import fix_text
except ImportError:
    # Fallback if ftfy not installed
    def fix_text(x: str) -> str:
        return x


class GreekNormalizer:
    """
    Comprehensive normalizer for Ancient Greek text.

    This class handles all aspects of Greek text normalization, providing
    options for different preprocessing strategies depending on the use case.

    Key Features:
    - Preserves or removes diacritics based on requirements
    - Handles polytonic (ancient) and monotonic (modern) Greek
    - Normalizes character variants for consistency
    - Provides detailed logging of transformations
    """

    # Unicode character constants for Greek
    GREEK_LOWER = 'αβγδεζηθικλμνξοπρστυφχψω'
    GREEK_UPPER = 'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'
    GREEK_FINAL_SIGMA = 'ς'
    GREEK_SIGMA = 'σ'
    GREEK_ANO_TELEIA = '·'  # Greek semicolon

    # Regular expressions for text processing
    COMBINING_MARKS_RE = re.compile(r'\p{M}+')  # All combining marks
    WHITESPACE_RE = re.compile(r'\s+')  # Multiple whitespace
    GREEK_WORD_RE = re.compile(r'\b[\p{Greek}]+\b')  # Greek words

    # Punctuation normalization mappings
    PUNCT_MAP = {
        '\u0387': '·',      # Greek ano teleia variant
        '\u037E': ';',      # Greek question mark → semicolon
        '\u2019': "'",      # Right single quotation
        '\u2018': "'",      # Left single quotation
        '\u201C': '"',      # Left double quotation
        '\u201D': '"',      # Right double quotation
        '…': '...',         # Ellipsis
    }

    def __init__(self,
                 keep_diacritics: bool = True,
                 lowercase: bool = True,
                 normalize_sigma: bool = True):
        """
        Initialize the Greek normalizer with specified options.

        Parameters:
            keep_diacritics: Whether to preserve diacritical marks
                           True: Keep all accents and breathing marks
                           False: Strip to base characters
            lowercase: Whether to convert to lowercase
            normalize_sigma: Whether to normalize final sigma to regular sigma
        """
        self.keep_diacritics = keep_diacritics
        self.lowercase = lowercase
        self.normalize_sigma = normalize_sigma

    def normalize(self, text: str) -> str:
        """
        Main normalization method applying all transformations.

        Processing pipeline:
        1. Fix encoding issues with ftfy
        2. Apply Unicode NFC normalization
        3. Normalize punctuation
        4. Handle sigma variants
        5. Apply case transformation
        6. Remove diacritics if specified
        7. Normalize whitespace

        Parameters:
            text: Input Greek text

        Returns:
            Normalized text string
        """
        if not text:
            return text

        # Step 1: Fix any encoding issues (mojibake, etc.)
        text = fix_text(text)

        # Step 2: Unicode normalization to NFC (Canonical Composition)
        # This ensures consistent representation of characters with diacritics
        text = unicodedata.normalize('NFC', text)

        # Step 3: Normalize punctuation marks
        text = self._normalize_punctuation(text)

        # Step 4: Handle sigma normalization
        if self.normalize_sigma:
            text = self._normalize_sigma(text)

        # Step 5: Case normalization
        if self.lowercase:
            text = text.lower()

        # Step 6: Diacritics handling
        if not self.keep_diacritics:
            text = self._strip_diacritics(text)
            # Re-normalize after stripping
            text = unicodedata.normalize('NFC', text)

        # Step 7: Whitespace normalization
        text = self._normalize_whitespace(text)

        return text

    def _normalize_punctuation(self, text: str) -> str:
        """
        Standardize punctuation marks to consistent forms.

        Ancient Greek texts often use different Unicode points for
        visually similar punctuation. This method maps them to
        standard forms for consistency.
        """
        result = []
        for char in text:
            # Use mapping if available, otherwise keep original
            result.append(self.PUNCT_MAP.get(char, char))
        return ''.join(result)

    def _normalize_sigma(self, text: str) -> str:
        """
        Convert final sigma (ς) to regular sigma (σ).

        In Ancient Greek, sigma has two lowercase forms:
        - σ: used at beginning/middle of words
        - ς: used at end of words

        For some NLP tasks, normalizing to a single form improves consistency.
        """
        return text.replace(self.GREEK_FINAL_SIGMA, self.GREEK_SIGMA)

    def _strip_diacritics(self, text: str) -> str:
        """
        Remove all diacritical marks from Greek text.

        This method:
        1. Decomposes characters into base + combining marks (NFD)
        2. Removes all combining marks
        3. Recomposes remaining characters

        Used when diacritics are not needed for the task (e.g., some
        machine learning models work better without diacritics).
        """
        # Decompose to NFD (base characters + combining marks)
        nfd_text = unicodedata.normalize('NFD', text)

        # Remove all combining marks (category Mn, Mc, Me)
        stripped = self.COMBINING_MARKS_RE.sub('', nfd_text)

        return stripped

    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace to single spaces.

        Handles:
        - Multiple spaces, tabs, newlines
        - Leading/trailing whitespace
        - Non-breaking spaces
        """
        # Replace all whitespace sequences with single space
        text = self.WHITESPACE_RE.sub(' ', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def process_with_alignment(self, text: str) -> Tuple[str, List[Tuple[int, int]]]:
        """
        Normalize text while maintaining character alignment mapping.

        This is useful for:
        - Preserving position information for annotations
        - Mapping normalized text back to original positions
        - Debugging normalization issues

        Returns:
            Tuple of (normalized_text, alignment_map)
            where alignment_map contains (original_pos, normalized_pos) pairs
        """
        # Track character positions through normalization
        alignment = []
        normalized = self.normalize(text)

        # Simple character-level alignment
        # (More sophisticated alignment would track through each transformation)
        orig_pos = 0
        norm_pos = 0

        # This is a simplified alignment - production code would track
        # positions through each transformation step
        for i, char in enumerate(normalized):
            if orig_pos < len(text):
                alignment.append((orig_pos, norm_pos))
                orig_pos += 1
                norm_pos += 1

        return normalized, alignment

    def explain_normalization(self, text: str) -> Dict[str, str]:
        """
        Provide step-by-step explanation of normalization process.

        Useful for debugging and understanding transformations.

        Returns:
            Dictionary with each transformation step and result
        """
        steps = {}
        current = text
        steps['original'] = current

        # Apply each transformation and record result
        current = fix_text(current)
        steps['after_ftfy'] = current

        current = unicodedata.normalize('NFC', current)
        steps['after_unicode_nfc'] = current

        current = self._normalize_punctuation(current)
        steps['after_punctuation'] = current

        if self.normalize_sigma:
            current = self._normalize_sigma(current)
            steps['after_sigma'] = current

        if self.lowercase:
            current = current.lower()
            steps['after_lowercase'] = current

        if not self.keep_diacritics:
            current = self._strip_diacritics(current)
            current = unicodedata.normalize('NFC', current)
            steps['after_strip_diacritics'] = current

        current = self._normalize_whitespace(current)
        steps['final'] = current

        return steps


class EnglishNormalizer:
    """
    Normalizer for English text in parallel corpora.

    While simpler than Greek normalization, consistent English preprocessing
    is important for alignment and translation quality.
    """

    PUNCT_MAP = {
        '\u2019': "'",      # Right single quotation
        '\u2018': "'",      # Left single quotation
        '\u201C': '"',      # Left double quotation
        '\u201D': '"',      # Right double quotation
        '…': '...',         # Ellipsis
        '–': '-',           # En dash
        '—': '-',           # Em dash
    }

    WHITESPACE_RE = re.compile(r'\s+')

    def __init__(self, lowercase: bool = False):
        """
        Initialize English normalizer.

        Parameters:
            lowercase: Whether to convert to lowercase
                      (Usually False for English to preserve proper nouns)
        """
        self.lowercase = lowercase

    def normalize(self, text: str) -> str:
        """
        Normalize English text.

        Processing:
        1. Fix encoding issues
        2. Unicode normalization
        3. Punctuation standardization
        4. Optional lowercasing
        5. Whitespace normalization
        """
        if not text:
            return text

        # Fix encoding issues
        text = fix_text(text)

        # Unicode normalization
        text = unicodedata.normalize('NFC', text)

        # Normalize punctuation
        text = self._normalize_punctuation(text)

        # Lowercase if specified
        if self.lowercase:
            text = text.lower()

        # Normalize whitespace
        text = self._normalize_whitespace(text)

        return text

    def _normalize_punctuation(self, text: str) -> str:
        """Standardize punctuation marks."""
        result = []
        for char in text:
            result.append(self.PUNCT_MAP.get(char, char))
        return ''.join(result)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace to single spaces."""
        text = self.WHITESPACE_RE.sub(' ', text)
        text = text.strip()
        return text


class ParallelNormalizer:
    """
    Normalizer for parallel Greek-English texts.

    Ensures consistent normalization across language pairs,
    which is crucial for alignment and translation model training.
    """

    def __init__(self,
                 keep_greek_diacritics: bool = True,
                 lowercase_greek: bool = True,
                 lowercase_english: bool = False,
                 normalize_sigma: bool = True):
        """
        Initialize parallel normalizer with language-specific options.
        """
        self.greek_normalizer = GreekNormalizer(
            keep_diacritics=keep_greek_diacritics,
            lowercase=lowercase_greek,
            normalize_sigma=normalize_sigma
        )
        self.english_normalizer = EnglishNormalizer(
            lowercase=lowercase_english
        )

    def normalize_pair(self, greek_text: str, english_text: str) -> Tuple[str, str]:
        """
        Normalize a parallel text pair.

        Returns:
            Tuple of (normalized_greek, normalized_english)
        """
        normalized_greek = self.greek_normalizer.normalize(greek_text)
        normalized_english = self.english_normalizer.normalize(english_text)
        return normalized_greek, normalized_english

    def normalize_corpus(self,
                        greek_texts: List[str],
                        english_texts: List[str]) -> Tuple[List[str], List[str]]:
        """
        Normalize entire parallel corpus.

        Parameters:
            greek_texts: List of Greek sentences
            english_texts: List of English sentences (aligned)

        Returns:
            Tuple of (normalized_greek_list, normalized_english_list)
        """
        if len(greek_texts) != len(english_texts):
            raise ValueError(f"Mismatched corpus sizes: {len(greek_texts)} vs {len(english_texts)}")

        normalized_greek = []
        normalized_english = []

        for greek, english in zip(greek_texts, english_texts):
            norm_greek, norm_english = self.normalize_pair(greek, english)
            normalized_greek.append(norm_greek)
            normalized_english.append(norm_english)

        return normalized_greek, normalized_english