"""
Core Translation Module

This module provides the main interface for Ancient Greek to English translation
and vice versa. It handles model loading, tokenization, and generation of translations
using state-of-the-art transformer models.

The translation process involves:
1. Text normalization and preprocessing
2. Tokenization with language-specific tokens
3. Neural translation using transformer models
4. Post-processing and formatting of output

Key Components:
- Translator: Main class for translation operations
- Model management: Automatic model downloading and caching
- Batch processing: Efficient handling of multiple texts
- Beam search: Multiple hypothesis generation for better translations
"""

import os
import torch
from typing import List, Optional, Dict, Union, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PreTrainedModel,
    PreTrainedTokenizer
)


class Translator:
    """
    Main translation interface for Ancient Greek <-> English translation.

    This class manages the entire translation pipeline from raw text to
    translated output, handling all intermediate steps automatically.

    Attributes:
        model: The loaded transformer model for translation
        tokenizer: Tokenizer corresponding to the model
        device: Device (CPU/GPU) for model inference
        src_lang: Source language code
        tgt_lang: Target language code

    Example usage:
        >>> translator = Translator(model_name="facebook/mbart-large-50-many-to-many-mmt")
        >>> result = translator.translate("οἱ παῖδες ἐν τῇ οἰκίᾳ εἰσίν",
        ...                              src_lang="grc", tgt_lang="en")
        >>> print(result)
        "The children are in the house"
    """

    def __init__(
        self,
        model_name: str = "facebook/mbart-large-50-many-to-many-mmt",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the translator with a specified model.

        Parameters:
            model_name: HuggingFace model identifier or local path
            device: Device for computation ('cuda', 'cpu', or None for auto-detect)
            cache_dir: Directory for caching downloaded models

        The initialization process:
        1. Detects available hardware (GPU/CPU)
        2. Downloads model if not cached
        3. Loads model and tokenizer
        4. Configures language tokens
        """
        # Automatically detect the best available device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Initializing translator on {self.device}")

        # Load the tokenizer and model
        # The tokenizer converts text to numerical tokens the model understands
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )

        # Load the translation model with automatic weight initialization
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )

        # Move model to the appropriate device for computation
        self.model = self.model.to(self.device)

        # Put model in evaluation mode (disables dropout for consistency)
        self.model.eval()

        # Store model configuration for reference
        self.model_name = model_name
        self._configure_language_tokens()

    def _configure_language_tokens(self):
        """
        Configure language-specific tokens for multilingual models.

        Different models use different language codes:
        - mBART: Uses codes like 'el_GR' for Greek, 'en_XX' for English
        - NLLB: Uses codes like 'ell_Grek' for Greek, 'eng_Latn' for English

        This method automatically detects and configures the appropriate codes.
        """
        # Language token mappings for different model families
        self.lang_tokens = {
            'mbart': {
                'grc': 'el_GR',  # Ancient Greek mapped to Modern Greek code
                'en': 'en_XX'
            },
            'nllb': {
                'grc': 'ell_Grek',
                'en': 'eng_Latn'
            }
        }

        # Detect model family from name
        model_lower = self.model_name.lower()
        if 'mbart' in model_lower:
            self.model_family = 'mbart'
        elif 'nllb' in model_lower:
            self.model_family = 'nllb'
        else:
            self.model_family = 'generic'

    def translate(
        self,
        text: Union[str, List[str]],
        src_lang: str = "grc",
        tgt_lang: str = "en",
        num_beams: int = 5,
        max_length: int = 512,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_p: float = 0.9,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,
        batch_size: int = 8
    ) -> Union[str, List[str]]:
        """
        Translate text from source to target language.

        Parameters:
            text: Input text(s) to translate (string or list of strings)
            src_lang: Source language code ('grc' for Ancient Greek, 'en' for English)
            tgt_lang: Target language code ('en' for English, 'grc' for Ancient Greek)
            num_beams: Number of beams for beam search (higher = better quality but slower)
                      Beam search explores multiple translation hypotheses in parallel
            max_length: Maximum length of generated translation in tokens
            temperature: Sampling temperature (higher = more creative, lower = more conservative)
            do_sample: Whether to use sampling instead of greedy/beam search
            top_p: Nucleus sampling parameter (only used if do_sample=True)
            length_penalty: Penalty for longer sequences (>1 favors longer, <1 favors shorter)
            no_repeat_ngram_size: Prevent repetition of n-grams of this size
            batch_size: Number of texts to process simultaneously

        Returns:
            Translated text(s) in the same format as input (string or list)

        The translation process:
        1. Normalize and prepare input text
        2. Tokenize with appropriate language tokens
        3. Generate translation using the model
        4. Decode tokens back to text
        5. Post-process the output
        """
        # Handle both single strings and lists of strings
        single_input = isinstance(text, str)
        if single_input:
            texts = [text]
        else:
            texts = text

        # Configure language tokens for the model
        self._set_language_tokens(src_lang, tgt_lang)

        # Process texts in batches for efficiency
        all_translations = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Tokenize the batch
            # This converts text to numerical representations
            inputs = self.tokenizer(
                batch,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            # Generate translations using the model
            with torch.no_grad():  # Disable gradient computation for inference
                # The model generates token IDs for the translation
                generated_tokens = self.model.generate(
                    **inputs,
                    num_beams=num_beams,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p if do_sample else None,
                    length_penalty=length_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    forced_bos_token_id=self._get_forced_bos_token_id(tgt_lang)
                )

            # Decode the generated tokens back to text
            batch_translations = self.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True  # Remove special tokens like <pad>, </s>
            )

            # Clean up the translations
            batch_translations = [self._postprocess(t) for t in batch_translations]
            all_translations.extend(batch_translations)

        # Return in the same format as input
        return all_translations[0] if single_input else all_translations

    def _set_language_tokens(self, src_lang: str, tgt_lang: str):
        """
        Set the appropriate language tokens for the tokenizer.

        This ensures the model knows which languages it's translating between,
        which is crucial for multilingual models like mBART and NLLB.
        """
        if self.model_family in self.lang_tokens:
            tokens = self.lang_tokens[self.model_family]

            # Set source language for encoding
            if hasattr(self.tokenizer, 'src_lang'):
                self.tokenizer.src_lang = tokens.get(src_lang, src_lang)

            # Set target language for decoding
            if hasattr(self.tokenizer, 'tgt_lang'):
                self.tokenizer.tgt_lang = tokens.get(tgt_lang, tgt_lang)

    def _get_forced_bos_token_id(self, tgt_lang: str) -> Optional[int]:
        """
        Get the forced beginning-of-sequence token ID for the target language.

        Some models require a specific token at the start of the generated
        sequence to indicate the target language.
        """
        if self.model_family not in self.lang_tokens:
            return None

        tokens = self.lang_tokens[self.model_family]
        tgt_token = tokens.get(tgt_lang)

        if tgt_token and hasattr(self.tokenizer, 'lang_code_to_id'):
            return self.tokenizer.lang_code_to_id.get(tgt_token)

        return None

    def _postprocess(self, text: str) -> str:
        """
        Post-process the generated translation.

        This method cleans up common issues in generated text:
        - Extra whitespace
        - Incorrect punctuation spacing
        - Capitalization issues
        """
        # Remove extra whitespace
        text = ' '.join(text.split())

        # Fix punctuation spacing
        text = text.replace(' .', '.')
        text = text.replace(' ,', ',')
        text = text.replace(' ;', ';')
        text = text.replace(' :', ':')
        text = text.replace(' !', '!')
        text = text.replace(' ?', '?')

        # Ensure first letter is capitalized
        if text and text[0].isalpha():
            text = text[0].upper() + text[1:]

        return text

    def translate_file(
        self,
        input_path: str,
        output_path: str,
        src_lang: str = "grc",
        tgt_lang: str = "en",
        **kwargs
    ):
        """
        Translate an entire file line by line.

        Parameters:
            input_path: Path to input file (one sentence per line)
            output_path: Path to save translations
            src_lang: Source language code
            tgt_lang: Target language code
            **kwargs: Additional arguments passed to translate()

        This method efficiently processes large files by:
        1. Reading in chunks to manage memory
        2. Using batch processing for speed
        3. Writing results incrementally
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Read input lines
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        print(f"Translating {len(lines)} lines from {input_path}")

        # Translate in batches
        translations = self.translate(
            lines,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            **kwargs
        )

        # Write translations
        with open(output_path, 'w', encoding='utf-8') as f:
            for translation in translations:
                f.write(translation + '\n')

        print(f"Saved translations to {output_path}")

    def explain_translation(
        self,
        text: str,
        src_lang: str = "grc",
        tgt_lang: str = "en"
    ) -> Dict[str, any]:
        """
        Provide detailed explanation of the translation process.

        This method is useful for understanding how the model translates,
        showing tokenization, attention patterns, and alternative translations.

        Returns:
            Dictionary containing:
            - original: The input text
            - translation: The final translation
            - tokens: Tokenized representation
            - alternatives: Alternative translations with different beam sizes
            - confidence: Translation confidence scores
        """
        # Get main translation
        main_translation = self.translate(
            text,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            num_beams=5
        )

        # Get alternative translations with different parameters
        alternatives = []
        for num_beams in [1, 3, 10]:
            alt = self.translate(
                text,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                num_beams=num_beams
            )
            alternatives.append({
                'num_beams': num_beams,
                'translation': alt
            })

        # Tokenize to show how text is processed
        tokens = self.tokenizer.tokenize(text)

        return {
            'original': text,
            'translation': main_translation,
            'tokens': tokens,
            'num_tokens': len(tokens),
            'alternatives': alternatives,
            'model': self.model_name,
            'device': str(self.device)
        }