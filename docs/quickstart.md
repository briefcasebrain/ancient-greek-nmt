# Quick Start Guide

## Basic Translation

The simplest way to translate Ancient Greek text:

```python
from ancient_greek_nmt.core.translator import Translator

# Initialize translator with default settings
translator = Translator()

# Translate a simple sentence
greek_text = "οἱ παῖδες ἐν τῇ οἰκίᾳ εἰσίν"
english = translator.translate(greek_text)
print(english)  # "The children are in the house"
```

## Text Preprocessing

Clean and normalize Ancient Greek text before translation:

```python
from ancient_greek_nmt.preprocessing.normalizer import GreekNormalizer

# Create normalizer
normalizer = GreekNormalizer(
    keep_diacritics=True,  # Preserve accent marks
    lowercase=True,         # Convert to lowercase
    normalize_sigma=True    # Normalize sigma variants
)

# Normalize text
text = "Τὸν ἄνδρα ὁρῶ"
normalized = normalizer.normalize(text)
print(normalized)  # "τὸν ἄνδρα ὁρῶ"
```

## Using Different Models

### mBART Model
```python
translator = Translator(model_name="facebook/mbart-large-50-many-to-many-mmt")
result = translator.translate(greek_text, src_lang="grc", tgt_lang="en")
```

### NLLB Model
```python
translator = Translator(model_name="facebook/nllb-200-distilled-600M")
result = translator.translate(greek_text, src_lang="grc_Grek", tgt_lang="eng_Latn")
```

## Batch Translation

Translate multiple texts efficiently:

```python
texts = [
    "γνῶθι σεαυτόν",
    "πάντα ῥεῖ",
    "ἓν οἶδα ὅτι οὐδὲν οἶδα"
]

translations = translator.translate_batch(texts)
for greek, english in zip(texts, translations):
    print(f"{greek} → {english}")
```

## Advanced Translation Options

```python
# Configure translation parameters
result = translator.translate(
    text=greek_text,
    num_beams=5,           # Beam search width
    max_length=128,        # Maximum output length
    temperature=0.8,       # Sampling temperature
    top_p=0.95,           # Nucleus sampling
    do_sample=False,      # Use deterministic decoding
    return_scores=True    # Return confidence scores
)

print(f"Translation: {result['translation']}")
print(f"Confidence: {result['score']:.2f}")
```

## Evaluation

Evaluate translation quality:

```python
from ancient_greek_nmt.evaluation.metrics import MetricCalculator

calculator = MetricCalculator()

# Evaluate a single translation
hypothesis = "The children are in the house"
reference = "The kids are at home"

metrics = calculator.evaluate_translation(hypothesis, reference)
print(f"BLEU: {metrics.bleu:.1f}")
print(f"chrF: {metrics.chrf:.1f}")
```

## Visualization

Visualize attention patterns:

```python
from ancient_greek_nmt.utils.visualizer import AttentionVisualizer

# Get translation with attention weights
result = translator.translate(greek_text, return_attention=True)

# Visualize attention
visualizer = AttentionVisualizer()
visualizer.plot_attention_heatmap(
    attention_weights=result['attention'],
    source_tokens=result['source_tokens'],
    target_tokens=result['target_tokens']
)
```

## Configuration Management

Use configuration files for reproducible experiments:

```python
from ancient_greek_nmt.configs.config import ConfigManager

# Load configuration
config_manager = ConfigManager()
config = config_manager.load_config("configs/standard.yaml")

# Use configuration
translator = Translator(**config.model_params)
```

## Common Use Cases

### Translating Literary Texts
```python
# Translate a passage from Homer
homer_text = "μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος"
translation = translator.translate(
    homer_text,
    domain="epic",  # Specify domain for better results
    num_beams=8      # Use larger beam for literary texts
)
```

### Academic Translation
```python
# Translate philosophical text with context
context = "From Plato's Republic:"
text = "δικαιοσύνη ἐστὶν ἀρετή"

translation = translator.translate(
    text,
    context=context,
    preserve_terms=["δικαιοσύνη", "ἀρετή"]  # Preserve key terms
)
```

### Batch Processing Files
```python
# Process a file with Greek texts
with open("greek_texts.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

translations = translator.translate_batch(lines, batch_size=32)

with open("translations.txt", "w", encoding="utf-8") as f:
    for translation in translations:
        f.write(translation + "\n")
```

## Tips for Best Results

1. **Preprocessing**: Always normalize your Greek text
2. **Model Selection**: Choose model based on your needs:
   - mBART: General purpose, good quality
   - NLLB: Faster, supports more languages
3. **Beam Search**: Use higher beam width (5-8) for better quality
4. **Batch Processing**: Process multiple texts together for efficiency
5. **Domain Adaptation**: Fine-tune on specific text types for best results

## Next Steps

- Explore [advanced examples](../examples/)
- Read the [API documentation](api_reference.md)
- Learn about [model training](training_guide.md)
- Check [troubleshooting guide](troubleshooting.md)