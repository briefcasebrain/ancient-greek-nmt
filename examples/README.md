# Ancient Greek NMT Examples

This directory contains example scripts demonstrating various features and use cases of the Ancient Greek Neural Machine Translation library.

## Example Scripts

### 1. Basic Translation (`basic_translation.py`)
- Simple translation of Greek text to English
- Text normalization
- Interactive translation mode
- Examples with famous Greek quotes

**Run:**
```bash
python basic_translation.py
```

### 2. Batch Processing (`batch_processing.py`)
- Efficient batch translation of multiple texts
- File processing capabilities
- Performance comparison of different batch sizes
- Parallel corpus evaluation

**Run:**
```bash
python batch_processing.py
```

### 3. Advanced Features (`advanced_features.py`)
- Different decoding strategies (beam search, sampling)
- Attention visualization
- Confidence scoring
- Model comparison

**Run:**
```bash
python advanced_features.py
```

### 4. Text Preprocessing (`text_preprocessing.py`)
- Various normalization options
- Unicode character analysis
- Handling mixed Greek/Latin text
- OCR error correction
- Dialectal variations

**Run:**
```bash
python text_preprocessing.py
```

### 5. Model Training (`train_model.py`)
- Data preparation for training
- Different training configurations
- Hyperparameter optimization
- Training monitoring and evaluation

**Run:**
```bash
python train_model.py
```

## Quick Start

1. Install the library:
```bash
cd ..
pip install -e .
```

2. Run any example:
```bash
cd examples
python basic_translation.py
```

## Sample Data

Some examples create sample data files:
- `sample_greek_texts.txt` - Sample Greek texts for batch processing
- `translations_output.txt` - Translation results
- `data/train.jsonl` - Training data in JSON Lines format
- `data/val.jsonl` - Validation data

## Requirements

All examples require the main library to be installed. Some advanced features may require additional dependencies:

```bash
# For visualization
pip install matplotlib seaborn

# For training
pip install datasets accelerate

# For full functionality
pip install -e "..[all]"
```

## Common Use Cases

### Translating a Single Text
```python
from ancient_greek_nmt.core.translator import Translator

translator = Translator()
translation = translator.translate("γνῶθι σεαυτόν")
print(translation)  # "Know thyself"
```

### Processing a File
```python
from ancient_greek_nmt.core.translator import Translator

translator = Translator()
with open("greek_texts.txt") as f:
    texts = f.readlines()

translations = translator.translate_batch(texts)
for greek, english in zip(texts, translations):
    print(f"{greek.strip()} → {english}")
```

### Normalizing Text
```python
from ancient_greek_nmt.preprocessing.normalizer import GreekNormalizer

normalizer = GreekNormalizer(lowercase=True)
normalized = normalizer.normalize("Τὸν Ἄνδρα ὁρῶ")
print(normalized)  # "τὸν ἄνδρα ὁρῶ"
```

## Tips

1. **Always normalize** Greek text before translation for best results
2. **Use batch processing** for multiple texts to improve efficiency
3. **Adjust beam size** based on quality vs speed requirements
4. **Monitor confidence scores** to identify uncertain translations
5. **Fine-tune models** on domain-specific data for better accuracy

## Troubleshooting

### Out of Memory
- Reduce batch size
- Use a smaller model
- Enable CPU mode if GPU memory is limited

### Slow Performance
- Use batch translation instead of individual calls
- Reduce beam search size
- Consider using NLLB models which are faster

### Poor Translation Quality
- Ensure proper text normalization
- Try different models (mBART vs NLLB)
- Consider fine-tuning on domain-specific data

## Further Resources

- [API Documentation](../docs/api_reference.md)
- [Training Guide](../docs/training_guide.md)
- [Quick Start Guide](../docs/quickstart.md)
- [Main README](../README.md)