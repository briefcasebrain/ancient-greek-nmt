# Ancient Greek Neural Machine Translation Library

## Complete Library Structure

This professional library provides a comprehensive solution for Ancient Greek to English translation using state-of-the-art neural machine translation techniques.

## Library Overview

The library has been restructured as a professional Python package with the following components:

### Core Modules

1. **`ancient_greek_nmt/core/translator.py`**
   - Main translation interface with extensive documentation
   - Handles model loading, tokenization, and generation
   - Supports multiple translation strategies (beam search, sampling)
   - Includes detailed explanations of each translation step

2. **`ancient_greek_nmt/preprocessing/normalizer.py`**
   - Comprehensive text normalization for Ancient Greek
   - Handles diacritics, sigma variants, and Unicode issues
   - Includes step-by-step processing explanations
   - Parallel corpus normalization utilities

3. **`ancient_greek_nmt/models/architectures.py`**
   - Visual explanations of transformer architecture
   - Attention mechanism documentation
   - Model size comparisons and trade-offs
   - Transfer learning from modern to ancient Greek

4. **`ancient_greek_nmt/training/trainer.py`**
   - Complete training pipeline with detailed comments
   - Hyperparameter explanations
   - Training strategies and best practices
   - Automatic mixed precision and distributed training support

5. **`ancient_greek_nmt/evaluation/metrics.py`**
   - Comprehensive evaluation metrics (BLEU, chrF, METEOR)
   - Error analysis tools
   - Detailed metric interpretations
   - Batch evaluation utilities

6. **`ancient_greek_nmt/utils/visualizer.py`**
   - Attention heatmap visualization
   - Training curve plotting
   - Error distribution analysis
   - Model performance comparisons

7. **`ancient_greek_nmt/configs/config.py`**
   - Centralized configuration management
   - Preset configurations for common scenarios
   - Configuration validation
   - YAML/JSON support

## Interactive Tutorial Notebook

**`notebooks/ancient_greek_translation_tutorial.ipynb`**

A comprehensive Jupyter notebook that includes:
- Step-by-step explanations of the translation process
- Interactive visualizations of model behavior
- Practical examples with famous Greek texts
- Performance analysis and metrics
- Attention pattern visualizations
- Training progress monitoring

## Installation

### Standard Installation
```bash
pip install -e .
```

### With Visualization Support
```bash
pip install -e ".[visualization]"
```

### For Development
```bash
pip install -e ".[development]"
```

### Complete Installation
```bash
pip install -e ".[all]"
```

## Usage Examples

### Basic Translation
```python
from ancient_greek_nmt.core.translator import Translator

# Initialize translator
translator = Translator(model_name="facebook/mbart-large-50-many-to-many-mmt")

# Translate Ancient Greek to English
greek_text = "οἱ παῖδες ἐν τῇ οἰκίᾳ εἰσίν"
english = translator.translate(greek_text, src_lang="grc", tgt_lang="en")
print(english)  # "The children are in the house"
```

### Text Normalization
```python
from ancient_greek_nmt.preprocessing.normalizer import GreekNormalizer

# Create normalizer
normalizer = GreekNormalizer(
    keep_diacritics=True,
    lowercase=True,
    normalize_sigma=True
)

# Normalize Ancient Greek text
text = "Τὸν ἄνδρα ὁρῶ"
normalized = normalizer.normalize(text)
print(normalized)  # "τὸν ἄνδρα ὁρῶ"

# Get step-by-step explanation
explanation = normalizer.explain_normalization(text)
for step, result in explanation.items():
    print(f"{step}: {result}")
```

### Model Training
```python
from ancient_greek_nmt.training.trainer import NMTTrainer, TrainingConfig

# Configure training
config = TrainingConfig(
    model_name="facebook/mbart-large-50-many-to-many-mmt",
    direction="grc2en",
    num_epochs=10,
    batch_size=8,
    learning_rate=5e-5
)

# Initialize trainer
trainer = NMTTrainer(config)

# Train model
trainer.train()
```

### Evaluation
```python
from ancient_greek_nmt.evaluation.metrics import evaluate_translation

# Evaluate translation quality
hypothesis = "The children are in the house"
reference = "The kids are at home"

scores = evaluate_translation(
    hypothesis=hypothesis,
    reference=reference,
    metrics=["bleu", "chrf"]
)
print(scores)
# {'bleu': 25.4, 'chrf': 67.8}
```

### Visualization
```python
from ancient_greek_nmt.utils.visualizer import AttentionVisualizer

# Visualize attention patterns
visualizer = AttentionVisualizer()
visualizer.plot_attention_heatmap(
    attention_weights=attention_matrix,
    source_tokens=greek_words,
    target_tokens=english_words,
    title="Translation Attention Patterns"
)
```

## Key Features

### For Students
- **Extensive Documentation**: Every function and class includes detailed explanations
- **Visual Learning**: Comprehensive visualizations of model behavior
- **Step-by-Step Examples**: Tutorial notebook with practical examples
- **Architecture Explanations**: Clear descriptions of how transformers work

### For Researchers
- **Modular Design**: Easy to extend and modify components
- **Configuration Management**: Flexible configuration system
- **Evaluation Tools**: Comprehensive metrics and analysis
- **Visualization Utilities**: Publication-ready plots and diagrams

### For Practitioners
- **Pre-trained Models**: Support for mBART, NLLB, and custom models
- **Batch Processing**: Efficient handling of large datasets
- **Error Analysis**: Tools to identify and fix translation issues
- **Production Ready**: Optimized for deployment

## Architecture Diagrams

The library includes visual explanations of:

1. **Transformer Architecture**
   - Encoder-decoder structure
   - Self-attention mechanisms
   - Cross-attention for alignment

2. **Training Process**
   - Data flow through the model
   - Gradient computation
   - Optimization steps

3. **Attention Patterns**
   - Word alignment visualization
   - Layer-wise attention evolution
   - Head-specific patterns

## Performance Metrics

The library tracks and visualizes:
- **BLEU Score**: N-gram overlap measurement
- **chrF Score**: Character-level F-score
- **Training Loss**: Model convergence tracking
- **Inference Speed**: Translation time analysis
- **Memory Usage**: Resource consumption monitoring

## Configuration Presets

### Quick Test
```python
config = config_manager.get_preset_config('quick_test')
# Fast training for testing (3 epochs, small batch)
```

### Standard Training
```python
config = config_manager.get_preset_config('standard')
# Balanced configuration for regular use
```

### High Quality
```python
config = config_manager.get_preset_config('high_quality')
# Best quality, longer training
```

### Low Resource
```python
config = config_manager.get_preset_config('low_resource')
# For limited GPU memory
```

## Educational Value

This library serves as a complete educational resource for understanding:

1. **Natural Language Processing**
   - Tokenization and text preprocessing
   - Neural network architectures
   - Training and optimization

2. **Ancient Greek Language**
   - Linguistic challenges and solutions
   - Diacritics and character handling
   - Morphological complexity

3. **Machine Learning Best Practices**
   - Code organization and documentation
   - Configuration management
   - Performance evaluation

## Learning Path

1. Start with the tutorial notebook for hands-on experience
2. Explore the preprocessing module to understand text handling
3. Study the model architecture explanations
4. Run training experiments with different configurations
5. Analyze results using the evaluation tools
6. Create custom visualizations for deeper insights

## Additional Resources

- Original pipeline script: `grc_nmt.py` (reference implementation)
- Sample data: `sample_data/` directory
- Configuration examples: `ancient_greek_nmt/configs/`
- Tutorial notebook: `notebooks/ancient_greek_translation_tutorial.ipynb`

## Contributing

The library is designed to be easily extensible. Key areas for contribution:
- Additional evaluation metrics
- New visualization types
- Model architecture variants
- Language-specific preprocessing
- Performance optimizations

## License

This library is provided for educational purposes. Please cite appropriately when using in academic work.

---

**Note**: This library has been carefully designed with extensive documentation and comments to help students understand both the technical implementation and the underlying concepts of neural machine translation.