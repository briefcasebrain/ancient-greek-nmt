# API Reference

## Core Module

### ancient_greek_nmt.core.translator

#### Class: `Translator`

Main translation interface for Ancient Greek to English translation.

```python
class Translator(model_name: str = "facebook/mbart-large-50-many-to-many-mmt",
                device: str = "auto",
                cache_dir: Optional[str] = None)
```

**Parameters:**
- `model_name` (str): Hugging Face model identifier
- `device` (str): Device to use ("auto", "cuda", "cpu", "mps")
- `cache_dir` (Optional[str]): Directory for model cache

**Methods:**

##### `translate()`
```python
def translate(text: str,
             src_lang: str = "grc",
             tgt_lang: str = "en",
             num_beams: int = 5,
             max_length: int = 128,
             temperature: float = 1.0,
             top_p: float = 1.0,
             do_sample: bool = False,
             return_scores: bool = False,
             return_attention: bool = False) -> Union[str, Dict]
```

Translate Ancient Greek text to English.

**Returns:** Translation string or dictionary with additional information

##### `translate_batch()`
```python
def translate_batch(texts: List[str],
                   batch_size: int = 8,
                   **kwargs) -> List[str]
```

Translate multiple texts efficiently.

## Preprocessing Module

### ancient_greek_nmt.preprocessing.normalizer

#### Class: `GreekNormalizer`

Text normalization for Ancient Greek.

```python
class GreekNormalizer(keep_diacritics: bool = True,
                     lowercase: bool = True,
                     normalize_sigma: bool = True,
                     remove_punctuation: bool = False)
```

**Methods:**

##### `normalize()`
```python
def normalize(text: str) -> str
```

Normalize Ancient Greek text according to configured rules.

##### `explain_normalization()`
```python
def explain_normalization(text: str) -> Dict[str, str]
```

Get step-by-step explanation of normalization process.

#### Class: `EnglishNormalizer`

Text normalization for English translations.

```python
class EnglishNormalizer(lowercase: bool = False,
                       remove_punctuation: bool = False,
                       expand_contractions: bool = True)
```

## Models Module

### ancient_greek_nmt.models.architectures

#### Class: `TransformerExplainer`

Educational tool for understanding transformer architecture.

```python
class TransformerExplainer()
```

**Methods:**

##### `visualize_encoder_decoder_flow()`
```python
def visualize_encoder_decoder_flow() -> str
```

Generate ASCII diagram of encoder-decoder architecture.

##### `explain_attention_mechanism()`
```python
def explain_attention_mechanism() -> str
```

Detailed explanation of attention mechanism.

#### Class: `ModelArchitectureVisualizer`

Visualization tools for model architecture.

```python
class ModelArchitectureVisualizer()
```

**Methods:**

##### `create_attention_heatmap_example()`
```python
def create_attention_heatmap_example() -> Dict
```

Generate example attention weights for visualization.

## Training Module

### ancient_greek_nmt.training.trainer

#### Class: `TrainingConfig`

Configuration for model training.

```python
@dataclass
class TrainingConfig:
    model_name: str = "facebook/mbart-large-50-many-to-many-mmt"
    direction: str = "grc2en"
    num_epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    output_dir: str = "./outputs"
```

#### Class: `NMTTrainer`

Training pipeline for neural machine translation.

```python
class NMTTrainer(config: TrainingConfig)
```

**Methods:**

##### `train()`
```python
def train(train_dataset: Dataset,
         eval_dataset: Optional[Dataset] = None) -> None
```

Train the translation model.

##### `evaluate()`
```python
def evaluate(dataset: Dataset) -> Dict[str, float]
```

Evaluate model on a dataset.

## Evaluation Module

### ancient_greek_nmt.evaluation.metrics

#### Class: `MetricCalculator`

Calculate translation quality metrics.

```python
class MetricCalculator()
```

**Methods:**

##### `evaluate_translation()`
```python
def evaluate_translation(hypothesis: str,
                        reference: str,
                        detailed: bool = False) -> MetricResult
```

Calculate BLEU, chrF, and other metrics.

##### `evaluate_batch()`
```python
def evaluate_batch(hypotheses: List[str],
                  references: List[str]) -> Dict[str, float]
```

Evaluate multiple translations.

#### Class: `MetricResult`

Container for evaluation metrics.

```python
@dataclass
class MetricResult:
    bleu: float
    chrf: float
    meteor: Optional[float] = None
    ter: Optional[float] = None
```

## Utils Module

### ancient_greek_nmt.utils.visualizer

#### Class: `AttentionVisualizer`

Visualization tools for attention patterns.

```python
class AttentionVisualizer()
```

**Methods:**

##### `plot_attention_heatmap()`
```python
def plot_attention_heatmap(attention_weights: np.ndarray,
                          source_tokens: List[str],
                          target_tokens: List[str],
                          title: str = "Attention Weights",
                          save_path: Optional[str] = None) -> None
```

Create heatmap visualization of attention weights.

##### `plot_training_curves()`
```python
def plot_training_curves(history: Dict[str, List[float]],
                        save_path: Optional[str] = None) -> None
```

Plot training and validation curves.

## Configuration Module

### ancient_greek_nmt.configs.config

#### Class: `ConfigManager`

Manage configuration files and presets.

```python
class ConfigManager()
```

**Methods:**

##### `load_config()`
```python
def load_config(path: str) -> Config
```

Load configuration from YAML or JSON file.

##### `save_config()`
```python
def save_config(config: Config, path: str) -> None
```

Save configuration to file.

##### `get_preset_config()`
```python
def get_preset_config(preset: str) -> Config
```

Get predefined configuration preset.

**Available Presets:**
- `"quick_test"`: Fast testing configuration
- `"standard"`: Balanced configuration
- `"high_quality"`: Best quality, slower
- `"low_resource"`: For limited GPU memory

## Type Definitions

### Common Types

```python
from typing import Union, List, Dict, Optional, Tuple
import numpy as np
import torch

# Type aliases
TokenList = List[str]
AttentionMatrix = np.ndarray
TranslationOutput = Union[str, Dict[str, Any]]
BatchOutput = List[TranslationOutput]
```

## Exceptions

### Custom Exceptions

```python
class TranslationError(Exception):
    """Base exception for translation errors"""
    pass

class ModelLoadError(TranslationError):
    """Error loading the translation model"""
    pass

class NormalizationError(Exception):
    """Error during text normalization"""
    pass

class ConfigurationError(Exception):
    """Error in configuration"""
    pass
```

## Constants

### Language Codes

```python
# mBART language codes
MBART_LANGS = {
    "grc": "grc",  # Ancient Greek
    "en": "en_XX"  # English
}

# NLLB language codes
NLLB_LANGS = {
    "grc": "grc_Grek",  # Ancient Greek
    "en": "eng_Latn"    # English
}
```

### Default Parameters

```python
DEFAULT_BEAM_SIZE = 5
DEFAULT_MAX_LENGTH = 128
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 5e-5
```