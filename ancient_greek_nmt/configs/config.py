"""
Configuration Management for Ancient Greek NMT

This module provides centralized configuration management for all components
of the translation system. It includes default settings, model configurations,
and utilities for loading/saving configurations.

Configuration Categories:
1. Model Configuration - Architecture and hyperparameters
2. Training Configuration - Learning settings and strategies
3. Data Configuration - Preprocessing and dataset settings
4. Inference Configuration - Translation generation parameters
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    """
    Model architecture and initialization configuration.

    These settings determine:
    - Which pre-trained model to use
    - Model size and capacity
    - Language-specific tokens
    - Architecture modifications
    """
    # Base model selection
    model_name: str = "facebook/mbart-large-50-many-to-many-mmt"
    model_type: str = "mbart"  # mbart, nllb, marian, custom

    # Model dimensions (for custom models)
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_encoder_layers: int = 12
    num_decoder_layers: int = 12
    intermediate_size: int = 4096
    vocab_size: int = 250054

    # Language configuration
    source_lang: str = "grc"  # Ancient Greek
    target_lang: str = "en"   # English
    src_lang_token: str = "el_GR"  # mBART token for Greek
    tgt_lang_token: str = "en_XX"  # mBART token for English
    forced_bos_token: Optional[str] = "en_XX"  # Force target language

    # Model behavior
    use_cache: bool = True
    tie_word_embeddings: bool = True
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    max_position_embeddings: int = 1024

    def get_model_kwargs(self) -> Dict[str, Any]:
        """Get keyword arguments for model initialization."""
        return {
            'dropout': self.dropout,
            'attention_dropout': self.attention_dropout,
            'use_cache': self.use_cache
        }


@dataclass
class TrainingConfig:
    """
    Training hyperparameters and optimization settings.

    These control:
    - How fast the model learns
    - When to stop training
    - Memory and computation trade-offs
    - Regularization strategies
    """
    # Basic training parameters
    num_epochs: int = 30
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    effective_batch_size: int = field(init=False)

    # Optimization
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Learning rate schedule
    lr_scheduler_type: str = "linear"  # linear, cosine, polynomial
    num_cycles: float = 0.5  # For cosine schedule

    # Regularization
    label_smoothing: float = 0.1
    dropout: float = 0.1

    # Mixed precision training
    fp16: bool = False
    bf16: bool = False
    fp16_opt_level: str = "O1"

    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "bleu"
    greater_is_better: bool = True

    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.01

    # Resource management
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True

    def __post_init__(self):
        """Calculate derived parameters."""
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps


@dataclass
class DataConfig:
    """
    Data processing and dataset configuration.

    Controls:
    - Text normalization strategies
    - Data augmentation techniques
    - Train/validation/test splits
    - Preprocessing pipelines
    """
    # Data paths
    train_file: str = "data/train.jsonl"
    validation_file: str = "data/val.jsonl"
    test_file: str = "data/test.jsonl"

    # Text preprocessing
    max_source_length: int = 512
    max_target_length: int = 512
    min_source_length: int = 1
    min_target_length: int = 1

    # Greek-specific preprocessing
    keep_diacritics: bool = True
    lowercase_greek: bool = True
    normalize_sigma: bool = True
    remove_punctuation: bool = False

    # English preprocessing
    lowercase_english: bool = False
    normalize_punctuation: bool = True

    # Data augmentation
    use_backtranslation: bool = False
    backtranslation_ratio: float = 0.5
    use_paraphrasing: bool = False
    add_noise: bool = False
    noise_probability: float = 0.1

    # Sampling and filtering
    max_samples: Optional[int] = None
    filter_by_length_ratio: bool = True
    min_length_ratio: float = 0.5
    max_length_ratio: float = 2.0

    # Curriculum learning
    use_curriculum: bool = False
    curriculum_strategy: str = "length"  # length, difficulty, random
    curriculum_steps: int = 10000


@dataclass
class InferenceConfig:
    """
    Translation generation and inference settings.

    These parameters control:
    - Translation quality vs speed trade-offs
    - Decoding strategies
    - Output formatting
    """
    # Decoding strategy
    num_beams: int = 5
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9

    # Generation constraints
    max_new_tokens: int = 256
    min_new_tokens: int = 1
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    repetition_penalty: float = 1.0

    # Advanced decoding
    use_beam_groups: bool = False
    num_beam_groups: int = 1
    diversity_penalty: float = 0.0
    constrained_decoding: bool = False

    # Output control
    return_scores: bool = False
    return_attention: bool = False
    num_return_sequences: int = 1

    # Batch processing
    batch_size: int = 32
    use_cuda: bool = True
    device_map: Optional[str] = None


class ConfigManager:
    """
    Manage configuration loading, saving, and validation.

    This class provides:
    - Configuration file I/O (JSON/YAML)
    - Default configurations for common scenarios
    - Configuration validation
    - Configuration merging and updates
    """

    def __init__(self, config_dir: str = "configs"):
        """
        Initialize configuration manager.

        Parameters:
            config_dir: Directory to store configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def create_default_configs(self) -> Dict[str, Any]:
        """
        Create default configuration set.

        Returns complete configuration dictionary with all settings.
        """
        return {
            'model': asdict(ModelConfig()),
            'training': asdict(TrainingConfig()),
            'data': asdict(DataConfig()),
            'inference': asdict(InferenceConfig())
        }

    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from file.

        Supports both JSON and YAML formats.

        Parameters:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Determine format from extension
        if config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                config = json.load(f)
        elif config_path.suffix in ['.yml', '.yaml']:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

        return config

    def save_config(
        self,
        config: Dict[str, Any],
        config_path: Union[str, Path],
        format: str = 'json'
    ):
        """
        Save configuration to file.

        Parameters:
            config: Configuration dictionary
            config_path: Path to save configuration
            format: File format ('json' or 'yaml')
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'json':
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        elif format == 'yaml':
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Configuration saved to {config_path}")

    def get_preset_config(self, preset: str) -> Dict[str, Any]:
        """
        Get pre-defined configuration for common scenarios.

        Available presets:
        - 'quick_test': Fast training for testing
        - 'standard': Balanced configuration
        - 'high_quality': Best quality, slower
        - 'low_resource': For limited GPU memory

        Parameters:
            preset: Name of configuration preset

        Returns:
            Configuration dictionary
        """
        presets = {
            'quick_test': {
                'training': {
                    'num_epochs': 3,
                    'batch_size': 4,
                    'eval_steps': 50,
                    'save_steps': 100
                },
                'model': {
                    'model_name': 'Helsinki-NLP/opus-mt-grk-en'
                }
            },
            'standard': self.create_default_configs(),
            'high_quality': {
                'model': {
                    'model_name': 'facebook/mbart-large-50-many-to-many-mmt'
                },
                'training': {
                    'num_epochs': 50,
                    'batch_size': 16,
                    'learning_rate': 3e-5,
                    'warmup_steps': 1000
                },
                'inference': {
                    'num_beams': 10,
                    'length_penalty': 1.2
                }
            },
            'low_resource': {
                'training': {
                    'batch_size': 2,
                    'gradient_accumulation_steps': 16,
                    'gradient_checkpointing': True,
                    'fp16': True
                },
                'model': {
                    'model_name': 'Helsinki-NLP/opus-mt-grk-en'
                }
            }
        }

        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")

        # Start with defaults and update with preset
        config = self.create_default_configs()
        self._deep_update(config, presets[preset])

        return config

    def _deep_update(self, base: Dict, update: Dict) -> Dict:
        """
        Recursively update nested dictionary.

        Parameters:
            base: Base dictionary
            update: Dictionary with updates

        Returns:
            Updated dictionary
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
        return base

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration for consistency and correctness.

        Checks:
        - Required fields are present
        - Values are in valid ranges
        - Incompatible options aren't combined

        Parameters:
            config: Configuration to validate

        Returns:
            True if valid, raises ValueError otherwise
        """
        # Check required sections
        required_sections = ['model', 'training', 'data', 'inference']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")

        # Validate training config
        training = config['training']
        if training['batch_size'] < 1:
            raise ValueError("Batch size must be >= 1")
        if training['learning_rate'] <= 0:
            raise ValueError("Learning rate must be > 0")
        if training['num_epochs'] < 1:
            raise ValueError("Number of epochs must be >= 1")

        # Validate data config
        data = config['data']
        if data['max_source_length'] < data['min_source_length']:
            raise ValueError("max_source_length must be >= min_source_length")

        # Check for incompatible options
        if training['fp16'] and training['bf16']:
            raise ValueError("Cannot use both fp16 and bf16")

        return True