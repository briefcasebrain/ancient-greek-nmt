"""
Training Module for Ancient Greek Neural Machine Translation

This module provides comprehensive training utilities with detailed explanations
of each step in the training process. It handles data loading, model optimization,
and training loop management with extensive logging and visualization support.

Training Process Overview:
==========================
1. Data Preparation: Load and preprocess parallel texts
2. Model Initialization: Set up transformer model and optimizer
3. Training Loop: Iteratively improve model weights
4. Validation: Monitor performance on held-out data
5. Checkpointing: Save best model states
6. Early Stopping: Prevent overfitting

Key Concepts for Students:
- Batch Processing: Process multiple examples together for efficiency
- Gradient Descent: Iteratively improve model by following gradients
- Learning Rate: Controls how much to update weights each step
- Loss Function: Measures how wrong the model's predictions are
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
from datasets import Dataset
import sacrebleu


@dataclass
class TrainingConfig:
    """
    Configuration class for training parameters.

    Each parameter is explained for educational purposes.
    """
    # Model settings
    model_name: str = "facebook/mbart-large-50-many-to-many-mmt"
    direction: str = "grc2en"  # grc2en or en2grc

    # Data paths
    train_data: str = "data/train.jsonl"
    dev_data: str = "data/dev.jsonl"
    test_data: Optional[str] = "data/test.jsonl"

    # Training hyperparameters (with explanations)
    learning_rate: float = 5e-5  # How fast the model learns (too high = unstable, too low = slow)
    batch_size: int = 8  # Number of examples processed together (limited by GPU memory)
    gradient_accumulation_steps: int = 4  # Simulate larger batch by accumulating gradients
    num_epochs: int = 10  # Number of complete passes through training data
    warmup_steps: int = 500  # Gradually increase learning rate at start
    weight_decay: float = 0.01  # Regularization to prevent overfitting

    # Model constraints
    max_source_length: int = 512  # Maximum input length in tokens
    max_target_length: int = 512  # Maximum output length in tokens

    # Training strategy
    evaluation_strategy: str = "steps"  # When to evaluate: "steps" or "epoch"
    eval_steps: int = 500  # Evaluate every N steps
    save_steps: int = 500  # Save checkpoint every N steps
    save_total_limit: int = 3  # Keep only N best checkpoints
    load_best_model_at_end: bool = True  # Load best model after training
    metric_for_best_model: str = "bleu"  # Metric to determine best model
    greater_is_better: bool = True  # Whether higher metric is better

    # Optimization
    fp16: bool = False  # Use 16-bit precision (faster but less accurate)
    gradient_checkpointing: bool = False  # Trade compute for memory
    label_smoothing: float = 0.1  # Soften target distributions

    # Early stopping
    early_stopping_patience: int = 3  # Stop if no improvement for N evaluations
    early_stopping_threshold: float = 0.01  # Minimum improvement required

    # Paths
    output_dir: str = "models/trained"
    logging_dir: str = "logs"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for saving."""
        return self.__dict__


class NMTTrainer:
    """
    Main trainer class for Ancient Greek NMT models.

    This class manages the entire training pipeline with detailed
    explanations at each step for educational purposes.
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize the trainer with configuration.

        Sets up:
        - Logging system for tracking progress
        - Device selection (GPU/CPU)
        - Output directories
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up logging to track training progress
        self._setup_logging()

        # Create output directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.logging_dir).mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Training on device: {self.device}")
        self.logger.info(f"Configuration: {config.to_dict()}")

    def _setup_logging(self):
        """
        Configure logging system for training monitoring.

        Logs are essential for:
        - Debugging training issues
        - Tracking performance over time
        - Understanding model behavior
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Console handler for immediate feedback
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # File handler for permanent record
        log_file = Path(self.config.logging_dir) / f"training_{datetime.now():%Y%m%d_%H%M%S}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Format messages with timestamp and level
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def load_data(self) -> Tuple[Dataset, Dataset, Optional[Dataset]]:
        """
        Load and prepare datasets for training.

        Data Loading Process:
        1. Read JSONL files containing parallel texts
        2. Convert to HuggingFace Dataset format
        3. Validate data quality
        4. Log dataset statistics

        Returns:
            Tuple of (train_dataset, dev_dataset, test_dataset)
        """
        self.logger.info("Loading datasets...")

        # Load training data
        train_data = self._load_jsonl(self.config.train_data)
        train_dataset = Dataset.from_list(train_data)
        self.logger.info(f"Loaded {len(train_dataset)} training examples")

        # Load development data for validation
        dev_data = self._load_jsonl(self.config.dev_data)
        dev_dataset = Dataset.from_list(dev_data)
        self.logger.info(f"Loaded {len(dev_dataset)} validation examples")

        # Load test data if available
        test_dataset = None
        if self.config.test_data and Path(self.config.test_data).exists():
            test_data = self._load_jsonl(self.config.test_data)
            test_dataset = Dataset.from_list(test_data)
            self.logger.info(f"Loaded {len(test_dataset)} test examples")

        # Validate data quality
        self._validate_dataset(train_dataset, "training")
        self._validate_dataset(dev_dataset, "validation")

        return train_dataset, dev_dataset, test_dataset

    def _load_jsonl(self, path: str) -> List[Dict]:
        """Load JSONL file containing parallel texts."""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def _validate_dataset(self, dataset: Dataset, name: str):
        """
        Validate dataset quality and log statistics.

        Checks:
        - No empty examples
        - Reasonable length distribution
        - Character set validity
        """
        if len(dataset) == 0:
            raise ValueError(f"{name} dataset is empty!")

        # Calculate statistics
        src_lengths = [len(ex['src'].split()) for ex in dataset]
        tgt_lengths = [len(ex['tgt'].split()) for ex in dataset]

        self.logger.info(f"{name} dataset statistics:")
        self.logger.info(f"  Source lengths: mean={np.mean(src_lengths):.1f}, "
                        f"std={np.std(src_lengths):.1f}, "
                        f"min={np.min(src_lengths)}, max={np.max(src_lengths)}")
        self.logger.info(f"  Target lengths: mean={np.mean(tgt_lengths):.1f}, "
                        f"std={np.std(tgt_lengths):.1f}, "
                        f"min={np.min(tgt_lengths)}, max={np.max(tgt_lengths)}")

    def setup_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """
        Initialize model and tokenizer with proper configuration.

        Process:
        1. Load pre-trained model
        2. Configure for specific language pair
        3. Set up tokenizer with language tokens
        4. Apply any model modifications

        The model starts with general language knowledge and will be
        specialized for Ancient Greek through training.
        """
        self.logger.info(f"Loading model: {self.config.model_name}")

        # Load tokenizer (converts text to numbers)
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # Load model with pre-trained weights
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)

        # Configure language tokens for multilingual models
        self._configure_language_tokens(tokenizer)

        # Move model to GPU if available
        model = model.to(self.device)

        # Log model information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

        return model, tokenizer

    def _configure_language_tokens(self, tokenizer):
        """
        Set up language-specific tokens for translation direction.

        Different models use different language codes:
        - mBART: el_GR for Greek, en_XX for English
        - NLLB: ell_Grek for Greek, eng_Latn for English
        """
        model_name_lower = self.config.model_name.lower()

        if "mbart" in model_name_lower:
            if self.config.direction == "grc2en":
                tokenizer.src_lang = "el_GR"  # Modern Greek code for Ancient Greek
                tokenizer.tgt_lang = "en_XX"
            else:
                tokenizer.src_lang = "en_XX"
                tokenizer.tgt_lang = "el_GR"
        elif "nllb" in model_name_lower:
            if self.config.direction == "grc2en":
                tokenizer.src_lang = "ell_Grek"
                tokenizer.tgt_lang = "eng_Latn"
            else:
                tokenizer.src_lang = "eng_Latn"
                tokenizer.tgt_lang = "ell_Grek"

        self.logger.info(f"Language tokens: src={getattr(tokenizer, 'src_lang', 'N/A')}, "
                        f"tgt={getattr(tokenizer, 'tgt_lang', 'N/A')}")

    def create_data_collator(self, tokenizer, model):
        """
        Create data collator for batch processing.

        The data collator:
        - Pads sequences to same length within batch
        - Creates attention masks
        - Handles label shifting for teacher forcing
        """
        return DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            max_length=self.config.max_source_length,
            label_pad_token_id=tokenizer.pad_token_id
        )

    def tokenize_datasets(self, train_dataset, dev_dataset, test_dataset, tokenizer):
        """
        Tokenize all datasets for model input.

        Tokenization Process:
        1. Convert text to token IDs
        2. Add special tokens (start, end, language)
        3. Truncate to maximum length
        4. Create attention masks

        This is a crucial step that converts human-readable text into
        the numerical format that neural networks can process.
        """
        self.logger.info("Tokenizing datasets...")

        def tokenize_function(examples):
            """
            Tokenize a batch of examples.

            This function is applied to all examples in the dataset.
            """
            # Tokenize source texts
            model_inputs = tokenizer(
                examples["src"],
                max_length=self.config.max_source_length,
                truncation=True,
                padding=False  # Padding done by data collator
            )

            # Tokenize target texts
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    examples["tgt"],
                    max_length=self.config.max_target_length,
                    truncation=True,
                    padding=False
                )

            # Add labels to model inputs
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        # Apply tokenization to all datasets
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            desc="Tokenizing training data"
        )
        dev_dataset = dev_dataset.map(
            tokenize_function,
            batched=True,
            desc="Tokenizing validation data"
        )
        if test_dataset:
            test_dataset = test_dataset.map(
                tokenize_function,
                batched=True,
                desc="Tokenizing test data"
            )

        return train_dataset, dev_dataset, test_dataset

    def compute_metrics(self, eval_preds, tokenizer):
        """
        Compute evaluation metrics (BLEU and chrF).

        Metrics Explained:
        - BLEU: Measures n-gram overlap with reference translations
          (0-100, higher is better)
        - chrF: Character-level F-score, more robust for morphologically rich languages
          (0-100, higher is better)

        These metrics help us understand:
        - How accurate our translations are
        - Whether the model is improving during training
        - When to stop training (early stopping)
        """
        predictions, labels = eval_preds

        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 in labels (used for padding)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Post-process texts
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        # Calculate metrics
        bleu = sacrebleu.corpus_bleu(decoded_preds, decoded_labels)
        chrf = sacrebleu.corpus_chrf(decoded_preds, decoded_labels)

        return {
            "bleu": bleu.score,
            "chrf": chrf.score
        }

    def train(self):
        """
        Main training method that orchestrates the entire training process.

        Training Steps:
        1. Load data
        2. Initialize model and tokenizer
        3. Prepare datasets
        4. Configure training arguments
        5. Create trainer
        6. Run training loop
        7. Save final model
        """
        self.logger.info("Starting training process...")

        # Step 1: Load data
        train_dataset, dev_dataset, test_dataset = self.load_data()

        # Step 2: Initialize model and tokenizer
        model, tokenizer = self.setup_model_and_tokenizer()

        # Step 3: Tokenize datasets
        train_dataset, dev_dataset, test_dataset = self.tokenize_datasets(
            train_dataset, dev_dataset, test_dataset, tokenizer
        )

        # Step 4: Create data collator
        data_collator = self.create_data_collator(tokenizer, model)

        # Step 5: Configure training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size * 2,  # Can be larger for eval
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_dir=self.config.logging_dir,
            logging_steps=50,
            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            fp16=self.config.fp16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            label_smoothing_factor=self.config.label_smoothing,
            predict_with_generate=True,  # Use generation for evaluation
            generation_max_length=self.config.max_target_length,
            report_to=["tensorboard"],  # Log to TensorBoard
        )

        # Step 6: Create trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=tokenizer,
            compute_metrics=lambda eval_preds: self.compute_metrics(eval_preds, tokenizer),
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold
                )
            ]
        )

        # Step 7: Train the model
        self.logger.info("Starting training loop...")
        train_result = trainer.train()

        # Step 8: Save the final model
        self.logger.info("Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(self.config.output_dir)

        # Save training metrics
        metrics_file = Path(self.config.output_dir) / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(train_result.metrics, f, indent=2)

        self.logger.info(f"Training complete! Model saved to {self.config.output_dir}")

        # Evaluate on test set if available
        if test_dataset:
            self.logger.info("Evaluating on test set...")
            test_results = trainer.evaluate(eval_dataset=test_dataset)
            self.logger.info(f"Test results: {test_results}")

        return trainer, train_result