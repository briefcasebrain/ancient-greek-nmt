#!/usr/bin/env python3
"""
Model Training Example

This script demonstrates how to train or fine-tune a translation model
on Ancient Greek data.
"""

import sys
sys.path.append('..')

import json
import os
from pathlib import Path
from typing import Dict, List
import numpy as np

from ancient_greek_nmt.training.trainer import NMTTrainer, TrainingConfig
from ancient_greek_nmt.preprocessing.normalizer import GreekNormalizer, EnglishNormalizer
from ancient_greek_nmt.evaluation.metrics import MetricCalculator


def prepare_training_data(data_dir: str = "./data"):
    """Prepare training data from raw files."""
    print("PREPARING TRAINING DATA")
    print("=" * 70)

    # Create data directory if it doesn't exist
    Path(data_dir).mkdir(exist_ok=True)

    # Sample training data
    train_data = [
        {"greek": "οἱ παῖδες ἐν τῇ οἰκίᾳ εἰσίν", "english": "The children are in the house"},
        {"greek": "γνῶθι σεαυτόν", "english": "Know thyself"},
        {"greek": "πάντα ῥεῖ καὶ οὐδὲν μένει", "english": "Everything flows and nothing remains"},
        {"greek": "ἓν οἶδα ὅτι οὐδὲν οἶδα", "english": "I know one thing, that I know nothing"},
        {"greek": "ὁ βίος βραχύς, ἡ δὲ τέχνη μακρή", "english": "Life is short, but art is long"},
        {"greek": "ἀρχὴ ἥμισυ παντός", "english": "The beginning is half of everything"},
        {"greek": "μηδὲν ἄγαν", "english": "Nothing in excess"},
        {"greek": "ἐγγύα πάρα δ' ἄτα", "english": "Give a pledge and trouble follows"},
        {"greek": "χαλεπὰ τὰ καλά", "english": "Beautiful things are difficult"},
        {"greek": "οὐδὲν κακὸν ἀμιγὲς καλοῦ", "english": "No evil without some good"}
    ]

    # Validation data
    val_data = [
        {"greek": "ὁ ἄνθρωπος φύσει πολιτικὸν ζῷον", "english": "Man is by nature a political animal"},
        {"greek": "εὖ ζῆν", "english": "To live well"},
        {"greek": "σοφὸς ὁ γινώσκων ἑαυτόν", "english": "Wise is he who knows himself"}
    ]

    # Initialize normalizers
    greek_norm = GreekNormalizer(keep_diacritics=True, lowercase=True)
    english_norm = EnglishNormalizer(lowercase=False)

    # Normalize and save training data
    train_file = f"{data_dir}/train.jsonl"
    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            normalized = {
                "greek": greek_norm.normalize(item["greek"]),
                "english": english_norm.normalize(item["english"])
            }
            f.write(json.dumps(normalized, ensure_ascii=False) + "\n")

    print(f"Created training file: {train_file}")
    print(f"  Training samples: {len(train_data)}")

    # Normalize and save validation data
    val_file = f"{data_dir}/val.jsonl"
    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_data:
            normalized = {
                "greek": greek_norm.normalize(item["greek"]),
                "english": english_norm.normalize(item["english"])
            }
            f.write(json.dumps(normalized, ensure_ascii=False) + "\n")

    print(f"Created validation file: {val_file}")
    print(f"  Validation samples: {len(val_data)}")

    return train_file, val_file


def demonstrate_training_configurations():
    """Show different training configurations."""
    print("\nTRAINING CONFIGURATIONS")
    print("=" * 70)

    configs = {
        "Quick Test": TrainingConfig(
            num_epochs=3,
            batch_size=2,
            learning_rate=5e-5,
            eval_steps=10,
            save_steps=20,
            logging_steps=5
        ),
        "Standard Training": TrainingConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=3e-5,
            warmup_steps=100,
            eval_steps=50,
            save_steps=100
        ),
        "Fine-tuning": TrainingConfig(
            num_epochs=5,
            batch_size=4,
            learning_rate=1e-5,
            warmup_steps=50,
            gradient_accumulation_steps=2
        ),
        "Low Resource": TrainingConfig(
            num_epochs=10,
            batch_size=2,
            learning_rate=2e-5,
            gradient_accumulation_steps=4,
            mixed_precision=True
        )
    }

    for name, config in configs.items():
        print(f"\n{name}:")
        print(f"  Epochs: {config.num_epochs}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Warmup steps: {config.warmup_steps}")
        if config.gradient_accumulation_steps > 1:
            effective_batch = config.batch_size * config.gradient_accumulation_steps
            print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
            print(f"  Effective batch size: {effective_batch}")


def simulate_training_loop():
    """Simulate a training loop with metrics."""
    print("\nSIMULATED TRAINING LOOP")
    print("=" * 70)

    # Simulate training for 5 epochs
    num_epochs = 5
    steps_per_epoch = 20

    # Track metrics
    train_losses = []
    val_losses = []
    val_bleu_scores = []

    print("\nTraining Progress:")
    print("-" * 50)

    for epoch in range(num_epochs):
        # Simulate training steps
        epoch_train_losses = []
        for step in range(steps_per_epoch):
            # Simulate decreasing loss
            loss = 4.0 * np.exp(-(epoch * steps_per_epoch + step) / 30) + 0.5
            loss += np.random.normal(0, 0.1)  # Add noise
            epoch_train_losses.append(loss)

            if step % 5 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Step {step}/{steps_per_epoch}, "
                      f"Loss: {loss:.4f}")

        # Calculate epoch metrics
        avg_train_loss = np.mean(epoch_train_losses)
        train_losses.append(avg_train_loss)

        # Simulate validation
        val_loss = avg_train_loss + np.random.normal(0.1, 0.05)
        val_losses.append(val_loss)

        # Simulate BLEU score (increasing over time)
        bleu = 15 * (1 - np.exp(-(epoch + 1) / 3)) + np.random.normal(0, 1)
        val_bleu_scores.append(max(0, bleu))

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Train Loss: {avg_train_loss:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Validation BLEU: {val_bleu_scores[-1]:.2f}")
        print("-" * 50)

    # Final summary
    print("\nTraining Complete!")
    print(f"  Final Train Loss: {train_losses[-1]:.4f}")
    print(f"  Final Val Loss: {val_losses[-1]:.4f}")
    print(f"  Final BLEU Score: {val_bleu_scores[-1]:.2f}")
    print(f"  Best BLEU Score: {max(val_bleu_scores):.2f}")


def demonstrate_hyperparameter_search():
    """Demonstrate hyperparameter search strategies."""
    print("\nHYPERPARAMETER SEARCH")
    print("=" * 70)

    # Define search space
    search_space = {
        'learning_rate': [1e-5, 3e-5, 5e-5],
        'batch_size': [4, 8, 16],
        'warmup_steps': [0, 100, 500],
        'num_beams': [3, 5, 8]
    }

    print("Search Space:")
    for param, values in search_space.items():
        print(f"  {param}: {values}")

    # Simulate grid search results
    print("\nSimulated Grid Search Results:")
    print("-" * 50)

    best_score = 0
    best_params = {}

    for lr in search_space['learning_rate']:
        for bs in search_space['batch_size']:
            for warmup in search_space['warmup_steps']:
                # Simulate a score based on hyperparameters
                score = 20 + np.random.normal(0, 5)
                # Favor certain combinations
                if lr == 3e-5:
                    score += 3
                if bs == 8:
                    score += 2
                if warmup == 100:
                    score += 1

                if score > best_score:
                    best_score = score
                    best_params = {
                        'learning_rate': lr,
                        'batch_size': bs,
                        'warmup_steps': warmup
                    }

    print(f"\nBest Configuration:")
    print(f"  Score (BLEU): {best_score:.2f}")
    for param, value in best_params.items():
        print(f"  {param}: {value}")


def create_training_script():
    """Create a complete training script."""
    print("\nGENERATING TRAINING SCRIPT")
    print("=" * 70)

    script_content = '''#!/usr/bin/env python3
"""
Custom training script for Ancient Greek NMT
Generated by the training example
"""

from ancient_greek_nmt.training.trainer import NMTTrainer, TrainingConfig
from ancient_greek_nmt.evaluation.metrics import MetricCalculator
from datasets import load_dataset
import wandb  # Optional: for experiment tracking

def main():
    # Initialize experiment tracking (optional)
    # wandb.init(project="ancient-greek-nmt")

    # Load datasets
    dataset = load_dataset("json", data_files={
        "train": "data/train.jsonl",
        "validation": "data/val.jsonl"
    })

    # Configure training
    config = TrainingConfig(
        model_name="facebook/mbart-large-50-many-to-many-mmt",
        direction="grc2en",
        num_epochs=10,
        batch_size=8,
        learning_rate=3e-5,
        warmup_steps=100,
        gradient_accumulation_steps=1,
        mixed_precision=True,
        eval_steps=50,
        save_steps=100,
        logging_steps=10,
        output_dir="./outputs/ancient_greek_model"
    )

    # Initialize trainer
    trainer = NMTTrainer(config)

    # Custom callbacks (optional)
    class CustomCallback:
        def on_epoch_end(self, epoch, logs):
            print(f"Completed epoch {epoch}")
            # wandb.log(logs)  # Log to Weights & Biases

    # Train model
    trainer.train(
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        callbacks=[CustomCallback()]
    )

    # Evaluate final model
    metrics = trainer.evaluate(dataset["validation"])
    print(f"Final evaluation: {metrics}")

    # Save model
    trainer.save_model("./models/ancient_greek_final")
    print("Training complete! Model saved.")

if __name__ == "__main__":
    main()
'''

    script_file = "custom_training.py"
    with open(script_file, "w") as f:
        f.write(script_content)

    print(f"Created training script: {script_file}")
    print("\nTo run the training script:")
    print(f"  python {script_file}")


def main():
    print("\n" + "=" * 70)
    print("MODEL TRAINING EXAMPLES")
    print("=" * 70)

    # Demo 1: Prepare training data
    train_file, val_file = prepare_training_data()

    # Demo 2: Show different configurations
    demonstrate_training_configurations()

    # Demo 3: Simulate training loop
    print("\n" + "=" * 70)
    simulate_training_loop()

    # Demo 4: Hyperparameter search
    print("\n" + "=" * 70)
    demonstrate_hyperparameter_search()

    # Demo 5: Generate training script
    print("\n" + "=" * 70)
    create_training_script()

    print("\n" + "=" * 70)
    print("Training demonstration complete!")
    print("\nNext steps:")
    print("1. Prepare your parallel corpus data")
    print("2. Adjust the training configuration")
    print("3. Run the generated training script")
    print("4. Monitor training with TensorBoard or Weights & Biases")
    print("5. Evaluate the trained model on test data")


if __name__ == "__main__":
    main()