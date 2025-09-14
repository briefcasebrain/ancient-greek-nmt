# Training Guide

## Overview

This guide covers training custom Ancient Greek translation models, fine-tuning existing models, and optimizing performance.

## Data Preparation

### Dataset Format

Prepare parallel text data in the following format:

```python
# JSON Lines format (.jsonl)
{"greek": "οἱ παῖδες ἐν τῇ οἰκίᾳ εἰσίν", "english": "The children are in the house"}
{"greek": "γνῶθι σεαυτόν", "english": "Know thyself"}
```

### Data Sources

Recommended sources for Ancient Greek parallel texts:
- Perseus Digital Library
- First1KGreek Project
- Open Greek and Latin Project
- Loeb Classical Library (with licensing)

### Data Preprocessing

```python
from ancient_greek_nmt.preprocessing.normalizer import GreekNormalizer, EnglishNormalizer
import json

# Initialize normalizers
greek_norm = GreekNormalizer(keep_diacritics=True)
english_norm = EnglishNormalizer()

# Process data file
processed_data = []
with open("raw_data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        processed_data.append({
            "greek": greek_norm.normalize(data["greek"]),
            "english": english_norm.normalize(data["english"])
        })

# Save processed data
with open("processed_data.jsonl", "w", encoding="utf-8") as f:
    for item in processed_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
```

## Basic Training

### Simple Training Script

```python
from ancient_greek_nmt.training.trainer import NMTTrainer, TrainingConfig
from datasets import load_dataset

# Load your dataset
dataset = load_dataset("json", data_files={
    "train": "train.jsonl",
    "validation": "val.jsonl"
})

# Configure training
config = TrainingConfig(
    model_name="facebook/mbart-large-50-many-to-many-mmt",
    direction="grc2en",
    num_epochs=10,
    batch_size=8,
    learning_rate=5e-5,
    output_dir="./outputs"
)

# Initialize trainer
trainer = NMTTrainer(config)

# Train model
trainer.train(
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"]
)
```

## Fine-tuning Strategies

### Domain-Specific Fine-tuning

Fine-tune for specific text types:

```python
# Configure for epic poetry
epic_config = TrainingConfig(
    model_name="facebook/mbart-large-50-many-to-many-mmt",
    direction="grc2en",
    num_epochs=5,  # Fewer epochs for fine-tuning
    batch_size=4,  # Smaller batch for longer texts
    learning_rate=2e-5,  # Lower learning rate
    max_length=256,  # Longer sequences for poetry
    output_dir="./models/epic"
)

# Configure for philosophical texts
philosophy_config = TrainingConfig(
    model_name="facebook/mbart-large-50-many-to-many-mmt",
    direction="grc2en",
    num_epochs=5,
    batch_size=8,
    learning_rate=2e-5,
    preserve_terminology=True,  # Keep philosophical terms
    output_dir="./models/philosophy"
)
```

### Progressive Training

Train in stages for better results:

```python
# Stage 1: General training
stage1_config = TrainingConfig(
    num_epochs=5,
    batch_size=16,
    learning_rate=5e-5
)

# Stage 2: Fine-tuning with smaller learning rate
stage2_config = TrainingConfig(
    num_epochs=3,
    batch_size=8,
    learning_rate=1e-5
)

# Stage 3: Final refinement
stage3_config = TrainingConfig(
    num_epochs=2,
    batch_size=4,
    learning_rate=5e-6
)
```

## Advanced Training Techniques

### Mixed Precision Training

```python
config = TrainingConfig(
    mixed_precision=True,  # Enable automatic mixed precision
    fp16=True,            # Use FP16 computation
    batch_size=16         # Can use larger batches with FP16
)
```

### Gradient Accumulation

For limited GPU memory:

```python
config = TrainingConfig(
    batch_size=2,  # Small batch per step
    gradient_accumulation_steps=8,  # Accumulate over 8 steps
    # Effective batch size = 2 * 8 = 16
)
```

### Learning Rate Scheduling

```python
config = TrainingConfig(
    learning_rate=5e-5,
    warmup_steps=500,  # Linear warmup
    lr_scheduler_type="cosine",  # Cosine decay
    num_train_epochs=10
)
```

### Data Augmentation

```python
from ancient_greek_nmt.training.augmentation import DataAugmenter

augmenter = DataAugmenter()

# Back-translation augmentation
augmented_data = augmenter.back_translate(
    dataset,
    intermediate_lang="la"  # Use Latin as intermediate
)

# Paraphrase augmentation
augmented_data = augmenter.paraphrase(
    dataset,
    num_paraphrases=2
)
```

## Distributed Training

### Multi-GPU Training

```python
# Using DataParallel
config = TrainingConfig(
    device="cuda",
    data_parallel=True,
    device_ids=[0, 1, 2, 3]  # Use 4 GPUs
)

# Using DistributedDataParallel (recommended)
config = TrainingConfig(
    distributed=True,
    local_rank=0,
    world_size=4
)
```

### Training on Multiple Nodes

```bash
# Node 1
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.1" \
    --master_port=1234 \
    train.py

# Node 2
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.1" \
    --master_port=1234 \
    train.py
```

## Hyperparameter Optimization

### Grid Search

```python
from itertools import product

# Define parameter grid
param_grid = {
    'learning_rate': [1e-5, 3e-5, 5e-5],
    'batch_size': [4, 8, 16],
    'num_beams': [3, 5, 8]
}

# Generate all combinations
for params in product(*param_grid.values()):
    config = TrainingConfig(
        learning_rate=params[0],
        batch_size=params[1],
        num_beams=params[2]
    )
    # Train and evaluate
```

### Bayesian Optimization

```python
from skopt import gp_minimize

def objective(params):
    lr, batch_size, warmup = params
    config = TrainingConfig(
        learning_rate=lr,
        batch_size=int(batch_size),
        warmup_steps=int(warmup)
    )
    # Train and return negative BLEU score
    return -trainer.evaluate()['bleu']

# Optimize
result = gp_minimize(
    objective,
    [(1e-6, 1e-4), (4, 32), (100, 1000)],
    n_calls=20
)
```

## Monitoring Training

### TensorBoard Integration

```python
config = TrainingConfig(
    logging_dir="./logs",
    logging_steps=10,
    tensorboard=True
)

# View in TensorBoard
# tensorboard --logdir=./logs
```

### Weights & Biases Integration

```python
import wandb

wandb.init(project="ancient-greek-nmt")

config = TrainingConfig(
    report_to="wandb",
    run_name="mbart-finetuning"
)
```

### Custom Callbacks

```python
from ancient_greek_nmt.training.callbacks import TrainingCallback

class CustomCallback(TrainingCallback):
    def on_epoch_end(self, epoch, logs):
        print(f"Epoch {epoch}: BLEU = {logs['bleu']:.2f}")

    def on_batch_end(self, batch, logs):
        if batch % 100 == 0:
            print(f"Batch {batch}: Loss = {logs['loss']:.4f}")

trainer = NMTTrainer(config, callbacks=[CustomCallback()])
```

## Model Export and Deployment

### Saving Models

```python
# Save after training
trainer.save_model("./final_model")

# Save best checkpoint
trainer.save_best_model("./best_model")
```

### Export to ONNX

```python
from ancient_greek_nmt.utils.export import export_to_onnx

export_to_onnx(
    model_path="./final_model",
    output_path="./model.onnx",
    optimize=True
)
```

### Quantization

```python
from ancient_greek_nmt.utils.quantization import quantize_model

# Dynamic quantization
quantized_model = quantize_model(
    model_path="./final_model",
    quantization_type="dynamic"
)

# Static quantization (requires calibration data)
quantized_model = quantize_model(
    model_path="./final_model",
    quantization_type="static",
    calibration_data=dataset["validation"]
)
```

## Best Practices

### Data Quality
1. Clean and normalize all text data
2. Remove duplicate entries
3. Validate parallel alignment
4. Balance dataset by text length

### Training Strategy
1. Start with pre-trained models
2. Use progressive learning rates
3. Monitor validation metrics closely
4. Save checkpoints frequently
5. Use early stopping to prevent overfitting

### Performance Optimization
1. Use mixed precision training
2. Enable gradient checkpointing for large models
3. Optimize batch size for your GPU
4. Use data loaders with multiple workers

### Evaluation
1. Use multiple metrics (BLEU, chrF, METEOR)
2. Perform human evaluation on samples
3. Test on diverse text types
4. Check for common error patterns

## Troubleshooting

### Out of Memory Errors
- Reduce batch size
- Enable gradient accumulation
- Use gradient checkpointing
- Switch to mixed precision

### Slow Training
- Check data loading bottlenecks
- Increase number of workers
- Use faster storage (SSD)
- Enable GPU acceleration

### Poor Performance
- Check data quality
- Adjust learning rate
- Increase training epochs
- Try different model architectures
- Add more training data

### Overfitting
- Add dropout layers
- Use weight decay
- Implement early stopping
- Augment training data
- Reduce model size