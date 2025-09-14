# Installation Guide

## System Requirements

- Python 3.8 or higher
- CUDA 11.0+ (optional, for GPU acceleration)
- 8GB RAM minimum (16GB recommended)
- 2GB free disk space for models

## Basic Installation

### Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/ancient-greek-nmt.git
cd ancient-greek-nmt

# Install in development mode
pip install -e .
```

### Installing with Optional Dependencies

```bash
# For visualization support
pip install -e ".[visualization]"

# For development (includes testing and linting tools)
pip install -e ".[development]"

# Install all optional dependencies
pip install -e ".[all]"
```

## Dependencies

### Core Dependencies
- `transformers>=4.30.0`: Hugging Face transformers library
- `torch>=2.0.0`: PyTorch deep learning framework
- `numpy>=1.21.0`: Numerical computing
- `pandas>=1.3.0`: Data manipulation
- `tqdm>=4.65.0`: Progress bars
- `sentencepiece>=0.1.99`: Tokenization
- `sacrebleu>=2.3.0`: BLEU metric calculation
- `PyYAML>=6.0`: Configuration file support

### Optional Dependencies

#### Visualization
- `matplotlib>=3.5.0`: Plotting library
- `seaborn>=0.12.0`: Statistical visualization
- `plotly>=5.14.0`: Interactive visualizations

#### Development
- `pytest>=7.3.0`: Testing framework
- `black>=23.3.0`: Code formatter
- `flake8>=6.0.0`: Linting
- `mypy>=1.3.0`: Type checking
- `jupyter>=1.0.0`: Notebook support

## GPU Support

### NVIDIA GPUs
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Apple Silicon (M1/M2)
```bash
# PyTorch with Metal Performance Shaders (MPS) support
pip install torch torchvision torchaudio
```

## Verification

Verify your installation:

```python
import ancient_greek_nmt
print(ancient_greek_nmt.__version__)

# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'ancient_greek_nmt'**
   - Ensure you've installed the package with `pip install -e .`
   - Check that you're in the correct virtual environment

2. **CUDA out of memory errors**
   - Reduce batch size in configuration
   - Use gradient accumulation
   - Try mixed precision training

3. **Slow performance on CPU**
   - Consider using a smaller model
   - Reduce batch size
   - Enable multi-threading: `export OMP_NUM_THREADS=4`

4. **Tokenizer errors**
   - Ensure sentencepiece is installed: `pip install sentencepiece`
   - Check internet connection for model downloads

## Docker Installation

A Dockerfile is provided for containerized deployment:

```bash
# Build the Docker image
docker build -t ancient-greek-nmt .

# Run the container
docker run -it --gpus all ancient-greek-nmt
```

## Next Steps

- Read the [Quick Start Guide](quickstart.md)
- Explore [example scripts](../examples/)
- Review the [API Documentation](api_reference.md)