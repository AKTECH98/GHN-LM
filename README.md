# GHN-3 for Language Models

This project contains adapted Graph HyperNetwork (GHN-3) implementations for both computer vision and language model training.

## Project Structure

```
├── ghn3/                    # Original GHN-3 for computer vision models
│   ├── ghn3/               # Core GHN-3 implementation
│   ├── train_ghn_ddp.py    # Training script for image models
│   ├── eval_ghn.py         # Evaluation script
│   └── examples/           # Example notebooks and scripts
│
├── lmghn3/                 # Adapted GHN-3 for language models
│   ├── CustomGHN3/         # Core GHN-3 implementation (adapted)
│   ├── language_models/    # Language model utilities
│   │   ├── lm_arch_loader.py      # Architecture dataset loader
│   │   ├── lm_architectures.py    # Curated LM architectures
│   │   ├── wikitext2_loader.py    # WikiText-2 dataset loader
│   │   └── tiny_lm_fixed.py       # Fixed language model (no weight tying)
│   ├── train_ghn_ddp.py    # Training script for language models
│   ├── eval_ghn.py         # Evaluation script
│   └── examples/           # Example notebooks and scripts
│
└── venv/                   # Python virtual environment
```

## Usage

### Training GHN-3 for Language Models

```bash
# Activate virtual environment
source venv/bin/activate

# Train GHN-3 for language models
python lmghn3/train_ghn_ddp.py \
    --vocab_size 50257 \
    --seq_len 256 \
    --ln \
    -e 20 \
    --opt adamw \
    --lr 4e-4 \
    --wd 1e-2 \
    -b 8 \
    --amp \
    -m 8 \
    --name ghn3lm \
    --hid 256 \
    --scheduler cosine-warmup
```

### Training GHN-3 for Computer Vision

```bash
# Train GHN-3 for image models (original)
python ghn3/train_ghn_ddp.py \
    -d imagenet \
    -D ./data \
    -n -v 50 --ln \
    -e 75 \
    --opt adamw \
    --lr 4e-4 \
    --wd 1e-2 \
    -b 128 \
    --amp \
    -m 8 \
    --name ghn3tm8 \
    --hid 64 \
    --scheduler cosine-warmup
```

## Key Features

- **Language Model Support**: Adapted GHN-3 to work with transformer-based language models
- **Fixed Weight Tying Issue**: Resolved parameter prediction conflicts with weight-tied layers
- **Curated Architectures**: Pre-defined set of language model architectures for training
- **WikiText-2 Integration**: Built-in support for WikiText-2 dataset
- **DDP Support**: Distributed training support for both implementations
- **Clean Training Output**: Suppressed warnings and verbose messages for clean training logs

## Requirements

- Python 3.11+
- PyTorch 2.8.0+
- Transformers library
- Datasets library
- Other dependencies in requirements.txt

## Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Notes

- The `lmghn3` folder contains the language model adapted version
- The `ghn3` folder contains the original computer vision version
- Language model training uses fixed architectures without weight tying to avoid parameter prediction conflicts
- Both implementations support distributed training with DDP
