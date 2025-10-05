# GHN Training for Language Models

This directory contains the training infrastructure for training Graph HyperNetworks (GHN) on language model architectures.

## Files

- `train_lm_ghn.py` - Main training script for GHN on language models
- `test_lm_ghn_training.py` - Test script to verify training setup
- `CustomGHN3/` - Custom components for GHN training
  - `graph_adapter.py` - Adapter to convert language models to graph format
  - `simple_trainer.py` - Simplified trainer (not used, kept for reference)

## Usage

### Basic Training

```bash
# Train GHN on language models with reasonable dataset
python lmghn3/train_lm_ghn.py --name ghn3-lm --epochs 50 --batch_size 4 --meta_batch_size 8 \
    --lr 1e-4 --wd 1e-2 --hid 64 --layers 3 --heads 8 --amp

# Train with full dataset (3M+ configurations)
python lmghn3/train_lm_ghn.py --name ghn3-lm-full --use_all_configs --epochs 100 \
    --batch_size 2 --meta_batch_size 4 --lr 5e-5 --wd 1e-2 --hid 128 --layers 4
```

### Test Training Setup

```bash
# Test that all components work correctly
python test_lm_ghn_training.py
```

## Key Features

- **Uses existing GHN-3**: Leverages the proven GHN-3 implementation from the ghn3 folder
- **No DDP**: Simplified single-GPU training without distributed training complexity
- **Language Model Support**: Works with RNN, LSTM, GRU, GPT Encoder, and Mini GPT models
- **Flexible Dataset**: Choose between reasonable (~17K) or full (3M+) model configurations
- **WikiText-2 Integration**: Uses WikiText-2 dataset for language modeling tasks

## Architecture

The training script:

1. **Loads Model Configurations**: Uses the model loader to get diverse language model architectures
2. **Loads Language Data**: Uses WikiText-2 dataset for training
3. **Creates GHN**: Uses the existing GHN-3 implementation
4. **Graph Adaptation**: Converts language models to graph format compatible with GHN
5. **Training Loop**: Trains GHN to predict parameters for language models

## Dependencies

- Uses existing GHN-3 from `lmghn3/CustomGHN3/` folder (NOT from ghn3/ folder)
- Uses model loader from `lmghn3/Dataloader/`
- Uses language models from `lmghn3/models/`
- Note: The `ghn3/` folder is only for reference, all training uses components from `lmghn3/`

## Configuration

Key parameters:
- `--use_all_configs`: Use full dataset (3M+ configs) vs reasonable (~17K)
- `--meta_batch_size`: Number of models per meta-batch
- `--batch_size`: Batch size for language data
- `--hid`, `--layers`, `--heads`: GHN architecture parameters

## Output

- Checkpoints saved to `logs/ghn_training/`
- TensorBoard logs for monitoring training
- Model configurations and training metadata
