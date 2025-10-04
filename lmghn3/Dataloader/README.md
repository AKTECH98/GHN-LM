# Model Loader for GHN Training

A comprehensive model configuration generator that creates diverse language model architectures for Graph HyperNetwork (GHN) training. This module generates thousands of different model configurations across multiple architecture types, providing a rich dataset for training GHNs to predict parameters for various language models.

## 🚀 Features

- **Multiple Architecture Types**: RNN, LSTM, GRU, GPT Encoder, and Mini GPT models
- **Comprehensive Parameter Coverage**: From tiny (3M params) to large (555M+ params) models
- **Two Dataset Sizes**: Reasonable (~17K configs) and Full (3M+ configs) options
- **Systematic Generation**: Every valid parameter combination is included
- **Memory Efficient**: Models are created on-demand during training
- **Ready for GHN Training**: Each model comes with metadata and parameter counts
- **Reproducible**: Uses seeds for consistent results across runs

## 📊 Architecture Types

### 1. RNN Models
- **Base Architecture**: Vanilla RNN with tanh activation
- **Parameters**: Hidden dimensions, number of layers, dropout, weight tying
- **Variants**: 448 (reasonable) / 9,504 (full) configurations

### 2. LSTM Models
- **Base Architecture**: Long Short-Term Memory networks
- **Parameters**: Hidden dimensions, number of layers, dropout, weight tying
- **Variants**: 448 (reasonable) / 9,504 (full) configurations

### 3. GRU Models
- **Base Architecture**: Gated Recurrent Units
- **Parameters**: Hidden dimensions, number of layers, dropout, weight tying
- **Variants**: 448 (reasonable) / 9,504 (full) configurations

### 4. GPT Encoder Models
- **Base Architecture**: Transformer encoder layers (GPT-style)
- **Parameters**: Hidden dimensions, layers, attention heads, feed-forward ratio, dropout, weight tying
- **Variants**: 7,680 (reasonable) / 1,484,912 (full) configurations

### 5. Mini GPT Models
- **Base Architecture**: GPT decoder with causal attention
- **Parameters**: Hidden dimensions, layers, attention heads, feed-forward ratio, dropout, weight tying
- **Variants**: 7,680 (reasonable) / 1,484,912 (full) configurations

## 🔧 Usage

### Basic Usage

```python
from lmghn3.Dataloader import create_reasonable_model_dataloader

# Create a reasonable dataset (~17K configurations)
dataloader, configs = create_reasonable_model_dataloader(
    vocab_size=50257,
    max_seq_len=512,
    batch_size=4,
    num_workers=0,
    device="cuda",  # or "cpu"
    seed=42
)

# Use in training loop
for models, metadatas in dataloader:
    for model, metadata in zip(models, metadatas):
        print(f"Training on {metadata['name']} ({metadata['num_params']:,} params)")
        # Your GHN training code here
```

### Advanced Usage

```python
from lmghn3.Dataloader import (
    create_reasonable_model_dataloader,
    create_full_model_dataloader,
    create_model_dataloader,
    ModelConfigGenerator
)

# Option 1: Reasonable dataset (recommended)
dataloader, configs = create_reasonable_model_dataloader(
    batch_size=4,
    seed=42
)

# Option 2: Full dataset (3M+ configurations - use with caution!)
dataloader, configs = create_full_model_dataloader(
    batch_size=2,  # Use smaller batch size for memory
    seed=42
)

# Option 3: Custom configuration
dataloader, configs = create_model_dataloader(
    batch_size=4,
    use_all_configs=False,  # True for full dataset
    seed=42
)

# Option 4: Direct generator usage
generator = ModelConfigGenerator(vocab_size=50257, max_seq_len=512, seed=42)
configs = generator.generate_reasonable_configs()  # or generate_all_configs()
```

## 📈 Dataset Statistics

### Reasonable Dataset (~17K configurations)
```
Total configurations: 16,704
├── RNN Models: 448 variants
├── LSTM Models: 448 variants  
├── GRU Models: 448 variants
├── GPT Encoder Models: 7,680 variants
└── Mini GPT Models: 7,680 variants

Parameter range: 3.3M - 555M parameters
Average parameters: ~60M per model
```

### Full Dataset (3M+ configurations)
```
Total configurations: 2,998,336
├── RNN Models: 9,504 variants
├── LSTM Models: 9,504 variants
├── GRU Models: 9,504 variants
├── GPT Encoder Models: 1,484,912 variants
└── Mini GPT Models: 1,484,912 variants

Parameter range: 3.3M - 555M+ parameters
Average parameters: ~60M per model
```

## 🎯 Parameter Ranges

### RNN/LSTM/GRU Models
- **Hidden Dimensions**: 64, 128, 256, 384, 512, 768, 1024
- **Layers**: 1, 2, 3, 4, 6, 8, 12, 16 (reasonable) / 1-32 (full)
- **Dropout**: 0.0, 0.1, 0.2, 0.3 (reasonable) / 0.0-0.5 (full)
- **Weight Tying**: True, False

### Transformer Models (GPT Encoder & Mini GPT)
- **Hidden Dimensions**: 64, 128, 256, 384, 512, 768, 1024
- **Layers**: 2, 3, 4, 6, 8, 12, 16, 24 (reasonable) / 2-64 (full)
- **Attention Heads**: 2, 4, 6, 8, 12, 16 (reasonable) / 1-32 (full)
- **Feed-Forward Ratio**: 2, 3, 4, 6, 8 (reasonable) / 1-32 (full)
- **Dropout**: 0.0, 0.1, 0.2 (reasonable) / 0.0-0.5 (full)
- **Weight Tying**: True, False

## 📝 Model Metadata

Each model comes with comprehensive metadata:

```python
metadata = {
    "name": "gpt-encoder-001-256d-4L-8h-ff4-d0.1-tTrue",
    "model_type": "gpt_encoder",
    "config": {
        "vocab_size": 50257,
        "d_model": 256,
        "n_layer": 4,
        "n_head": 8,
        "d_ff": 1024,
        "max_seq_len": 512,
        "p_drop": 0.1,
        "tie_weights": True
    },
    "num_params": 25,847,296
}
```

## 🔍 Model Naming Convention

Models are named with a descriptive format:
```
{model_type}-{id:03d}-{d_model}d-{n_layer}L-{n_head}h-ff{d_ff_ratio}-d{dropout:.1f}-t{tie_weights}
```

Examples:
- `rnn-001-256d-4L-d0.1-tTrue`
- `lstm-045-512d-8L-d0.2-tFalse`
- `gpt-encoder-123-384d-6L-12h-ff4-d0.1-tTrue`
- `mini-gpt-456-768d-12L-16h-ff6-d0.2-tFalse`

## ⚡ Performance Considerations

### Memory Usage
- **Reasonable Dataset**: ~17K models, manageable memory footprint
- **Full Dataset**: 3M+ models, requires significant memory
- **Recommendation**: Start with reasonable dataset, scale up as needed

### Batch Size Recommendations
- **Reasonable Dataset**: batch_size=4-8
- **Full Dataset**: batch_size=1-2
- **GPU Memory**: Adjust based on your hardware

### Training Time
- **Reasonable Dataset**: Hours to days depending on GHN complexity
- **Full Dataset**: Days to weeks for complete training

## 🛠️ Integration with GHN Training

```python
from lmghn3.Dataloader import create_reasonable_model_dataloader
from your_ghn_module import GHNTrainer

# Create model dataset
dataloader, configs = create_reasonable_model_dataloader(
    batch_size=4,
    seed=42
)

# Initialize GHN trainer
ghn_trainer = GHNTrainer()

# Training loop
for epoch in range(num_epochs):
    for models, metadatas in dataloader:
        for model, metadata in zip(models, metadatas):
            # Train GHN to predict parameters for this model
            loss = ghn_trainer.train_step(model, metadata)
            print(f"Epoch {epoch}, Model {metadata['name']}, Loss: {loss:.4f}")
```

## 🔧 Customization

### Adding New Architecture Types
1. Add new model class to `models.py`
2. Add configuration generation method to `ModelConfigGenerator`
3. Update `ModelDataset.__getitem__()` to handle new type
4. Add to `generate_all_configs()` method

### Modifying Parameter Ranges
Edit the parameter lists in the generation methods:
```python
# In generate_rnn_configs()
d_models = [64, 128, 256, 384, 512]  # Add/remove sizes
n_layers = [1, 2, 3, 4, 6, 8]        # Add/remove layer counts
dropouts = [0.0, 0.1, 0.2, 0.3]     # Add/remove dropout values
```

## 📚 Dependencies

- PyTorch
- torch.utils.data
- typing
- random
- sys
- os

## 🤝 Contributing

To add new features or fix issues:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This module is part of the GHN-LM project. Please refer to the main project license.

## 🆘 Support

For questions or issues:
1. Check the examples above
2. Review the parameter ranges and naming conventions
3. Test with the reasonable dataset first
4. Open an issue with detailed error messages

---

**Happy GHN Training! 🚀**
