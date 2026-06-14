# Benchmark configurations

All benchmark configs live in [`configs/benchmarks/`](../configs/benchmarks/). Each YAML file defines model architecture, training hyperparameters, and data settings.

## File list

| Config | Architecture | Approx. scale |
|--------|-------------|---------------|
| `benchmark_1_tiny.yaml` | gpt_encoder | ~0.1M params |
| `benchmark_2_small.yaml` | gpt_encoder | ~1M params |
| `benchmark_3_medium.yaml` | gpt_encoder | ~5M params |
| `benchmark_4_large.yaml` | gpt_encoder | ~15M params |
| `benchmark_5_mini_gpt.yaml` | mini_gpt | ~25M params |
| `benchmark_6_mini_gpt_tiny.yaml` | mini_gpt | small |
| `benchmark_7_mini_gpt_small.yaml` | mini_gpt | small-medium |
| `benchmark_8_mini_gpt_medium.yaml` | mini_gpt | medium |
| `benchmark_9_mini_gpt_large.yaml` | mini_gpt | large |
| `benchmark_10_mini_gpt_xl.yaml` | mini_gpt | XL |

Configs 1–4 use `gpt_encoder` (TransformerEncoder with causal mask). Configs 5–10 use `mini_gpt` (explicit decoder blocks).

## YAML schema

Each file has three sections:

### `model`

| Field | Type | Description |
|-------|------|-------------|
| `model_type` | string | `gpt_encoder` or `mini_gpt` |
| `vocab_size` | int | Token vocabulary size (50257 for GPT-2 tokenizer) |
| `d_model` | int | Hidden dimension |
| `n_layer` | int | Number of transformer layers |
| `n_head` | int | Attention heads (must divide `d_model`) |
| `d_ff` | int | Feed-forward hidden size |
| `max_seq_len` | int | Maximum sequence length |
| `p_drop` | float | Dropout probability |

### `training`

| Field | Type | Description |
|-------|------|-------------|
| `epochs` | int | Maximum training epochs |
| `batch_size` | int | Batch size |
| `learning_rate` | float | AdamW learning rate |
| `weight_decay` | float | L2 regularization |
| `warmup_steps` | int | LR warmup steps |
| `max_grad_norm` | float | Gradient clipping threshold |
| `save_interval` | int | Epochs between checkpoint saves |
| `eval_interval` | int | Epochs between validation runs |
| `log_interval` | int | Steps between log prints |
| `device` | string | `"cuda"` or `"cpu"` |
| `mixed_precision` | bool | Use AMP |
| `gradient_accumulation_steps` | int | Accumulation before optimizer step |
| `seed` | int | Random seed (optional) |
| `early_stopping_patience` | int | Epochs without improvement before stop |
| `early_stopping_min_delta` | float | Minimum val-loss improvement |

### `data`

| Field | Type | Description |
|-------|------|-------------|
| `seq_len` | int | WikiText-2 sequence length |
| `num_workers` | int | DataLoader worker processes |

## Example

```yaml
model:
  model_type: "gpt_encoder"
  vocab_size: 50257
  d_model: 64
  n_layer: 2
  n_head: 2
  d_ff: 256
  max_seq_len: 64
  p_drop: 0.1

training:
  epochs: 10
  batch_size: 64
  learning_rate: 0.001
  weight_decay: 0.01
  warmup_steps: 50
  max_grad_norm: 1.0
  save_interval: 2
  eval_interval: 1
  log_interval: 10
  device: "cuda"
  mixed_precision: true
  gradient_accumulation_steps: 1
  seed: 42
  early_stopping_patience: 3
  early_stopping_min_delta: 0.001

data:
  seq_len: 64
  num_workers: 4
```

## Loading configs in Python

```python
from capstone.data.config_loader import load_config_file, list_benchmark_configs
from capstone.paths import CONFIGS_DIR

# List available configs
print(list_benchmark_configs())

# Load a specific config
model_cfg, train_cfg, data_cfg = load_config_file(
    str(CONFIGS_DIR / "benchmark_1_tiny.yaml")
)
```

## Creating a model from config

```python
from capstone.lm.create_model import create_model

model = create_model(model_cfg, vocab_size=50257)
```

Or from the CLI:

```bash
python -m capstone.lm.create_model --config benchmark_1_tiny --list_configs
```
