# Training

## Prerequisites

```bash
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

All paths below assume you run commands from the repository root.

## Stage 1: Train the hypernetwork

The GHN learns to predict weights for many LM architecture variants sampled from `capstone.data.lm_arch_loader`.

```bash
python scripts/train_ghn.py \
  --model_name ghn3lm \
  --epochs 20 \
  --batch_size 8 \
  --seq_len 256
```

Training uses WikiText-2 batches paired with randomly sampled architectures (GPT encoder and Mini GPT variants). Outputs:

| Output | Location |
|--------|----------|
| Checkpoints | `Experiment/{model_name}/` |
| TensorBoard | `tensor_log/{model_name}/` |
| Run config | `Experiment/{model_name}/config.json` |

Common flags (via ppuda `init_config`):

- `--model_name` — job ID and experiment folder name
- `--epochs`, `--batch_size`, `--lr`, `--wd` — optimization
- `--max_d_model`, `--max_layers` — limit architecture size for GPU memory
- `--include_embeddings` — include embedding layers in GHN prediction (off by default)

Multi-GPU: use `slurm/train_ghn_multigpu.sh` or pass `--multigpu` where supported.

## Stage 2: Train benchmark language models

Each benchmark config in `configs/benchmarks/` defines a fixed architecture. Train with one of four initialization methods:

| Init method | Description |
|-------------|-------------|
| `default` | PyTorch default initialization |
| `he` | He (Kaiming) init, scaled for GELU transformers |
| `xavier` | Xavier (Glorot) uniform init |
| `ghn` | Weights predicted by the trained hypernetwork |

```bash
# Default init
python scripts/train_lm.py \
  --config configs/benchmarks/benchmark_1_tiny.yaml \
  --init_method default

# GHN init (requires checkpoint)
python scripts/train_lm.py \
  --config configs/benchmarks/benchmark_1_tiny.yaml \
  --init_method ghn \
  --ghn_checkpoint GHN_Models/20917896.pt
```

List available configs:

```bash
python scripts/train_lm.py --list_configs
```

Outputs per run:

| Output | Location |
|--------|----------|
| Best model | `Experiment/{job_id}/best_model.pt` |
| Epoch checkpoints | `Experiment/{job_id}/epoch_*.pt` |
| Run metadata | `Experiment/{job_id}/config.json` (includes `init_method`) |
| TensorBoard | `tensor_log/{job_id}/` |

The `JOB_ID` environment variable (set by SLURM scripts) determines the experiment folder name.

## SLURM

Scripts in `slurm/` wrap the commands above for RIT's cluster:

```bash
sbatch slurm/train_ghn.sh
INIT_METHOD=default sbatch slurm/train_lm.sh
INIT_METHOD=he sbatch slurm/train_lm.sh
INIT_METHOD=ghn sbatch slurm/train_lm_ghn_init.sh
```

Batch runners for all ten configs:

```bash
./slurm/run_all_configs_all_inits.sh
./slurm/run_first_10_configs_ghn_init.sh
```

Edit the `CONFIG_FILE` or `CONFIG` variable at the top of each script to change which benchmark is trained.

## TensorBoard

```bash
tensorboard --logdir tensor_log
```

Or use `slurm/start_tensorboard.sh` if configured for your cluster.
