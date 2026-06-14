# GHN-3 for Language Models

Graph HyperNetwork (GHN-3) training and evaluation for transformer language models on WikiText-2.

This repository trains a hypernetwork to predict weights for language model architectures, then benchmarks those predictions against standard initialization methods (default, He, Xavier) across ten model scales.

## Quick start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Train a hypernetwork
python scripts/train_ghn.py --model_name ghn3lm --epochs 20

# Train a benchmark LM with GHN init
python scripts/train_lm.py \
  --config configs/benchmarks/benchmark_1_tiny.yaml \
  --init_method ghn \
  --ghn_checkpoint GHN_Models/20917896.pt

# Evaluate and plot results
python scripts/evaluate_metrics.py --all-configs --plot
python scripts/evaluate_metrics.py --plot-all
```

## Project structure

```
Capstone/
├── pyproject.toml              # editable install (pip install -e .)
├── requirements.txt
├── configs/benchmarks/         # 10 benchmark YAML configs
├── scripts/                    # CLI entry points
│   ├── train_ghn.py
│   ├── train_lm.py
│   └── evaluate_metrics.py
├── slurm/                      # cluster job wrappers
├── docs/                       # project documentation
├── src/capstone/               # Python package
│   ├── paths.py                # repo-root path constants
│   ├── ghn/                    # hypernetwork (Graphormer + decoders)
│   ├── lm/                     # GPT encoder & Mini GPT models
│   ├── data/                   # WikiText-2 + architecture loader
│   └── eval/                   # metrics extraction, plotting, migration
├── Results/                    # evaluation output (gitignored)
├── Experiment/                 # training checkpoints (gitignored)
├── tensor_log/                 # TensorBoard logs (gitignored)
└── data/                       # HuggingFace dataset cache (gitignored)
```

## Workflow

```
train_ghn.py  →  GHN checkpoint in Experiment/ or GHN_Models/
       ↓
train_lm.py   →  Experiment/{job_id}/  (per init method)
       ↓
evaluate_metrics.py  →  Results/metrics/{config}.json
       ↓
--plot-all      →  Results/plots/*.png
```

## Documentation

| Document | Description |
|----------|-------------|
| [docs/training.md](docs/training.md) | GHN and LM training, init methods, SLURM |
| [docs/evaluation.md](docs/evaluation.md) | Metrics schema, plotting, legacy migration |
| [docs/configuration.md](docs/configuration.md) | Benchmark YAML format and model list |
| [docs/README.md](docs/README.md) | Full documentation index |

Capstone research reports in `Report/` (gitignored) and `docs/report.pdf` may exist locally as deliverables.

## SLURM (RIT cluster)

```bash
sbatch slurm/train_ghn.sh
INIT_METHOD=he sbatch slurm/train_lm.sh
sbatch slurm/train_lm_ghn_init.sh
sbatch slurm/evaluate_metrics.sh
./slurm/run_first_10_configs_evaluation.sh
```

SLURM scripts change to the repo root, activate the venv, and run `pip install -e .` before calling `scripts/`.

## Requirements

- Python 3.11+
- PyTorch 2.8+
- See [requirements.txt](requirements.txt) for pinned dependencies
