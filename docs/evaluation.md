# Evaluation

## Overview

`scripts/evaluate_metrics.py` extracts three metrics from completed training runs:

1. **Perplexity at intervals** — validation perplexity at epochs 1, 2, 5, 10, 20, 50 (from TensorBoard)
2. **Convergence** — epoch when validation loss stopped improving
3. **Test score** — perplexity of the best checkpoint on the WikiText-2 test set

Results are written to `Results/metrics/{config_name}.json` using a canonical schema keyed by init method.

## Commands

```bash
# Single config
python scripts/evaluate_metrics.py \
  --config configs/benchmarks/benchmark_1_tiny.yaml \
  --plot

# All ten benchmark configs
python scripts/evaluate_metrics.py --all-configs --plot

# Aggregate comparison charts from existing Results/
python scripts/evaluate_metrics.py --plot-all

# One-time migration of legacy JSON
python scripts/evaluate_metrics.py --migrate
```

### Flags

| Flag | Description |
|------|-------------|
| `--config PATH` | Evaluate one YAML config |
| `--all-configs` | Loop all files in `configs/benchmarks/` |
| `--intervals` | Comma-separated epochs for perplexity (default: `1,2,5,10,20,50`) |
| `--plot` | Save per-config curve PNG to `Results/plots/` |
| `--plot-all` | Generate aggregate bar/scatter charts |
| `--migrate` | Merge legacy JSON into `Results/metrics/` |
| `--output-dir` | Override results root (default: `Results/`) |
| `--device` | Evaluation device (default: `cuda`) |

## Results layout

```
Results/
├── metrics/
│   ├── benchmark_1_tiny.json
│   ├── benchmark_2_small.json
│   └── ...                          # one file per config
├── plots/
│   ├── benchmark_1_tiny_curves.png  # per-config (--plot)
│   ├── test_performance_comparison.png
│   ├── training_improvement_comparison.png
│   └── convergence_epoch_comparison.png
└── summary/
    └── all_configs.json             # combined index
```

## Metrics JSON schema

Each file in `Results/metrics/` follows this structure:

```json
{
  "config_file": "configs/benchmarks/benchmark_1_tiny.yaml",
  "config_name": "benchmark_1_tiny",
  "num_parameters": 123456,
  "results_by_init_method": {
    "default": {
      "init_method": "default",
      "job_id": "Benchmark_1_tiny_...",
      "experiment_dir": "Experiment/...",
      "perplexity_at_intervals": [
        {"epoch": 1, "train_perplexity": 826.6, "val_perplexity": 401.6}
      ],
      "convergence": {
        "converged": true,
        "convergence_epoch": 7,
        "convergence_perplexity": 236.0
      },
      "test_evaluation": {
        "test_loss": 5.50,
        "test_perplexity": 244.0
      },
      "best_model": {
        "epoch": 7,
        "val_loss": 5.46,
        "val_perplexity": 236.0
      }
    },
    "ghn": { }
  }
}
```

Init method keys match `train_lm.py`: `default`, `he`, `xavier`, `ghn`.

## Experiment discovery

The evaluator finds runs in `Experiment/` by matching the config name in the directory name and reading `init_method` from `config.json`. For older runs without that field, init method is inferred from the job ID prefix (e.g. `Benchmark_*` → `default`, `GHN-I_*` → `ghn`).

TensorBoard logs are read from `tensor_log/` first, with fallback to `Final_tensors/`.

## Legacy migration

`--migrate` merges JSON from these legacy locations into `Results/metrics/`:

| Source | Contents |
|--------|----------|
| `Evaluations/*_comparison.json` | Benchmark vs GHN comparison runs |
| `Report/benchmark_*_all_inits_evaluation.json` | Per-init evaluation reports |

Legacy init aliases (`GHN-I`, `GHN-T`, `ghn-i`) are normalized to `ghn`. Existing PNGs in `Evaluations/` are copied to `Results/plots/`.

## SLURM

```bash
sbatch slurm/evaluate_metrics.sh
./slurm/run_first_10_configs_evaluation.sh
```

Set `CONFIG_FILE` in `evaluate_metrics.sh` to target a specific benchmark.
