# Example Metrics Output Format

This document shows the structure of the saved metrics JSON files.

## 1. Individual Metrics File (per init method)

**Location:** `Experiment/{experiment_dir}/metrics.json`

**Example:** `Experiment/Benchmark_1_tiny_1763943806/metrics.json`

```json
{
  "config_file": "LM/configs/benchmark_1_tiny.yaml",
  "config_name": "benchmark_1_tiny",
  "init_method": "default",
  "job_id": "Benchmark_1_tiny_1763943806",
  "experiment_dir": "Experiment/Benchmark_1_tiny_1763943806",
  "perplexity_at_intervals": [
    {
      "epoch": 1,
      "train_perplexity": 245.32,
      "val_perplexity": 280.45
    },
    {
      "epoch": 2,
      "train_perplexity": 198.21,
      "val_perplexity": 220.13
    },
    {
      "epoch": 5,
      "train_perplexity": 145.67,
      "val_perplexity": 165.89
    },
    {
      "epoch": 10,
      "train_perplexity": 98.45,
      "val_perplexity": 112.34
    },
    {
      "epoch": 20,
      "train_perplexity": null,
      "val_perplexity": 85.23
    },
    {
      "epoch": 50,
      "train_perplexity": null,
      "val_perplexity": null
    }
  ],
  "convergence": {
    "converged": true,
    "convergence_epoch": 15,
    "convergence_loss": 4.523,
    "convergence_perplexity": 92.10,
    "epochs_to_convergence": 15,
    "best_epoch": 12,
    "total_epochs": 20
  },
  "test_evaluation": {
    "test_loss": 4.612,
    "test_perplexity": 100.73,
    "total_tokens": 1234567,
    "num_batches": 234
  },
  "best_model": {
    "epoch": 12,
    "val_loss": 4.523,
    "val_perplexity": 92.10
  }
}
```

## 2. Combined Metrics File (all init methods)

**Location:** `Experiment/{config_name}_all_metrics.json`

**Example:** `Experiment/benchmark_1_tiny_all_metrics.json`

```json
{
  "config_file": "LM/configs/benchmark_1_tiny.yaml",
  "config_name": "benchmark_1_tiny",
  "results_by_init_method": {
    "default": {
      "config_file": "LM/configs/benchmark_1_tiny.yaml",
      "config_name": "benchmark_1_tiny",
      "init_method": "default",
      "job_id": "Benchmark_1_tiny_1763943806",
      "experiment_dir": "Experiment/Benchmark_1_tiny_1763943806",
      "perplexity_at_intervals": [
        {
          "epoch": 1,
          "train_perplexity": 245.32,
          "val_perplexity": 280.45
        },
        {
          "epoch": 2,
          "train_perplexity": 198.21,
          "val_perplexity": 220.13
        },
        {
          "epoch": 5,
          "train_perplexity": 145.67,
          "val_perplexity": 165.89
        },
        {
          "epoch": 10,
          "train_perplexity": 98.45,
          "val_perplexity": 112.34
        }
      ],
      "convergence": {
        "converged": true,
        "convergence_epoch": 15,
        "convergence_loss": 4.523,
        "convergence_perplexity": 92.10,
        "epochs_to_convergence": 15,
        "best_epoch": 12,
        "total_epochs": 20
      },
      "test_evaluation": {
        "test_loss": 4.612,
        "test_perplexity": 100.73,
        "total_tokens": 1234567,
        "num_batches": 234
      },
      "best_model": {
        "epoch": 12,
        "val_loss": 4.523,
        "val_perplexity": 92.10
      }
    },
    "he": {
      "config_file": "LM/configs/benchmark_1_tiny.yaml",
      "config_name": "benchmark_1_tiny",
      "init_method": "he",
      "job_id": "BenchmarkHEInit_1_tiny_1763943927",
      "experiment_dir": "Experiment/BenchmarkHEInit_1_tiny_1763943927",
      "perplexity_at_intervals": [
        {
          "epoch": 1,
          "train_perplexity": 238.45,
          "val_perplexity": 275.12
        },
        {
          "epoch": 2,
          "train_perplexity": 192.34,
          "val_perplexity": 215.67
        },
        {
          "epoch": 5,
          "train_perplexity": 142.89,
          "val_perplexity": 162.45
        },
        {
          "epoch": 10,
          "train_perplexity": 96.23,
          "val_perplexity": 110.12
        }
      ],
      "convergence": {
        "converged": true,
        "convergence_epoch": 14,
        "convergence_loss": 4.489,
        "convergence_perplexity": 89.05,
        "epochs_to_convergence": 14,
        "best_epoch": 11,
        "total_epochs": 20
      },
      "test_evaluation": {
        "test_loss": 4.578,
        "test_perplexity": 97.34,
        "total_tokens": 1234567,
        "num_batches": 234
      },
      "best_model": {
        "epoch": 11,
        "val_loss": 4.489,
        "val_perplexity": 89.05
      }
    },
    "xavier": {
      "config_file": "LM/configs/benchmark_1_tiny.yaml",
      "config_name": "benchmark_1_tiny",
      "init_method": "xavier",
      "job_id": "BenchmarkXavier_1_tiny_1763943987",
      "experiment_dir": "Experiment/BenchmarkXavier_1_tiny_1763943987",
      "perplexity_at_intervals": [
        {
          "epoch": 1,
          "train_perplexity": 240.12,
          "val_perplexity": 278.34
        },
        {
          "epoch": 2,
          "train_perplexity": 195.67,
          "val_perplexity": 218.45
        },
        {
          "epoch": 5,
          "train_perplexity": 144.23,
          "val_perplexity": 164.12
        },
        {
          "epoch": 10,
          "train_perplexity": 97.89,
          "val_perplexity": 111.67
        }
      ],
      "convergence": {
        "converged": true,
        "convergence_epoch": 16,
        "convergence_loss": 4.556,
        "convergence_perplexity": 95.23,
        "epochs_to_convergence": 16,
        "best_epoch": 13,
        "total_epochs": 20
      },
      "test_evaluation": {
        "test_loss": 4.634,
        "test_perplexity": 102.89,
        "total_tokens": 1234567,
        "num_batches": 234
      },
      "best_model": {
        "epoch": 13,
        "val_loss": 4.556,
        "val_perplexity": 95.23
      }
    },
    "ghn": {
      "config_file": "LM/configs/benchmark_1_tiny.yaml",
      "config_name": "benchmark_1_tiny",
      "init_method": "ghn",
      "job_id": "GHNInit_1_tiny_1763944050",
      "experiment_dir": "Experiment/GHNInit_1_tiny_1763944050",
      "perplexity_at_intervals": [
        {
          "epoch": 1,
          "train_perplexity": 230.45,
          "val_perplexity": 265.78
        },
        {
          "epoch": 2,
          "train_perplexity": 185.34,
          "val_perplexity": 208.12
        },
        {
          "epoch": 5,
          "train_perplexity": 138.67,
          "val_perplexity": 158.45
        },
        {
          "epoch": 10,
          "train_perplexity": 92.12,
          "val_perplexity": 105.67
        }
      ],
      "convergence": {
        "converged": true,
        "convergence_epoch": 13,
        "convergence_loss": 4.412,
        "convergence_perplexity": 82.45,
        "epochs_to_convergence": 13,
        "best_epoch": 10,
        "total_epochs": 20
      },
      "test_evaluation": {
        "test_loss": 4.501,
        "test_perplexity": 90.12,
        "total_tokens": 1234567,
        "num_batches": 234
      },
      "best_model": {
        "epoch": 10,
        "val_loss": 4.412,
        "val_perplexity": 82.45
      }
    }
  }
}
```

## File Locations Summary

After running:
```bash
python evaluate_metrics.py --config LM/configs/benchmark_1_tiny.yaml
```

You will get:

1. **Individual metrics files** (one per init method):
   - `Experiment/Benchmark_1_tiny_1763943806/metrics.json` (default)
   - `Experiment/BenchmarkHEInit_1_tiny_1763943927/metrics.json` (he)
   - `Experiment/BenchmarkXavier_1_tiny_1763943987/metrics.json` (xavier)
   - `Experiment/GHNInit_1_tiny_1763944050/metrics.json` (ghn)

2. **Combined metrics file**:
   - `Experiment/benchmark_1_tiny_all_metrics.json` (all init methods together)

## Notes

- `train_perplexity` may be `null` if not available in TensorBoard logs (only checkpoints available)
- `val_perplexity` is always available (from TensorBoard or checkpoints)
- `convergence_epoch` indicates when the model stopped improving
- `test_evaluation` is computed on the best model checkpoint
- All perplexity values are calculated as `exp(loss)`

