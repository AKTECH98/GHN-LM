# Documentation

Index for the GHN-3 language model capstone project.

## Guides

- [Training](training.md) — train the hypernetwork and benchmark language models
- [Evaluation](evaluation.md) — extract metrics, generate plots, migrate legacy results
- [Configuration](configuration.md) — benchmark YAML configs and model architectures

## Repository layout

| Path | Purpose |
|------|---------|
| `src/capstone/` | Installable Python package (`pip install -e .`) |
| `src/capstone/paths.py` | Absolute paths to configs, results, experiments, data |
| `configs/benchmarks/` | Ten benchmark model YAML files |
| `scripts/` | Command-line entry points |
| `slurm/` | SLURM batch scripts for RIT cluster |
| `Results/` | Canonical evaluation JSON and plots (gitignored) |
| `Experiment/` | Training checkpoints and run configs (gitignored) |
| `tensor_log/` | TensorBoard event files (gitignored) |
| `GHN_Models/` | Saved hypernetwork checkpoints (gitignored) |

## Python package modules

| Module | Contents |
|--------|----------|
| `capstone.ghn` | Graph HyperNetwork: graph builder, Graphormer, weight prediction |
| `capstone.lm` | Language models (`gpt_encoder`, `mini_gpt`) and training loop |
| `capstone.data` | WikiText-2 loader, YAML config parser, GHN architecture dataset |
| `capstone.eval` | Metrics evaluator, experiment discovery, plotting, legacy migration |

## Local deliverables (gitignored)

These folders are not tracked in git but may exist on your machine:

- `Report/` — capstone research reports and figures
- `Evaluations/` — legacy comparison JSON and plots (superseded by `Results/`)
- `docs/report.pdf` — final report PDF

Use `python scripts/evaluate_metrics.py --migrate` to merge legacy JSON from those folders into `Results/`.
