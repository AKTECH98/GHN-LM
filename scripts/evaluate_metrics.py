#!/usr/bin/env python3
"""
Evaluate metrics from completed training runs and generate comparison plots.

Usage:
    python scripts/evaluate_metrics.py --config configs/benchmarks/benchmark_1_tiny.yaml
    python scripts/evaluate_metrics.py --all-configs --plot
    python scripts/evaluate_metrics.py --plot-all
    python scripts/evaluate_metrics.py --migrate
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from capstone.data.config_loader import list_benchmark_configs
from capstone.eval.consolidate import migrate_legacy_results
from capstone.eval.discovery import find_experiment_dirs_by_init_method
from capstone.eval.evaluator import MetricsEvaluator
from capstone.eval.plotting import plot_all_aggregate, plot_config_curves
from capstone.eval.schema import (
    build_summary,
    config_metrics_path,
    empty_config_metrics,
    load_metrics,
    merge_init_results,
    metrics_dir,
    plots_dir,
    save_metrics,
    summary_dir,
)

from capstone.paths import CONFIGS_DIR, RESULTS_DIR

DEFAULT_RESULTS_ROOT = RESULTS_DIR
INIT_METHODS = ("default", "he", "xavier", "ghn")


def evaluate_config(
    config_file: Path,
    results_root: Path,
    target_epochs: List[int],
    device: str,
    plot: bool = False,
) -> Optional[Dict]:
    config_name = config_file.stem
    print(f"\n📋 Finding experiments for config: {config_name}")
    experiment_dirs = find_experiment_dirs_by_init_method(config_name)

    if not experiment_dirs:
        print(f"   ⚠️  No experiment directories found for {config_name}")
        return None

    for init_method, exp_dir in experiment_dirs.items():
        print(f"      {init_method}: {exp_dir.name}")

    with open(config_file) as f:
        yaml_config = yaml.safe_load(f)
    training_config = yaml_config.get("training", {})
    convergence_patience = training_config.get("early_stopping_patience", 3)
    convergence_threshold = training_config.get("early_stopping_min_delta", 0.001)

    record = empty_config_metrics(str(config_file), config_name)

    for init_method, experiment_dir in experiment_dirs.items():
        print(f"\n{'=' * 60}")
        print(f"📊 Evaluating init_method={init_method}")
        print(f"{'=' * 60}")

        evaluator = MetricsEvaluator(
            config_file=config_file,
            experiment_dir=experiment_dir,
            device=device,
        )
        metrics = evaluator.evaluate_all_metrics(
            target_epochs=target_epochs,
            convergence_threshold=convergence_threshold,
            convergence_patience=convergence_patience,
        )
        merge_init_results(record, init_method, metrics)

    output_path = config_metrics_path(results_root, config_name)
    save_metrics(record, output_path)
    print(f"\n💾 Metrics saved to: {output_path}")

    if plot:
        plot_config_curves(record, plots_dir(results_root) / f"{config_name}_curves.png")

    _print_summary(record, config_name)
    return record


def _print_summary(record: Dict, config_name: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"📊 Summary: {config_name}")
    print(f"{'=' * 60}")

    by_init = record.get("results_by_init_method", {})
    print(f"{'Init':<10} {'Test PPL':<12} {'Conv Epoch':<12}")
    print("-" * 40)
    for init_method in INIT_METHODS:
        if init_method not in by_init:
            continue
        result = by_init[init_method]
        test_ppl = result.get("test_evaluation", {}).get("test_perplexity", "N/A")
        conv_epoch = result.get("convergence", {}).get("convergence_epoch", "N/A")
        test_str = f"{test_ppl:.2f}" if isinstance(test_ppl, (int, float)) else str(test_ppl)
        print(f"{init_method:<10} {test_str:<12} {conv_epoch!s:<12}")


def refresh_summary(results_root: Path) -> None:
    all_metrics = {}
    root = metrics_dir(results_root)
    if not root.exists():
        return
    for path in sorted(root.glob("*.json")):
        data = load_metrics(path)
        config_name = data.get("config_name", path.stem)
        all_metrics[config_name] = data
    save_metrics(build_summary(all_metrics), summary_dir(results_root) / "all_configs.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate metrics from completed training runs")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--all-configs", action="store_true", help="Evaluate all benchmark configs")
    parser.add_argument("--intervals", type=str, default="1,2,5,10,20,50",
                        help="Comma-separated epochs for perplexity extraction")
    parser.add_argument("--device", type=str, default="cuda", help="Evaluation device")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_RESULTS_ROOT),
                        help="Results output root (default: Results/)")
    parser.add_argument("--plot", action="store_true", help="Generate per-config curve plots")
    parser.add_argument("--plot-all", action="store_true", help="Generate aggregate comparison plots")
    parser.add_argument("--migrate", action="store_true", help="Migrate legacy JSON into Results/")
    args = parser.parse_args()

    results_root = Path(args.output_dir)
    target_epochs = [int(x.strip()) for x in args.intervals.split(",")]

    if args.migrate:
        stats = migrate_legacy_results(results_root)
        print("\n✅ Migration complete:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        refresh_summary(results_root)

    if args.plot_all:
        plot_all_aggregate(results_root)
        return

    if args.all_configs:
        configs = list_benchmark_configs()
        for config_name in configs:
            config_file = CONFIGS_DIR / f"{config_name}.yaml"
            evaluate_config(config_file, results_root, target_epochs, args.device, plot=args.plot)
        refresh_summary(results_root)
        if args.plot:
            plot_all_aggregate(results_root)
        return

    if args.config:
        config_file = Path(args.config)
        if not config_file.exists():
            parser.error(f"Config file not found: {config_file}")
        evaluate_config(config_file, results_root, target_epochs, args.device, plot=args.plot)
        refresh_summary(results_root)
        return

    if not args.migrate:
        parser.error("Provide --config, --all-configs, --plot-all, or --migrate")


if __name__ == "__main__":
    main()
