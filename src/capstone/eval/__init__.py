"""Evaluation metrics extraction, consolidation, and plotting."""

from capstone.eval.consolidate import migrate_legacy_results
from capstone.eval.discovery import find_experiment_dirs_by_init_method, find_tensorboard_dir
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

__all__ = [
    "MetricsEvaluator",
    "find_experiment_dirs_by_init_method",
    "find_tensorboard_dir",
    "migrate_legacy_results",
    "plot_all_aggregate",
    "plot_config_curves",
    "build_summary",
    "config_metrics_path",
    "empty_config_metrics",
    "load_metrics",
    "merge_init_results",
    "metrics_dir",
    "plots_dir",
    "save_metrics",
    "summary_dir",
]
