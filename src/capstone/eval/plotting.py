"""Plotting utilities for benchmark vs GHN init comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from capstone.data.config_loader import load_config_file
from capstone.eval.schema import load_metrics, metrics_dir, plots_dir
from capstone.lm.create_model import create_model
from capstone.paths import CONFIGS_DIR, LEGACY_EVALUATIONS_DIR

CONFIG_ORDER: List[Tuple[str, str]] = [
    ("benchmark_1_tiny", "Tiny"),
    ("benchmark_2_small", "Small"),
    ("benchmark_3_medium", "Medium"),
    ("benchmark_4_large", "Large"),
    ("benchmark_5_mini_gpt", "Mini GPT"),
    ("benchmark_6_mini_gpt_tiny", "Mini GPT Tiny"),
    ("benchmark_7_mini_gpt_small", "Mini GPT Small"),
    ("benchmark_8_mini_gpt_medium", "Mini GPT Medium"),
    ("benchmark_9_mini_gpt_large", "Mini GPT Large"),
    ("benchmark_10_mini_gpt_xl", "Mini GPT XL"),
]

DISPLAY_BY_CONFIG = {cfg: label for cfg, label in CONFIG_ORDER}


def calculate_model_parameters(config_file: Path) -> int:
    model_config, _, _ = load_config_file(str(config_file))
    vocab_size = getattr(model_config, "vocab_size", 50257)
    model = create_model(model_config, vocab_size)
    total_params = sum(p.numel() for p in model.parameters())
    del model
    return total_params


def load_all_metrics(results_root: Path) -> Dict[str, Dict[str, Any]]:
    root = metrics_dir(results_root)
    if not root.exists():
        return {}

    loaded: Dict[str, Dict[str, Any]] = {}
    for path in sorted(root.glob("*.json")):
        data = load_metrics(path)
        config_name = data.get("config_name", path.stem)
        loaded[config_name] = data
    return loaded


def _get_init_result(metrics: Dict[str, Any], init_method: str) -> Optional[Dict[str, Any]]:
    by_init = metrics.get("results_by_init_method", {})
    if init_method in by_init:
        return by_init[init_method]
    return None


def extract_comparison_metrics(all_metrics: Dict[str, Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
    test_results: List[Dict] = []
    training_results: List[Dict] = []

    for config_name, display_name in CONFIG_ORDER:
        metrics = all_metrics.get(config_name)
        if not metrics:
            continue

        default = _get_init_result(metrics, "default")
        ghn = _get_init_result(metrics, "ghn")
        if not default or not ghn:
            continue

        default_test = default.get("test_evaluation", {}).get("test_perplexity")
        ghn_test = ghn.get("test_evaluation", {}).get("test_perplexity")
        if default_test is not None and ghn_test is not None:
            improvement = ((default_test - ghn_test) / default_test) * 100
            test_results.append(
                {
                    "name": display_name,
                    "benchmark": default_test,
                    "ghn": ghn_test,
                    "improvement": improvement,
                    "winner": "GHN Init" if improvement > 0 else "Benchmark",
                }
            )

        default_best_val = _best_val_perplexity(default)
        ghn_best_val = _best_val_perplexity(ghn)
        if default_best_val is not None and ghn_best_val is not None:
            train_improvement = ((default_best_val - ghn_best_val) / default_best_val) * 100
            training_results.append(
                {
                    "name": display_name,
                    "benchmark": default_best_val,
                    "ghn": ghn_best_val,
                    "improvement": train_improvement,
                    "winner": "GHN Init" if train_improvement > 0 else "Benchmark",
                }
            )

    return test_results, training_results


def _best_val_perplexity(result: Dict[str, Any]) -> Optional[float]:
    intervals = result.get("perplexity_at_intervals") or result.get("perplexity_all_epochs") or []
    values = [entry["val_perplexity"] for entry in intervals if entry.get("val_perplexity") is not None]
    if values:
        return min(values)
    best_model = result.get("best_model") or {}
    return best_model.get("val_perplexity")


def extract_convergence_data(
    all_metrics: Dict[str, Dict[str, Any]],
    config_base_path: Path = CONFIGS_DIR,
) -> List[Dict]:
    convergence_results: List[Dict] = []

    for config_name, display_name in CONFIG_ORDER:
        metrics = all_metrics.get(config_name)
        if not metrics:
            continue

        default = _get_init_result(metrics, "default")
        ghn = _get_init_result(metrics, "ghn")
        if not default or not ghn:
            continue

        default_conv = default.get("convergence", {}).get("convergence_epoch")
        ghn_conv = ghn.get("convergence", {}).get("convergence_epoch")
        if default_conv is None or ghn_conv is None:
            continue

        config_file = config_base_path / f"{config_name}.yaml"
        num_params = metrics.get("num_parameters")
        if num_params is None and config_file.exists():
            try:
                num_params = calculate_model_parameters(config_file)
            except Exception:
                num_params = 0

        convergence_results.append(
            {
                "name": display_name,
                "num_params": num_params or 0,
                "benchmark_conv_epoch": default_conv,
                "ghn_conv_epoch": ghn_conv,
            }
        )

    convergence_results.sort(key=lambda x: x["num_params"])
    return convergence_results


def create_test_performance_graph(results: List[Dict], output_path: Path) -> None:
    if not results:
        return

    names = [r["name"] for r in results]
    improvements = [r["improvement"] for r in results]
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    x = np.arange(len(names))
    colors = ["#06A77D" if imp > 0 else "#D00000" for imp in improvements]
    bars = ax.bar(x, improvements, color=colors, alpha=0.8, width=0.6)

    if len(improvements) > 1:
        z = np.polyfit(x, improvements, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), "r--", alpha=0.7, linewidth=2.5, label=f"Trend: {z[0]:.2f}% per step")

    for bar, imp in zip(bars, improvements):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{imp:+.1f}%",
            ha="center",
            va="bottom" if imp > 0 else "top",
            fontsize=12,
            fontweight="bold",
        )

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_xlabel("Model Configuration (Increasing Complexity →)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Improvement (%)", fontsize=14, fontweight="bold")
    ax.set_title("GHN Init Test Performance Improvement Over Default Init", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    if len(improvements) > 1:
        ax.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Test performance graph saved to: {output_path}")


def create_training_improvement_graph(results: List[Dict], output_path: Path) -> None:
    if not results:
        return

    names = [r["name"] for r in results]
    improvements = [r["improvement"] for r in results]
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    x = np.arange(len(names))
    colors = ["#06A77D" if imp > 0 else "#D00000" for imp in improvements]
    bars = ax.bar(x, improvements, color=colors, alpha=0.8, width=0.6)

    if len(improvements) > 1:
        z = np.polyfit(x, improvements, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), "r--", alpha=0.7, linewidth=2.5, label=f"Trend: {z[0]:.2f}% per step")

    for bar, imp in zip(bars, improvements):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{imp:+.1f}%",
            ha="center",
            va="bottom" if imp > 0 else "top",
            fontsize=12,
            fontweight="bold",
        )

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_xlabel("Model Configuration (Increasing Complexity →)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Improvement (%)", fontsize=14, fontweight="bold")
    ax.set_title("GHN Init Validation Perplexity Improvement Over Default Init", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    if len(improvements) > 1:
        ax.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Training improvement graph saved to: {output_path}")


def create_convergence_graph(results: List[Dict], output_path: Path) -> None:
    if not results:
        return

    num_params = [r["num_params"] for r in results]
    benchmark_epochs = [r["benchmark_conv_epoch"] for r in results]
    ghn_epochs = [r["ghn_conv_epoch"] for r in results]
    names = [r["name"] for r in results]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.scatter(num_params, benchmark_epochs, s=120, alpha=0.7, color="#2E86AB", label="Default", marker="o")
    ax.scatter(num_params, ghn_epochs, s=120, alpha=0.7, color="#A23B72", label="GHN Init", marker="s")
    ax.plot(num_params, benchmark_epochs, "--", alpha=0.5, color="#2E86AB")
    ax.plot(num_params, ghn_epochs, "--", alpha=0.5, color="#A23B72")

    ax.set_xlabel("Number of Parameters", fontsize=14, fontweight="bold")
    ax.set_ylabel("Convergence Epoch", fontsize=14, fontweight="bold")
    ax.set_title("Convergence Epoch vs Model Size", fontsize=16, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    def format_params(x, _pos):
        if x >= 1e6:
            return f"{x / 1e6:.2f}M"
        if x >= 1e3:
            return f"{x / 1e3:.1f}K"
        return f"{int(x)}"

    ax.xaxis.set_major_formatter(FuncFormatter(format_params))
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Convergence graph saved to: {output_path}")


def plot_config_curves(metrics: Dict[str, Any], output_path: Path) -> None:
    """Plot val perplexity curves for default vs ghn for one config."""
    by_init = metrics.get("results_by_init_method", {})
    default = by_init.get("default")
    ghn = by_init.get("ghn")
    if not default or not ghn:
        return

    def series(result: Dict[str, Any]) -> Tuple[List[int], List[float]]:
        intervals = result.get("perplexity_at_intervals") or result.get("perplexity_all_epochs") or []
        epochs = []
        vals = []
        for entry in intervals:
            if entry.get("val_perplexity") is not None:
                epochs.append(entry["epoch"])
                vals.append(entry["val_perplexity"])
        return epochs, vals

    d_epochs, d_vals = series(default)
    g_epochs, g_vals = series(ghn)
    if not d_epochs and not g_epochs:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    if d_epochs:
        ax.plot(d_epochs, d_vals, "o-", label="Default", color="#2E86AB")
    if g_epochs:
        ax.plot(g_epochs, g_vals, "s-", label="GHN Init", color="#A23B72")

    config_name = metrics.get("config_name", "config")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Perplexity")
    ax.set_title(f"Training Curves: {config_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Config curves saved to: {output_path}")


def plot_all_aggregate(results_root: Path) -> None:
    all_metrics = load_all_metrics(results_root)
    if not all_metrics:
        print(f"❌ No metrics found in {metrics_dir(results_root)}")
        return

    out = plots_dir(results_root)
    test_results, training_results = extract_comparison_metrics(all_metrics)
    convergence_results = extract_convergence_data(all_metrics)

    if test_results:
        create_test_performance_graph(test_results, out / "test_performance_comparison.png")
    if training_results:
        create_training_improvement_graph(training_results, out / "training_improvement_comparison.png")
    if convergence_results:
        create_convergence_graph(convergence_results, out / "convergence_epoch_comparison.png")


def copy_legacy_plots(results_root: Path) -> None:
    """Copy existing PNGs from Evaluations/ into Results/plots/ when present."""
    source = LEGACY_EVALUATIONS_DIR
    if not source.exists():
        return
    dest = plots_dir(results_root)
    dest.mkdir(parents=True, exist_ok=True)
    for png in source.glob("*.png"):
        target = dest / png.name
        if not target.exists():
            target.write_bytes(png.read_bytes())
            print(f"📋 Copied plot: {png.name} -> {target}")
