"""Migrate and merge legacy evaluation JSON into canonical Results/ layout."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from capstone.eval.schema import (
    build_summary,
    config_metrics_path,
    empty_config_metrics,
    merge_init_results,
    metrics_dir,
    normalize_init_method,
    plots_dir,
    save_metrics,
    summary_dir,
)
from capstone.eval.plotting import copy_legacy_plots
from capstone.paths import (
    CONFIGS_DIR,
    LEGACY_EVALUATIONS_DIR,
    LEGACY_NEWRESULTS_DIR,
    LEGACY_REPORT_DIR,
    RESULTS_DIR,
)

# Map old Report config indices to current benchmark config names.
REPORT_CONFIG_INDEX = {
    1: "benchmark_1_tiny",
    2: "benchmark_2_small",
    3: "benchmark_3_medium",
    4: "benchmark_4_large",
    5: "benchmark_5_mini_gpt",
    6: "benchmark_6_mini_gpt_tiny",
    7: "benchmark_7_mini_gpt_small",
    8: "benchmark_8_mini_gpt_medium",
    9: "benchmark_9_mini_gpt_large",
}


def _ensure_config_record(
    store: Dict[str, Dict[str, Any]],
    config_name: str,
    config_file: Optional[str] = None,
) -> Dict[str, Any]:
    if config_name not in store:
        store[config_name] = empty_config_metrics(
            config_file or str(CONFIGS_DIR / f"{config_name}.yaml"),
            config_name,
        )
    return store[config_name]


def _normalize_perplexity_intervals(result: Dict[str, Any]) -> None:
    if "perplexity_at_intervals" not in result and "perplexity_all_epochs" in result:
        result["perplexity_at_intervals"] = result.pop("perplexity_all_epochs")


def _ingest_init_method_block(
    store: Dict[str, Dict[str, Any]],
    config_name: str,
    config_file: str,
    init_method: str,
    result: Dict[str, Any],
    num_parameters: Optional[int] = None,
) -> None:
    record = _ensure_config_record(store, config_name, config_file)
    if num_parameters is not None:
        record["num_parameters"] = num_parameters
    payload = dict(result)
    _normalize_perplexity_intervals(payload)
    merge_init_results(record, init_method, payload)


def ingest_newresults(store: Dict[str, Dict[str, Any]], root: Path = LEGACY_NEWRESULTS_DIR) -> int:
    count = 0
    if not root.exists():
        return count

    for path in root.glob("*_all_metrics.json"):
        with open(path) as f:
            data = json.load(f)
        config_name = data.get("config_name", path.stem.replace("_all_metrics", ""))
        config_file = data.get("config_file", str(CONFIGS_DIR / f"{config_name}.yaml"))
        by_init = data.get("results_by_init_method", {})
        for init_method, result in by_init.items():
            _ingest_init_method_block(store, config_name, config_file, init_method, result)
            count += 1
    return count


def ingest_evaluations_all_metrics(store: Dict[str, Dict[str, Any]], root: Path = LEGACY_EVALUATIONS_DIR) -> int:
    count = 0
    if not root.exists():
        return count

    for path in root.glob("*_all_metrics.json"):
        with open(path) as f:
            data = json.load(f)
        config_name = data.get("config_name", path.stem.replace("_all_metrics", ""))
        config_file = data.get("config_file", str(CONFIGS_DIR / f"{config_name}.yaml"))

        by_category = data.get("results_by_category", {})
        for category, result in by_category.items():
            _ingest_init_method_block(store, config_name, config_file, category, result)
            count += 1

        by_init = data.get("results_by_init_method", {})
        for init_method, result in by_init.items():
            _ingest_init_method_block(store, config_name, config_file, init_method, result)
            count += 1
    return count


def ingest_evaluations_comparison(store: Dict[str, Dict[str, Any]], root: Path = LEGACY_EVALUATIONS_DIR) -> int:
    count = 0
    if not root.exists():
        return count

    for path in root.glob("*_comparison.json"):
        if path.name == "all_benchmark_ghn_comparisons.json":
            continue
        with open(path) as f:
            data = json.load(f)
        config_name = data.get("config_name", path.stem.replace("_comparison", ""))
        config_file = data.get("config_file", str(CONFIGS_DIR / f"{config_name}.yaml"))

        benchmark = data.get("benchmark")
        ghn = data.get("ghn_init")
        if benchmark:
            _ingest_init_method_block(store, config_name, config_file, "default", _comparison_to_result(benchmark, config_name, config_file, "default"))
            count += 1
        if ghn:
            _ingest_init_method_block(store, config_name, config_file, "ghn", _comparison_to_result(ghn, config_name, config_file, "ghn"))
            count += 1
    return count


def ingest_combined_comparisons(store: Dict[str, Dict[str, Any]], root: Path = LEGACY_EVALUATIONS_DIR) -> int:
    path = root / "all_benchmark_ghn_comparisons.json"
    count = 0
    if not path.exists():
        return count

    with open(path) as f:
        combined = json.load(f)

    for key, data in combined.items():
        config_name = data.get("config_name")
        if not config_name:
            match = re.search(r"benchmark_(\d+)", key)
            if match:
                idx = int(match.group(1))
                config_name = REPORT_CONFIG_INDEX.get(idx)
        if not config_name:
            continue
        config_file = data.get("config_file", str(CONFIGS_DIR / f"{config_name}.yaml"))
        benchmark = data.get("benchmark")
        ghn = data.get("ghn_init")
        if benchmark:
            _ingest_init_method_block(store, config_name, config_file, "default", _comparison_to_result(benchmark, config_name, config_file, "default"))
            count += 1
        if ghn:
            _ingest_init_method_block(store, config_name, config_file, "ghn", _comparison_to_result(ghn, config_name, config_file, "ghn"))
            count += 1
    return count


def _comparison_to_result(
    block: Dict[str, Any],
    config_name: str,
    config_file: str,
    init_method: str,
) -> Dict[str, Any]:
    curves = block.get("training_curves", {})
    val_ppl = curves.get("val_perplexity") or curves.get("val_loss")
    intervals = []
    if val_ppl:
        for epoch, value in val_ppl:
            intervals.append({"epoch": int(epoch) + 1, "train_perplexity": None, "val_perplexity": float(value)})

    return {
        "config_file": config_file,
        "config_name": config_name,
        "init_method": init_method,
        "job_id": block.get("job_id"),
        "experiment_dir": block.get("experiment_dir"),
        "perplexity_at_intervals": intervals,
        "convergence": {
            "converged": True,
            "convergence_epoch": block.get("convergence_epoch"),
            "convergence_loss": block.get("best_val_loss"),
            "convergence_perplexity": block.get("best_val_perplexity"),
        },
        "test_evaluation": block.get("test_metrics", {}),
        "best_model": {
            "epoch": (block.get("checkpoint_info") or {}).get("epoch"),
            "val_loss": block.get("best_val_loss"),
            "val_perplexity": block.get("best_val_perplexity"),
        },
    }


def ingest_report_evaluations(store: Dict[str, Dict[str, Any]], root: Path = LEGACY_REPORT_DIR) -> int:
    count = 0
    if not root.exists():
        return count

    for path in root.glob("benchmark_*_all_inits_evaluation.json"):
        with open(path) as f:
            data = json.load(f)

        config_name = data.get("config_name", "")
        if config_name.startswith("benchmark_") and config_name.count("_") == 1:
            match = re.search(r"benchmark_(\d+)$", config_name)
            if match:
                config_name = REPORT_CONFIG_INDEX.get(int(match.group(1)), config_name)

        config_file = data.get("config_file", str(CONFIGS_DIR / f"{config_name}.yaml"))
        if not config_name.startswith("benchmark_"):
            continue

        num_parameters = data.get("num_parameters")
        by_init = data.get("results_by_init_method", {})
        for init_method, result in by_init.items():
            _ingest_init_method_block(
                store,
                config_name,
                config_file,
                init_method,
                result,
                num_parameters=num_parameters,
            )
            count += 1
    return count


def migrate_legacy_results(results_root: Path = RESULTS_DIR) -> Dict[str, int]:
    """Merge all legacy JSON sources into Results/metrics/."""
    store: Dict[str, Dict[str, Any]] = {}
    stats = {
        "newresults": ingest_newresults(store),
        "evaluations_all_metrics": ingest_evaluations_all_metrics(store),
        "evaluations_comparison": ingest_evaluations_comparison(store),
        "evaluations_combined": ingest_combined_comparisons(store),
        "report": ingest_report_evaluations(store),
    }

    metrics_dir(results_root).mkdir(parents=True, exist_ok=True)
    plots_dir(results_root).mkdir(parents=True, exist_ok=True)
    summary_dir(results_root).mkdir(parents=True, exist_ok=True)

    for config_name, record in store.items():
        save_metrics(record, config_metrics_path(results_root, config_name))

    summary = build_summary(store)
    save_metrics(summary, summary_dir(results_root) / "all_configs.json")
    copy_legacy_plots(results_root)

    stats["configs_written"] = len(store)
    return stats
