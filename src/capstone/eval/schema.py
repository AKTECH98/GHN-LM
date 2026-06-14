"""Canonical metrics JSON schema helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

INIT_METHODS = ("default", "he", "xavier", "ghn")

# Legacy init keys mapped to canonical names during migration.
LEGACY_INIT_ALIASES: Dict[str, str] = {
    "benchmark": "default",
    "benchmark_": "default",
    "benchmark_16__": "default",
    "ghn-i": "ghn",
    "ghn-i_16-2_": "ghn",
    "ghn-i_64-2_": "ghn",
    "ghn_init": "ghn",
    "ghn-t": "ghn",
    "GHN-I": "ghn",
    "GHN-T": "ghn",
    "GHN-I_16-2_": "ghn",
    "GHN-I_64-2_": "ghn",
}


def normalize_init_method(key: str) -> str:
    """Map legacy init keys to canonical init_method names."""
    if not key:
        return "default"
    lowered = key.lower()
    if lowered in INIT_METHODS:
        return lowered
    if key in LEGACY_INIT_ALIASES:
        return LEGACY_INIT_ALIASES[key]
    if lowered in LEGACY_INIT_ALIASES:
        return LEGACY_INIT_ALIASES[lowered]
    return lowered


def empty_config_metrics(config_file: str, config_name: str) -> Dict[str, Any]:
    return {
        "config_file": config_file,
        "config_name": config_name,
        "num_parameters": None,
        "results_by_init_method": {},
    }


def merge_init_results(
    target: Dict[str, Any],
    init_method: str,
    result: Dict[str, Any],
) -> None:
    """Merge a single init result into a config metrics document."""
    canonical = normalize_init_method(init_method)
    result = dict(result)
    result["init_method"] = canonical
    existing = target.setdefault("results_by_init_method", {})
    if canonical not in existing:
        existing[canonical] = result
        return
    # Prefer the result with test_evaluation data, or the newer experiment_dir name.
    current = existing[canonical]
    if result.get("test_evaluation") and not current.get("test_evaluation"):
        existing[canonical] = result
    elif (result.get("experiment_dir") or "") > (current.get("experiment_dir") or ""):
        existing[canonical] = result


def save_metrics(metrics: Dict[str, Any], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return output_path


def load_metrics(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def metrics_dir(results_root: Path) -> Path:
    return Path(results_root) / "metrics"


def plots_dir(results_root: Path) -> Path:
    return Path(results_root) / "plots"


def summary_dir(results_root: Path) -> Path:
    return Path(results_root) / "summary"


def config_metrics_path(results_root: Path, config_name: str) -> Path:
    return metrics_dir(results_root) / f"{config_name}.json"


def build_summary(all_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "configs": sorted(all_metrics.keys()),
        "metrics_by_config": all_metrics,
    }
