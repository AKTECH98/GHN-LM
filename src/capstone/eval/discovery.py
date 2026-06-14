"""Experiment and TensorBoard discovery for evaluation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional

from capstone.eval.schema import INIT_METHODS, normalize_init_method
from capstone.paths import EXPERIMENT_DIR, TENSORLOG_DIRS

EXPERIMENT_ROOT = EXPERIMENT_DIR
TENSORBOARD_DIRS = TENSORLOG_DIRS

# Job ID prefix patterns -> init method (longest match first).
JOB_ID_PREFIX_RULES = (
    (re.compile(r"^GHN-I_64-2_", re.I), "ghn"),
    (re.compile(r"^GHN-I_16-2_", re.I), "ghn"),
    (re.compile(r"^GHN-I[_-]", re.I), "ghn"),
    (re.compile(r"^GHN-T[_-]", re.I), "ghn"),
    (re.compile(r"^GHN_init[_-]", re.I), "ghn"),
    (re.compile(r"^GHN[_-]", re.I), "ghn"),
    (re.compile(r"^Xavier[_-]", re.I), "xavier"),
    (re.compile(r"^He[_-]", re.I), "he"),
    (re.compile(r"^Benchmark_16__", re.I), "default"),
    (re.compile(r"^Benchmark[_-]", re.I), "default"),
    (re.compile(r"^Mini_GPT[_-]", re.I), "default"),
)


def config_suffix(config_name: str) -> str:
    """Return the suffix after 'benchmark_' for directory matching."""
    name = config_name.lower()
    if name.startswith("benchmark_"):
        return name.replace("benchmark_", "", 1)
    return name


def infer_init_method_from_job_id(job_id: str) -> str:
    for pattern, init_method in JOB_ID_PREFIX_RULES:
        if pattern.match(job_id):
            return init_method
    return "default"


def read_experiment_init_method(experiment_dir: Path) -> str:
    config_path = experiment_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        init_method = config.get("init_method")
        if init_method:
            return normalize_init_method(init_method)
        job_id = config.get("job_id", experiment_dir.name)
        return infer_init_method_from_job_id(job_id)
    return infer_init_method_from_job_id(experiment_dir.name)


def experiment_matches_config(experiment_dir: Path, config_name: str) -> bool:
    suffix = config_suffix(config_name)
    name = experiment_dir.name.lower()
    return suffix in name or config_name.lower() in name


def find_experiment_dirs_by_init_method(
    config_name: str,
    experiment_root: Path = EXPERIMENT_ROOT,
) -> Dict[str, Path]:
    """
    Find the most recent experiment directory per init method for a config.

    Returns:
        Mapping of canonical init_method -> experiment directory path.
    """
    if not experiment_root.exists():
        return {}

    candidates: Dict[str, Path] = {}
    for experiment_dir in experiment_root.iterdir():
        if not experiment_dir.is_dir():
            continue
        if not experiment_matches_config(experiment_dir, config_name):
            continue

        init_method = read_experiment_init_method(experiment_dir)
        if init_method not in INIT_METHODS:
            init_method = normalize_init_method(init_method)
        if init_method not in INIT_METHODS:
            continue

        existing = candidates.get(init_method)
        if existing is None or experiment_dir.stat().st_mtime > existing.stat().st_mtime:
            candidates[init_method] = experiment_dir

    return candidates


def find_tensorboard_dir(job_id: str) -> Optional[Path]:
    """Find TensorBoard log directory for a job_id."""
    for base in TENSORBOARD_DIRS:
        log_dir = base / job_id
        if log_dir.exists():
            return log_dir
    return None
