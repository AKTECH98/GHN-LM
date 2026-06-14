"""Repository path constants."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs" / "benchmarks"
RESULTS_DIR = REPO_ROOT / "Results"
EXPERIMENT_DIR = REPO_ROOT / "Experiment"
TENSORLOG_DIRS = (REPO_ROOT / "tensor_log", REPO_ROOT / "Final_tensors")
LEGACY_EVALUATIONS_DIR = REPO_ROOT / "Evaluations"
LEGACY_REPORT_DIR = REPO_ROOT / "Report"
LEGACY_NEWRESULTS_DIR = REPO_ROOT / "NewResults"
DATA_DIR = REPO_ROOT / "data"
GHN_MODELS_DIR = REPO_ROOT / "GHN_Models"
