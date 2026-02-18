# src/utils/paths.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MLRUNS_DIR = RESULTS_DIR / "mlruns"
CONFIGS_DIR = PROJECT_ROOT / "configs"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENT_NAME = "Sentiment"

def get_figure_path(subdir: str, filename: str) -> Path:
    path = FIGURES_DIR / subdir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def get_data_path(subdir: str, filename: str) -> Path:
    path = DATA_DIR / subdir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def get_results_path(subdir: str, filename: str) -> Path:
    path = RESULTS_DIR / subdir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def get_mlflow_uri() -> str:
    """Return absolute file URI for MLflow tracking."""
    return f"file://{MLRUNS_DIR.absolute()}"