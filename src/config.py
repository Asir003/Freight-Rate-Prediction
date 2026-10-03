from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RANDOM_SEED = 42

TRAIN_PATH = PROJECT_ROOT / "train-test.csv"
VALIDATION_PATH = PROJECT_ROOT / "validation.csv"
TEMPLATE_PATH = PROJECT_ROOT / "validation-predictions-template.csv"
DECEMBER_INPUT_PATH = PROJECT_ROOT / "december-chart-inputs.csv"

VALIDATION_PREDICTIONS_PATH = PROJECT_ROOT / "validation_predictions.csv"
DECEMBER_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "december_chart_inputs.csv"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODEL_PATH = OUTPUTS_DIR / "final_pipeline.joblib"
METRICS_PATH = OUTPUTS_DIR / "experiment_metrics.csv"
FINAL_METRICS_PATH = OUTPUTS_DIR / "final_metrics.json"

TARGET_COL = "posted_rate"
ID_COL = "load_id"

TIME_VALID_START = "2025-10-01"

RATE_FLOOR = 1.0
RATE_CEILING = 50_000.0

EXPECTED_VAL_ROWS = 12_000
DECEMBER_COLUMNS = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
    "predicted_rate",
]
