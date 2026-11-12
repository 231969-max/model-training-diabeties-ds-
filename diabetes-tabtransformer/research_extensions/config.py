"""Shared configuration and constants for isolated TabTransformer research extensions."""
import os

# Paths (relative to diabetes-tabtransformer/ or resolved dynamically)
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "diabetes_dataset.csv")
BASELINE_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "outputs", "models", "best_tabtransformer.pt")
EXT_OUTPUT_DIR = os.path.join(PACKAGE_DIR, "outputs")
EXT_FIGURES_DIR = os.path.join(EXT_OUTPUT_DIR, "figures")
EXT_METRICS_DIR = os.path.join(EXT_OUTPUT_DIR, "metrics")
EXT_MODELS_DIR = os.path.join(EXT_OUTPUT_DIR, "models")

# Reproducibility & Split Constants (must match baseline exactly)
SEED = 42
CORRUPTION_SEEDS = [42, 101, 202]
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Feature Definitions
PAPER_CAT_COLS = ["gender", "smoking_status", "family_history_diabetes", "employment_status"]
PAPER_NUM_COLS = ["age", "bmi", "glucose_fasting", "cholesterol_total", "hba1c"]

ALL_CAT_COLS = [
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "employment_status",
    "smoking_status",
    "family_history_diabetes",
    "hypertension_history",
    "cardiovascular_history",
]

ALL_NUM_COLS = [
    "age",
    "alcohol_consumption_per_week",
    "physical_activity_minutes_per_week",
    "diet_score",
    "sleep_hours_per_day",
    "screen_time_hours_per_day",
    "bmi",
    "waist_to_hip_ratio",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "cholesterol_total",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "triglycerides",
    "glucose_fasting",
    "glucose_postprandial",
    "insulin_level",
    "hba1c",
]

TARGET_BINARY = "diagnosed_diabetes"
LEAKAGE_COLS = ["diabetes_risk_score", "diabetes_stage"]

# Corruption Grids
MCAR_RATES = [0.05, 0.10, 0.20, 0.30]
NOISE_SIGMAS = [0.05, 0.10, 0.25, 0.50]
TARGETED_DROP_SETS = {
    "drop_glucose_fasting": ["glucose_fasting"],
    "drop_hba1c": ["hba1c"],
    "drop_glucose_and_hba1c": ["glucose_fasting", "hba1c"],
}
COMBINED_SCENARIOS = [
    {"name": "mcar_10_noise_0.10", "mcar_rate": 0.10, "noise_sigma": 0.10},
    {"name": "mcar_20_noise_0.25", "mcar_rate": 0.20, "noise_sigma": 0.25},
]

# SHAP Evaluation Config
SHAP_SAMPLE_SIZE = 500
SHAP_BACKGROUND_SIZE = 50
SHAP_MAX_EVALS = 180
SHAP_TOP_K_LIST = [3, 5]

