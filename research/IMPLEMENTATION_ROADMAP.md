# Implementation Roadmap

> **Current Status:** Stages 1–7 (Scaffolding, Baseline Audit, Consensus Feature Selection, Ablation Models, SHAP Stability, Multi-Seed Validation, and Forensic Audit) **COMPLETED**. Full observed results, negative findings, and confidence intervals are documented in [EXPERIMENTAL_RESULTS.md](file:///c:/Users/hp/.gemini/antigravity-ide/scratch/model-training-diabeties-ds-/research/EXPERIMENTAL_RESULTS.md).

---

## Stage 1 — Environment Verification & Modular Scaffolding [COMPLETED]
- **Dependencies:** Installed `shap==0.54.1` (`slicer==0.0.8`) without modifying CPU-only `torch==2.14.1+cpu`, `scikit-learn==1.9.1`, `pandas==3.0.6`, `numpy==2.5.3`, or `scipy==1.18.1`.
- **Scaffolding:** Created isolated module [diabetes-tabtransformer/research_extensions/](file:///c:/Users/hp/.gemini/antigravity-ide/scratch/model-training-diabeties-ds-/diabetes-tabtransformer/research_extensions) leaving `src/`, `main.py`, `outputs/`, `app.py`, `frontend/`, and `diabetes_random_forest_reproduction/` untouched.

## Stage 2 — Baseline Audit: Calibration, Thresholds & Corruptions (`A0`, `A1`) [COMPLETED]
- Executed via [run_baseline_audit.py](file:///c:/Users/hp/.gemini/antigravity-ide/scratch/model-training-diabeties-ds-/diabetes-tabtransformer/research_extensions/run_baseline_audit.py).
- Verified baseline parity (`91.93%` acc, `0.9418` ROC-AUC, `0.0088` ECE, `0.0670` Brier).
- Audited Validation-fitted Temperature, Platt, and Isotonic calibration; discovered Isotonic saturation collapse under MCAR/noise shift (`0.9189 → 0.8489` AUC at MCAR 20%) and catastrophic baseline failure on `drop_hba1c` (`60.00%` acc, `0.3837` ECE).

## Stage 3 — Baseline SHAP & Explanation-Stability Benchmark [COMPLETED]
- Computed `shap.PermutationExplainer` values (`n=500` stratified test patients, `bg=25`, `max_evals=110`) across clean and 7 corrupted regimes.
- Observed baseline SHAP cosine collapse (`0.0608`) under `drop_hba1c` due to `6.52%` median imputation.

## Stage 4 — Consensus Feature Selection Over All 28 Clean Predictors (`A4`) [COMPLETED]
- Executed strictly on `69,999`-row `Train` split via [consensus_fs.py](file:///c:/Users/hp/.gemini/antigravity-ide/scratch/model-training-diabeties-ds-/diabetes-tabtransformer/research_extensions/consensus_fs.py) (ANOVA, MI, RF, L1-LR).
- Identified 8 consensus predictors (`family_history_diabetes`, `hba1c`, `glucose_fasting`, `age`, `physical_activity_minutes_per_week`, `bmi`, `glucose_postprandial`, `systolic_bp`).

## Stage 5 — Robustness-Aware TabTransformer, Classical Baselines & SHAP Stability (`A2`–`A5`) [COMPLETED]
- Executed via [run_ablation_models.py](file:///c:/Users/hp/.gemini/antigravity-ide/scratch/model-training-diabeties-ds-/diabetes-tabtransformer/research_extensions/run_ablation_models.py) and [run_ablation_shap.py](file:///c:/Users/hp/.gemini/antigravity-ide/scratch/model-training-diabeties-ds-/diabetes-tabtransformer/research_extensions/run_ablation_shap.py).
- Proved `A2` (mask only, clean training) is insufficient, whereas `A3` and especially `A4` (Consensus-8 + Missingness Mask + Stochastic Masking/Noise + Temperature Calibration) deliver large, statistically significant accuracy, AUC, ECE, and SHAP-consistency gains across MCAR, targeted lab-drop, and combined corruptions.

## Stage 6 — Verification & Integrity Audit [COMPLETED]
- Confirmed zero data leakage (imputers/scalers/encoders/selectors on `Train`; calibrators/PRC thresholds on `Val`; `Test` untouched).
- Confirmed original baseline files and `outputs/models/best_tabtransformer.pt` remain 100% unmodified via `git status`.

## Stage 7 — Multi-Seed Reproducibility & Forensic Audit [COMPLETED]
- Executed [audit_forensics.py](file:///c:/Users/hp/.gemini/antigravity-ide/scratch/model-training-diabeties-ds-/diabetes-tabtransformer/research_extensions/audit_forensics.py) and [run_multiseed_validation.py](file:///c:/Users/hp/.gemini/antigravity-ide/scratch/model-training-diabeties-ds-/diabetes-tabtransformer/research_extensions/run_multiseed_validation.py) across seeds `42, 101, 202`.
- Audited near-deterministic ADA clinical cutoffs (92.11% match rate) and root cause of baseline median imputation collapse.
- Computed Mean $\pm$ SD across all 11 evaluation regimes, McNemar's tests ($p < 10^{-28}$ under missingness), and 1,000-bootstrap 95% confidence intervals.


