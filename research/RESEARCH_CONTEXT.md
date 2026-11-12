# Research Context — Diabetes Mellitus Prediction

## 1. Purpose of This Document
This document establishes the verified research baseline, dataset inventory, codebase architecture, and proposed research direction for our academic study on Diabetes Mellitus prediction. All existing baseline code, checkpoints, and web application components remain **untouched and preserved**.

---

## 2. Existing Base Research Paper

- **Title:** Classification of Health Indicators for Diabetes Mellitus Prediction Using a TabTransformer Model on Clinical Tabular Data
- **Authors:** Al Khaidar, Sri Kurnia (Master of Information Technology Study Program, Malikussaleh University, Aceh, Indonesia)
- **Journal:** *Journal of Computer Science Research (JoCoSiR)*, Vol. 2, No. 2, pp. 9–15
- **DOI:** [http://doi.org/10.65126/jocosir.v2i2.52](http://doi.org/10.65126/jocosir.v2i2.52)
- **Local Source:** [Classification_of_Health_Indicators_for_Diabetes_M (1).pdf](file:///c:/Users/hp/.gemini/antigravity-ide/scratch/model-training-diabeties-ds-/Classification_of_Health_Indicators_for_Diabetes_M%20(1).pdf)

### 2.1 Published Methodology (Base Paper)
- **Dataset:** Kaggle Clinical Diabetes Dataset (`diabetes_dataset.csv`), 100,000 patient records. *(Note: The paper's abstract mentions "more than 35" attributes and Table 2 mentions 26 predictor features [6 categorical + 20 numeric], whereas Section 3.2 and Figure 2 explicitly specify 9 primary predictors: 4 categorical + 5 numerical.)*
- **Target Variable:** `diagnosed_diabetes` (binary: `0` = Non-Diabetic, `1` = Diabetic). The paper also mentions `diabetes_stage` as a secondary multiclass variable, though Table 3 reports binary classification metrics for `diagnosed_diabetes`.
- **Architecture:** TabTransformer
  - Categorical embeddings: 32 dimensions per categorical feature
  - Transformer encoder blocks: 4 layers, 8 attention heads, feedforward dimension 128, GELU activation, dropout 0.1
  - Concatenation of flattened contextual categorical embeddings with normalized numerical features
  - MLP classifier: 2 hidden layers `[64, 32]`, ReLU activation, dropout 0.2, single logit output
- **Optimization & Loss:** `AdamW` (`lr=1e-3`, `weight_decay=1e-5` from Figure 4), `ReduceLROnPlateau` (`mode='min'`, `factor=0.5`, `patience=1` from Figure 4), `BCEWithLogitsLoss`, classification threshold `0.50`.
- **Undocumented in Base Paper:** Exact train/val/test split ratios, batch size, total epoch count / early stopping rule, and random seed.

### 2.2 Base Paper Reported Benchmark Results
| Metric | Class 0 (Non-Diabetic) | Class 1 (Diabetic) | Overall |
| :--- | :---: | :---: | :---: |
| **Accuracy** | — | — | **82.55%** |
| **ROC-AUC** | — | — | **0.9009** |
| **Precision** | 0.7716 | **0.8637** | — |
| **Recall** | 0.8006 | **0.8420** | — |
| **F1-score** | 0.7858 | **0.8527** | — |

---

## 3. Verified Local Repository Inspection & Baseline Results

### 3.1 Workspace Overview
The repository contains four distinct sub-components:
1. **`diabetes-tabtransformer/` (Primary Baseline):** PyTorch TabTransformer reproduction of Al Khaidar & Sri Kurnia ([main.py](file:///c:/Users/hp/.gemini/antigravity-ide/scratch/model-training-diabeties-ds-/diabetes-tabtransformer/main.py)).
2. **`diabetes_random_forest_reproduction/` (Sibling Audit):** Reproduction and leakage audit of Syahla et al. (2025, JUTITI, DOI: `10.55606/jutiti.v5i3.6338`) on `diabetes_stage` ([README.md](file:///c:/Users/hp/.gemini/antigravity-ide/scratch/model-training-diabeties-ds-/diabetes_random_forest_reproduction/README.md)).
3. **Root Classical Scripts (`step4_preprocessing.py`–`step9_model_comparison.py`):** 60/40 split Random Forest (91.97% acc) and Logistic Regression (86.00% acc) pipelines using 47 OHE features (including `diabetes_risk_score`).
4. **Web Deployment (`app.py` & `frontend/`):** Flask REST API + Angular 22 clinical dashboard serving the Random Forest and Logistic Regression models. *Per Research Rule 13, frontend/backend remain untouched.*

### 3.2 Actual Dataset Inspection (`diabetes-tabtransformer/data/diabetes_dataset.csv`)
Direct programmatic inspection of `diabetes_dataset.csv` confirms:
- **Dimensions:** `100,000` rows × `31` columns (`patient_id` is **not** present in the CSV).
- **Data Integrity:** `0` missing values (`nulls = 0` across all 31 columns); `0` duplicate rows (also documented in [ML_Lab_Log.md](file:///c:/Users/hp/.gemini/antigravity-ide/scratch/model-training-diabeties-ds-/ML_Lab_Log.md)).
- **Binary Target (`diagnosed_diabetes`):**
  - `1` (Diabetic): `59,998` (`60.00%`)
  - `0` (Non-Diabetic): `40,002` (`40.00%`)
- **Multiclass Variable (`diabetes_stage`):**
  - `Type 2`: `59,774` (`59.77%`) — 100% have `diagnosed_diabetes = 1`
  - `Pre-Diabetes`: `31,845` (`31.85%`) — 100% have `diagnosed_diabetes = 0`
  - `No Diabetes`: `7,981` (`7.98%`) — 100% have `diagnosed_diabetes = 0`
  - `Gestational`: `278` (`0.28%`) — `120` negative, `158` positive
  - `Type 1`: `122` (`0.12%`) — `56` negative, `66` positive
- **Complete Feature Inventory (31 Columns):**
  - **28 Legitimate Predictor Columns:**
    - *6 Object Categoricals:* `gender` (3), `ethnicity` (5), `education_level` (4), `income_level` (5), `employment_status` (4), `smoking_status` (3)
    - *3 Binary Clinical History Flags:* `family_history_diabetes`, `hypertension_history`, `cardiovascular_history`
    - *19 Continuous/Discrete Numericals:* `age`, `alcohol_consumption_per_week`, `physical_activity_minutes_per_week`, `diet_score`, `sleep_hours_per_day`, `screen_time_hours_per_day`, `bmi`, `waist_to_hip_ratio`, `systolic_bp`, `diastolic_bp`, `heart_rate`, `cholesterol_total`, `hdl_cholesterol`, `ldl_cholesterol`, `triglycerides`, `glucose_fasting`, `glucose_postprandial`, `insulin_level`, `hba1c`
  - **1 Derived Composite Risk Index (Excluded as Leakage in TabTransformer):** `diabetes_risk_score` (range `2.7`–`67.2`)
  - **2 Target Variables:** `diabetes_stage` (excluded to prevent target leakage when predicting binary diagnosis) and `diagnosed_diabetes` (binary target)

### 3.3 Local TabTransformer Implementation Details (`diabetes-tabtransformer/`)
- **Features Used (9 of 28 predictors):**
  - Categorical (4): `['gender', 'smoking_status', 'family_history_diabetes', 'employment_status']` with cardinalities `[3, 3, 2, 4]`.
  - Numerical (5): `['age', 'bmi', 'glucose_fasting', 'cholesterol_total', 'hba1c']`.
  - *Note:* 19 legitimate predictors (including `glucose_postprandial`, `insulin_level`, `systolic_bp`, `diastolic_bp`, `waist_to_hip_ratio`, `ldl_cholesterol`, `hdl_cholesterol`, `triglycerides`, lifestyle and comorbidity flags) are currently unused by the 9-feature baseline.
- **Data Splitting:** Stratified 70% Train (`69,999`), 15% Val (`15,001`), 15% Test (`15,000`), `random_state=42`.
- **Preprocessing (`src/preprocessing.py`):** `LabelEncoder` and `StandardScaler` are fitted strictly on the training split. *(Methodological note for future extensions: the fallback `fillna` lines 63–71 in `src/preprocessing.py` run prior to `train_test_split`; while harmless on the clean dataset with 0 nulls, any future missing-data experiments must fit imputers strictly inside the training split to avoid leakage.)*
- **Training Configuration:** `batch_size=256`, `eval_batch_size=512`, `num_epochs=15`, `AdamW(lr=1e-3, weight_decay=1e-5)`, `ReduceLROnPlateau(mode='min', factor=0.5, patience=1)`, `BCEWithLogitsLoss`, `seed=42`, best checkpoint retained by minimum validation loss (`Epoch 15`, `val_loss=0.2149`).

### 3.4 Independently Verified Local Baseline Results
Re-evaluating the saved checkpoint ([best_tabtransformer.pt](file:///c:/Users/hp/.gemini/antigravity-ide/scratch/model-training-diabeties-ds-/diabetes-tabtransformer/outputs/models/best_tabtransformer.pt)) on the `15,000`-record test split confirms exact agreement with [test_metrics.json](file:///c:/Users/hp/.gemini/antigravity-ide/scratch/model-training-diabeties-ds-/diabetes-tabtransformer/outputs/metrics/test_metrics.json):

| Metric | Paper Reported | Local Reproduced (Verified) | Delta |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | 82.55% | **91.93%** (`0.9193`) | +9.38% |
| **ROC-AUC** | 0.9009 | **0.9418** | +0.0409 |
| **Class 0 Precision** | 0.7716 | **0.8344** | +0.0628 |
| **Class 0 Recall** | 0.8006 | **0.9960** | +0.1954 |
| **Class 0 F1-score** | 0.7858 | **0.9081** | +0.1223 |
| **Class 1 Precision** | 0.8637 | **0.9969** | +0.1332 |
| **Class 1 Recall** | 0.8420 | **0.8682** | +0.0262 |
| **Class 1 F1-score** | 0.8527 | **0.9281** | +0.0754 |
| **Macro F1 / Weighted F1** | — | **0.9181 / 0.9201** | — |

**Test Confusion Matrix (`N = 15,000`):**
- True Negatives (`TN`): `5,976` | False Positives (`FP`): `24`
- False Negatives (`FN`): `1,186` | True Positives (`TP`): `7,814`

**Key Diagnostic Observation:** At the default probability threshold (`0.50`), the reproduced TabTransformer achieves near-zero false positives (`FP = 24`, Class 1 Precision `0.9969`) but produces **`1,186` false negatives** (`13.18%` of diabetic cases missed). Furthermore, the baseline provides no probability calibration metrics (ECE, Brier score), no post-hoc explainability (SHAP), and no evaluation under missing or noisy clinical inputs.

---

## 4. Proposed Research Direction
- **Working Title:** *Robustness-Aware Explainable TabTransformer for Diabetes Prediction*
- **Core Objective:** Investigate and mitigate how missing clinical measurements and random measurement noise degrade (a) predictive accuracy/discrimination, (b) probability calibration and clinical decision utility, and (c) global and patient-level SHAP explanation stability in TabTransformer, while evaluating leakage-safe consensus feature selection over the full 28-feature clinical pool.
- **Preservation Guarantee:** All baseline files in `diabetes-tabtransformer/` and baseline metrics (`91.93%` acc, `0.9418` AUC) remain strictly preserved as the clean reference benchmark.
- **Experimental Implementation & Observed Results:** Completed in isolated [diabetes-tabtransformer/research_extensions/](file:///c:/Users/hp/.gemini/antigravity-ide/scratch/model-training-diabeties-ds-/diabetes-tabtransformer/research_extensions) and documented in [EXPERIMENTAL_RESULTS.md](file:///c:/Users/hp/.gemini/antigravity-ide/scratch/model-training-diabeties-ds-/research/EXPERIMENTAL_RESULTS.md).

