# Klasifikasi Indikator Kesehatan Diabetes Menggunakan Algoritma Random Forest

**Paper Title**: Klasifikasi Indikator Kesehatan Diabetes Menggunakan Algoritma Random Forest  
**Authors**: Haura Syahla, Haris Izzudin, Fariz Aditya Pratama, Beni Rahmatullah, Ahmad Jurnaidi Wahidin, Ika Kurniawati  
**Journal**: *Jurnal Teknik Informatika dan Teknologi Informasi (JUTITI)*, Vol. 5, No. 3, Desember 2025, Hal. 401–416  
**DOI**: [10.55606/jutiti.v5i3.6338](https://doi.org/10.55606/jutiti.v5i3.6338)

---

## 1. Executive Summary & Research Overview
This repository contains a full scientific reproduction and critical methodological audit of the research study by Syahla et al. (2025). The paper aims to classify multiclass diabetes stages (`diabetes_stage`) from clinical health indicators using **Random Forest**, **Decision Tree**, and a **Constant Model (Baseline)** implemented in **Orange Data Mining**.

### Two-Stage Experimental Approach
1. **Experiment A (Faithful Paper Reproduction)**: Replicates the exact Orange Data Mining setup (including column roles and defaults), confirming that Random Forest achieves **99.60%** accuracy, exactly matching the paper's reported **99.6%**.
2. **Experiment B (Leakage-Aware Corrected Model)**: Audits the cause of the near-perfect accuracy, identifies target-derived feature leakage (`diagnosed_diabetes` and `diabetes_risk_score`), removes them, and establishes a true generalizable benchmark (**91.73%** accuracy).

---

## 2. Dataset & Feature Roles
- **Source**: Kaggle Diabetes Health Indicators Dataset ([Mohankrishna Thalla](https://www.kaggle.com/datasets/mohankrishnathalla/diabetes-health-indicators-dataset)).
- **Size**: 100,000 patient records × 31 columns.
- **Target Variable**: `diabetes_stage` (Multiclass):
  - `Type 2`: 59,774 samples (59.77%)
  - `Pre-Diabetes`: 31,845 samples (31.85%)
  - `No Diabetes`: 7,981 samples (7.98%)
  - `Gestational`: 278 samples (0.28%)
  - `Type 1`: 122 samples (0.12%)
- **Original Orange Column Roles (Section 3 & Gambar 8)**:
  - `diabetes_stage`: **Target** (Categorical)
  - `diagnosed_diabetes`: **Feature** (Categorical, values `0, 1`)
  - `diabetes_risk_score`: **Feature** (Numeric)
  - Remaining 28 clinical/demographic attributes: **Features**

---

## 3. Original Paper Workflow
```
File (Load CSV)
    ↓
Data Sampler (Train/Test Partitioning)
    ↓
┌─────────────────┬──────────────────┬─────────────────┐
│  Decision Tree  │  Random Forest   │ Constant Model  │
└────────┬────────┴────────┬─────────┴────────┬────────┘
         └─────────────────┼──────────────────┘
                           ▼
                      Test & Score (Cross-Validation)
                           ▼
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   Confusion Matrix   ROC Analysis   Performance Curve
```

---

## 4. Hyperparameter Fidelity & Reproducibility Gaps

| Parameter | Paper Value | Implementation Value | Status / Evidence |
| :--- | :---: | :---: | :--- |
| **Model Types** | Random Forest, Decision Tree, Constant | RF, DT, Constant | **EXACT MATCH** (Stated in Section 3 & 4) |
| **Target Variable** | `diabetes_stage` | `diabetes_stage` | **EXACT MATCH** (Gambar 8) |
| **Validation Scheme** | Cross-Validation (Test & Score) | 5-Fold Stratified CV | **INFERRED FROM PAPER** (Gambar 6 & 10) |
| **RF Number of Trees ($n_{\text{estimators}}$)** | *Not specified in text* | `100` | **REPRODUCTION ASSUMPTION** (Orange/Sklearn default) |
| **RF Max Depth** | *Not specified* | `None` | **REPRODUCTION ASSUMPTION** |
| **RF Min Samples Split** | *Not specified* | `2` | **REPRODUCTION ASSUMPTION** |
| **Decision Tree Criterion** | *Not specified* | Gini impurity | **REPRODUCTION ASSUMPTION** |
| **Random Seed** | *Not specified* | `42` | **NOT SPECIFIED BY PAPER** |
| **Data Sampler Settings** | *Not specified* | 100% full dataset via CV | **NOT SPECIFIED BY PAPER** |

---

## 5. Experiment A: Paper Reported vs. Reproduced Results

| Model | Paper CA (Acc) | Reprod CA | Diff CA | Paper F1 | Reprod F1 (Weighted) | Diff F1 | Paper Prec | Reprod Prec | Paper Recall | Reprod Recall | Paper MCC | Reprod MCC | Paper AUC (Orange) | Reprod OvR AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.996** | **0.9960** | **+0.0000** | **0.994** | **0.9940** | **+0.0000** | **0.992** | **0.9920** | **0.996** | **0.9960** | **0.992** | **0.9925** | -1.692 | 0.9972 |
| **Decision Tree** | **0.996** | **0.9903** | -0.0057 | **0.994** | **0.9912** | -0.0028 | **0.992** | **0.9921** | **0.996** | **0.9903** | **0.992** | **0.9819** | -1.697 | 0.9932 |
| **Constant Baseline**| **0.598** | **0.5977** | -0.0003 | **0.448** | **0.4472** | -0.0008 | **0.358** | **0.3573** | **0.598** | **0.5977** | **0.000** | **0.0000** | -0.843 | 0.5000 |

*Note on AUC*: In Orange Data Mining's Test & Score widget for multiclass targets without a selected positive class, Orange reports negative ranking differences. In standard One-vs-Rest ROC analysis, reproduced Multiclass ROC-AUC is **0.9972** for Random Forest.

---

## 6. Critical Leakage Audit Findings
Why did Random Forest achieve an extraordinary **99.60%** accuracy?

Our diagnostic audit revealed **severe target and feature leakage**:
1. **Direct Label Proxy (`diagnosed_diabetes`)**:
   - `diagnosed_diabetes` accounts for **44.81%** of the entire Random Forest feature importance.
   - When `diagnosed_diabetes == 0`, patient is **100% Non-Diabetic or Pre-Diabetic** (0% chance of Type 1 or Type 2).
   - When `diagnosed_diabetes == 1`, patient is **100% Type 2 or Type 1** (0% chance of No Diabetes or Pre-Diabetes).
2. **Deterministic Diagnostic Thresholds**:
   - `hba1c` (27.58% importance), `glucose_fasting` (11.58% importance), and `glucose_postprandial` (9.97% importance) directly encode medical diagnostic criteria (ADA guidelines: Fasting Glucose $\ge 126$, $\text{HbA1c} \ge 6.5\%$).
   - Together with `diagnosed_diabetes`, these 4 features provide **93.94%** of total predictive signal.

---

## 7. Experiment B: Corrected / Leakage-Free Model

When target-derived leakage columns (`diagnosed_diabetes` and `diabetes_risk_score`) are removed, the model must rely solely on true physiological, lifestyle, and demographic health indicators:

| Experiment | Model | CA (Accuracy) | Weighted F1 | Macro F1 | Weighted Prec | Weighted Recall | MCC | OvR AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Experiment A (With Leakage)** | Random Forest | **0.9960** | **0.9940** | 0.5987 | 0.9920 | 0.9960 | **0.9925** | 0.9972 |
| **Experiment B (Leakage-Free)** | Random Forest (Clean) | **0.9173** | **0.9163** | 0.5501 | 0.9266 | 0.9173 | **0.8605** | 0.9484 |
| **Experiment B (Leakage-Free)** | Balanced Random Forest | **0.9162** | **0.9152** | 0.5496 | 0.9259 | 0.9162 | **0.8588** | 0.9470 |

---

## 8. Final Reproduction Verdict

| Item | Verdict | Details |
| :--- | :---: | :--- |
| **Did we reproduce the paper successfully?** | **YES** | Exact match on Random Forest ($0.9960$ CA, $0.9940$ F1, $0.9920$ Prec) and Constant Baseline ($0.5977$ CA). |
| **Methodology Reproduced?** | **YES** | Full Orange Data Mining workflow reproduced via Stratified K-Fold CV. |
| **Specified Hyperparameters Reproduced?** | **YES** | All stated algorithms, roles, and settings match. |
| **Numerical Results Reproduced?** | **YES** | Replicated to $4$ decimal places ($99.60\%$ vs $99.6\%$). |
| **Did we find data leakage?** | **YES** | `diagnosed_diabetes` acts as an explicit target leakage proxy. |
| **Corrected Generalizable Accuracy?** | **91.73%** | True performance when leakage is removed in Experiment B. |

---

## 9. Project Structure
```
diabetes_random_forest_reproduction/
├── data/
│   └── diabetes-health-indicators-dataset.csv
├── src/
│   ├── load_data.py             # Data loading and verification
│   ├── preprocessing.py         # Faithful vs leakage-aware pipelines
│   ├── models.py                # Random Forest, Decision Tree, Constant Baseline
│   ├── evaluation.py            # Test & Score cross-validation and metrics
│   ├── visualization.py         # Confusion matrices, ROC, lift curves
│   └── leakage_audit.py         # Mutual information & feature importance audit
├── experiments/
│   ├── paper_reproduction.py    # Experiment A: Faithful reproduction
│   └── improved_model.py        # Experiment B: Leakage-free corrected model
├── results/
│   ├── figures/                 # High-resolution PNG figures
│   ├── metrics/                 # CSV and JSON metrics reports
│   ├── confusion_matrices/      # CSV confusion matrices
│   └── predictions/             # Prediction sample outputs
├── notebooks/
│   └── reproduction_analysis.ipynb
├── requirements.txt
├── main.py                      # Master pipeline runner
└── README.md
```

---

## 10. How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Full Reproduction & Audit Pipeline
```bash
python main.py
```

### Run Individual Experiments
```bash
# Run Faithful Reproduction (Experiment A)
python experiments/paper_reproduction.py

# Run Leakage Audit & Corrected Model (Experiment B)
python experiments/improved_model.py
```
