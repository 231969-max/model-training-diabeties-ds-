# Clinical Diabetes Risk Prediction System & Automated CI/CD Pipeline

[![CI/CD Pipeline](https://github.com/231969-max/model-training-diabeties-ds-/actions/workflows/deploy.yml/badge.svg)](https://github.com/231969-max/model-training-diabeties-ds-/actions/workflows/deploy.yml)
[![Live GitHub Pages](https://img.shields.io/badge/Live%20Website-GitHub%20Pages-brightgreen)](https://231969-max.github.io/model-training-diabeties-ds-/)
[![Frontend](https://img.shields.io/badge/Frontend-Angular%20Standalone-dd0031)](frontend)
[![Backend](https://img.shields.io/badge/Backend-Flask%20REST%20API-000000)](app.py)

**Live Deployed Application**: [https://231969-max.github.io/model-training-diabeties-ds-/](https://231969-max.github.io/model-training-diabeties-ds-/)

---

## 🚀 Lab Requirement: CI/CD Automated Deployment

### Automated Workflow Pipeline Architecture:
$$\text{Code Modification} \longrightarrow \text{Git Commit \& Push} \longrightarrow \text{GitHub Actions CI (Test \& Build)} \longrightarrow \text{Automatic CD (GitHub Pages)} \longrightarrow \text{Live Website Updated}$$

1. **Frontend**: Standalone **Angular** clinical dashboard (`frontend/`) with real-time probability meters, clinical preset profiles, and dual-model evaluator (Random Forest & Logistic Regression).
2. **Backend**: Python **Flask REST API** (`app.py`) providing `/api/predict`, `/api/health`, and `/api/models` with CORS integration.
3. **CI Validation**: GitHub Actions automatically triggers on push to `main`:
   - Runs Flask backend tests (`python test_app.py`)
   - Installs frontend packages (`npm ci`)
   - Runs Angular unit tests (`npm run test:ci` via Vitest)
   - Builds optimized production bundles with base-href (`npm run build:gh-pages`)
4. **CD Deployment**: Automatically deploys the production bundle to GitHub Pages (`actions/deploy-pages@v4` & `peaceiris/actions-gh-pages@v4`).

### 🔬 Live Lab Demonstration Step-by-Step:
1. Open VS Code in this repository.
2. Make a small visible change in any frontend file:
   - In `frontend/src/app/app.html`, change the banner title or subtitle text.
   - Or in `frontend/src/app/app.css`, change an accent color (e.g. `--primary` or `--accent-emerald`).
   - Or in `frontend/src/app/app.ts`, change `pipelineVersion` or `labVerificationBadge`.
3. Commit and push the change to GitHub:
   ```bash
   git add .
   git commit -m "feat: lab live demo code update"
   git push origin main
   ```
4. Open the **Actions** tab on GitHub:
   - Observe **CI - Validation & Build** succeed.
   - Observe **CD - Deployment to GitHub Pages** deploy automatically.
5. Refresh the live site: [https://231969-max.github.io/model-training-diabeties-ds-/](https://231969-max.github.io/model-training-diabeties-ds-/) and verify the change is live!

---

## 1. Research Objective
This repository contains a full, reproducible implementation of the research paper **"Classification of Health Indicators for Diabetes Mellitus Prediction Using a TabTransformer Model on Clinical Tabular Data"**.

The objective is to establish an artificial intelligence-based clinical decision support model capable of accurately predicting diabetes mellitus status (`diagnosed_diabetes`: `0 = non-diabetic`, `1 = diabetic`) using high-dimensional heterogeneous clinical tabular data. The TabTransformer architecture applies self-attention mechanisms over categorical feature embeddings while fusing normalized continuous indicators into a multi-layer perceptron (MLP) classification network.

---

## 2. Dataset Overview
- **Source**: Kaggle Clinical Tabular Diabetes Dataset.
- **Profiles**: 100,000 unique patient records.
- **Raw Dimension**: 100,000 rows × 31 columns.
- **Target Variable**: `diagnosed_diabetes` (Binary: 0 = Non-Diabetic [40.0%], 1 = Diabetic [60.0%]).
- **Secondary Target**: `diabetes_stage` (Multi-class: Type 2, Pre-Diabetes, No Diabetes, Gestational, Type 1).

---

## 3. Features & Clinical Indicators
The TabTransformer architecture consumes 9 primary clinical indicators:

### Categorical Features (4)
1. `gender` (Male, Female, Other)
2. `smoking_status` (Never, Former, Current)
3. `family_history_diabetes` (0 = No, 1 = Yes)
4. `employment_status` (Employed, Unemployed, Retired, Student)

### Numerical Features (5)
1. `age` (Years)
2. `bmi` (Body Mass Index, kg/m²)
3. `glucose_fasting` (Fasting blood glucose level, mg/dL)
4. `cholesterol_total` (Total serum cholesterol, mg/dL)
5. `hba1c` (Glycated hemoglobin percentage, %)

---

## 4. Preprocessing Pipeline
1. **Leakage Removal**: Explicitly drops `diabetes_risk_score` (calculated diagnostic index) and `diabetes_stage` (disease severity outcome) prior to modeling.
2. **Missing-Value Handling**: Median imputation for numerical attributes, mode imputation for categorical attributes.
3. **Stratified Splitting**: 70% Training (69,999 records), 15% Validation (15,001 records), 15% Testing (15,000 records).
4. **Numerical Standardization**: `StandardScaler` fitted strictly on training data and applied to validation/testing data.
5. **Categorical Integer Encoding**: `LabelEncoder` fitted strictly on training data.

---

## 5. TabTransformer Architecture Pipeline
```
Input CSV (100,000 records)
    ↓
Data Cleaning & Leakage Removal
    ↓
Categorical & Numerical Separation
    ↓
Categorical Embeddings (4 × 32-dim vectors)
    ↓
4-Layer Multi-Head Self-Attention Transformer Blocks (8 heads, GELU, FF=128, Dropout=0.1)
    ↓
Contextual Categorical Representation (Flattened to 128-dim)
    ↓
Concatenation with 5 Normalized Numerical Features (Total = 133-dim vector)
    ↓
2-Layer MLP Classifier [133 → 64 → 32 → 1] (ReLU, Dropout=0.2)
    ↓
Output Logits (BCEWithLogitsLoss / Sigmoid ≥ 0.5)
```

---

## 6. Comprehensive Hyperparameter Fidelity Table

| Hyperparameter | Paper Value | Implementation Value | Status | Evidence / Reference in Paper |
| :--- | :---: | :---: | :---: | :--- |
| **Categorical Embedding Dimension** | `32` | `32` | **EXACT MATCH** | Section 3.2: *"categorical features with a 32-dimensional embedding"* |
| **Transformer Encoder Layers** | `4` | `4` | **EXACT MATCH** | Section 3.2: *"transformer block consists of 4 layers"* |
| **Attention Heads ($n_{\text{head}}$)** | `8` | `8` | **EXACT MATCH** | Section 3.2: *"with 8 attention heads"* |
| **Feedforward Dimension ($d_{\text{ff}}$)** | `128` | `128` | **EXACT MATCH** | Section 3.2: *"a feedforward dimension of 128"* |
| **Transformer Activation** | `GELU` | `GELU` | **EXACT MATCH** | Section 3.2: *"GELU activation"* |
| **Transformer Dropout** | `0.1` | `0.1` | **EXACT MATCH** | Section 3.2: *"and a dropout of 0.1"* |
| **MLP Hidden Dimensions** | `[64, 32]` | `[64, 32]` | **EXACT MATCH** | Section 3.2: *"two hidden layers [64, 32]"* |
| **MLP Activation** | `ReLU` | `ReLU` | **EXACT MATCH** | Section 3.2: *"ReLU activation"* |
| **MLP Dropout** | `0.2` | `0.2` | **EXACT MATCH** | Section 3.2: *"dropout 0.2"* |
| **Loss Function** | `BCEWithLogitsLoss` | `BCEWithLogitsLoss` | **EXACT MATCH** | Section 4.4 & Figure 4: `criterion = nn.BCEWithLogitsLoss()` |
| **Optimizer Type** | `AdamW` | `AdamW` | **EXACT MATCH** | Section 4.4 & Figure 4: `torch.optim.AdamW(...)` |
| **Learning Rate ($\alpha$)** | `1e-3` | `1e-3` | **INFERRED FROM PAPER** | Figure 4 code snippet: `lr=1e-3` |
| **Optimizer Weight Decay** | `1e-5` | `1e-5` | **INFERRED FROM PAPER** | Figure 4 code snippet: `weight_decay=1e-5` |
| **LR Scheduler Type** | `ReduceLROnPlateau` | `ReduceLROnPlateau` | **EXACT MATCH** | Section 4.4 & Figure 4: `torch.optim.lr_scheduler.ReduceLROnPlateau` |
| **Scheduler Factor** | `0.5` | `0.5` | **INFERRED FROM PAPER** | Figure 4 code snippet: `factor=0.5` |
| **Scheduler Patience** | `1` | `1` | **INFERRED FROM PAPER** | Figure 4 code snippet: `patience=1` |
| **Scheduler Mode** | `'min'` | `'min'` | **INFERRED FROM PAPER** | Figure 4 code snippet: `mode='min'` |
| **Classification Threshold** | `0.5` | `0.5` | **EXACT MATCH** | Section 4.4 & Figure 4: `(probs >= 0.5).astype(int)` |
| **Batch Size** | *Not stated* | `256` (Train), `512` (Eval) | **NOT SPECIFIED BY PAPER** | Assumed standard mini-batch size |
| **Number of Epochs** | *Not stated* | `15` | **NOT SPECIFIED BY PAPER** | Assumed sufficient epochs for loss stabilization |
| **Train/Val/Test Split Ratio**| *Not stated* | `70% / 15% / 15%` | **NOT SPECIFIED BY PAPER** | Standard stratified cross-validation partition |
| **Random Seed** | *Not stated* | `42` | **NOT SPECIFIED BY PAPER** | Standard reproducibility seed across PyTorch/NumPy |
| **Weight Initialization** | *Not stated* | PyTorch Default | **NOT SPECIFIED BY PAPER** | Standard PyTorch default initialization |
| **Minimum Learning Rate** | *Not stated* | `0.0` (PyTorch default) | **NOT SPECIFIED BY PAPER** | Default floor in `ReduceLROnPlateau` |
| **Early Stopping** | *Not stated* | Best Val Checkpoint | **NOT SPECIFIED BY PAPER** | Best checkpoint retained on minimum validation loss |

---

## 7. Experimental Results & Paper Comparison

| Metric / Target | Paper Reported Result | Reproduced Result (Test Set) | Difference | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | **82.55%** | **91.93%** | **+9.38%** | Exceeded |
| **Class 0 (Non-Diabetic) Precision** | 0.7716 | **0.8344** | +0.0628 | Exceeded |
| **Class 0 (Non-Diabetic) Recall** | 0.8006 | **0.9960** | +0.1954 | Exceeded |
| **Class 0 (Non-Diabetic) F1-score** | 0.7858 | **0.9081** | +0.1223 | Exceeded |
| **Class 1 (Diabetic) Precision** | 0.8637 | **0.9969** | +0.1332 | Exceeded |
| **Class 1 (Diabetic) Recall** | 0.8420 | **0.8682** | +0.0262 | Exceeded |
| **Class 1 (Diabetic) F1-score** | **0.8527** | **0.9281** | **+0.0754** | Exceeded |
| **ROC-AUC Score** | **0.9009** | **0.9418** | **+0.0409** | Exceeded |

### Test Classification Report
```
                  precision    recall  f1-score   support

Non-Diabetic (0)     0.8344    0.9960    0.9081      6000
    Diabetic (1)     0.9969    0.8682    0.9281      9000

        accuracy                         0.9193     15000
       macro avg     0.9157    0.9321    0.9181     15000
    weighted avg     0.9319    0.9193    0.9201     15000

ROC-AUC Score: 0.9418
```

---

## 8. Exact-Result Discrepancy Investigation
While methodology and architecture are replicated with 100% fidelity, the reproduced numerical metrics show higher performance (91.93% accuracy vs 82.55%). The root causes are:

1. **Number of Training Epochs & Early Stopping (Primary Factor)**:
   - At **Epoch 1**, the validation accuracy is **81.71%** (extremely close to the paper's reported **82.55%**).
   - As training continues through Epochs 5–15 with `ReduceLROnPlateau`, the TabTransformer optimizes further, achieving **91.93%**.
2. **Train/Val/Test Split Ratio (Secondary Factor)**:
   - The paper never specifies its split proportion or cross-validation fold size. A different split alters test class proportions and evaluation metrics.
3. **Feature Selection Nuance in Paper**:
   - Table 2 of the paper lists 26 input features (6 categorical + 20 numerical), while Section 3.2 and Figure 2 specify 9 features (4 categorical + 5 numerical). When tested on all 26 features at early epochs, validation accuracy is ~87.9%–89.4%.
4. **Random Seed & Hardware Initialization**:
   - Random seed, CPU/GPU hardware environment, and weight initialization were omitted in the original publication.

---

## 9. Final Reproduction Test Answers

1. **Does the implementation use every explicitly specified hyperparameter exactly as the paper?**  
   **YES**. Every dimension, head count, layer count, activation function, dropout rate, loss function, optimizer, scheduler, and decision threshold matches the paper.
2. **Does the preprocessing match what can be determined from the paper?**  
   **YES**. Identity columns and data leakage features (`diabetes_risk_score`, `diabetes_stage`) are removed, numericals are scaled via `StandardScaler` fitted on training data, and categoricals are integer-encoded for embeddings.
3. **Does the architecture match the paper?**  
   **YES**. 4 embedding lookup tables ($d=32$) $\to$ 4-layer Transformer Encoder (8 heads, $d_{\text{ff}}=128$, GELU, dropout 0.1) $\to$ contextual categorical representations concatenated with 5 standardized numerical features $\to$ 2-layer MLP (`[64, 32]`, ReLU, dropout 0.2) $\to$ output logits.
4. **Does the training procedure match the paper?**  
   **YES**. `BCEWithLogitsLoss` loss minimization, `AdamW` optimization, and `ReduceLROnPlateau` scheduling responding to validation loss.
5. **Do the results match the paper numerically?**  
   **NO**. The reproduced model achieves higher performance (91.93% accuracy vs. 82.55%, 0.9418 AUC vs. 0.9009 AUC).
6. **What specific undocumented information prevents exact numerical reproduction?**  
   - Exact number of training epochs / early stopping criteria.
   - Exact train/val/test split ratio and cross-validation scheme.
   - Random seed and initialization state.
7. **Is it scientifically honest to call this an "exact reproduction"?**  
   It is scientifically honest to designate this as an **Exact Methodological Reproduction**, but **NOT an Exact Numerical Reproduction**. Faking or forcing the metrics to 82.55% would violate research integrity.

---

## 10. Final Reproduction Verdict

| Evaluation Category | Verdict | Notes |
| :--- | :---: | :--- |
| **Methodology Reproduced** | **YES** | Complete data pipeline, leakage removal, scaling, and embeddings match |
| **Specified Hyperparameters Reproduced** | **YES** | All 18 architectural & optimization hyperparameters match exactly or are faithful inferences |
| **Training Procedure Reproduced** | **YES** | `train_epoch`, `eval_epoch`, `AdamW`, `ReduceLROnPlateau`, `BCEWithLogitsLoss` implemented as published |
| **Numerical Results Reproduced** | **NO** | Reaches 91.93% test accuracy vs 82.55% reported (explained by epoch convergence and split factors) |
| **Exact Numerical Reproduction Possible from Paper** | **NO** | Missing epoch count/stopping criteria, split ratios, and random seed prevent bit-exact numerical replication |

---

## 11. Complete Monorepo Project Structure
```
model-training-diabeties-ds-/
├── .github/
│   └── workflows/
│       └── deploy.yml                       # GitHub Actions CI/CD Pipeline
├── frontend/                                # Angular Standalone Frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── app.ts                       # Dual-model logic, presets & inference
│   │   │   ├── app.html                     # Clinical dashboard UI & verification banner
│   │   │   ├── app.css                      # Modern responsive styling
│   │   │   ├── app.spec.ts                  # Vitest unit test suite
│   │   │   ├── app.config.ts                # Application configuration & HTTP client
│   │   │   └── app.routes.ts                # Routing configuration
│   │   ├── index.html                       # HTML template with Google Fonts
│   │   ├── styles.css                       # Global design system & tokens
│   │   └── main.ts                          # App bootstrap
│   ├── angular.json                         # Angular CLI configuration
│   ├── package.json                         # Frontend dependencies & scripts
│   └── tsconfig.json                        # TypeScript configuration
├── app.py                                   # Flask Backend REST API & HTML Dashboard
├── test_app.py                              # Automated Flask unit & API tests
├── requirements.txt                         # Root Python backend dependencies
├── random_forest_diabetes_model.pkl         # 100-Tree Random Forest trained pipeline
├── logistic_regression_diabetes_model.pkl   # Logistic Regression baseline pipeline
├── templates/
│   └── index.html                           # Original Flask Jinja template
├── static/
│   ├── style.css                            # Original Flask stylesheet
│   └── random_forest_roc_curve.png          # Verified ROC Curve artifact
├── diabetes_random_forest_reproduction/     # JUTITI Random Forest research paper reproduction
├── diabetes-tabtransformer/                 # JoCoSiR TabTransformer research paper reproduction
└── README.md                                # Full Documentation & Lab Guide
```

---

## 12. Local Development & Testing Guide

### 1. Run the Flask Backend API
```bash
# Install backend dependencies
pip install -r requirements.txt

# Run backend unit tests
python test_app.py

# Start Flask development server
python app.py
# Server runs on: http://127.0.0.1:5000
# Health check:  http://127.0.0.1:5000/api/health
```

### 2. Run the Angular Frontend
```bash
# Navigate to frontend directory
cd frontend

# Install npm dependencies
npm install

# Run frontend unit tests (Vitest)
npm run test:ci

# Start Angular local development server
npm start
# Open browser at: http://localhost:4200
```

### 3. Build Production Bundle for GitHub Pages
```bash
cd frontend
npm run build:gh-pages
```

---

## 13. Summary of CI/CD Automated Workflow for Evaluator
| Stage | Component | Tools / Runners | Verification Check |
|---|---|---|---|
| **Code Edit** | HTML / CSS / TS | VS Code | Quick edit to header or styling |
| **Push** | Git Version Control | Git CLI (`git push origin main`) | Triggers GitHub Actions workflow |
| **CI: Backend Test** | Python Flask API | Python 3.10 / `pytest` / `test_app.py` | 100% assertions passed |
| **CI: Frontend Test** | Angular 22 & Vitest | Node.js 22 / Vitest | 5/5 unit tests passed |
| **CI: Build** | Production Build | `@angular/build` with base-href | Generates optimized static bundle |
| **CD: Deploy** | GitHub Pages Deployment | `actions/deploy-pages@v4` & `gh-pages` | Publishes live site with 0 downtime |

---

## 14. Research Novelty: Robustness-Aware, Calibrated & Explanation-Stable TabTransformer

### 14.1 Research Title
**Robustness-Aware, Calibrated, and Explanation-Stable TabTransformer for Clinical Diabetes Prediction**

Detailed theoretical background, literature gap analysis, and implementation roadmap are documented in the [`research/`](research/) suite:
- [RESEARCH_CONTEXT.md](research/RESEARCH_CONTEXT.md): Base paper hyperparameter and local baseline replication.
- [LITERATURE_REVIEW.md](research/LITERATURE_REVIEW.md): In-depth critical survey of 10+ contemporary papers (2020–2026).
- [LITERATURE_COMPARISON.md](research/LITERATURE_COMPARISON.md): Methodological comparison matrix across recent architectures.
- [RESEARCH_GAP.md](research/RESEARCH_GAP.md): Critical analysis of unaddressed failure modes under clinical data corruption.
- [PROPOSED_METHODOLOGY.md](research/PROPOSED_METHODOLOGY.md): Architectural design of `RobustTabTransformer` ($A_4$).
- [EXPERIMENTAL_RESULTS.md](research/EXPERIMENTAL_RESULTS.md): Complete raw experimental tables, statistical tests, negative results, and audit logs.
- [IMPLEMENTATION_ROADMAP.md](research/IMPLEMENTATION_ROADMAP.md): Staged execution milestones and integrity checklists.

---

### 14.2 Research Contribution
While deep tabular architectures like **TabTransformer** achieve high clean-data classification accuracy, real-world clinical screening and telehealth deployments rarely guarantee complete, noise-free laboratory profiles. In clinical workflows, fasting glucose or HbA1c may be omitted due to patient non-compliance, emergency triage, or assay unavailability.

The proposed **$A_4$ (RobustCons TabTransformer)** architecture addresses these vulnerabilities by integrating:
1. **Consensus-Based Feature Selection**: Fitted strictly on training data across 28 non-leakage candidate features using four independent criteria (ANOVA F-statistic, Mutual Information, Random Forest Gini importance, and $L_1$-penalized Logistic Regression), isolating an optimal 8-feature subset (`family_history_diabetes`, `hba1c`, `glucose_fasting`, `age`, `physical_activity_minutes_per_week`, `bmi`, `glucose_postprandial`, `systolic_bp`).
2. **Missingness-Aware Representations**: Continuous features are paired with explicit binary missingness indicator masks $m_{\text{num}} \in \{0, 1\}$, embedded via a linear projection layer and concatenated to the numerical representation. Categorical features reserve a dedicated `[MISSING]` token embedding ($K+1$), replacing silent mode imputation with structural awareness of missing data.
3. **Stochastic Corruption Training**: During training, batches are subjected to stochastic feature masking ($p_{\text{mask}} = 0.15$) and Gaussian measurement noise ($\sigma = 0.10$), regularizing self-attention mechanisms to learn redundant glycemic pathways rather than over-relying on a single dominant laboratory marker.
4. **Shift-Safe Probability Calibration**: Validation-only fitted **Temperature Scaling** ensures predicted probabilities reflect true empirical clinical risk without non-parametric binning saturation under distribution shift.
5. **Post-Hoc Decision Threshold Optimization**: Validation Precision-Recall Curve (PRC) optimization addresses clinical false-negative penalties.
6. **Explanation-Stability Verification**: Systematic auditing of post-hoc attribution consistency via `shap.PermutationExplainer` under missingness and assay noise.

---

### 14.3 Multi-Seed Experimental Benchmark ($A_0$ Baseline vs. $A_4$ Robust)
To guarantee rigorous academic reproducibility, both the baseline ($A_0$) and proposed model ($A_4$) were independently trained and evaluated across **three random seeds (`42, 101, 202`)** on the untouched test partition ($N = 15,000$). 

The table below presents the verified **Mean Accuracy** across corruption regimes:

| Scenario | A0 Baseline Accuracy | A4 Robust Accuracy |
|---|---:|---:|
| Clean | 92.05% | 91.15% |
| MCAR 20% | 85.58% | 89.02% |
| MCAR 30% | 82.18% | 87.43% |
| HbA1c missing | 60.01% | 84.31% |
| Glucose + HbA1c missing | 60.00% | 82.28% |
| MCAR 20% + Noise 0.25 | 80.57% | 85.25% |

#### Comprehensive Evaluation Metrics (Mean ± Standard Deviation Across Seeds 42, 101, 202)
From [multiseed_a0_vs_a4_summary.csv](diabetes-tabtransformer/research_extensions/outputs/metrics/multiseed_a0_vs_a4_summary.csv):

| Stress Scenario | Model | Accuracy | ROC-AUC | PR-AUC | Brier Score (↓) | ECE (↓) | False Negatives | False Positives |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Test** | $A_0$ Baseline | **0.9205 ± 0.0032** | **0.9454 ± 0.0045** | **0.9729 ± 0.0021** | **0.0656 ± 0.0024** | **0.0083 ± 0.0022** | 1,160.0 ± 50.3 | 32.3 ± 7.2 |
| | $A_4$ RobustCons | 0.9115 ± 0.0046 | 0.9443 ± 0.0031 | 0.9721 ± 0.0015 | 0.0710 ± 0.0028 | 0.0152 ± 0.0010 | 1,134.0 ± 50.5 | 193.3 ± 20.5 |
| **MCAR 20%** | $A_0$ Baseline | 0.8558 ± 0.0038 | 0.9226 ± 0.0062 | 0.9573 ± 0.0039 | 0.1271 ± 0.0081 | 0.0776 ± 0.0078 | 958.3 ± 35.0 | 1,204.0 ± 32.1 |
| | $A_4$ RobustCons | **0.8902 ± 0.0050** | **0.9375 ± 0.0043** | **0.9680 ± 0.0020** | **0.0841 ± 0.0028** | **0.0157 ± 0.0035** | 1,179.0 ± 41.1 | 467.7 ± 38.4 |
| **MCAR 30%** | $A_0$ Baseline | 0.8218 ± 0.0073 | 0.8992 ± 0.0109 | 0.9417 ± 0.0069 | 0.1601 ± 0.0136 | 0.1158 ± 0.0144 | 852.0 ± 41.1 | 1,820.7 ± 70.9 |
| | $A_4$ RobustCons | **0.8743 ± 0.0042** | **0.9320 ± 0.0043** | **0.9643 ± 0.0020** | **0.0932 ± 0.0024** | **0.0191 ± 0.0021** | 1,180.3 ± 34.0 | 705.3 ± 31.8 |
| **Missing HbA1c** | $A_0$ Baseline | 0.6001 ± 0.0002 | 0.6147 ± 0.0658 | 0.7607 ± 0.0409 | 0.3706 ± 0.0325 | 0.3666 ± 0.0371 | 0.0 ± 0.0 | 5,998.3 ± 2.9 |
| | $A_4$ RobustCons | **0.8431 ± 0.0052** | **0.9166 ± 0.0040** | **0.9528 ± 0.0022** | **0.1119 ± 0.0024** | **0.0243 ± 0.0042** | 1,383.3 ± 114.4 | 970.7 ± 80.1 |
| **Missing Gluc + HbA1c**| $A_0$ Baseline | 0.6000 ± 0.0000 | 0.4305 ± 0.0676 | 0.5579 ± 0.0426 | 0.3764 ± 0.0244 | 0.3645 ± 0.0389 | 0.0 ± 0.0 | 6,000.0 ± 0.0 |
| | $A_4$ RobustCons | **0.8228 ± 0.0029** | **0.9013 ± 0.0031** | **0.9418 ± 0.0019** | **0.1248 ± 0.0011** | **0.0298 ± 0.0101** | 1,373.3 ± 117.8 | 1,284.0 ± 81.3 |
| **Combined Corruption**| $A_0$ Baseline | 0.8057 ± 0.0041 | 0.8773 ± 0.0071 | 0.9168 ± 0.0047 | 0.1673 ± 0.0089 | 0.1221 ± 0.0106 | 1,314.7 ± 50.0 | 1,599.7 ± 22.5 |
| | $A_4$ RobustCons | **0.8525 ± 0.0045** | **0.9183 ± 0.0033** | **0.9547 ± 0.0015** | **0.1109 ± 0.0017** | **0.0407 ± 0.0002** | 1,439.0 ± 35.7 | 773.3 ± 32.9 |

> **Clean Trade-Off Analysis**: Stochastic corruption training incurs a modest **$-0.90\%$ clean accuracy penalty** ($91.15\%$ vs. $92.05\%$) while preserving identical clean discriminative ability (ROC-AUC $0.9443$ vs. $0.9454$). In exchange, $A_4$ gains massive fault tolerance: under missing HbA1c, accuracy increases by **$+24.30\%$** and Expected Calibration Error decreases **$15\times$** ($0.0243$ vs. $0.3666$).

---

### 14.4 Forensic Dataset & Diagnostic Rule Audit
A thorough forensic audit of the 100,000 patient records was conducted ([audit_forensics_report.json](diabetes-tabtransformer/research_extensions/outputs/metrics/audit_forensics_report.json)):
1. **ADA Diagnostic Cutoffs & Label Determinism**:
   - $P(\text{diagnosed\_diabetes} = 1 \mid \text{hba1c} \ge 6.5) = 1.0000$ (51,105 patients, exactly 0 false positives).
   - $P(\text{diagnosed\_diabetes} = 1 \mid \text{glucose\_fasting} \ge 126) = 1.0000$ (14,528 patients, exactly 0 false positives).
   - Standard clinical American Diabetes Association (ADA) threshold rules match **92.11% of all labels** (92,108 / 100,000 rows).
   - The remaining 7,890 diabetic patients below cutoffs represent clinically realistic non-deterministic diagnoses (7,852 Type 2 cases managed with lifestyle/medication, 17 Type 1, 21 Gestational).
2. **Semi-Synthetic Nature of the Dataset**:
   - These findings confirm that the underlying Kaggle dataset has strong **threshold-driven, semi-synthetic characteristics**. It must not be characterized as an uncurated natural clinical cohort.
3. **Synthetic Target Leakage Removal**:
   - Linear regression of `diabetes_risk_score` against basic indicators yielded $R^2 = 0.9735$, proving it is a synthetically generated composite risk score that leaks target labels. It was strictly excluded from all feature selections.
4. **Root Cause of Baseline Catastrophic Collapse**:
   - The cohort median for `hba1c` is **`6.52%`** (above the diagnostic threshold).
   - Under naive median imputation ($A_0$), whenever HbA1c is omitted, `6.52%` is imputed for 100% of patients.
   - The baseline classifies all patients as diabetic ($0.0\%$ specificity, 6,000 false positives, collapsing accuracy to exactly $60.00\%$, which equals the cohort positive prevalence). $A_4$ avoids this via explicit missingness indicator masks and stochastic training.

---

### 14.5 Validation Protocols & Statistical Significance
1. **Strict Data Leakage Prevention**:
   - The 70/15/15 split ($N = 69,999 / 15,001 / 15,000$) ensures zero patient overlap ($\text{Train} \cap \text{Val} = \emptyset, \text{Train} \cap \text{Test} = \emptyset$).
   - Imputers, scalers, encoders, and consensus feature selectors are fitted **strictly on training data**.
   - Temperature scaling calibrators and PRC decision thresholds are fitted **strictly on validation data**.
   - The test set was touched only once at evaluation time.
2. **Paired McNemar's Tests (A0 vs. A4)**:
   - **MCAR 20%**: $p = 1.74 \times 10^{-35}$ (Seed 42), $5.57 \times 10^{-34}$ (Seed 101), $4.37 \times 10^{-29}$ (Seed 202).
   - **Missing HbA1c**: $p < 10^{-300}$ across all three seeds.
   - **Noise $\sigma=0.25$**: $p = 1.65 \times 10^{-4}$ (Seed 42), $p = 0.0111$ (Seed 101), $p = 3.87 \times 10^{-3}$ (Seed 202).
   - **Combined Corruption**: $p = 8.94 \times 10^{-47}$ (Seed 42), $5.95 \times 10^{-38}$ (Seed 101), $2.61 \times 10^{-37}$ (Seed 202).
3. **1,000-Iteration Stratified Bootstrap 95% Confidence Intervals**:
   - Confirm completely non-overlapping intervals favoring $A_4$ under corruption (e.g., Seed 42 Drop HbA1c accuracy: $A_0 = [0.5923, 0.6079]$ vs. $A_4 = [0.8353, 0.8465]$; ECE: $A_0 = [0.3760, 0.3914]$ vs. $A_4 = [0.0180, 0.0271]$).

---

### 14.6 Explainability & SHAP Stability Audit
Audited across $n = 500$ stratified test patients using `shap.PermutationExplainer` ([a4_robustcons8_tabtransformer_shap_global_importance.csv](diabetes-tabtransformer/research_extensions/outputs/metrics/a4_robustcons8_tabtransformer_shap_global_importance.csv)):
- **Additivity Verification**: Max absolute additivity error is $< 5.3 \times 10^{-8}$ (mean $8.51 \times 10^{-9}$), confirming mathematical convergence.
- **Attribution Redistribution**: When HbA1c is omitted, baseline $A_0$ continues attributing $0.3573$ mean $|SHAP|$ to the *missing* feature (due to static median imputation), causing local explanation cosine similarity to collapse to **$0.0608$** and global Top-3 Jaccard to $0.5000$.
- In contrast, $A_4$ dynamically shifts attribution to observed metabolic indicators (`glucose_postprandial`: $0.0318 \to 0.1527$; `glucose_fasting`: $0.0532 \to 0.0820$), maintaining a local explanation cosine similarity of **$0.5989$** ($10\times$ improvement) and preserving global Top-3 Jaccard stability at **$1.0000$**.

---

### 14.7 Research Limitations & Threats to Validity
1. **Semi-Synthetic Label Characteristics**: The 100,000-record dataset displays near-deterministic ADA rule adherence (92.11% match). Results cannot be cited as proof of real-world clinical effectiveness without external validation on natural EHR cohorts (e.g., MIMIC-IV, UK Biobank, or NHANES).
2. **Simulation of Clinical Corruption**: Synthetic MCAR missingness and Gaussian noise model measurement failure and assay variance, but do not fully reflect complex clinical Missing Not At Random (MNAR) processes (e.g., tests ordered selectively based on patient acuity).
3. **Calibration Under Shift**: Non-parametric calibration (Isotonic Regression) fails catastrophically under covariate shift (validation AUC dropped from $0.9473 \to 0.8569$; test NLL exploded to $1.458$). Parametric scaling (Temperature Scaling) must be used.

