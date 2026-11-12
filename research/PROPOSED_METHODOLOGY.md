# Proposed Methodology

> **Design Principle:** All new experiments will live in an isolated experimental module (e.g., `diabetes-tabtransformer/experiments/` or a new `robust_tabtransformer/` module) using the **exact same 70% / 15% / 15% stratified split (`random_state=42`)** as the preserved baseline, ensuring 100% fair comparisons and zero modification to the original baseline implementation or web app.

---

## 1. Strict Leakage-Free Data Hygiene Protocol
Addressing leakage pitfalls identified in our literature review (Syahla et al., 2025; Rossi et al., 2026) and fixing the latent pre-split `fillna` ordering in `src/preprocessing.py`:
1. **Leakage Exclusion:** Drop `diabetes_risk_score` and `diabetes_stage` prior to all modeling. Target is `diagnosed_diabetes` (`0`/`1`).
2. **Split First:** Perform stratified 70% Train (`69,999`) / 15% Val (`15,001`) / 15% Test (`15,000`) split with `random_state=42` *before* any imputation, scaling, encoding, feature selection, or calibration.
3. **Train-Only Fitting:** Fit `SimpleImputer` (median/mode or iterative/KNN imputer), `StandardScaler`, `LabelEncoder`, and consensus feature selectors **strictly on Train**.
4. **Validation-Only Calibration & Threshold Selection:** Fit Temperature Scaling, Platt Scaling, Isotonic Regression, and PRC optimal thresholds **strictly on Val**.
5. **Untouched Test Set:** Evaluate final clean and corrupted scenarios on the untouched **Test** set (`N = 15,000`).

---

## 2. Proposed Architecture: Robustness-Aware Calibrated TabTransformer

```
Clean / Incomplete / Noisy Patient Record (x_num, x_cat)
    │
    ├──► [Train-Only Imputer + Missingness Indicator Mask m_num ∈ {0,1}^d_num]
    │
    ├──► Categorical Features x_cat (with dedicated [MISSING] token index per feature)
    │       └──► Learnable Embeddings (d = 32)
    │       └──► 4-Layer Multi-Head Self-Attention Transformer (8 heads, d_ff=128, GELU, p=0.1)
    │       └──► Flatten → [B, n_cat × 32]
    │
    ├──► Numerical Features: Standardized x_num_scaled ⊕ Missingness Mask m_num
    │       └──► (Optional linear/mask projection or direct concatenation)
    │
    ▼
Concatenated Representation ──► 2-Layer MLP [64 → 32 → 1] (ReLU, p=0.2) ──► Raw Logit z
                                                                              │
                                              ┌───────────────────────────────┴────────────────────┐
                                              ▼ (Val-Fitted Post-Hoc Calibration)                  ▼
                                  Calibrated Probability p_cal ∈ (0,1)             SHAP Explainer (Clean & Perturbed)
                                              │                                            │
                                              ▼ (PRC-Optimized Threshold τ*)               ▼
                                  Clinical Risk Tier / Binary Decision          Global Rank & Local Drift Audit
```

### Key Architectural & Training Additions Over Baseline
1. **Explicit Missingness Awareness:**
   - Each categorical feature adds a dedicated `[MISSING]` token index (`cardinality + 1`), preventing missing categories from silently colliding with category `0`.
   - Each numerical feature is paired with a binary missingness indicator `m_num` concatenated alongside the standardized value (`x_num_scaled`), allowing the MLP to down-weight imputed values and adjust uncertainty when a key lab (e.g., `hba1c`) is absent.
2. **Stochastic Corruption-Augmented Training (Optional Variant):**
   - During training *only*, randomly mask features with probability `p_mask ~ Uniform(0, 0.25)` and inject mild Gaussian jitter `N(0, σ_train²)` on observed numericals, encouraging the Transformer attention and MLP to learn redundant cross-feature pathways (e.g., leveraging `bmi`, `age`, `glucose_postprandial` when `hba1c` is masked).
3. **Post-Hoc Calibration Layer (Fitted on Validation Set):**
   - **Temperature Scaling:** `p = sigmoid(z / T)` where scalar `T > 0` minimizes NLL on `Val`.
   - **Platt Scaling:** Logistic regression fitted on validation logits `z`.
   - **Isotonic Regression:** Non-parametric monotonic mapping fitted on validation probabilities.
4. **Consensus Feature Selection Module (Optional 28-Feature Extension):**
   - Run 4 diverse selectors strictly on `Train` across all 28 non-leakage predictors:
     1. ANOVA F-score (linear univariate)
     2. Mutual Information (non-linear dependency)
     3. Random Forest / Permutation importance (tree-based multivariate)
     4. L1-penalized Logistic Regression or RFE (sparse linear / wrapper)
   - Aggregate via majority consensus (`≥ 3 of 4` selectors) and compare `9` baseline features vs. `Consensus-K` features vs. `All 28` features under clean and missing-feature tests.

---

## 3. Controlled Corruption & Stress-Testing Protocol (Test Time)

To simulate real-world clinical and telehealth data imperfections reproducibly (using fixed seeds on the Test set, repeated across multiple corruption seeds for confidence intervals):

1. **Missingness Regimes (Applied to Test Features Before Imputation):**
   - **MCAR (Missing Completely at Random):** Missing rates `r ∈ {5%, 10%, 20%, 30%}` across input features.
   - **Targeted Clinical Missingness (Realistic MAR/Lab-Drop):** Selectively dropping high-cost laboratory assays (`hba1c`, `glucose_fasting`, `cholesterol_total` — or `insulin_level`, `ldl_cholesterol` in the 28-feature setting) to test graceful degradation when primary diagnostic markers are unavailable.
2. **Measurement Noise Regimes (Applied to Standardized Numerical Features):**
   - Additive zero-mean Gaussian noise `ε ~ N(0, σ²)` with `σ ∈ {0.05, 0.10, 0.25, 0.50}` (in standardized units).
   - Combined corruption: `15%` missingness + `σ = 0.15` noise.
3. **Comparative Classical Baselines (Same 70/15/15 Split):**
   - Logistic Regression and Random Forest trained on the exact same 70/15/15 split and evaluated under the identical corruption sweeps for fair cross-paradigm benchmarking.

---

## 4. Comprehensive Evaluation Metrics

### Axis A — Predictive Discrimination
- Accuracy, Precision (Class 0 & Class 1), Recall / Sensitivity (Class 1), Specificity (Class 0), F1-score (Class 1 & Macro F1), **ROC-AUC**, and **PR-AUC** (Average Precision).

### Axis B — Probabilistic Calibration & Clinical Utility
- **Brier Score:** $\text{BS} = \frac{1}{N}\sum_{i=1}^{N}(p_i - y_i)^2$ (proper scoring rule combining calibration + refinement).
- **Expected Calibration Error (ECE):** 10-bin or 15-bin weighted confidence-vs-accuracy gap.
- **Negative Log-Likelihood (NLL / Log-Loss).**
- **Reliability Diagrams:** Visual calibration curves across probability deciles.
- **PRC-Optimized Thresholding & Decision Curve Analysis (DCA):** Net clinical benefit $\text{NB}(p_t) = \frac{\text{TP}}{N} - \frac{\text{FP}}{N}\left(\frac{p_t}{1 - p_t}\right)$ across clinically plausible risk thresholds $p_t \in [0.10, 0.80]$, quantifying reduction in missed diabetic cases (`FN`) relative to `0.50`.

### Axis C — Explainability & Explanation Stability
- **Global SHAP Importance:** Mean `|SHAP|` across a stratified test sample (e.g., `n = 1,000` test patients, using `shap.KernelExplainer` / `GradientExplainer` / `PermutationExplainer` depending on installed tooling).
- **Local Patient-Level SHAP:** Waterfall/force attribution vectors for representative TN, TP, FN, and borderline cases.
- **Quantitative Stability Under Corruption (Novel Audit):**
  - **Global Rank Stability:** Kendall's $\tau$ and Spearman's $\rho$ between clean-test global SHAP rankings and corrupted-test global SHAP rankings.
  - **Top-$k$ Feature Consistency:** Jaccard similarity of top-$k$ (`k = 3, 5`) drivers per patient between clean and noisy/missing inputs.
  - **Local Attribution Drift:** Mean cosine similarity and normalized $L_2$ distance $\|\phi(x_{\text{corrupt}}) - \phi(x_{\text{clean}})\|_2 / \|\phi(x_{\text{clean}})\|_2$ across identical patients.
- **Statistical Significance:** McNemar's test (paired binary predictions) and bootstrap 95% confidence intervals for AUC, Brier, and ECE comparisons.

---

## 5. Planned Ablation Studies

| Ablation ID | Model Variant | Feature Set | Training | Calibration | Purpose |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **A0 (Preserved)** | Baseline TabTransformer | 9 Paper | Clean | None (`0.50`) | Verified clean & corrupted benchmark |
| **A1** | Baseline TabTransformer + Calibration | 9 Paper | Clean | Temp / Platt / Isotonic + PRC $\tau^*$ | Isolate pure calibration & threshold gain |
| **A2** | Missingness-Aware TabTransformer | 9 Paper | Clean + Mask/Token | Temp / Platt / Isotonic | Isolate explicit missingness representation |
| **A3** | Robustness-Aware TabTransformer (Full) | 9 Paper | Corruption-Aug + Mask | Temp / Platt / Isotonic | Measure joint robustness + calibration + XAI stability |
| **A4** | Robustness-Aware TabTransformer (28 / Consensus) | Consensus-$K$ (from 28) | Corruption-Aug + Mask | Temp / Platt / Isotonic | Test whether extra/consensus predictors buffer lab missingness |
| **A5** | Classical ML Baselines (RF, LR) | 9 Paper & Consensus-$K$ | Clean / Augmented | Platt / Isotonic | Fair tree/linear vs. TabTransformer comparison |

---

## 6. Executed Implementation Notes & Key Design Confirmations
- All experiments (`A0`–`A5`) were executed in [diabetes-tabtransformer/research_extensions/](file:///c:/Users/hp/.gemini/antigravity-ide/scratch/model-training-diabeties-ds-/diabetes-tabtransformer/research_extensions) with full results recorded in [EXPERIMENTAL_RESULTS.md](file:///c:/Users/hp/.gemini/antigravity-ide/scratch/model-training-diabeties-ds-/research/EXPERIMENTAL_RESULTS.md).
- **Calibration Note:** Empirical testing in Stage 2 showed Validation-fitted Isotonic Regression suffers step-function saturation under MCAR/noise distribution shifts (dropping ROC-AUC from `0.9189` to `0.8489` at MCAR 20%), whereas **Temperature Scaling** (and Platt Scaling) preserves ranking monotonicity while keeping ECE low (`0.0145–0.0205` under MCAR 30% and `drop_hba1c`).
- **SHAP Evaluation Note:** Computed via `shap.PermutationExplainer` on `n = 500` stratified test patients (`bg = 25`, `max_evals = 110`) with `NaN` sentinels triggering missingness masks (`m_num = 1.0`).

