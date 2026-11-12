# Research Gap Analysis & Recommended Novelty

## 1. Specific Research Problem
Deep tabular architectures such as **TabTransformer** achieve strong clean-data discrimination on large clinical diabetes datasets (locally verified baseline: **91.93% accuracy**, **0.9418 ROC-AUC** on `N = 100,000` records). However, real-world clinical screening and telehealth workflows rarely supply complete, noise-free laboratory panels: measurements such as `hba1c`, `glucose_fasting`, `insulin_level`, or `cholesterol_total` are frequently missing or subject to assay/device measurement noise.

Furthermore, our audit of the baseline TabTransformer reveals two critical clinical deployment barriers:
1. **Asymmetric Clinical Error Profile at Fixed Threshold (`0.50`):** While positive precision is `0.9969` (only `24` false positives out of `15,000` test patients), the model misses **`1,186` diabetic patients (`13.18%` false-negative rate)**, with no calibration or decision-threshold optimization.
2. **Unverified Reliability Under Data Imperfections:** It is unknown how TabTransformer's (a) **discrimination**, (b) **probability calibration (ECE, Brier score) & clinical net benefit (DCA)**, and (c) **SHAP explanation stability** behave—or co-degrade—when clinical inputs suffer from missingness or measurement noise, and whether lightweight robustness-aware modifications can preserve all three simultaneously.

---

## 2. Why Simple Additions Are Not Enough (Literature Guardrails)
Based on our review of 2025–2026 literature:
- **Adding SHAP/LIME to TabTransformer** is already published by Mukherjee & De (May 2026, *Med. Novel Technol. Devices*).
- **Applying consensus feature selection** is already published by Natarajan et al. (2025, *Sci. Rep.*) and Chen et al. (2026, *DIGITAL HEALTH*).
- **Calibrating classical ML (LR/RF/XGBoost) with Platt/Isotonic + DCA + SHAP** is already published by Della Corte et al. (2025, *Electronics*), Rossi et al. (2026, *MTAP*), and Khan et al. (2026, *Sci. Rep.*).

Therefore, a defensible contribution **cannot** be "TabTransformer + SHAP" or "Feature Selection + TabTransformer."

---

## 3. Identified Research Gaps

### Gap 1: No Joint Robustness–Calibration–Explanation Auditing of TabTransformer in Diabetes Prediction
Existing TabTransformer diabetes studies (Al Khaidar & Sri Kurnia; Mukherjee & De, 2026) evaluate clean-data classification metrics only. Even foundational TabTransformer robustness tests (Huang et al., 2020) measured only ROC-AUC. **No prior study** quantifies how missing clinical features and Gaussian measurement noise simultaneously degrade:
- Predictive discrimination (Accuracy, Recall, F1, ROC-AUC, PR-AUC),
- Probabilistic reliability (Brier score, Expected Calibration Error [ECE], Reliability Diagrams, DCA net benefit), and
- Post-hoc XAI stability (Global SHAP rank correlation $\tau$/$\rho$ and Local patient-level SHAP drift).

### Gap 2: Lack of Missingness-Aware and Calibration-Preserving Mechanisms in Clinical TabTransformer
Standard TabTransformer pipelines replace missing continuous values with static training medians and encode unseen/missing categories silently (or map to index `0`, as in `src/preprocessing.py`), stripping away the **missingness signal** itself and causing silent overconfidence or miscalibration when key biomarkers (`hba1c`, `glucose_fasting`) are imputed.

### Gap 3: Unexplored Interaction Between Feature Selection and Missing-Data Fault Tolerance
Consensus feature selection methods (Natarajan et al., 2025; Chen et al., 2026) optimize for **clean-data parsimony** (`46 → 8` features). In our 31-column dataset, the baseline TabTransformer uses only `9` of `28` non-leakage predictors. It remains an open, untested question whether a compact consensus feature set vs. a consensus-plus-redundant metabolic set exhibits better **graceful degradation** when critical diagnostic features are missing or noisy at inference time.

---

## 4. Research Questions

- **RQ1 (Degradation Profiling):** How severely do controlled missing-data mechanisms (MCAR and clinically realistic MAR across `5%–30%` missingness) and Gaussian measurement noise ($\sigma \in \{0.05, 0.10, 0.25, 0.50\}$) degrade the baseline TabTransformer's discrimination, probability calibration (ECE, Brier), clinical net benefit (DCA), and SHAP explanation consistency?
- **RQ2 (Robustness-Aware Mitigation):** Does augmenting TabTransformer with **missingness-aware representations** (explicit binary missingness masks + `[MISSING]` categorical tokens + train-time stochastic feature masking/noise regularization) improve predictive retention and explanation stability under test-time corruptions without degrading clean-data performance?
- **RQ3 (Calibration & Threshold Reliability):** Do post-hoc calibration methods (**Temperature Scaling**, **Platt Scaling**, **Isotonic Regression**) and **PRC-optimized clinical thresholding** fitted strictly on validation data reduce ECE/Brier error, lower the baseline's high false-negative count (`1,186` missed diabetics), and maintain positive DCA net benefit across both clean and corrupted test regimes?
- **RQ4 (Feature-Set Fault Tolerance):** Compared to the 9-feature baseline, does leakage-safe consensus feature selection across all 28 clean clinical predictors improve clean performance and provide redundancy-driven resilience when primary glycemic markers (`hba1c` or `glucose_fasting`) are missing or noisy?

---

## 5. Recommended Novelty (Strongest & Most Feasible)

We recommend framing the primary contribution as:

> **A Robustness-Aware, Calibrated, and Explanation-Stable TabTransformer Framework for Clinical Diabetes Prediction**

Specifically, the **three tightly coupled pillars** that distinguish this work from all 2025–2026 studies are:
1. **Triple-Axis Stress-Testing Protocol (Performance × Calibration × XAI Stability):** Moving beyond clean-data accuracy and descriptive SHAP plots by introducing quantitative **explanation-stability metrics** (Kendall's $\tau$, Top-$k$ Jaccard, local cosine/L2 attribution drift) alongside **calibration reliability** (ECE, Brier, DCA) under systematic missingness and noise perturbations on `N = 100,000` records.
2. **Missingness- & Noise-Aware TabTransformer Architecture:** Extending (while preserving) the baseline TabTransformer with missingness indicator embeddings/masks for numericals, explicit `[MISSING]` embeddings for categoricals, and stochastic corruption training—so self-attention and the MLP adapt gracefully to incomplete/noisy patient profiles at inference time.
3. **Validation-Only Calibration & Clinical Thresholding Under Shift:** Combining Temperature Scaling / Platt / Isotonic calibration with PRC-guided thresholding and DCA to explicitly solve both the baseline's high false-negative rate (`FN = 1,186`) and calibration drift under corrupted clinical inputs, plus an optional consensus feature-selection ablation across the 28 non-leakage predictors.

### Why This Contribution Matters
- **Clinical Safety:** In screening, a silent drop in recall or an overconfident probability (`0.95` predicted risk when true empirical risk under missing `hba1c` is `0.60`) causes real patient harm.
- **Trustworthy XAI:** Clinicians cannot trust SHAP explanations if small measurement noise (`±0.10` SD) flips the top contributing biomarkers for a patient.
- **High Feasibility:** Fully implementable in PyTorch/scikit-learn/SciPy on the existing `100,000`-row dataset with zero external dataset dependencies and zero disruption to the preserved baseline or web app.

