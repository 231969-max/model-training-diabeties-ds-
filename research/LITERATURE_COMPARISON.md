# Literature Comparison & Synthesis Matrix

This document compares our base paper and 2025–2026 literature across architecture, dataset scale, feature selection, explainability, probability calibration, threshold analysis, leakage hygiene, and robustness to missing/noisy clinical data.

---

## 1. Comparative Matrix of Reviewed Studies

| Study (Year) | Dataset & Scale | Core Model(s) | Feature Selection | XAI Method | Probability Calibration | Threshold / DCA | Missing & Noise Stress-Test | XAI Stability Under Corruption |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Al Khaidar & Kurnia** (Base) | Kaggle Diabetes (`100k`, 9/26 feats) | **TabTransformer** | Manual (9 feats) | None | None | Fixed `0.50` | None | None |
| **Huang et al.** (2020) | 15 UCI/Public Tabular sets | **TabTransformer** | None | None | None | Fixed `0.50` | **Yes** (AUC under missing/noise) | None |
| **Natarajan et al.** (Feb 2025) | PIMA (`768`, 8 feats) | Classical ML (XGBoost, NB, RF, MLP) | **AdaptDiab** (ANOVA, Fisher, Var, PB) | None | None | Fixed `0.50` | None | None |
| **Tiwari et al.** (Aug 2025) | Big Health Data (Multiclass) | ML + Deep Attention | Feature engineering | Post-hoc XAI | None | Multiclass | None | None |
| **Netayawijit et al.** (Oct 2025) | Kaggle subset (`1,500`) | RF + SMOTE + Classical ML | None | **SHAP** (Global & Local) | None | Fixed `0.50` | None | None |
| **Della Corte et al.** (Nov 2025) | MIMIC-IV EHR (Incident T2DM) | LR, RF (Leakage-safe CV) | Clinical EHR set | SHAP + Perm. Imp. | **Platt + Isotonic** (Brier, Rel. Diag.) | **DCA** | None | None |
| **Syahla et al.** (Dec 2025) | Kaggle Diabetes (`100k`, `diabetes_stage`) | RF, DT (Orange) | None (contained leak) | None | None | Multiclass | None | None |
| **Mukherjee & De** (May 2026) | PIMA (`768`) & ESDRP (`520`) | **TabTransformer** + TabNet, RF, LR | None | **SHAP + LIME** | None | Fixed / Conceptual alerts | Raw ESDRP only (no synthetic sweep) | None |
| **Chen et al.** (2026) | Kaggle Diabetes (`1,879`, 46 feats) | 7-Model Soft Voting Ensemble | **Consensus** (RFE, RF, ANOVA, MI + PCA) | Global FS only (notes no SHAP) | None | Fixed `0.50` | None | None |
| **Rossi et al. — D.R.E.A.M.** (2026) | PIMA+BRFSS (`442k`) + NHANES ext. | RF, XGBoost, LightGBM | Clinical interactions | **SHAP** (Global & Local) | **Isotonic** (no Brier/ECE table) | **PRC Opt.** (`0.389`) | None | None |
| **Khan et al.** (2026) | PIMA (`768`, 8 feats) | XGBoost, AdaBoost, LR, NB | None | **SHAP** + LR Coeffs | **Calib. curves** (Brier, ECE) | Discussed, fixed `0.50` | Zero imputation (clean test) | CV fold stability only (clean) |
| **OUR PROPOSED WORK** | **Kaggle Diabetes (`100k`, 28 predictors)** | **Robustness-Aware TabTransformer** | **Leakage-Safe Consensus FS (Ablation)** | **SHAP (Global, Local & Stability)** | **Temp. Scaling / Platt / Isotonic (ECE, Brier)** | **PRC Opt. + DCA** | **MCAR/MAR Missingness + Gaussian Noise Sweeps** | **SHAP Rank & Local Drift Metrics** |

---

## 2. Techniques Already Investigated in Literature (Do Not Claim as Novel)

1. **TabTransformer on the 100k Kaggle Diabetes Dataset:** Already published by Al Khaidar & Sri Kurnia.
2. **TabTransformer + SHAP + LIME for Diabetes Prediction:** Already published by Mukherjee & De (May 2026). Simply attaching SHAP or LIME to TabTransformer is **insufficient** for novelty.
3. **Ensemble / Consensus Feature Selection for Diabetes:** Already published by Natarajan et al. (2025, *AdaptDiab*) and Chen et al. (2026, *DIGITAL HEALTH*).
4. **SMOTE + SHAP + Stratified CV:** Already published by Netayawijit et al. (2025) and Khan et al. (2026).
5. **Post-hoc Calibration (Platt / Isotonic) + Brier + DCA for Classical ML in Diabetes:** Already published by Della Corte et al. (2025) and Rossi et al. (2026).
6. **Clean-Data TabTransformer Robustness to Missingness/Noise (AUC only):** Demonstrated on general UCI benchmarks by Huang et al. (2020), but **never** evaluated for clinical probability calibration, clinical net benefit (DCA), or SHAP explanation stability in diabetes prediction.

---

## 3. Critical Synthesis: Where Existing Studies Fall Short

1. **Disconnection Between Robustness, Calibration, and XAI:**
   - Studies that evaluate **TabTransformer** (Al Khaidar & Kurnia; Mukherjee & De 2026) ignore probability calibration (ECE, Brier, DCA) and never stress-test what happens when clinical biomarkers are missing or noisy.
   - Studies that evaluate **calibration and DCA** (Della Corte et al. 2025; Rossi et al. 2026; Khan et al. 2026) only study classical/tree models (LR, RF, XGBoost) under clean, complete test conditions.
   - **No study examines whether TabTransformer's predicted probabilities stay calibrated or whether its SHAP explanations stay consistent when clinical inputs are degraded by missingness or measurement noise.**
2. **Small-Sample Generalizability Limitations:**
   - Four of the five recent 2025–2026 diabetes studies operate on tiny datasets: PIMA (`N = 768`), UCI ESDRP (`N = 520`), Kaggle subset (`N = 1,500`), or `N = 1,879`. Our `N = 100,000` dataset provides 50×–130× larger statistical power for fine-grained calibration binning, subgroup auditing, and controlled perturbation sweeps.
3. **Unresolved Feature-Subset vs. Fault-Tolerance Trade-off:**
   - The base TabTransformer uses only `9` of `28` legitimate predictors (omitting `glucose_postprandial`, `insulin_level`, `waist_to_hip_ratio`, `systolic_bp`, `diastolic_bp`, `triglycerides`, etc.). While Chen et al. (2026) and Natarajan et al. (2025) show compact consensus subsets perform well on *clean* data, nobody has tested whether compact vs. redundant clinical feature sets degrade differently when a primary lab measurement (e.g., `hba1c` or `glucose_fasting`) is missing!
4. **Data-Leakage Pitfalls in Literature:**
   - Syahla et al. (2025) leaked `diagnosed_diabetes` into `diabetes_stage`.
   - Rossi et al. (2026) applied SMOTE *before* train/test splitting.
   - Even our base reproduction's `src/preprocessing.py` has a latent pre-split `fillna` call (inert on 0-null data, dangerous once missingness is introduced).
   - Strict train-only fitting of imputers, scalers, encoders, feature selectors, and validation-only fitting of calibrators/thresholds is essential for scientific rigor.

---

## 4. What Our Proposed Work Adds
Our work bridges these disconnected pillars into a single, leakage-safe framework on large-scale (`N = 100,000`) clinical tabular data:
1. **Joint Auditing of Discrimination, Calibration, and SHAP Stability Under Clinical Corruptions** for TabTransformer (and comparative tree/linear baselines).
2. **Missingness-Aware & Corruption-Augmented TabTransformer Mitigation** (explicit missingness indicator embeddings/masks + train-time corruption regularization) to preserve performance, calibration, and explanation consistency without test-time data leakage.
3. **Post-Hoc Neural Calibration (Temperature Scaling, Platt, Isotonic) + PRC Thresholding + DCA** specifically targeting the baseline's high false-negative rate (`1,186` missed diabetics at `0.50`) and calibration drift under noisy/incomplete inputs.
4. **Ablation of Feature Selection Under Missingness:** Comparing the 9-feature baseline against a leakage-safe consensus-selected feature set from all 28 clean predictors under both clean and missing-feature conditions.

