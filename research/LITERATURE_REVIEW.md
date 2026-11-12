# Literature Review — Diabetes Mellitus Prediction (2020–2026)

This review synthesizes the five assigned reference papers, four additional full-text papers located in the workspace (`researchpapers/` and sibling audit), and the foundational TabTransformer paper. Published findings are strictly separated from local repository observations.

---

## 1. Base Paper & Foundational Architecture

### 1.1 Al Khaidar & Sri Kurnia — Base TabTransformer Study
- **Full Title:** Classification of Health Indicators for Diabetes Mellitus Prediction Using a TabTransformer Model on Clinical Tabular Data
- **Authors:** Al Khaidar, Sri Kurnia
- **Journal / Year:** *Journal of Computer Science Research (JoCoSiR)*, Vol. 2, No. 2, pp. 9–15 (cites references up to 2025)
- **DOI:** [http://doi.org/10.65126/jocosir.v2i2.52](http://doi.org/10.65126/jocosir.v2i2.52)
- **Objective:** Classify binary diabetes status (`diagnosed_diabetes`) from a 100,000-record Kaggle clinical tabular dataset using TabTransformer.
- **Methodology:** Removes `patient_id` and leakage attributes (`diabetes_risk_score`, `diabetes_stage`); normalizes numericals and embeds categoricals (`d=32`) through 4 Transformer layers (8 heads, `d_ff=128`, GELU, dropout 0.1) concatenated with numericals into an MLP `[64, 32]` (ReLU, dropout 0.2), trained with `BCEWithLogitsLoss`, `AdamW`, and `ReduceLROnPlateau`.
- **Reported Results:** Accuracy `82.55%`, ROC-AUC `0.9009`, Class 1 Precision `0.8637`, Class 1 Recall `0.8420`, Class 1 F1 `0.8527`.
- **Limitations:** Omits training epochs, split ratios, and random seed; exhibits internal discrepancy between 9 features (Section 3.2/Fig. 2) and 26 features (Table 2); lacks explainability (SHAP/LIME), probability calibration, threshold analysis, and robustness testing under missing/noisy data.
- **Relevance:** Establishes our preserved baseline architecture and dataset.

### 1.2 Huang et al. (2020) — Foundational TabTransformer
- **Full Title:** TabTransformer: Tabular Data Modeling Using Contextual Embeddings
- **Authors:** Xin Huang, Ashish Khetan, Milan Cvitkovic, Zohar S. Karnin
- **Venue / Year:** arXiv preprint `arXiv:2012.06678` (2020)
- **Link:** [https://arxiv.org/abs/2012.06678](https://arxiv.org/abs/2012.06678)
- **Objective & Methodology:** Introduces self-attention over categorical embeddings to produce contextual representations robust to tabular noise and missingness across 15 tabular benchmarks, plus semi-supervised pretraining.
- **Key Findings & Limitations:** Outperforms MLPs by ≥1.0% mean AUC and shows superior resilience to synthetic missingness and feature noise compared to context-free MLPs. However, it evaluates discrimination (AUC) only—not probability calibration (ECE/Brier), clinical decision utility, or explanation stability under corruption.
- **Relevance:** Provides the theoretical justification for why TabTransformer attention may retain predictive signal under corruption, motivating an investigation of whether calibration and SHAP attributions also remain reliable.

---

## 2. Assigned Literature References (Papers 1–5)

### 2.1 Paper 1 — Natarajan et al. (February 2025): Adaptive Ensemble Feature Selection
- **Full Title:** An adaptive ensemble feature selection technique for model-agnostic diabetes prediction
- **Authors:** K. Natarajan, Dhanalakshmi Baskaran, Selvakumar Kamalanathan
- **Journal / Date:** *Scientific Reports*, February 2025
- **DOI:** [https://doi.org/10.1038/s41598-025-91282-8](https://doi.org/10.1038/s41598-025-91282-8)
- **Objective:** Reduce feature redundancy, noise, and selection instability via a model-agnostic adaptive ensemble feature selector (**AdaptDiab**).
- **Methodology:** Partitions data into multiple subsets and combines four diverse filter/statistical selectors—**ANOVA F-score**, **Fisher score**, **Variance Threshold**, and **Point-Biserial correlation**—using an adaptive combiner function. Evaluated across classical ML classifiers (e.g., XGBoost, Gaussian NB, LR, RF, GBM, MLP) using cross-validation on benchmark data (Pima Indians Diabetes Dataset).
- **Reported Results:** Achieves 100% feature-selection stability across repeated samplings; with XGBoost, reports Accuracy `89.2%`, AUC `0.93`, Precision `0.88`, Recall `0.87`, F1 `0.87`.
- **Limitations:** Evaluated on small benchmark data (PIMA, 768 records, 8 features); does not incorporate deep tabular Transformers, probability calibration, SHAP explainability, or test-time corruption/missingness stress tests.
- **Relevance:** Demonstrates how combining complementary filter statistics improves predictor stability; warns us not to claim ensemble feature selection itself as novel, and motivates testing consensus feature selection with TabTransformer across our 28 clinical indicators.

### 2.2 Paper 2 — Netayawijit et al. (October 2025): SMOTE + SHAP + Statistical Testing
- **Full Title:** Interpretable Machine Learning Framework for Diabetes Prediction: Integrating SMOTE Balancing with SHAP Explainability for Clinical Decision Support
- **Authors:** Pathamakorn Netayawijit, Wirapong Chansanam, Kanda Sorn-In
- **Journal / Date:** *Healthcare*, Vol. 13, Issue 20, Article 2588, October 2025
- **DOI:** [https://doi.org/10.3390/healthcare13202588](https://doi.org/10.3390/healthcare13202588)
- **Objective:** Address class imbalance and black-box opacity in clinical diabetes decision support.
- **Methodology:** Uses a stratified subset of `1,500` records from a Kaggle diabetes prediction dataset, applies SMOTE balancing, evaluates multiple ML models (including RF, XGBoost, SVM, LR, DT) via 5-fold stratified cross-validation, applies McNemar's test for pairwise statistical significance, and computes global/local SHAP explanations.
- **Reported Results:** Random Forest + SMOTE achieves Accuracy `96.91%` (95% CI: 95.4–98.2%), AUC `0.998`, sensitivity `99.5%`, specificity `97.3%`.
- **Limitations:** Uses a small 1,500-row subset (explicitly limiting population generalizability); relies on tree/classical models rather than TabTransformer; does not evaluate probability calibration, clinical threshold sensitivity, or robustness/SHAP stability under missing or noisy measurements.
- **Relevance:** Provides a rigorous protocol for 5-fold CV, McNemar's statistical testing, and global + local SHAP explanations, while underscoring why accuracy cannot be compared directly across different dataset subsets.

### 2.3 Paper 3 — Della Corte et al. (November 2025): Leakage-Safe Calibration on MIMIC-IV
- **Full Title:** Deployable Machine Learning on MIMIC-IV: Leakage-Safe Prediction and Calibration for Incident Diabetes
- **Authors:** Antonio Della Corte, Enrico Barbierato, Alice Gatti
- **Journal / Date:** *Electronics*, Vol. 14, Issue 22, Article 4415, November 2025
- **DOI:** [https://doi.org/10.3390/electronics14224415](https://doi.org/10.3390/electronics14224415)
- **Objective:** Build an auditable, deployment-ready, leakage-safe pipeline predicting 365-day incident diabetes from MIMIC-IV EHR admissions.
- **Methodology:** Enforces strict temporal and patient-level data-splitting guards; fits preprocessing strictly inside inner CV folds for Logistic Regression and Random Forest; applies post-hoc **Platt scaling** and **Isotonic regression**; audits models using **Brier score**, **reliability diagrams**, **Decision Curve Analysis (DCA)**, and SHAP/permutation importance.
- **Reported Results:** Calibrated Logistic Regression achieves modest, leakage-free real-world EHR discrimination (`AUROC ≈ 0.70`, `AUPRC ≈ 0.31`) with superior calibration and positive net clinical benefit in DCA compared to Random Forest.
- **Limitations:** Evaluates only classical LR and RF (no deep tabular architectures like TabTransformer); does not study how missing data or sensor/lab noise affect calibration or SHAP stability.
- **Relevance:** Directly provides the calibration-first evaluation toolkit (Platt scaling, Isotonic regression, Brier score, reliability diagrams, DCA) needed to audit and improve our TabTransformer's probability estimates.

### 2.4 Paper 4 — Tiwari et al. (August 2025): Multiclass Risk Stratification
- **Full Title:** Artificial intelligence-based multiclass diabetes risk stratification for big data embedded with explainability: From machine learning to attention models
- **Authors:** Ekta Tiwari, Siddharth Gupta, Anudeep Pavulla, Mustafa Al-Maini, Rajesh Singh, Esma R. Isenovic, Sumit Chaudhary, John L. Laird, Laura Mantella, Amer M. Johri, Luca Saba, Jasjit S. Suri
- **Journal / Date:** *Biomedical Signal Processing and Control*, Vol. 106, Article 107672, August 2025
- **DOI:** [https://doi.org/10.1016/j.bspc.2025.107672](https://doi.org/10.1016/j.bspc.2025.107672)
- **Objective:** Move beyond binary diabetes classification toward big-data multiclass risk stratification using ML, deep learning, and attention-based models coupled with XAI.
- **Methodology:** Benchmarks traditional ML classifiers alongside deep attention architectures on large-scale tabular health data with post-hoc XAI explanations.
- **Limitations & Local Dataset Inspection Caveat:** While multiclass stratification is clinically attractive, inspecting `diabetes_stage` in our local 100,000-row dataset (and the sibling audit in `diabetes_random_forest_reproduction/`) reveals severe class imbalance: `Gestational` (`278` rows, `0.28%`) and `Type 1` (`122` rows, `0.12%`) yield `0.0000` F1-score under standard Random Forest training (`Macro F1 = 0.5501`), and `diabetes_stage` has deterministic overlap with `diagnosed_diabetes` (`Type 2` = 100% positive; `Pre-Diabetes` and `No Diabetes` = 100% negative).
- **Relevance:** Confirms that multiclass `diabetes_stage` should not replace binary `diagnosed_diabetes` as the primary benchmark without explicit minority-class/clinical-validity scoping.

### 2.5 Paper 5 — Mukherjee & De (May 2026): XAI-IoT TabTransformer
- **Full Title:** XAI-IoT based real-time diabetes prediction using TabTransformer
- **Authors:** Sunetra Mukherjee, Debashis De
- **Journal / Date:** *Medicine in Novel Technology and Devices*, Vol. 30, Article 100435, May 2026
- **DOI:** [https://doi.org/10.1016/j.medntd.2026.100435](https://doi.org/10.1016/j.medntd.2026.100435)
- **Local Source:** [1-s2.0-S259009352600010X-main.pdf](file:///c:/Users/hp/.gemini/antigravity-ide/scratch/model-training-diabeties-ds-/researchpapers/1-s2.0-S259009352600010X-main.pdf)
- **Objective:** Propose a continuous IoT glucose-monitoring framework combining TabTransformer (Adam optimizer) with SHAP and LIME for real-time diabetes alerts.
- **Methodology:** Compares TabTransformer (30 epochs, Adam, dropout, batch norm), TabNet, Random Forest (100 trees, 10-fold CV), and Logistic Regression on PIMA (`768` samples, 80/20 split, median imputation of zeros, MinMax + StandardScaler) and tests cross-population generalization on the UCI Early Stage Diabetes Risk Prediction (ESDRP) dataset (`520` samples, 70/30 split, raw unprocessed). Applies SHAP summary/bar plots and LIME local explanations.
- **Reported Results:** On PIMA: TabTransformer achieves `94.50%` accuracy, `0.9320` precision, `0.9600` recall, `0.9458` F1, `0.99` AUC (outperforming TabNet `87.50%`, RF `75.97%`, LR `71.43%`). On UCI ESDRP: `97%` accuracy, `1.00` AUC.
- **Limitations:**
  1. Very small benchmark datasets (`N = 768` and `N = 520`), far smaller than clinical scale (`N = 100,000`).
  2. No actual IoT sensor-noise or missing-packet stress experiments—only tests clean PIMA and raw ESDRP.
  3. Zero probability calibration evaluation (no ECE, Brier score, reliability diagrams, or DCA), despite proposing clinician-customizable alert thresholds.
  4. Zero quantitative evaluation of SHAP/LIME explanation stability under noisy or missing inputs.
  5. Note that PIMA features are all numerical/integer, meaning TabTransformer's categorical self-attention mechanism is under-utilized compared to heterogeneous clinical tables.
- **Relevance:** **Critical boundary condition.** Because Mukherjee & De (2026) already combine TabTransformer + SHAP + LIME for diabetes, merely adding SHAP/LIME to our TabTransformer is **not novel**. Our novelty must lie in **quantifying and mitigating robustness, calibration, and explanation-stability degradation under controlled missingness and measurement noise** at large scale.

---

## 3. Additional 2025–2026 Literature in Workspace & Recent Search

### 3.1 Chen et al. (2026) — Consensus-Based Feature Selection + Soft Voting
- **Full Title:** A hybrid ensemble approach for diabetes prediction using consensus-based feature selection
- **Authors:** JianHao Chen, Yung-Wey Chong, LiLi Wang, ZhongJian Tang, Xiao Zhang, Lee-Fueng Yap
- **Journal / Date:** *DIGITAL HEALTH*, Vol. 12, pp. 1–23, 2026
- **DOI:** [https://doi.org/10.1177/20552076261453207](https://doi.org/10.1177/20552076261453207)
- **Local Source:** [chen-et-al-2026-a-hybrid-ensemble-approach-for-diabetes-prediction-using-consensus-based-feature-selection.pdf](file:///c:/Users/hp/.gemini/antigravity-ide/scratch/model-training-diabeties-ds-/researchpapers/chen-et-al-2026-a-hybrid-ensemble-approach-for-diabetes-prediction-using-consensus-based-feature-selection.pdf)
- **Methodology:** Analyzes `1,879` Kaggle patients (`46` features), engineers 6 clinical interaction features (`HbA1c/FBS`, `TG/HDL`, `Age×BMI`, `BP_diff`, High-Risk Clinical Score, Lifestyle Score), removes `|r| ≥ 0.8` multicollinear raw features, applies 4 supervised selectors (RFE, RF, ANOVA, Mutual Information) + PCA with the one-standard-error rule and `≥3/4` majority consensus voting (reducing `46 → 8` features), and trains a 7-model Soft Voting ensemble tuned via 5-fold stratified GridSearchCV.
- **Reported Results:** `0.948` AUC (95% CI: 0.926–0.970), `92.6%` accuracy, `91.8%` precision, `89.4%` recall, `90.6%` F1 on `n = 376` held-out test set.
- **Limitations:** Explicitly notes small single-source dataset (`N = 1,879`), residual multicollinearity, and **lack of instance-level XAI (SHAP/LIME)**; uses classical ML ensembles only (no TabTransformer), no probability calibration, and no missingness/noise stress testing.
- **Relevance:** Directly supports testing consensus feature selection (ANOVA, MI, RF, RFE) on our 28-feature clinical pool—and raises an important question: does a compact consensus set help or hurt *robustness* when key lab features go missing?

### 3.2 Rossi et al. (2026) — D.R.E.A.M.: Calibrated Risk + PRC Thresholding + SHAP
- **Full Title:** D.R.E.A.M: diabetes risk via explainable AI modeling
- **Authors:** Domenico Rossi, Alessia Auriemma Citarella, Fabiola De Marco, Luigi Di Biasi, Huiru Zheng, Genoveffa Tortora
- **Journal / Date:** *Multimedia Tools and Applications*, Vol. 85, Article 145, 2026
- **DOI:** [https://doi.org/10.1007/s11042-026-21240-7](https://doi.org/10.1007/s11042-026-21240-7)
- **Local Source:** [s11042-026-21240-7.pdf](file:///c:/Users/hp/.gemini/antigravity-ide/scratch/model-training-diabeties-ds-/researchpapers/s11042-026-21240-7.pdf)
- **Methodology:** Harmonizes PIMA (`768`) and BRFSS 2015 (`441,456` respondents), engineers interaction terms (`BMI_Age`, `Glucose_Age`, `BP_Chol`, etc.), balances classes via SMOTE, evaluates RF/XGBoost/LightGBM via 5-fold CV, calibrates probabilities using **Isotonic regression**, optimizes decision thresholds using **Precision–Recall Curve (PRC)** F1-maximization (`0.389` vs default `0.50`), stratifies into 4 risk tiers, provides SHAP explanations, and validates externally on NHANES 2017–2020.
- **Reported Results:** Harmonized test RF: AUC `0.8417`, F1 `0.78`, Recall `0.89`, Precision `0.70` at threshold `0.389`. NHANES external: AUC `0.826`, F1 `0.75`, Recall `0.80`.
- **Limitations:** (1) Applies SMOTE *before* train/test splitting (Section 3.3)—a methodological data-leakage flaw we must strictly avoid; (2) uses tree ensembles only (no TabTransformer); (3) does not report quantitative calibration metrics (ECE, Brier score); (4) does not test missingness/noise degradation or SHAP stability.
- **Relevance:** Strongly validates PRC-based threshold optimization (`< 0.50`) to reduce missed diabetic cases—directly relevant to our baseline TabTransformer's `1,186` false negatives!

### 3.3 Khan et al. (2026) — Leakage-Free SMOTE + Calibration + SHAP
- **Full Title:** A comparative and interpretable machine learning framework for reliable diabetes risk prediction
- **Authors:** Talha Farooq Khan, Mariyam Saeed, Majid Hussain, Lal Khan, Mohammad Zubair Khan, Adnan Nadeem, Turki Alghamdi
- **Journal / Date:** *Scientific Reports*, Vol. 16, Article 31218, 2026
- **DOI:** [https://doi.org/10.1038/s41598-026-61038-z](https://doi.org/10.1038/s41598-026-61038-z)
- **Local Source:** [s41598-026-61038-z.pdf](file:///c:/Users/hp/.gemini/antigravity-ide/scratch/model-training-diabeties-ds-/researchpapers/s41598-026-61038-z.pdf)
- **Methodology:** Evaluates LR, Naïve Bayes, AdaBoost, and XGBoost on PIDD (`768` patients, 8 features) with physiological-zero imputation, standardization, SMOTE strictly *inside* stratified 10-fold CV folds (testing on untouched original unbalanced folds), calibration curves (referencing Brier Score and ECE), ROC-AUC, PR-AUC, and fold-level SHAP stability checks.
- **Reported Results:** XGBoost achieves `94%` accuracy, `0.73` precision, `0.86` recall, `0.78` F1 on untouched folds (explicitly revising down from an earlier `97%` figure that leaked SMOTE into evaluation).
- **Limitations:** Single small dataset (PIDD, `768` rows); classical/boosting models only (no TabTransformer); no post-hoc recalibration comparison; no systematic missingness or noise stress-testing of accuracy, calibration, or SHAP stability.
- **Relevance:** Reinforces strict leakage-free methodology and demonstrates that calibration + SHAP stability are emerging 2026 expectations in *Scientific Reports*.

### 3.4 Syahla et al. (2025) & Local Audit (`diabetes_random_forest_reproduction/`)
- **Full Title:** Klasifikasi Indikator Kesehatan Diabetes Menggunakan Algoritma Random Forest
- **Authors:** Haura Syahla, Haris Izzudin, Fariz Aditya Pratama, Beni Rahmatullah, Ahmad Jurnaidi Wahidin, Ika Kurniawati
- **Journal / Date:** *JUTITI*, Vol. 5, No. 3, pp. 401–416, December 2025 | DOI: [10.55606/jutiti.v5i3.6338](https://doi.org/10.55606/jutiti.v5i3.6338)
- **Audit Finding:** Reported `99.6%` accuracy predicting `diabetes_stage` on the exact same 100k Kaggle dataset, which our local sibling audit proved was caused by target leakage from `diagnosed_diabetes`. Removing `diagnosed_diabetes` and `diabetes_risk_score` drops RF accuracy to `91.73%` and exposes `0.5501` Macro F1 (`0.0000` F1 on `Type 1` and `Gestational`).

### 3.5 Additional 2025–2026 Findings on SHAP Stability & Tabular Calibration
- Recent 2025–2026 XAI and tabular deep learning studies note two critical vulnerabilities directly relevant to clinical deployment:
  1. **Calibration Drift in Deep Tabular Models:** High AUROC in neural tabular models does not guarantee calibrated clinical probabilities; post-hoc calibration (Temperature Scaling, Platt Scaling, Isotonic Regression) and proper scoring rules (Brier score, ECE) are increasingly mandated.
  2. **SHAP Attribution Sensitivity Under Noise/Imputation:** While SHAP rankings can be stable across clean CV folds (Kendall's $\tau \approx 0.88$–$0.92$ on simple benchmarks), introducing Gaussian noise or stochastic imputation in complex clinical data causes measurable Jaccard/rank drift in local and global attributions—an area rarely studied for **TabTransformer**.

