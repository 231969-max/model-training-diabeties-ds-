# Clinical Diabetes Risk Prediction — Machine Learning & Web Dashboard

An end-to-end Machine Learning clinical classification system and web application built on **100,000 patient records**. This project compares an ensemble model (**Random Forest Classifier**) and a linear probability baseline (**Logistic Regression**) across **39 raw patient features** (standardized and one-hot encoded) to predict diabetes diagnosis (`diagnosed_diabetes`), deployed with an interactive, clinical-grade web dashboard.

// Changing this section
Logistic regression model used

---

## 📊 Project Overview & Highlights

* **Dataset Size:** 100,000 records × 41 columns (100% complete, zero missing values)
* **Dataset Preservation:** Original `diabetes.csv` is strictly preserved unmodified; enhanced features reside in `diabetes_processed.csv`.
* **10 Enhanced Non-Leaking Features Added:**
  1. `daily_water_intake_liters` (Hydration level)
  2. `stress_level` (Categorical: Low / Moderate / High)
  3. `fruit_vegetable_servings_per_day` (Dietary quality indicator)
  4. `annual_health_checkups` (Preventive care frequency)
  5. `waist_circumference_cm` (Central adiposity marker)
  6. `daily_steps` (Objective physical mobility)
  7. `resting_respiratory_rate` (Cardiopulmonary baseline)
  8. `vitamin_d_level_ng_ml` (Micronutrient metabolic indicator)
  9. `salt_intake_level` (Categorical: Low / Moderate / High)
  10. `medication_adherence_score` (Therapeutic compliance index)
* **Target Variable:** `diagnosed_diabetes` (Binary: `0` = Healthy / Non-Diabetic, `1` = Diagnosed Diabetes)
* **Target Leakage Prevention:** `diabetes_stage` is strictly excluded from all model feature spaces ($X$).
* **Data Partition:** Stratified 60% Training (60,000 samples) / 40% Testing (40,000 samples) (`random_state=42`).
* **Self-Contained Pipelines:** Complete preprocessors (`StandardScaler` for 31 numerical features + `OneHotEncoder` for 8 categorical features) serialized inside `.pkl` models.

---

## 🏆 Dual-Model Performance Comparison (40,000 Test Records)

Both models are trained and evaluated on the exact same **39-feature enhanced space** for fair, side-by-side clinical evaluation:

| Metric | Random Forest (100 Trees) | Logistic Regression (Linear) | Better Model | Clinical Implication |
| :--- | :---: | :---: | :---: | :--- |
| **Architecture** | Non-Linear Ensemble | Linear Baseline | — | Non-linear tree splits capture complex metabolic thresholds |
| **Accuracy** | **91.97%** | 85.97% | **Random Forest** | +6.00 pp higher overall classification accuracy |
| **Precision** | **99.83%** | 87.50% | **Random Forest** | Near-zero false alarms (**36 FP vs. 3,064 FP**) |
| **Recall (Sensitivity)** | 86.77% | **89.37%** | **Logistic Regression** | Higher sensitivity for broad initial population screening |
| **F1-Score** | **92.84%** | 88.43% | **Random Forest** | Superior harmonic balance of precision and recall |
| **ROC-AUC** | **0.9426** | 0.9341 | **Random Forest** | Higher discrimination across all clinical probability cutoffs |

### Empirical Confusion Matrices (40,000 Test Cases)

* **Random Forest Classifier:**
  $$\begin{bmatrix} \text{TN}=15,965 & \text{FP}=36 \\ \text{FN}=3,175 & \text{TP}=20,824 \end{bmatrix}$$
* **Logistic Regression:**
  $$\begin{bmatrix} \text{TN}=12,937 & \text{FP}=3,064 \\ \text{FN}=2,550 & \text{TP}=21,449 \end{bmatrix}$$

---

## 🌐 Step 10: Clinical Web Dashboard & Interactive Model Toggle

The web application (`app.py`) provides an interactive interface featuring:
* **Interactive Model Toggle:** A segmented pill toggle switch and selectable cards enabling the user to choose between **Random Forest** and **Logistic Regression** on demand.
* **Dynamic Prediction Engine:** Automatically applies `StandardScaler` and `OneHotEncoder` to raw form inputs and outputs class labels with posterior probabilities.
* **Cross-Model Check:** Automatically generates shadow predictions from the secondary model to provide comparative clinical perspective.
* **One-Click Presets:** Instant loading for typical **Healthy** and **Diabetic** patient profiles for rapid clinical verification.
* **Light Hospital Theme:** Clinical styling adhering to modern hospital EHR standards (`#F5F7FA` light-gray background, `#FFFFFF` cards, `#2563EB` medical blue, `#1F2937` typography).

---

## 📁 Repository Structure

```text
├── app.py                                 # Flask web application with dual-model toggle engine
├── diabetes.csv                           # Original pristine dataset (100k rows, 31 cols - preserved)
├── diabetes_processed.csv                 # Enhanced dataset (100k rows, 41 cols - 0 missing values)
├── random_forest_diabetes_model.pkl       # Serialized Random Forest pipeline (64.06 MB)
├── logistic_regression_diabetes_model.pkl # Serialized Logistic Regression pipeline (8.08 KB)
├── save_random_forest_model.py            # Random Forest pipeline training & serialization script
├── save_logistic_regression_model.py      # Logistic Regression pipeline training & serialization script
├── step4_preprocessing.py                 # Feature scaling & one-hot encoding audits
├── step5_logistic_regression.py           # Logistic Regression baseline training
├── step6_logistic_evaluation.py           # Logistic Regression evaluation metrics
├── step7_random_forest.py                 # Dataset preparation (10 new features) & Random Forest training
├── step8_random_forest_evaluation.py      # Random Forest comprehensive evaluation & ROC curve generation
├── step9_model_comparison.py              # Step-by-step comparative metrics
├── test_app.py                            # Flask server unit test suite (GET & POST routes)
├── test_saved_model.py                    # Standalone CLI validation of serialized .pkl pipelines
├── random_forest_roc_curve.png            # High-resolution ROC curve plot (AUC = 0.9426)
├── static/
│   ├── style.css                          # Clinical hospital dashboard stylesheet
│   └── random_forest_roc_curve.png        # Static ROC visualization asset
├── templates/
│   └── index.html                         # Responsive HTML5 dashboard template with model toggle
├── commands.txt                           # Complete inventory of terminal commands
├── ML_Lab_Log.md                          # Comprehensive permanent lab log & viva preparation notes
└── README.md                              # Project documentation
```

---

## 🚀 How to Run the Project Locally

### 1. Launch the Clinical Web Application
```powershell
python app.py
```
* Access the live dashboard at: **`http://127.0.0.1:5000`**

### 2. Run Standalone Inference Tests
```powershell
# Verify both saved .pkl pipelines on raw sample patient records
python test_saved_model.py

# Run web app integration test suite
python test_app.py
```

### 3. Re-train & Save Pipeline Models
```powershell
# Train & serialize the Random Forest pipeline
python save_random_forest_model.py

# Train & serialize the Logistic Regression pipeline
python save_logistic_regression_model.py
```

### 4. Run Step-by-Step Lab Scripts
```powershell
# Step 7: Dataset preprocessing, missing value imputation & 10 feature generation
python step7_random_forest.py

# Step 8: Comprehensive Random Forest evaluation & ROC curve generation
python step8_random_forest_evaluation.py
```

---

## 📜 Academic Integrity & Lab Compliance

1. **No Data Leakage:** `diabetes_stage` was strictly quarantined from model training.
2. **Strict Split Integrity:** All statistical scalers and encoders were fitted solely on training partitions.
3. **Reproducibility:** All random seeds are fixed (`random_state=42`) for consistent results across environments.
