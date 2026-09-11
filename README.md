# Machine Learning Classification Lab — Diabetes Prediction

An end-to-end Machine Learning classification pipeline built on a dataset of **100,000 patient records**. This project compares a linear baseline (**Logistic Regression**) against an ensemble model (**Random Forest Classifier**) to predict diabetes diagnosis (`diagnosed_diabetes`), following best practices in data preprocessing, leakage prevention, and clinical evaluation.

---

## 📊 Project Highlights

* **Dataset Size:** 100,000 records
* **Feature Dimension:** 47 features after One-Hot Encoding and Standard Scaling (exceeds the $\ge 35$ feature lab requirement)
* **Target Variable:** `diagnosed_diabetes` (Binary: `0` = Healthy, `1` = Diagnosed Diabetes)
* **Data Partition:** Strict 60% Training (60,000 samples) / 40% Testing (40,000 samples) stratified split (`random_state=42`)
* **Target Leakage Handling:** `diabetes_stage` was strictly excluded from features because staging is a diagnostic consequence that deterministically leaks the label (`Type 2` $\rightarrow$ 100% diabetic).
* **Leak-Free Preprocessing:** All `ColumnTransformer` scalers and encoders are fitted **strictly on training data** and applied to test data.

---

## 🏆 Final Model Comparison

Evaluated on **40,000 unseen test records**:

| Metric | Logistic Regression | Random Forest | Better Performing Model | Note |
|---|---:|---:|---|---|
| **Accuracy** | 86.00% | **91.99%** | **Random Forest** | Higher is better (+5.99 pp) |
| **Precision** | 87.55% | **99.85%** | **Random Forest** | Higher is better (+12.30 pp) |
| **Recall (Sensitivity)** | **89.38%** | 86.77% | **Logistic Regression** | Higher is better (+2.61 pp) |
| **False Positives (FP)** | 3,050 | **31** | **Random Forest** | Lower is better (-3,019 false alarms) |
| **False Negatives (FN)** | **2,549** | 3,174 | **Logistic Regression** | Lower is better (-625 missed cases) |

### Confusion Matrices

* **Logistic Regression:**
  $$\begin{bmatrix} \text{TN}=12,951 & \text{FP}=3,050 \\ \text{FN}=2,549 & \text{TP}=21,450 \end{bmatrix}$$
* **Random Forest:**
  $$\begin{bmatrix} \text{TN}=15,970 & \text{FP}=31 \\ \text{FN}=3,174 & \text{TP}=20,825 \end{bmatrix}$$

---

## 💡 Key Clinical Takeaway

* **Random Forest** is the **stronger overall model**, achieving **91.99% accuracy** and near-perfect **99.85% precision** while reducing false alarms from 3,050 to just 31.
* **Logistic Regression** retains an advantage in **Recall (89.38% vs. 86.77%)**, which is valuable in early-stage public health screening where minimizing missed diagnoses (False Negatives) is the primary priority.

---

## 📁 Repository Structure

```text
├── diabetes.csv                       # Dataset (100,000 rows, 31 columns)
├── ML_Lab_Log.md                      # Comprehensive permanent lab log & viva preparation guide
├── commands.txt                       # Quick-reference terminal commands
├── step4_preprocessing.py             # Feature scaling (StandardScaler) & encoding (OneHotEncoder)
├── step5_logistic_regression.py       # Logistic Regression pipeline training & test inference
├── step6_logistic_evaluation.py       # Logistic Regression metrics & confusion matrix
├── step7_random_forest.py             # Random Forest pipeline training (100 trees)
├── step8_random_forest_evaluation.py  # Random Forest metrics & confusion matrix
└── step9_model_comparison.py          # Empirical side-by-side comparative analysis
```

---

## 🚀 How to Run the Scripts Manually

Navigate to the repository folder:

```powershell
cd c:\Users\231969\Downloads\archive
```

Run using the Anaconda Python environment:

```powershell
# Step 4: Preprocessing pipeline & feature dimension audit
& "C:\ProgramData\anaconda3\python.exe" step4_preprocessing.py

# Step 5: Train Logistic Regression model
& "C:\ProgramData\anaconda3\python.exe" step5_logistic_regression.py

# Step 6: Evaluate Logistic Regression (Accuracy, Precision, Recall, CM)
& "C:\ProgramData\anaconda3\python.exe" step6_logistic_evaluation.py

# Step 7: Train Random Forest model (100 trees)
& "C:\ProgramData\anaconda3\python.exe" step7_random_forest.py

# Step 8: Evaluate Random Forest (Accuracy, Precision, Recall, CM)
& "C:\ProgramData\anaconda3\python.exe" step8_random_forest_evaluation.py

# Step 9: Run side-by-side model comparison & viva summary
& "C:\ProgramData\anaconda3\python.exe" step9_model_comparison.py
```

For detailed mathematical formulas, step-by-step logs, and viva Q&A explanations, refer to [`ML_Lab_Log.md`](ML_Lab_Log.md).
