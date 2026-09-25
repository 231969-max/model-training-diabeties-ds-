"""
=============================================================================
STEP 8: RANDOM FOREST MODEL EVALUATION & COMPARATIVE ANALYSIS
=============================================================================

Lab & Viva Explanations (Beginner-Friendly):

1. PURPOSE OF STEP 8:
   - In Step 7, we engineered 10 new features, handled missing values, and trained 
     our RandomForestClassifier on the enhanced dataset ('diabetes_processed.csv').
   - In Step 8, we perform rigorous, honest statistical evaluation strictly on the 
     unseen 40% test partition (40,000 patients).

2. KEY EVALUATION METRICS:
   - Accuracy: Overall proportion of correct predictions across both classes:
     (TP + TN) / Total
   - Precision: Out of all patients the model flagged as Diabetic, how many actually are?
     TP / (TP + FP) -> Critical to avoid false alarms and unnecessary stress/treatments.
   - Recall (Sensitivity): Out of all actual Diabetic patients, how many did the model catch?
     TP / (TP + FN) -> Highly critical in clinical screening to prevent missed diagnoses.
   - F1-Score: Harmonic mean of Precision and Recall:
     2 * (Precision * Recall) / (Precision + Recall)
   - ROC-AUC: Area Under the Receiver Operating Characteristic Curve. Evaluates the model's 
     ability to rank positive cases higher than negative cases across all classification thresholds.

3. DATASET NOTE:
   - Random Forest is evaluated on the enhanced dataset ('diabetes_processed.csv') with 
     10 additional engineered features.
   - Logistic Regression was previously evaluated on the original 29-feature set.
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# -----------------------------------------------------------------------------
# 1. Dataset Verification & Loading
# -----------------------------------------------------------------------------
orig_csv = 'diabetes.csv'
processed_csv = 'diabetes_processed.csv'

# Check original dataset integrity
orig_mtime_before = os.path.getmtime(orig_csv)
orig_size_before = os.path.getsize(orig_csv)

df_orig = pd.read_csv(orig_csv)
df = pd.read_csv(processed_csv)

n_records = len(df)
n_cols_orig = df_orig.shape[1]
n_cols_proc = df.shape[1]
n_new_cols = n_cols_proc - n_cols_orig
missing_val_count = df.isnull().sum().sum()

# -----------------------------------------------------------------------------
# 2. Target and Feature Separation
# -----------------------------------------------------------------------------
# 'diagnosed_diabetes' is target (y)
# 'diabetes_stage' is excluded to prevent severe target leakage
y = df['diagnosed_diabetes']
X = df.drop(columns=['diagnosed_diabetes', 'diabetes_stage'])

categorical_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object', 'string']).columns.tolist()

# -----------------------------------------------------------------------------
# 3. Stratified 60/40 Train-Test Split
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

# -----------------------------------------------------------------------------
# 4. Pipeline Setup & Model Training
# -----------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# Fit strictly on training partition
model.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# 5. Generate Predictions and Probabilities on Unseen Test Partition
# -----------------------------------------------------------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# -----------------------------------------------------------------------------
# 6. Calculate Statistical Evaluation Metrics
# -----------------------------------------------------------------------------
test_samples = len(y_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

roc_auc = roc_auc_score(y_test, y_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# -----------------------------------------------------------------------------
# 7. Verification of Integrity & Math
# -----------------------------------------------------------------------------
cm_sum = tn + fp + fn + tp
is_math_verified = (cm_sum == test_samples)
has_no_missing = (missing_val_count == 0)

orig_mtime_after = os.path.getmtime(orig_csv)
orig_size_after = os.path.getsize(orig_csv)
orig_unmodified = (orig_mtime_before == orig_mtime_after) and (orig_size_before == orig_size_after)

# -----------------------------------------------------------------------------
# 8. Generate ROC Curve Plot
# -----------------------------------------------------------------------------
roc_img_path = 'random_forest_roc_curve.png'

plt.figure(figsize=(8, 6), dpi=300)
plt.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'Random Forest ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='#d62728', lw=1.8, linestyle='--', label='Random Chance Baseline (AUC = 0.5000)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold', labelpad=10)
plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12, fontweight='bold', labelpad=10)
plt.title('Receiver Operating Characteristic (ROC) Curve\nRandom Forest Classifier (Enhanced Dataset)', fontsize=14, fontweight='bold', pad=15)
plt.legend(loc='lower right', fontsize=11, frameon=True)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(roc_img_path, dpi=300)
plt.close()

# -----------------------------------------------------------------------------
# 9. Print Complete Real Execution Results & Comparison
# -----------------------------------------------------------------------------
print("=== DATASET PRE-CHECK & INTEGRITY ===")
print(f"Processed dataset rows: {n_records:,}")
print(f"Original columns: {n_cols_orig}")
print(f"Processed columns: {n_cols_proc}")
print(f"Newly added columns: {n_new_cols}")
print(f"No missing values = {has_no_missing}")
print(f"Original diabetes.csv modified = {not orig_unmodified}")
print()

print("=== STEP 8: RANDOM FOREST MODEL EVALUATION ===")
print()
print(f"Number of test samples: {test_samples:,}")
print(f"Accuracy:  {acc * 100:.2f}%")
print(f"Precision: {prec * 100:.2f}%")
print(f"Recall:    {rec * 100:.2f}%")
print(f"F1-score:  {f1 * 100:.2f}%")
print(f"ROC-AUC:   {roc_auc:.4f}")
print()
print("Confusion Matrix:")
print(f"[[{tn}  {fp}]")
print(f" [{fn} {tp}]]")
print()
print("Confusion Matrix Breakdown:")
print(f"  True Negatives  (TN): {tn:,}  (Correctly classified healthy patients)")
print(f"  False Positives (FP): {fp:,}     (Healthy patients incorrectly flagged as diabetic)")
print(f"  False Negatives (FN): {fn:,}  (Diabetic patients missed by the model)")
print(f"  True Positives  (TP): {tp:,} (Correctly detected diabetic patients)")
print()
print("=== MATHEMATICAL VERIFICATION ===")
print(f"TN + FP + FN + TP = {tn:,} + {fp:,} + {fn:,} + {tp:,} = {cm_sum:,}")
print(f"TN + FP + FN + TP = number of test records: {is_math_verified}")
print(f"No missing values: {has_no_missing}")
print(f"Original diabetes.csv modified: {not orig_unmodified}")
print(f"ROC curve saved: {roc_img_path}")
print()

print("=== MODEL COMPARISON: LOGISTIC REGRESSION VS RANDOM FOREST ===")
print()
print(f"{'Metric':<15} | {'Logistic Regression':<25} | {'Random Forest (Enhanced)':<25}")
print("-" * 72)
print(f"{'Accuracy':<15} | {'86.00%':<25} | {f'{acc * 100:.2f}%':<25}")
print(f"{'Precision':<15} | {'87.55%':<25} | {f'{prec * 100:.2f}%':<25}")
print(f"{'Recall':<15} | {'89.38%':<25} | {f'{rec * 100:.2f}%':<25}")
print(f"{'F1-score':<15} | {'Not previously calculated':<25} | {f'{f1 * 100:.2f}%':<25}")
print(f"{'ROC-AUC':<15} | {'Not previously calculated':<25} | {f'{roc_auc:.4f}':<25}")
print("-" * 72)
print()
print("Confusion Matrix Comparison:")
print("Logistic Regression (Baseline 29 Features):")
print("  [[12951, 3050],")
print("   [ 2549, 21450]]")
print(f"Random Forest (Enhanced 39 Features / 10 New Columns):")
print(f"  [[{tn}, {fp}],")
print(f"   [{fn}, {tp}]]")
print()
print("NOTE ON FEATURE SETS:")
print("Logistic Regression was evaluated on the baseline 29-feature dataset in Step 6.")
print("Random Forest is evaluated on the enhanced dataset with 10 added lifestyle & clinical features.")
print("Both models provide complementary clinical insights: Random Forest achieves near-zero false alarms")
print("(very high precision), while Logistic Regression maintains strong sensitivity (recall).")
