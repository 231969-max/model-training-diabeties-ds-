"""
=============================================================================
Step 6: Logistic Regression Model Evaluation
=============================================================================

Viva & Lab Concepts (Explained in Simple Terms):

1. WHY DO WE EVALUATE ON THE TEST SET?
   Evaluating on the 40,000 unseen test samples measures how well the model 
   generalizes to new patients. It proves whether the model actually learned 
   general clinical patterns rather than simply memorizing training data.

2. WHAT IS A CONFUSION MATRIX (CM)?
   A 2x2 grid that categorizes every test prediction against the actual truth:
   - True Negative (TN): Patient is healthy (0), model correctly predicted healthy (0).
   - False Positive (FP): Patient is healthy (0), but model incorrectly predicted diabetic (1). (False Alarm / Type I Error)
   - False Negative (FN): Patient has diabetes (1), but model missed it and predicted healthy (0). (Dangerous Miss / Type II Error)
   - True Positive (TP): Patient has diabetes (1), and model correctly identified diabetes (1).

3. KEY CLASSIFICATION METRICS & FORMULAS:
   - ACCURACY (AC):
     Formula: (TP + TN) / (TP + TN + FP + FN)
     Meaning: The proportion of all test predictions that were correct.
   - PRECISION (PR):
     Formula: TP / (TP + FP)
     Meaning: When the model predicts a patient has diabetes, how often is it right?
     High precision minimizes false alarms.
   - RECALL (Sensitivity):
     Formula: TP / (TP + FN)
     Meaning: Out of all patients who ACTUALLY have diabetes, what percentage did 
     the model catch? In medicine, high recall is vital because missing a sick 
     patient (FN) can delay critical treatment.
=============================================================================
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)

# -----------------------------------------------------------------------------
# 1. Load Dataset
# -----------------------------------------------------------------------------
csv_path = 'diabetes.csv'
initial_mtime = os.path.getmtime(csv_path)

df = pd.read_csv(csv_path)

# -----------------------------------------------------------------------------
# 2. Target and Feature Separation (Excluding Target Leakage)
# -----------------------------------------------------------------------------
y = df['diagnosed_diabetes']
X = df.drop(columns=['diagnosed_diabetes', 'diabetes_stage'])

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# -----------------------------------------------------------------------------
# 3. 60/40 Stratified Split
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

# -----------------------------------------------------------------------------
# 4. Preprocessing and Model Pipeline
# -----------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# -----------------------------------------------------------------------------
# 5. Model Training (Training Data ONLY)
# -----------------------------------------------------------------------------
model.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# 6. Generate Predictions on the Unseen Test Set
# -----------------------------------------------------------------------------
y_pred = model.predict(X_test)

# -----------------------------------------------------------------------------
# 7. Calculate Official Evaluation Metrics
# -----------------------------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()
cm_sum = tn + fp + fn + tp
is_sum_verified = (cm_sum == len(y_test))

# Verify original file was not modified
final_mtime = os.path.getmtime(csv_path)
csv_unmodified = (initial_mtime == final_mtime)

# -----------------------------------------------------------------------------
# 8. Print Formatted Evaluation Output
# -----------------------------------------------------------------------------
print("=== STEP 6: LOGISTIC REGRESSION EVALUATION ===")
print()
print("Model: Logistic Regression")
print()
print(f"Test samples: {len(y_test)}")
print()
print(f"Accuracy (AC):  {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision (PR): {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall:         {recall:.4f} ({recall * 100:.2f}%)")
print()
print("Confusion Matrix (CM):")
print(cm)
print()
print("Individual Confusion Matrix Values:")
print(f"True Negatives (TN):  {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP):  {tp}")
print()
print("Metric Interpretations:")
print(f"- Accuracy:         {accuracy:.2%} of all predictions were correct.")
print(f"- Precision:        Among all predictions of diabetes, {precision:.2%} were actually diabetic.")
print(f"- Recall:           Among all patients who actually have diabetes, the model detected {recall:.2%}.")
print("- Confusion Matrix: Shows correct (TN, TP) and incorrect (FP, FN) predictions for both classes.")
print()
print("=== VERIFICATION ===")
print(f"TN + FP + FN + TP = {cm_sum} (Expected: 40000) -> {is_sum_verified}")
print(f"Original diabetes.csv modified: {not csv_unmodified}")
