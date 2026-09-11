"""
=============================================================================
Step 8: Random Forest Model Evaluation Pipeline
=============================================================================

Viva & Lab Concepts (Explained in Simple Terms):

1. WHY DO WE EVALUATE ON THE TEST SET?
   Evaluating the trained Random Forest on the 40,000 unseen test samples measures 
   how well an ensemble of decision trees generalizes to new patients. It confirms 
   whether non-linear tree patterns translate into higher classification performance.

2. CONFUSION MATRIX (CM) EXPLAINED FOR RANDOM FOREST:
   - True Negatives (TN): Correctly predicted no diabetes (healthy patient).
   - False Positives (FP): Predicted diabetes when the actual class was no diabetes (False Alarm / Type I Error).
   - False Negatives (FN): Predicted no diabetes when the actual class was diabetes (Missed Case / Type II Error).
   - True Positives (TP): Correctly predicted diabetes.

3. KEY METRICS & FORMULAS:
   - ACCURACY: (TP + TN) / (TP + TN + FP + FN)
     Percentage of all test predictions that were correct.
   - PRECISION: TP / (TP + FP)
     When the model predicts diabetes, how often that prediction is correct.
   - RECALL: TP / (TP + FN)
     Percentage of actual diabetes cases that the model successfully detects.
=============================================================================
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
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
# 'diagnosed_diabetes' = target
# 'diabetes_stage' = excluded because of target leakage
y = df['diagnosed_diabetes']
X = df.drop(columns=['diagnosed_diabetes', 'diabetes_stage'])

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# -----------------------------------------------------------------------------
# 3. Stratified 60/40 Train-Test Split
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

# -----------------------------------------------------------------------------
# 4. Preprocessing Pipeline
# -----------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# -----------------------------------------------------------------------------
# 5. Build Random Forest Pipeline
# -----------------------------------------------------------------------------
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# -----------------------------------------------------------------------------
# 6. Train on Training Data ONLY and Predict on Test Data
# -----------------------------------------------------------------------------
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -----------------------------------------------------------------------------
# 7. Calculate Official Evaluation Metrics
# -----------------------------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

# Verifications
total_tested = tn + fp + fn + tp
is_total_verified = (total_tested == len(y_test))

actual_class_0 = (y_test == 0).sum()
actual_class_1 = (y_test == 1).sum()
is_class_0_verified = ((tn + fp) == actual_class_0)
is_class_1_verified = ((fn + tp) == actual_class_1)

num_features = len(model.named_steps['preprocessor'].get_feature_names_out())

final_mtime = os.path.getmtime(csv_path)
csv_unmodified = (initial_mtime == final_mtime)

# -----------------------------------------------------------------------------
# 8. Print Results Clearly
# -----------------------------------------------------------------------------
print("=== STEP 8: RANDOM FOREST EVALUATION ===")
print()
print("Model: Random Forest Classifier")
print("Number of trees: 100")
print()
print(f"Test samples: {len(y_test)}")
print(f"Features after preprocessing: {num_features}")
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
print("=== CONFUSION MATRIX VERIFICATION ===")
print(f"TN + FP + FN + TP = {total_tested}")
print(f"Verification: {is_total_verified}")
print(f"TN + FP = {tn + fp} (actual Class 0 test samples: {actual_class_0}) -> Verified: {is_class_0_verified}")
print(f"FN + TP = {fn + tp} (actual Class 1 test samples: {actual_class_1}) -> Verified: {is_class_1_verified}")
print()
print("=== METRIC INTERPRETATIONS ===")
print(f"- Accuracy:  {accuracy * 100:.2f}% of all test predictions were correct.")
print(f"- Precision: When the model predicts diabetes, that prediction is correct {precision * 100:.2f}% of the time.")
print(f"- Recall:    The model successfully detects {recall * 100:.2f}% of all actual diabetes cases.")
print(f"- TN:        {tn} patients correctly predicted as having no diabetes.")
print(f"- FP:        {fp} patients predicted as diabetic when the actual class was no diabetes (False Alarms).")
print(f"- FN:        {fn} patients predicted as having no diabetes when the actual class was diabetes (Missed Diagnoses).")
print(f"- TP:        {tp} patients correctly predicted as diabetic.")
print()
print("=== DATASET INTEGRITY ===")
print(f"Original diabetes.csv modified: {not csv_unmodified}")
