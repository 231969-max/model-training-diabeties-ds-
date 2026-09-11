"""
=============================================================================
Step 7: Random Forest Model Training Pipeline
=============================================================================

Viva & Lab Concepts (Explained in Simple Terms):

1. WHAT IS A RANDOM FOREST?
   A Random Forest is an ensemble machine learning algorithm that builds a collection 
   ("forest") of many individual Decision Trees and combines their predictions.
   Instead of relying on a single tree that might overfit or make mistakes, the forest 
   combines the intelligence of multiple trees to make a much more accurate and 
   stable prediction.

2. HOW DOES IT WORK (BAGGING & MAJORITY VOTING)?
   - Bootstrap Aggregating (Bagging): Each decision tree in the forest is trained on 
     a random subset of the training data (sampled with replacement).
   - Feature Randomness: At each split point in a tree, only a random subset of 
     features is considered. This ensures the trees are diverse and don't all look identical.
   - Majority Voting: When predicting for a new patient:
     * Every individual tree votes: 1 (Diabetic) or 0 (Healthy).
     * The forest counts the votes.
     * The class with the majority of votes becomes the final prediction.

3. WHY USE RANDOM FOREST AFTER LOGISTIC REGRESSION?
   - Non-Linear Pattern Recognition: While Logistic Regression assumes a linear 
     boundary between features and log-odds, Random Forest can effortlessly capture 
     complex non-linear interactions (e.g. high BMI *combined with* high glucose 
     and family history).
   - Robustness: It handles complex tabular datasets exceptionally well, resists 
     overfitting through averaging, and provides a benchmark to compare against our 
     linear baseline.

4. WHY PIPELINE?
   Wrapping ColumnTransformer and RandomForestClassifier in a single scikit-learn 
   Pipeline ensures zero data leakage, automatic transformation of test data, 
   and clean, reproducible code.
=============================================================================
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# -----------------------------------------------------------------------------
# 1. Load Dataset
# -----------------------------------------------------------------------------
csv_path = 'diabetes.csv'
initial_mtime = os.path.getmtime(csv_path)

df = pd.read_csv(csv_path)

# -----------------------------------------------------------------------------
# 2. Target and Feature Separation (Excluding Target Leakage)
# -----------------------------------------------------------------------------
# 'diagnosed_diabetes' is our classification target (y)
# 'diabetes_stage' is excluded because it causes severe target leakage
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
# n_estimators=100 builds an ensemble of 100 decision trees
# random_state=42 guarantees reproducibility
# n_jobs=-1 utilizes all available CPU cores for parallel training
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# -----------------------------------------------------------------------------
# 6. Train Model on Training Data ONLY
# -----------------------------------------------------------------------------
model.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# 7. Generate Predictions on Test Set
# -----------------------------------------------------------------------------
y_pred = model.predict(X_test)

# Verify counts and integrity
features_before = X.shape[1]
features_after = len(model.named_steps['preprocessor'].get_feature_names_out())
pred_count = len(y_pred)
is_count_verified = (pred_count == len(y_test))

final_mtime = os.path.getmtime(csv_path)
csv_unmodified = (initial_mtime == final_mtime)

# -----------------------------------------------------------------------------
# 8. Print Formatted Execution Output
# -----------------------------------------------------------------------------
print("=== STEP 7: RANDOM FOREST MODEL TRAINING ===")
print()
print("Model: Random Forest Classifier")
print()
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print()
print(f"Input features before preprocessing: {features_before}")
print(f"Features after preprocessing: {features_after}")
print()
print("Number of trees (n_estimators): 100")
print()
print("Training completed: True")
print("Predictions generated: True")
print(f"Number of test predictions: {pred_count}")
print()
print("Example predictions (first 10 test samples):")
for i in range(10):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    print(f"Sample {i+1:2d} -> Actual: {actual} | Predicted: {predicted}")

print()
print("=== VERIFICATION ===")
print(f"Number of predictions = {pred_count}")
print(f"Prediction count verification: {is_count_verified}")
print(f"Original diabetes.csv modified: {not csv_unmodified}")
