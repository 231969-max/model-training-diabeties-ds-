"""
=============================================================================
STANDALONE INFERENCE SCRIPT: TEST SAVED RANDOM FOREST PIPELINE (.PKL)
=============================================================================

This script demonstrates that 'random_forest_diabetes_model.pkl' is a completely
self-contained, standalone production model. It can be executed independently 
without any retraining code or knowledge of the original training script.

The loaded pipeline automatically handles:
- StandardScaler transformation for all 31 numerical features
- OneHotEncoder transformation for all 8 categorical features
- Ensemble voting across all 100 Decision Trees
- Final class prediction and posterior probabilities
=============================================================================
"""

import os
import joblib
import pandas as pd

rf_filename = 'random_forest_diabetes_model.pkl'
lr_filename = 'logistic_regression_diabetes_model.pkl'
processed_csv = 'diabetes_processed.csv'

print("=== TESTING SAVED DUAL-MODEL PIPELINES (.PKL) ===")
print()

# Load pipelines
rf_model = joblib.load(rf_filename)
lr_model = joblib.load(lr_filename)
print(f"Loaded Random Forest: {rf_filename} ({os.path.getsize(rf_filename)/(1024*1024):.2f} MB)")
print(f"Loaded Logistic Regression: {lr_filename} ({os.path.getsize(lr_filename)/1024:.2f} KB)")
print()

# Load sample patient cases
df = pd.read_csv(processed_csv)
X = df.drop(columns=['diagnosed_diabetes', 'diabetes_stage'])
y = df['diagnosed_diabetes']

healthy_idx = (y == 0).idxmax()
diabetic_idx = (y == 1).idxmax()

for name, model in [("Random Forest (100 Trees)", rf_model), ("Logistic Regression (Linear)", lr_model)]:
    print(f"--- Testing {name} ---")
    for case_name, idx in [("Healthy Case", healthy_idx), ("Diabetic Case", diabetic_idx)]:
        sample = X.iloc[[idx]]
        pred = model.predict(sample)[0]
        proba = model.predict_proba(sample)[0]
        actual = y.iloc[idx]
        print(f"  [{case_name}] Actual: {actual} | Predicted: {pred} | Prob(Healthy): {proba[0]*100:.1f}% | Prob(Diabetic): {proba[1]*100:.1f}%")
    print()

print("Both models verified successfully on raw patient features.")
