"""
=============================================================================
STEP 9: MODEL SERIALIZATION (SAVING FULL PIPELINE AS A .PKL FILE)
=============================================================================

Lab & Viva Explanations (Beginner-Friendly):

1. WHAT IS A .PKL (PICKLE) FILE?
   - A .pkl file is a binary serialized representation of a Python object stored on disk.
   - In machine learning, training a model creates complex data structures (decision trees, 
     split thresholds, categorical encoders, scaling means/variances) inside Python's RAM.
   - A .pkl file freezes and preserves this exact trained state onto the hard drive.

2. WHY SAVE THE COMPLETE SCIKIT-LEARN PIPELINE?
   - In production (web apps, mobile apps, REST APIs), new patients arrive with raw, 
     unscaled numbers (e.g. glucose = 145, age = 52) and string categories (e.g. gender = "Male").
   - If we only saved the RandomForestClassifier, the application would crash because 
     the classifier expects pre-scaled, one-hot encoded numeric vectors.
   - By saving the ENTIRE Pipeline (ColumnTransformer + RandomForestClassifier), the loaded 
     .pkl model accepts raw patient dictionaries/DataFrames, automatically scales and encodes them, 
     and outputs the final prediction in a single call.

3. REUSABILITY & DEPLOYMENT:
   - Web frameworks (Streamlit, Flask, FastAPI) or microservices can load 
     'random_forest_diabetes_model.pkl' in milliseconds without needing to retrain 
     on 100,000 records every time.
=============================================================================
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# -----------------------------------------------------------------------------
# 1. Load Enhanced Processed Dataset
# -----------------------------------------------------------------------------
orig_csv_path = 'diabetes.csv'
processed_csv_path = 'diabetes_processed.csv'
model_filename = 'random_forest_diabetes_model.pkl'

initial_orig_mtime = os.path.getmtime(orig_csv_path)

df = pd.read_csv(processed_csv_path)

# Separate Target and Features
# 'diagnosed_diabetes' is target (y)
# 'diabetes_stage' is strictly excluded to prevent target leakage
y = df['diagnosed_diabetes']
X = df.drop(columns=['diagnosed_diabetes', 'diabetes_stage'])

categorical_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object', 'string']).columns.tolist()

# -----------------------------------------------------------------------------
# 2. Train-Test Split (60% Training / 40% Testing)
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

# -----------------------------------------------------------------------------
# 3. Build Full Scikit-Learn Pipeline
# -----------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

rf_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', rf_classifier)
])

# -----------------------------------------------------------------------------
# 4. Fit Pipeline on Training Data
# -----------------------------------------------------------------------------
print("Training full Random Forest Pipeline on 60,000 records...")
model_pipeline.fit(X_train, y_train)
print("Pipeline training completed successfully.")

# -----------------------------------------------------------------------------
# 5. Serialize and Save Pipeline to .PKL
# -----------------------------------------------------------------------------
joblib.dump(model_pipeline, model_filename)
print(f"Serialized model successfully saved to '{model_filename}'.")

# -----------------------------------------------------------------------------
# 6. Verification of Serialized .PKL Model
# -----------------------------------------------------------------------------
file_exists = os.path.exists(model_filename)
file_size_bytes = os.path.getsize(model_filename)
file_size_mb = file_size_bytes / (1024 * 1024)

# Reload model from disk to test independent persistence
loaded_pipeline = joblib.load(model_filename)
is_pipeline_instance = isinstance(loaded_pipeline, Pipeline)
has_preprocessor = 'preprocessor' in loaded_pipeline.named_steps
has_classifier = 'classifier' in loaded_pipeline.named_steps
loaded_rf = loaded_pipeline.named_steps['classifier']
n_trees = loaded_rf.n_estimators

# Verify original CSV was not modified
final_orig_mtime = os.path.getmtime(orig_csv_path)
orig_unmodified = (initial_orig_mtime == final_orig_mtime)

# -----------------------------------------------------------------------------
# 7. Test Inference on Sample Records Using the Loaded Model
# -----------------------------------------------------------------------------
sample_record = X_test.iloc[[0]]
actual_label = y_test.iloc[0]

pred_class = loaded_pipeline.predict(sample_record)[0]
pred_proba = loaded_pipeline.predict_proba(sample_record)[0]

# -----------------------------------------------------------------------------
# 8. Print Complete Terminal Output
# -----------------------------------------------------------------------------
print()
print("=== STEP 9: RANDOM FOREST MODEL SERIALIZATION ===")
print()
print(f"Model filename: {model_filename}")
print(f"File exists: {file_exists}")
print(f"File size: {file_size_bytes:,} bytes ({file_size_mb:.2f} MB)")
print()
print("Pipeline Structure Verification:")
print(f"  Is scikit-learn Pipeline: {is_pipeline_instance}")
print(f"  Contains Preprocessor (StandardScaler + OneHotEncoder): {has_preprocessor}")
print(f"  Contains Classifier (RandomForestClassifier): {has_classifier}")
print(f"  Number of trees in loaded forest: {n_trees}")
print()
print("Sample Inference Test on Loaded .pkl Model:")
print(f"  Actual ground truth class: {actual_label}")
print(f"  Predicted diabetes class: {pred_class}")
print(f"  Probability of class 0 (Healthy):  {pred_proba[0]:.4f} ({pred_proba[0]*100:.2f}%)")
print(f"  Probability of class 1 (Diabetic): {pred_proba[1]:.4f} ({pred_proba[1]*100:.2f}%)")
print()
print("=== INTEGRITY VERIFICATION ===")
print(f"Loaded model functional: True")
print(f"Original diabetes.csv modified: {not orig_unmodified}")
