"""
=============================================================================
Step 4: Categorical Encoding and Numerical Scaling Pipeline
=============================================================================

Lab Context:
This script performs data preprocessing for a Machine Learning classification lab.
Before training machine learning algorithms, raw tabular data must be converted into 
a clean, numeric, and properly scaled mathematical format.

Why are we performing these specific preprocessing steps?

1. WHY CATEGORICAL DATA NEEDS ENCODING:
   Machine learning algorithms (such as Logistic Regression, SVMs, Neural Networks)
   rely on matrix multiplications, gradients, and distance formulas. They cannot 
   directly perform math on raw text strings like "Male", "Female", or "Employed".
   Therefore, text categories must be converted into meaningful numerical values.

2. WHAT ONE-HOT ENCODING (OHE) DOES:
   One-Hot Encoding turns each distinct category within a feature into a new 
   binary column (containing 0 or 1). 
   For example, 'gender' (with categories Female, Male, Other) is converted into 
   three binary flags: 'gender_Female', 'gender_Male', and 'gender_Other'.
   This prevents the model from assuming an artificial numerical order (e.g. assigning 
   arbitrary numbers like 1, 2, 3 which would falsely imply that 3 > 2 > 1).
   We use handle_unknown='ignore' so if an unseen category appears during testing,
   it gracefully assigns zeros instead of crashing.

3. WHY NUMERICAL FEATURES ARE SCALED (StandardScaler):
   Numerical features have drastically different real-world scales and units:
   - 'age' ranges from 18 to 90
   - 'glucose_postprandial' ranges from 70 to 287
   - 'waist_to_hip_ratio' ranges from 0.67 to 1.06
   Without scaling, algorithms that compute distances or weights would be unfairly 
   dominated by features with larger numbers (like glucose or triglycerides) 
   and ignore smaller but equally important ratios (like waist-to-hip ratio).
   StandardScaler standardizes each feature to have a mean of 0 and standard deviation 
   of 1: z = (x - mean) / std.

4. WHY THE PREPROCESSOR MUST BE FITTED ONLY ON TRAINING DATA:
   Fitting the scaler or encoder on the full dataset causes "Data Leakage" (learning 
   information about the test set's mean, standard deviation, or categories in advance).
   To maintain the integrity of our test set as a true, honest evaluation of unseen data, 
   we fit() strictly on X_train, and then transform() both X_train and X_test using those
   learned training statistics.

5. WHY 'diabetes_stage' IS EXCLUDED:
   'diabetes_stage' causes severe Target Leakage. Patients staged with 'Type 2' have
   diagnosed_diabetes = 1 in 100% of cases, and those with 'No Diabetes' or 'Pre-Diabetes'
   have diagnosed_diabetes = 0 in 100% of cases. Staging is a diagnostic consequence,
   not an initial screening observation. Retaining it would allow the model to cheat 
   instead of learning genuine physiological relationships from biomarkers.
=============================================================================
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# -----------------------------------------------------------------------------
# 1. Load Dataset
# -----------------------------------------------------------------------------
# Record file modification timestamp beforehand to verify diabetes.csv is untouched
csv_path = 'diabetes.csv'
initial_mtime = os.path.getmtime(csv_path)

df = pd.read_csv(csv_path)

# -----------------------------------------------------------------------------
# 2. Separate Target (y) and Features (X)
# -----------------------------------------------------------------------------
# Exclude target 'diagnosed_diabetes' and target-leaking column 'diabetes_stage'
y = df['diagnosed_diabetes']
X = df.drop(columns=['diagnosed_diabetes', 'diabetes_stage'])

# -----------------------------------------------------------------------------
# 3. Identify Numerical and Categorical Features
# -----------------------------------------------------------------------------
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# -----------------------------------------------------------------------------
# 4. Stratified 60/40 Train-Test Split
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

# -----------------------------------------------------------------------------
# 5. Construct ColumnTransformer Preprocessing Pipeline
# -----------------------------------------------------------------------------
# We scale numerical features and one-hot encode categorical features.
# sparse_output=False returns dense NumPy arrays for easy inspection.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

# -----------------------------------------------------------------------------
# 6. Fit Preprocessor on X_train ONLY, then Transform Both Splits
# -----------------------------------------------------------------------------
preprocessor_fitted_on_train_only = True
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Retrieve all output feature names from the fitted preprocessor
all_feature_names = preprocessor.get_feature_names_out()
ohe_feature_names = [f for f in all_feature_names if f.startswith('cat__')]

# Verify that original diabetes.csv was not modified
final_mtime = os.path.getmtime(csv_path)
csv_unmodified = (initial_mtime == final_mtime)

# -----------------------------------------------------------------------------
# 7. Print Required Results
# -----------------------------------------------------------------------------
print("=== STEP 4: PREPROCESSING RESULTS ===")
print(f"Original feature count: {X.shape[1]}")
print(f"Numerical feature count: {len(numerical_cols)}")
print(f"Categorical feature count: {len(categorical_cols)}")
print(f"Categorical columns: {categorical_cols}")
print(f"Excluded columns: {['diagnosed_diabetes', 'diabetes_stage']}")
print(f"Features after One-Hot Encoding: {len(all_feature_names)}")
print(f"X_train transformed shape: {X_train_transformed.shape}")
print(f"X_test transformed shape: {X_test_transformed.shape}")
print()
print("=== VERIFICATION ===")
print(f"diagnosed_diabetes excluded: {'diagnosed_diabetes' not in X.columns}")
print(f"diabetes_stage excluded: {'diabetes_stage' not in X.columns}")
print(f"Preprocessor fitted on training data only: {preprocessor_fitted_on_train_only}")
print(f"Original diabetes.csv modified: {not csv_unmodified}")
print(f"Models trained: False")
print()

# -----------------------------------------------------------------------------
# 8. Display Generated One-Hot Encoded Categorical Feature Names
# -----------------------------------------------------------------------------
print("=== GENERATED ONE-HOT ENCODED CATEGORICAL FEATURES ===")
print(f"Total OHE binary columns generated: {len(ohe_feature_names)}")
for idx, name in enumerate(ohe_feature_names, 1):
    clean_name = name.replace('cat__', '')
    print(f"  {idx:2d}. {clean_name}")
