"""
=============================================================================
Step 5: Logistic Regression Model Training Pipeline
=============================================================================

Viva & Lab Concepts (Explained in Simple Terms):

1. WHAT IS LOGISTIC REGRESSION?
   Despite having "regression" in its name, Logistic Regression is a supervised 
   Machine Learning algorithm used for CLASSIFICATION (predicting discrete classes 
   such as 0 or 1).
   
2. HOW DOES IT WORK?
   - First, it calculates a linear combination of the weighted input features:
     z = w1*x1 + w2*x2 + ... + b
   - Next, it passes this value through a mathematical S-shaped curve called the 
     SIGMOID function:
     P(Y = 1 | X) = 1 / (1 + e^(-z))
   - This squashes any real number into a probability between 0.0 and 1.0.
   - Finally, a decision threshold (default is 0.50) is applied to make the 
     final classification:
     * If Probability >= 0.50  --> Predict 1 (Diagnosed Diabetes)
     * If Probability <  0.50  --> Predict 0 (Not Diagnosed / Healthy)

3. WHY USE LOGISTIC REGRESSION AS OUR FIRST / BASELINE MODEL?
   - Simplicity and Interpretability: It provides a clear, transparent baseline 
     where we can observe linear relationships between patient biomarkers and disease.
   - Fast and Computationally Efficient: Fits 60,000 records in seconds.
   - Benchmark Comparison: In machine learning best practices, you always start with 
     a simpler baseline model before trying complex algorithms (like Random Forest or 
     XGBoost). This allows you to evaluate whether added complexity truly yields 
     better performance.

4. WHY USE AN SKLEARN PIPELINE?
   A Pipeline bundles preprocessing (StandardScaler + OneHotEncoder) and the model 
   (LogisticRegression) into a single cohesive object.
   - Calling pipeline.fit(X_train, y_train) automatically fits the scaler and encoder 
     on X_train and trains the classifier.
   - Calling pipeline.predict(X_test) automatically transforms X_test using the 
     parameters learned from X_train and generates predictions.
   - This completely eliminates accidental data leakage and ensures clean code.
=============================================================================
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# -----------------------------------------------------------------------------
# 1. Load Dataset
# -----------------------------------------------------------------------------
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
# 3. Identify Numerical and Categorical Columns
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
# 5. Build ColumnTransformer Preprocessor (from Step 4)
# -----------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# -----------------------------------------------------------------------------
# 6. Build Single Sklearn Pipeline (Preprocessing + Logistic Regression)
# -----------------------------------------------------------------------------
# max_iter=1000 ensures gradient descent reaches full convergence without warnings.
# random_state=42 ensures identical, reproducible results.
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# -----------------------------------------------------------------------------
# 7. Train Model on Training Data ONLY
# -----------------------------------------------------------------------------
model.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# 8. Generate Predictions on the Unseen Test Set
# -----------------------------------------------------------------------------
y_pred = model.predict(X_test)

# Number of features produced by preprocessor
num_features = len(model.named_steps['preprocessor'].get_feature_names_out())

# Verify original file was untouched
final_mtime = os.path.getmtime(csv_path)
csv_unmodified = (initial_mtime == final_mtime)

# -----------------------------------------------------------------------------
# 9. Print Required Summary Output
# -----------------------------------------------------------------------------
print("=== STEP 5: LOGISTIC REGRESSION ===")
print()
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"Number of input features after preprocessing: {num_features}")
print()
print("Model:")
print("Logistic Regression")
print()
print(f"Training completed: True")
print(f"Predictions generated: True")
print(f"Number of test predictions: {len(y_pred)}")
print()

# -----------------------------------------------------------------------------
# 10. Print Example Predictions (Showing both classes)
# -----------------------------------------------------------------------------
print("Example predictions (first 10 test samples):")
for i in range(10):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    print(f"Sample {i+1:2d} -> Actual: {actual} | Predicted: {predicted}")

print()
# Show a couple of Class 0 examples to demonstrate negative predictions
print("Example predictions for confirmed negative cases (Class 0):")
zero_indices = [i for i, val in enumerate(y_test) if val == 0][:3]
for idx in zero_indices:
    actual = y_test.iloc[idx]
    predicted = y_pred[idx]
    print(f"Test index {idx:5d} -> Actual: {actual} | Predicted: {predicted}")
