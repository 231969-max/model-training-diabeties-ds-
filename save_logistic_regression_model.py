"""
=============================================================================
STEP 9B: TRAIN & SERIALIZE LOGISTIC REGRESSION PIPELINE ON ENHANCED DATASET
=============================================================================
Ensures Logistic Regression is trained on the exact same 39-feature enhanced
dataset (diabetes_processed.csv) as the Random Forest model for 100% fair
side-by-side clinical comparison.
=============================================================================
"""

import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def main():
    print("=" * 80)
    print("SAVING LOGISTIC REGRESSION MODEL PIPELINE (ENHANCED DATASET)")
    print("=" * 80)

    dataset_path = 'diabetes_processed.csv'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset {dataset_path} not found.")

    print(f"Loading dataset: {dataset_path} ...")
    df = pd.read_csv(dataset_path)

    target_col = 'diagnosed_diabetes'
    exclude_cols = [target_col, 'diabetes_stage']

    X = df.drop(columns=exclude_cols)
    y = df[target_col]

    categorical_cols = X.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

    print(f"Total features: {len(X.columns)} ({len(numerical_cols)} numerical, {len(categorical_cols)} categorical)")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    print("Fitting Logistic Regression pipeline on entire dataset ...")
    full_pipeline.fit(X, y)

    model_output_path = 'logistic_regression_diabetes_model.pkl'
    print(f"Saving serialized pipeline to: {model_output_path} ...")
    joblib.dump(full_pipeline, model_output_path)
    file_size_kb = os.path.getsize(model_output_path) / 1024
    print(f"Successfully saved {model_output_path} ({file_size_kb:.2f} KB)")

if __name__ == '__main__':
    main()
