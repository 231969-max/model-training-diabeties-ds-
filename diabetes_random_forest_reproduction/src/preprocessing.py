import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

TARGET_COLUMN = 'diabetes_stage'
LEAKAGE_COLUMNS = ['diagnosed_diabetes', 'diabetes_risk_score']

def preprocess_faithful(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Reproduces the exact Orange Data Mining feature setup from the paper (Section 3 & Gambar 8).
    - Target: diabetes_stage
    - Features: ALL 30 remaining columns, including 'diagnosed_diabetes' and 'diabetes_risk_score'.
    - No artificial scaling, outlier removal, or SMOTE (faithful to Orange workflow).
    """
    df_clean = df.copy()
    y = df_clean[TARGET_COLUMN].copy()
    X = df_clean.drop(columns=[TARGET_COLUMN]).copy()
    
    # Encode categorical features as discrete numeric codes (as done internally by Orange tree algorithms)
    cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    for col in cat_cols:
        X[col] = pd.factorize(X[col])[0]
        
    feature_names = X.columns.tolist()
    return X, y, feature_names

def preprocess_leakage_aware(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Preprocesses data for Experiment B (Improved / Leakage-Aware):
    - Removes explicit target-derived leakage features: 'diagnosed_diabetes' and 'diabetes_risk_score'.
    - Encodes categorical predictors.
    """
    df_clean = df.copy()
    y = df_clean[TARGET_COLUMN].copy()
    
    cols_to_drop = [TARGET_COLUMN] + [col for col in LEAKAGE_COLUMNS if col in df_clean.columns]
    X = df_clean.drop(columns=cols_to_drop).copy()
    
    cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    for col in cat_cols:
        X[col] = pd.factorize(X[col])[0]
        
    feature_names = X.columns.tolist()
    return X, y, feature_names
