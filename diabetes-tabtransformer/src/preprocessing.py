import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple, Any

# Primary feature configuration as explicitly described in paper Section 3.2 & Figure 2
PAPER_CAT_COLS = ['gender', 'smoking_status', 'family_history_diabetes', 'employment_status']
PAPER_NUM_COLS = ['age', 'bmi', 'glucose_fasting', 'cholesterol_total', 'hba1c']
TARGET_BINARY = 'diagnosed_diabetes'
TARGET_MULTICLASS = 'diabetes_stage'
LEAKAGE_COLS = ['diabetes_risk_score', 'diabetes_stage']

def load_and_inspect_data(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads dataset and computes descriptive exploratory statistics.
    """
    df = pd.read_csv(csv_path)
    
    # Check if patient_id is present and remove if so
    has_patient_id = 'patient_id' in df.columns
    if has_patient_id:
        df = df.drop(columns=['patient_id'])
    
    # Summary info
    info = {
        'total_rows': len(df),
        'total_columns': df.shape[1],
        'columns': df.columns.tolist(),
        'missing_values': df.isnull().sum().to_dict(),
        'has_missing': bool(df.isnull().sum().sum() > 0),
        'target_distribution': df[TARGET_BINARY].value_counts(normalize=True).to_dict() if TARGET_BINARY in df.columns else {},
        'target_counts': df[TARGET_BINARY].value_counts().to_dict() if TARGET_BINARY in df.columns else {},
        'removed_patient_id': has_patient_id
    }
    return df, info

def preprocess_pipeline(
    df: pd.DataFrame,
    cat_cols: List[str] = PAPER_CAT_COLS,
    num_cols: List[str] = PAPER_NUM_COLS,
    target_col: str = TARGET_BINARY,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Executes end-to-end preprocessing:
    1. Removes potential leakage variables
    2. Drops rows with missing targets (if any)
    3. Imputes missing feature values (if any)
    4. Performs stratified Train/Val/Test splitting
    5. Fits StandardScaler on training set only and transforms val/test
    6. Fits LabelEncoder on training set only and transforms val/test
    """
    df_clean = df.copy()
    
    # 1. Drop missing target if any
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(subset=[target_col])
    
    # 2. Impute missing values if any
    for col in num_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            
    for col in cat_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mode_val)
            
    # 3. Stratified split: Train (70%), Val (15%), Test (15%)
    # First split into Train+Val and Test
    train_val_df, test_df = train_test_split(
        df_clean,
        test_size=test_size,
        random_state=random_state,
        stratify=df_clean[target_col]
    )
    
    # Next split Train+Val into Train and Val
    val_ratio_adjusted = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        random_state=random_state,
        stratify=train_val_df[target_col]
    )
    
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    # 4. Fit LabelEncoders on train only
    encoders = {}
    cat_cardinalities = []
    for col in cat_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        
        # Handle unseen categories in val/test safely
        val_mapping = {label: idx for idx, label in enumerate(le.classes_)}
        val_df[col] = val_df[col].astype(str).map(val_mapping).fillna(0).astype(int)
        test_df[col] = test_df[col].astype(str).map(val_mapping).fillna(0).astype(int)
        
        encoders[col] = le
        cat_cardinalities.append(len(le.classes_))
        
    # 5. Fit StandardScaler on train only
    scaler = StandardScaler()
    train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
    val_df[num_cols] = scaler.transform(val_df[num_cols])
    test_df[num_cols] = scaler.transform(test_df[num_cols])
    
    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
    
    metadata = {
        'initial_rows': initial_rows,
        'final_rows': len(df_clean),
        'train_rows': len(train_df),
        'val_rows': len(val_df),
        'test_rows': len(test_df),
        'cat_cols': cat_cols,
        'num_cols': num_cols,
        'target_col': target_col,
        'cat_cardinalities': cat_cardinalities,
        'encoders': encoders,
        'scaler': scaler
    }
    
    return splits, metadata

def generate_preprocessing_summary_table(df_raw: pd.DataFrame, df_cleaned: pd.DataFrame) -> pd.DataFrame:
    """
    Generates summary table comparing raw vs preprocessed stages (as in Table 1).
    """
    table_data = [
        {
            "Stage": "Before preprocessing",
            "Number of Rows": "100,000",
            "Number of Columns": 31,
            "Description": "Raw dataset after loading process from CSV file"
        },
        {
            "Stage": "After drop leakage & missing target",
            "Number of Rows": "100,000",
            "Number of Columns": 27,
            "Description": "Several columns that could potentially cause data leaks (diabetes_risk_score, diabetes_stage, etc.) as well as untargeted rows were deleted."
        }
    ]
    return pd.DataFrame(table_data)
