import os
import pandas as pd
from typing import Tuple, Dict, Any

TARGET_COLUMN = 'diabetes_stage'
DIAGNOSED_COLUMN = 'diagnosed_diabetes'

def load_dataset(csv_path: str = "data/diabetes-health-indicators-dataset.csv") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads the clinical diabetes health indicators dataset and verifies its structural properties.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[ERROR] Dataset file not found at: {csv_path}. Reproduction cannot proceed without data.")
        
    df = pd.read_csv(csv_path)
    
    # Check shape, types, missing values
    n_rows, n_cols = df.shape
    missing_dict = df.isnull().sum().to_dict()
    total_missing = sum(missing_dict.values())
    
    cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    unique_vals = {c: df[c].unique().tolist() for c in cat_cols}
    
    target_counts = df[TARGET_COLUMN].value_counts().to_dict() if TARGET_COLUMN in df.columns else {}
    target_norm = df[TARGET_COLUMN].value_counts(normalize=True).to_dict() if TARGET_COLUMN in df.columns else {}
    
    metadata = {
        'num_rows': n_rows,
        'num_cols': n_cols,
        'column_names': df.columns.tolist(),
        'missing_values': missing_dict,
        'total_missing': total_missing,
        'categorical_columns': cat_cols,
        'numerical_columns': num_cols,
        'unique_values_categorical': unique_vals,
        'target_class_counts': target_counts,
        'target_class_proportions': target_norm
    }
    
    return df, metadata

def print_dataset_summary(metadata: Dict[str, Any]):
    """
    Prints a formatted summary of the dataset.
    """
    print("=" * 80)
    print(" DATASET VERIFICATION & EXPLORATION SUMMARY")
    print("=" * 80)
    print(f"Total Rows:    {metadata['num_rows']:,}")
    print(f"Total Columns: {metadata['num_cols']}")
    print(f"Total Missing: {metadata['total_missing']}")
    print(f"\nTarget Variable: '{TARGET_COLUMN}' Class Distribution:")
    for cls, cnt in metadata['target_class_counts'].items():
        pct = metadata['target_class_proportions'][cls] * 100
        print(f"  - {cls:<15}: {cnt:>6,} samples ({pct:>5.2f}%)")
    print(f"\nCategorical Columns ({len(metadata['categorical_columns'])}): {metadata['categorical_columns']}")
    print(f"Numerical Columns ({len(metadata['numerical_columns'])}): {len(metadata['numerical_columns'])} attributes")
    print("=" * 80)

if __name__ == "__main__":
    df, meta = load_dataset()
    print_dataset_summary(meta)
