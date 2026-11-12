"""Leakage-free data splitting, encoding, scaling, and missingness-aware preprocessing."""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import PAPER_CAT_COLS, PAPER_NUM_COLS, SEED, TARGET_BINARY, TEST_SIZE, VAL_SIZE


def load_raw_dataset(csv_path: str) -> pd.DataFrame:
    """Loads the raw CSV dataset without modifying on-disk files."""
    df = pd.read_csv(csv_path)
    if "patient_id" in df.columns:
        df = df.drop(columns=["patient_id"])
    return df


def split_dataset_identical(
    df: pd.DataFrame,
    target_col: str = TARGET_BINARY,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = SEED,
) -> Dict[str, pd.DataFrame]:
    """
    Produces the EXACT same stratified 70/15/15 Train/Val/Test split as baseline
    src/preprocessing.py (split BEFORE any imputation, scaling, encoding, or FS).
    """
    df_clean = df.dropna(subset=[target_col]).copy()
    train_val_df, test_df = train_test_split(
        df_clean,
        test_size=test_size,
        random_state=random_state,
        stratify=df_clean[target_col],
    )
    val_ratio_adjusted = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        random_state=random_state,
        stratify=train_val_df[target_col],
    )
    return {
        "train": train_df.copy(),
        "val": val_df.copy(),
        "test": test_df.copy(),
    }


def _normalize_cat_series(series: pd.Series) -> pd.Series:
    """Normalizes non-null categorical values to strings ('0.0' -> '0') while preserving NaNs."""
    if pd.api.types.is_numeric_dtype(series.dtype):
        def _fmt(v):
            if pd.isna(v):
                return np.nan
            fv = float(v)
            return str(int(fv)) if fv.is_integer() else str(v)
        return series.map(_fmt)
    return series.where(series.isna(), series.astype(str))


def fit_preprocessor(
    train_df: pd.DataFrame,
    cat_cols: List[str] = PAPER_CAT_COLS,
    num_cols: List[str] = PAPER_NUM_COLS,
    add_missing_cat_token: bool = False,
) -> Dict[str, Any]:
    """
    Fits imputers, LabelEncoders, and StandardScaler strictly on training data.
    When add_missing_cat_token=True, reserves index `len(classes_)` for missing/unseen categories.
    """
    num_medians = {col: float(train_df[col].median()) for col in num_cols}
    cat_modes = {col: str(_normalize_cat_series(train_df[col]).dropna().mode().iloc[0]) for col in cat_cols}

    encoders: Dict[str, LabelEncoder] = {}
    cat_mappings: Dict[str, Dict[str, int]] = {}
    missing_cat_ids: Dict[str, int] = {}
    cat_cardinalities: List[int] = []

    train_cat_encoded = pd.DataFrame(index=train_df.index)
    for col in cat_cols:
        le = LabelEncoder()
        series_str = _normalize_cat_series(train_df[col]).fillna(cat_modes[col]).astype(str)
        train_cat_encoded[col] = le.fit_transform(series_str)
        mapping = {label: int(idx) for idx, label in enumerate(le.classes_)}
        encoders[col] = le
        cat_mappings[col] = mapping
        if add_missing_cat_token:
            miss_id = len(le.classes_)
            missing_cat_ids[col] = miss_id
            cat_cardinalities.append(len(le.classes_) + 1)
        else:
            missing_cat_ids[col] = 0
            cat_cardinalities.append(len(le.classes_))

    train_num_filled = train_df[num_cols].copy()
    for col in num_cols:
        train_num_filled[col] = train_num_filled[col].fillna(num_medians[col])

    scaler = StandardScaler()
    scaler.fit(train_num_filled)

    return {
        "cat_cols": list(cat_cols),
        "num_cols": list(num_cols),
        "num_medians": num_medians,
        "cat_modes": cat_modes,
        "encoders": encoders,
        "cat_mappings": cat_mappings,
        "missing_cat_ids": missing_cat_ids,
        "cat_cardinalities": cat_cardinalities,
        "scaler": scaler,
        "add_missing_cat_token": add_missing_cat_token,
    }


def transform_features(
    df: pd.DataFrame,
    prep: Dict[str, Any],
    noise_std_num: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Transforms a dataframe (possibly containing NaNs) using train-fitted preprocessor.
    Returns:
        x_num: standardized numerical array [N, n_num] (plus optional additive Gaussian noise on observed/all)
        x_cat: integer-encoded categorical array [N, n_cat]
        m_num: binary missingness mask [N, n_num] (1.0 where original value was NaN, 0.0 otherwise)
        m_cat: binary missingness mask [N, n_cat] (1.0 where original value was NaN, 0.0 otherwise)
    """
    cat_cols: List[str] = prep["cat_cols"]
    num_cols: List[str] = prep["num_cols"]
    n_rows = len(df)

    # Missingness masks recorded prior to imputation
    m_num = df[num_cols].isna().to_numpy(dtype=np.float32) if num_cols else np.zeros((n_rows, 0), dtype=np.float32)
    m_cat = df[cat_cols].isna().to_numpy(dtype=np.float32) if cat_cols else np.zeros((n_rows, 0), dtype=np.float32)

    # Impute & scale numericals
    if num_cols:
        num_df = df[num_cols].copy()
        for col in num_cols:
            if num_df[col].isna().any():
                num_df[col] = num_df[col].fillna(prep["num_medians"][col])
        x_num = prep["scaler"].transform(num_df).astype(np.float32)
        if noise_std_num is not None:
            x_num = (x_num + noise_std_num).astype(np.float32)
    else:
        x_num = np.zeros((n_rows, 0), dtype=np.float32)

    # Impute or assign [MISSING] token for categoricals
    if cat_cols:
        x_cat = np.zeros((n_rows, len(cat_cols)), dtype=np.int64)
        for j, col in enumerate(cat_cols):
            mapping = prep["cat_mappings"][col]
            is_na = df[col].isna().to_numpy()
            norm_s = _normalize_cat_series(df[col])
            if prep["add_missing_cat_token"]:
                # Map valid values, assign dedicated [MISSING] token where NaN or unseen
                vals = norm_s.astype(str).map(mapping)
                arr = vals.fillna(prep["missing_cat_ids"][col]).to_numpy(dtype=np.int64, copy=True)
                arr[is_na] = prep["missing_cat_ids"][col]
            else:
                # Baseline-compatible: impute missing with train mode, unseen fallback to 0
                filled = norm_s.fillna(prep["cat_modes"][col]).astype(str)
                arr = filled.map(mapping).fillna(0).to_numpy(dtype=np.int64, copy=True)
            x_cat[:, j] = arr
    else:
        x_cat = np.zeros((n_rows, 0), dtype=np.int64)

    return {
        "x_num": x_num,
        "x_cat": x_cat,
        "m_num": m_num,
        "m_cat": m_cat,
    }

