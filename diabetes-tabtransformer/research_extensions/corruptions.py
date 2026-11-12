"""Controlled, seeded clinical data corruption generators (MCAR, MAR/Targeted, Gaussian noise)."""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def inject_mcar_missingness(
    df: pd.DataFrame,
    feature_cols: List[str],
    rate: float,
    seed: int = 42,
) -> pd.DataFrame:
    """Injects Missing Completely At Random (MCAR) NaNs across feature_cols at given rate."""
    corrupted = df.copy()
    if rate <= 0.0:
        return corrupted
    rng = np.random.default_rng(seed)
    mask = rng.random((len(corrupted), len(feature_cols))) < rate
    for j, col in enumerate(feature_cols):
        col_mask = mask[:, j]
        if col_mask.any():
            # Cast integer columns to float if needed to hold NaN cleanly
            if pd.api.types.is_integer_dtype(corrupted[col].dtype):
                corrupted[col] = corrupted[col].astype(np.float64)
            corrupted.loc[corrupted.index[col_mask], col] = np.nan
    return corrupted


def inject_targeted_missingness(
    df: pd.DataFrame,
    drop_cols: List[str],
) -> pd.DataFrame:
    """Simulates missing clinical lab panel(s) by setting specified columns to NaN for all rows."""
    corrupted = df.copy()
    for col in drop_cols:
        if col in corrupted.columns:
            if pd.api.types.is_integer_dtype(corrupted[col].dtype):
                corrupted[col] = corrupted[col].astype(np.float64)
            corrupted[col] = np.nan
    return corrupted


def generate_gaussian_noise(
    n_rows: int,
    n_num: int,
    sigma: float,
    seed: int = 42,
) -> Optional[np.ndarray]:
    """Generates zero-mean Gaussian measurement noise in standardized units."""
    if sigma <= 0.0 or n_num == 0:
        return None
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=sigma, size=(n_rows, n_num)).astype(np.float32)


def build_corrupted_inputs(
    df: pd.DataFrame,
    prep: Dict,
    mcar_rate: float = 0.0,
    drop_cols: Optional[List[str]] = None,
    noise_sigma: float = 0.0,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Applies missingness (before imputation) and Gaussian noise (after scaling) cleanly."""
    from .data_pipeline import transform_features

    feature_cols = prep["cat_cols"] + prep["num_cols"]
    df_corr = df.copy()
    if mcar_rate > 0.0:
        df_corr = inject_mcar_missingness(df_corr, feature_cols, rate=mcar_rate, seed=seed)
    if drop_cols:
        df_corr = inject_targeted_missingness(df_corr, drop_cols=drop_cols)

    noise = generate_gaussian_noise(len(df_corr), len(prep["num_cols"]), sigma=noise_sigma, seed=seed + 1000)
    transformed = transform_features(df_corr, prep, noise_std_num=noise)

    # Zero out noise on values that were missing (imputed values represent fallback, or keep realistic)
    # Note: Keeping noise on observed values only when missingness + noise are combined:
    if noise is not None and transformed["m_num"].any():
        # Re-transform without noise on missing entries so imputed medians aren't doubly corrupted
        # Wait: in practice, sensor noise applies to observed measurements, while missing entries are imputed.
        transformed_clean_imp = transform_features(df_corr, prep, noise_std_num=None)
        obs_mask = 1.0 - transformed["m_num"]
        transformed["x_num"] = (
            transformed_clean_imp["x_num"] + noise * obs_mask
        ).astype(np.float32)

    return transformed

