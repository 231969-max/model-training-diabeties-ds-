"""Leakage-free Consensus Feature Selection across all 28 clean clinical predictors (Train-only)."""
import os
from typing import Any, Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .config import ALL_CAT_COLS, ALL_NUM_COLS, SEED, TARGET_BINARY


def run_consensus_feature_selection(
    train_df: pd.DataFrame,
    cat_cols: List[str] = ALL_CAT_COLS,
    num_cols: List[str] = ALL_NUM_COLS,
    target_col: str = TARGET_BINARY,
    top_k_per_selector: int = 10,
    min_votes: int = 3,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Fits 4 diverse feature selectors strictly on training data:
      1. ANOVA F-score (linear univariate; OHE max-aggregated for categoricals)
      2. Mutual Information (non-linear dependency; discrete-aware)
      3. Random Forest Gini Importance (non-linear tree ensemble; OHE sum-aggregated)
      4. L1-Regularized Logistic Regression (sparse linear multivariate; OHE L2-norm aggregated)
    Returns (summary_df, selected_cat_cols, selected_num_cols).
    """
    y_train = train_df[target_col].to_numpy(dtype=int)
    all_features = list(cat_cols) + list(num_cols)

    # Prepare OHE + Standardized matrix for ANOVA, RF, and L1-LR
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    x_cat_ohe = ohe.fit_transform(train_df[cat_cols].astype(str))
    ohe_feature_groups: List[Tuple[str, np.ndarray]] = []
    offset = 0
    for col, cats in zip(cat_cols, ohe.categories_):
        idxs = np.arange(offset, offset + len(cats))
        ohe_feature_groups.append((col, idxs))
        offset += len(cats)

    scaler = StandardScaler()
    x_num_std = scaler.fit_transform(train_df[num_cols].to_numpy(dtype=np.float64))
    num_offset = x_cat_ohe.shape[1]
    for j, col in enumerate(num_cols):
        ohe_feature_groups.append((col, np.array([num_offset + j])))

    X_ohe = np.hstack([x_cat_ohe, x_num_std])

    # 1. ANOVA F-score
    f_vals, _ = f_classif(X_ohe, y_train)
    f_vals = np.nan_to_num(f_vals, nan=0.0)
    anova_scores = {col: float(np.max(f_vals[idxs])) for col, idxs in ohe_feature_groups}

    # 2. Mutual Information (OrdinalEncoder for discrete categoricals + continuous numericals)
    ord_enc = OrdinalEncoder()
    x_cat_ord = ord_enc.fit_transform(train_df[cat_cols].astype(str))
    X_ord = np.hstack([x_cat_ord, train_df[num_cols].to_numpy(dtype=np.float64)])
    discrete_mask = np.array([True] * len(cat_cols) + [False] * len(num_cols))
    # Subsample 25,000 stratified rows if large to keep MI deterministic and fast (<10s)
    rng = np.random.default_rng(seed)
    if len(y_train) > 25000:
        idx0 = np.where(y_train == 0)[0]
        idx1 = np.where(y_train == 1)[0]
        n1 = int(round(25000 * len(idx1) / len(y_train)))
        n0 = 25000 - n1
        sub_idx = np.sort(np.concatenate([rng.choice(idx0, n0, replace=False), rng.choice(idx1, n1, replace=False)]))
    else:
        sub_idx = np.arange(len(y_train))
    mi_vals = mutual_info_classif(X_ord[sub_idx], y_train[sub_idx], discrete_features=discrete_mask, random_state=seed)
    mi_scores = {col: float(mi_vals[i]) for i, col in enumerate(all_features)}

    # 3. Random Forest Importance (sum across OHE dummy columns)
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=16,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_ohe, y_train)
    rf_imp = rf.feature_importances_
    rf_scores = {col: float(np.sum(rf_imp[idxs])) for col, idxs in ohe_feature_groups}

    # 4. L1-Regularized Logistic Regression (L2 norm across OHE dummy columns)
    lr_l1 = LogisticRegression(
        C=0.05,
        l1_ratio=1.0,
        solver="liblinear",
        max_iter=1000,
        random_state=seed,
    )
    lr_l1.fit(X_ohe, y_train)
    coefs = np.abs(lr_l1.coef_[0])
    l1_scores = {col: float(np.linalg.norm(coefs[idxs])) for col, idxs in ohe_feature_groups}

    # Normalize scores to [0, 1] and compute Top-K votes
    def _rank_dict(scores: Dict[str, float]) -> Dict[str, int]:
        ordered = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
        return {c: rank + 1 for rank, c in enumerate(ordered)}

    ranks_anova = _rank_dict(anova_scores)
    ranks_mi = _rank_dict(mi_scores)
    ranks_rf = _rank_dict(rf_scores)
    ranks_l1 = _rank_dict(l1_scores)

    rows = []
    for col in all_features:
        v_anova = int(ranks_anova[col] <= top_k_per_selector)
        v_mi = int(ranks_mi[col] <= top_k_per_selector)
        v_rf = int(ranks_rf[col] <= top_k_per_selector)
        v_l1 = int(ranks_l1[col] <= top_k_per_selector)
        votes = v_anova + v_mi + v_rf + v_l1
        mean_rank = (ranks_anova[col] + ranks_mi[col] + ranks_rf[col] + ranks_l1[col]) / 4.0
        rows.append(
            {
                "feature": col,
                "type": "categorical" if col in cat_cols else "numerical",
                "in_paper_9": int(col in (["gender", "smoking_status", "family_history_diabetes", "employment_status", "age", "bmi", "glucose_fasting", "cholesterol_total", "hba1c"])),
                "anova_score": anova_scores[col],
                "anova_rank": ranks_anova[col],
                "mi_score": mi_scores[col],
                "mi_rank": ranks_mi[col],
                "rf_score": rf_scores[col],
                "rf_rank": ranks_rf[col],
                "l1_lr_score": l1_scores[col],
                "l1_lr_rank": ranks_l1[col],
                "vote_anova": v_anova,
                "vote_mi": v_mi,
                "vote_rf": v_rf,
                "vote_l1_lr": v_l1,
                "consensus_votes": votes,
                "mean_rank": mean_rank,
                "selected_consensus": int(votes >= min_votes),
            }
        )

    df_fs = pd.DataFrame(rows).sort_values(["consensus_votes", "mean_rank"], ascending=[False, True]).reset_index(drop=True)
    selected = df_fs[df_fs["selected_consensus"] == 1]["feature"].tolist()
    sel_cat = [c for c in cat_cols if c in selected]
    sel_num = [c for c in num_cols if c in selected]
    return df_fs, sel_cat, sel_num


def plot_consensus_heatmap(df_fs: pd.DataFrame, save_path: str):
    """Plots normalized selector importance heatmap and consensus votes across all 28 predictors."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cols_raw = ["anova_score", "mi_score", "rf_score", "l1_lr_score"]
    norm_df = pd.DataFrame(index=df_fs["feature"])
    for c, label in zip(cols_raw, ["ANOVA F", "Mutual Info", "Random Forest", "L1-LogReg"]):
        vals = df_fs[c].to_numpy(dtype=float)
        norm_df[label] = vals / max(np.max(vals), 1e-12)
    norm_df["Votes (/4)"] = df_fs["consensus_votes"].to_numpy(dtype=float) / 4.0

    fig, ax = plt.subplots(figsize=(9.5, 9.5), dpi=300)
    annot_mat = norm_df.copy().astype(object)
    for i, feat in enumerate(df_fs["feature"]):
        for j, col_lbl in enumerate(["ANOVA F", "Mutual Info", "Random Forest", "L1-LogReg"]):
            annot_mat.iloc[i, j] = f"{norm_df.iloc[i, j]:.2f}"
        annot_mat.iloc[i, 4] = f"{int(df_fs.iloc[i]['consensus_votes'])}/4"

    sns.heatmap(
        norm_df,
        annot=annot_mat,
        fmt="",
        cmap="YlGnBu",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Normalized Importance Score"},
    )
    ax.set_title(
        "Train-Only Consensus Feature Selection Across All 28 Clinical Predictors",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("Clinical Feature", fontsize=11, fontweight="semibold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

