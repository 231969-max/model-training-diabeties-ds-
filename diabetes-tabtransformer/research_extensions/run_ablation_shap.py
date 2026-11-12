"""Stage 3C: SHAP Explainability & Explanation-Stability Audit for RobustTabTransformer (A3, A4) vs. Baseline (A0)."""
import os
import sys
import time
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

EXTENSIONS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EXTENSIONS_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))

from src.utils import set_seed

from research_extensions.config import (
    DATA_PATH,
    EXT_FIGURES_DIR,
    EXT_METRICS_DIR,
    EXT_MODELS_DIR,
    PAPER_CAT_COLS,
    PAPER_NUM_COLS,
    SEED,
    SHAP_BACKGROUND_SIZE,
    SHAP_MAX_EVALS,
    SHAP_SAMPLE_SIZE,
    SHAP_TOP_K_LIST,
    TARGET_BINARY,
)
from research_extensions.corruptions import build_corrupted_inputs
from research_extensions.data_pipeline import (
    fit_preprocessor,
    load_raw_dataset,
    split_dataset_identical,
    transform_features,
)
from research_extensions.explainability import (
    build_tabtransformer_predict_fn,
    compute_shap_stability_metrics,
    compute_shap_values,
    select_stratified_indices,
)
from research_extensions.metrics_calibration import PostHocCalibrator
from research_extensions.plotting import plot_shap_global_bar, plot_shap_stability_bars
from research_extensions.robust_model import RobustTabTransformer, infer_robust_logits


def run_shap_audit_for_model(
    model_id: str,
    model: RobustTabTransformer,
    prep: Dict,
    splits: Dict[str, pd.DataFrame],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    cat_cols: List[str],
    num_cols: List[str],
    device: torch.device,
) -> pd.DataFrame:
    """Computes clean + corrupted SHAP attributions and stability metrics for a RobustTabTransformer."""
    feature_names = cat_cols + num_cols
    imp_scaled = prep["scaler"].transform(pd.DataFrame([prep["num_medians"]])[num_cols])[0].astype(np.float32)

    # Fit Temperature calibrator strictly on Val (consistent with A3/A4)
    val_clean = transform_features(splits["val"], prep)
    val_logits = infer_robust_logits(model, val_clean["x_num"], val_clean["x_cat"], val_clean["m_num"], device)
    cal = PostHocCalibrator("temperature").fit(val_logits, y_val)

    tr_clean = transform_features(splits["train"], prep)
    te_clean = transform_features(splits["test"], prep)

    bg_idx = select_stratified_indices(y_train, SHAP_BACKGROUND_SIZE, seed=SEED)
    test_idx = select_stratified_indices(y_test, SHAP_SAMPLE_SIZE, seed=SEED)

    bg_x = np.hstack([tr_clean["x_cat"][bg_idx], tr_clean["x_num"][bg_idx]])
    clean_x = np.hstack([te_clean["x_cat"][test_idx], te_clean["x_num"][test_idx]])

    predict_fn = build_tabtransformer_predict_fn(
        model=model,
        n_cat=len(cat_cols),
        n_num=len(num_cols),
        device=device,
        calibrator=cal,
        use_missing_mask=True,
        imputed_scaled_medians=imp_scaled,
        batch_size=2048,
    )

    t0 = time.time()
    shap_clean = compute_shap_values(
        predict_fn, background=bg_x, samples=clean_x, feature_names=feature_names, seed=SEED, max_evals=SHAP_MAX_EVALS
    )
    print(f"[{model_id} Clean SHAP] shape={shap_clean.shape} in {time.time()-t0:.1f}s")

    # Save clean beeswarm plot
    plt.figure(figsize=(9.5, 5.8), dpi=300)
    shap.summary_plot(shap_clean, clean_x, feature_names=feature_names, show=False)
    plt.title(f"{model_id} — Clean Test SHAP Summary (n={SHAP_SAMPLE_SIZE})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(EXT_FIGURES_DIR, f"{model_id.lower()}_shap_beeswarm_clean.png"), dpi=300, bbox_inches="tight")
    plt.close()

    shap_scenarios = [
        {"name": "mcar_10", "mcar_rate": 0.10, "drop_cols": None, "noise_sigma": 0.0},
        {"name": "mcar_20", "mcar_rate": 0.20, "drop_cols": None, "noise_sigma": 0.0},
        {"name": "drop_glucose_fasting", "mcar_rate": 0.0, "drop_cols": ["glucose_fasting"], "noise_sigma": 0.0},
        {"name": "drop_hba1c", "mcar_rate": 0.0, "drop_cols": ["hba1c"], "noise_sigma": 0.0},
        {"name": "noise_0.10", "mcar_rate": 0.0, "drop_cols": None, "noise_sigma": 0.10},
        {"name": "noise_0.25", "mcar_rate": 0.0, "drop_cols": None, "noise_sigma": 0.25},
        {"name": "mcar_20_noise_0.25", "mcar_rate": 0.20, "drop_cols": None, "noise_sigma": 0.25},
    ]

    df_test_sub = splits["test"].iloc[test_idx].copy()
    global_imp_dict = {"Clean": np.mean(np.abs(shap_clean), axis=0)}
    stab_rows = []

    for sc in shap_scenarios:
        t_sc = time.time()
        corr_sub = build_corrupted_inputs(
            df_test_sub, prep, mcar_rate=sc["mcar_rate"], drop_cols=sc["drop_cols"], noise_sigma=sc["noise_sigma"], seed=SEED
        )
        # Put NaN sentinel in numerical columns where m_num == 1 so predict_fn activates m_num=1
        x_num_nan = corr_sub["x_num"].astype(np.float64).copy()
        x_num_nan[corr_sub["m_num"] > 0.5] = np.nan
        corrupt_x = np.hstack([corr_sub["x_cat"].astype(np.float64), x_num_nan])

        shap_corrupt = compute_shap_values(
            predict_fn, background=bg_x, samples=corrupt_x, feature_names=feature_names, seed=SEED, max_evals=SHAP_MAX_EVALS
        )
        stab = compute_shap_stability_metrics(shap_clean, shap_corrupt, top_k_list=SHAP_TOP_K_LIST)
        global_imp_dict[sc["name"]] = np.array(stab["global_mean_abs_shap_corrupt"])
        stab_rows.append(
            {
                "model": model_id,
                "scenario": sc["name"],
                **{k: v for k, v in stab.items() if not isinstance(v, list)},
            }
        )
        print(
            f"  [{model_id} | {sc['name']}] Top3-Jaccard={stab['local_jaccard_top3_mean']:.4f}, "
            f"LocalTau={stab['local_kendall_tau_mean']:.4f}, "
            f"Cosine={stab['local_cosine_sim_mean']:.4f}, "
            f"RelL2Drift={stab['local_rel_l2_drift_mean']:.4f} ({time.time()-t_sc:.1f}s)"
        )

    df_gi = pd.DataFrame({"feature": feature_names, **{k: v for k, v in global_imp_dict.items()}})
    df_gi.to_csv(os.path.join(EXT_METRICS_DIR, f"{model_id.lower()}_shap_global_importance.csv"), index=False)

    plot_shap_global_bar(
        {k: global_imp_dict[k] for k in ["Clean", "mcar_20", "drop_hba1c", "noise_0.25", "mcar_20_noise_0.25"]},
        feature_names=feature_names,
        save_path=os.path.join(EXT_FIGURES_DIR, f"{model_id.lower()}_shap_global_comparison.png"),
        title=f"{model_id} — Global SHAP Importance Shift Under Corruptions",
    )
    return pd.DataFrame(stab_rows)


def main():
    set_seed(SEED)
    device = torch.device("cpu")
    print("=" * 85)
    print(" STAGE 3C: SHAP EXPLANATION-STABILITY AUDIT (ROBUST A3 & A4 VS. BASELINE A0)")
    print("=" * 85)

    df_raw = load_raw_dataset(DATA_PATH)
    splits = split_dataset_identical(df_raw, target_col=TARGET_BINARY, random_state=SEED)
    y_train = splits["train"][TARGET_BINARY].to_numpy(dtype=int)
    y_val = splits["val"][TARGET_BINARY].to_numpy(dtype=int)
    y_test = splits["test"][TARGET_BINARY].to_numpy(dtype=int)

    # A3 (9 features)
    prep_rob9 = fit_preprocessor(splits["train"], PAPER_CAT_COLS, PAPER_NUM_COLS, add_missing_cat_token=True)
    model_a3 = RobustTabTransformer(prep_rob9["cat_cardinalities"], len(PAPER_NUM_COLS), use_missing_mask=True).to(device)
    model_a3.load_state_dict(torch.load(os.path.join(EXT_MODELS_DIR, "a3_robust_9feat.pt"), map_location=device)["model_state_dict"])
    df_stab_a3 = run_shap_audit_for_model(
        "A3_Robust9_TabTransformer", model_a3, prep_rob9, splits, y_train, y_val, y_test, PAPER_CAT_COLS, PAPER_NUM_COLS, device
    )

    # A4 (Consensus-8 features)
    df_fs = pd.read_csv(os.path.join(EXT_METRICS_DIR, "consensus_feature_selection.csv"))
    cons_rows = df_fs[df_fs["selected_consensus"] == 1]
    CONS_CAT_COLS = cons_rows[cons_rows["type"] == "categorical"]["feature"].tolist()
    CONS_NUM_COLS = cons_rows[cons_rows["type"] == "numerical"]["feature"].tolist()
    prep_cons = fit_preprocessor(splits["train"], CONS_CAT_COLS, CONS_NUM_COLS, add_missing_cat_token=True)
    model_a4 = RobustTabTransformer(prep_cons["cat_cardinalities"], len(CONS_NUM_COLS), use_missing_mask=True).to(device)
    model_a4.load_state_dict(torch.load(os.path.join(EXT_MODELS_DIR, "a4_robust_consensus8.pt"), map_location=device)["model_state_dict"])
    df_stab_a4 = run_shap_audit_for_model(
        "A4_RobustCons8_TabTransformer", model_a4, prep_cons, splits, y_train, y_val, y_test, CONS_CAT_COLS, CONS_NUM_COLS, device
    )

    # Combine with Baseline A0 SHAP Stability
    df_stab_a0 = pd.read_csv(os.path.join(EXT_METRICS_DIR, "baseline_shap_stability.csv"))
    df_all_stab = pd.concat([df_stab_a0, df_stab_a3, df_stab_a4], ignore_index=True)
    df_all_stab.to_csv(os.path.join(EXT_METRICS_DIR, "ablation_shap_stability_comparison.csv"), index=False)

    plot_shap_stability_bars(
        df_all_stab,
        scenario_col="scenario",
        group_col="model",
        save_path=os.path.join(EXT_FIGURES_DIR, "ablation_shap_stability_comparison.png"),
        title="SHAP Explanation Stability Comparison: Baseline (A0) vs. RobustTabTransformer (A3 & A4)",
    )
    print("\n" + "=" * 85)
    print(" STAGE 3C SHAP STABILITY AUDIT COMPLETED SUCCESSFULLY!")
    print("=" * 85)


if __name__ == "__main__":
    main()

