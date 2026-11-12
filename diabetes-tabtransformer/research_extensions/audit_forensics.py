"""Forensic Validation Audit: Leakage Checks, Synthetic Label Check, Corruption Checks, and SHAP Sensitivity."""
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import torch

EXTENSIONS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EXTENSIONS_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))

from src.model import TabTransformer
from src.utils import set_seed

from research_extensions.config import (
    BASELINE_CHECKPOINT_PATH,
    DATA_PATH,
    EXT_METRICS_DIR,
    EXT_MODELS_DIR,
    PAPER_CAT_COLS,
    PAPER_NUM_COLS,
    SEED,
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
from research_extensions.metrics_calibration import (
    PostHocCalibrator,
    compute_full_evaluation,
    sigmoid,
)
from research_extensions.robust_model import RobustTabTransformer


def infer_base(model, inp):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(inp["x_num"]), 2048):
            e = min(s + 2048, len(inp["x_num"]))
            out.append(model(torch.tensor(inp["x_num"][s:e]), torch.tensor(inp["x_cat"][s:e])).numpy())
    return np.concatenate(out)


def main():
    set_seed(SEED)
    device = torch.device("cpu")
    df = load_raw_dataset(DATA_PATH)
    splits = split_dataset_identical(df, TARGET_BINARY, random_state=SEED)

    report = {}

    # 1. Split disjointness & class balance verification
    tr_i, va_i, te_i = set(splits["train"].index), set(splits["val"].index), set(splits["test"].index)
    report["split_audit"] = {
        "train_n": len(tr_i),
        "val_n": len(va_i),
        "test_n": len(te_i),
        "overlap_train_val": len(tr_i & va_i),
        "overlap_train_test": len(tr_i & te_i),
        "overlap_val_test": len(va_i & te_i),
        "train_pos_rate": float(splits["train"][TARGET_BINARY].mean()),
        "val_pos_rate": float(splits["val"][TARGET_BINARY].mean()),
        "test_pos_rate": float(splits["test"][TARGET_BINARY].mean()),
    }

    # 2. Below-ADA positive analysis (forensic label generation check)
    y = df[TARGET_BINARY]
    c_hb = df["hba1c"] >= 6.5
    c_fpg = df["glucose_fasting"] >= 126
    c_ppg = df["glucose_postprandial"] >= 200
    any_ada = c_hb | c_fpg | c_ppg
    sub_no_ada = df[~any_ada]
    report["synthetic_label_audit"] = {
        "p_y1_given_hba1c_ge_6_5": float(y[c_hb].mean()),
        "n_hba1c_ge_6_5": int(c_hb.sum()),
        "p_y1_given_fpg_ge_126": float(y[c_fpg].mean()),
        "n_fpg_ge_126": int(c_fpg.sum()),
        "p_y1_given_ppg_ge_200": float(y[c_ppg].mean()),
        "n_ppg_ge_200": int(c_ppg.sum()),
        "any_ada_exact_match_rate": float((any_ada.astype(int) == y).mean()),
        "any_ada_fp_negatives": int((y[any_ada] == 0).sum()),
        "below_ada_count": int(len(sub_no_ada)),
        "below_ada_pos_rate": float(sub_no_ada[TARGET_BINARY].mean()),
        "below_ada_stage_crosstab": pd.crosstab(
            sub_no_ada["diabetes_stage"], sub_no_ada[TARGET_BINARY]
        ).to_dict(),
    }

    # 3. Corruption generator verification
    prep_rob = fit_preprocessor(splits["train"], PAPER_CAT_COLS, PAPER_NUM_COLS, add_missing_cat_token=True)
    te_clean = transform_features(splits["test"], prep_rob)
    te_mcar20 = build_corrupted_inputs(splits["test"], prep_rob, mcar_rate=0.20, seed=42)
    te_drop_hb = build_corrupted_inputs(splits["test"], prep_rob, drop_cols=["hba1c"], seed=42)
    te_noise25 = build_corrupted_inputs(splits["test"], prep_rob, noise_sigma=0.25, seed=42)
    noise_diff = te_noise25["x_num"] - te_clean["x_num"]

    report["corruption_audit"] = {
        "mcar20_empirical_num_missing_rate": float(te_mcar20["m_num"].mean()),
        "mcar20_empirical_cat_missing_rate": float(te_mcar20["m_cat"].mean()),
        "mcar20_cat_missing_token_rate": float(
            np.mean([np.mean(te_mcar20["x_cat"][:, j] == prep_rob["missing_cat_ids"][c]) for j, c in enumerate(PAPER_CAT_COLS)])
        ),
        "drop_hba1c_m_num_per_col": {c: float(v) for c, v in zip(PAPER_NUM_COLS, te_drop_hb["m_num"].mean(axis=0))},
        "noise_0.25_empirical_mean": float(noise_diff.mean()),
        "noise_0.25_empirical_std": float(noise_diff.std()),
    }

    # 4. Untouched Validation-Only Calibrator Selection (Clean Val + Corrupted Val)
    prep_base = fit_preprocessor(splits["train"], PAPER_CAT_COLS, PAPER_NUM_COLS, add_missing_cat_token=False)
    model_a0 = TabTransformer(prep_base["cat_cardinalities"], len(PAPER_NUM_COLS), 32, 8, 4, 128, 0.1, [64, 32], 0.2, 1)
    model_a0.load_state_dict(torch.load(BASELINE_CHECKPOINT_PATH, map_location="cpu")["model_state_dict"])
    model_a0.eval()

    val_c = transform_features(splits["val"], prep_base)
    val_m20 = build_corrupted_inputs(splits["val"], prep_base, mcar_rate=0.20, seed=42)
    val_dhb = build_corrupted_inputs(splits["val"], prep_base, drop_cols=["hba1c"], seed=42)
    vl_c, vl_m20, vl_dhb = infer_base(model_a0, val_c), infer_base(model_a0, val_m20), infer_base(model_a0, val_dhb)
    yv = splits["val"][TARGET_BINARY].to_numpy(dtype=int)

    val_cal_audit = {}
    for m in ["none", "temperature", "platt", "isotonic"]:
        cal = PostHocCalibrator(m).fit(vl_c, yv)
        ev_c = compute_full_evaluation(yv, cal.predict_proba(vl_c))
        ev_m = compute_full_evaluation(yv, cal.predict_proba(vl_m20))
        ev_d = compute_full_evaluation(yv, cal.predict_proba(vl_dhb))
        val_cal_audit[m] = {
            "val_clean_auc": ev_c["roc_auc"],
            "val_clean_ece": ev_c["ece"],
            "val_clean_nll": ev_c["nll"],
            "val_mcar20_auc": ev_m["roc_auc"],
            "val_mcar20_ece": ev_m["ece"],
            "val_mcar20_nll": ev_m["nll"],
            "val_drophba1c_auc": ev_d["roc_auc"],
            "val_drophba1c_nll": ev_d["nll"],
            "mean_val_nll_clean_mcar20": float(0.5 * (ev_c["nll"] + ev_m["nll"])),
        }
    report["val_only_calibrator_selection_audit"] = val_cal_audit

    # 5. SHAP Additivity & Approximation Sensitivity Check (n=100 stratified test patients)
    y_train = splits["train"][TARGET_BINARY].to_numpy(dtype=int)
    y_test = splits["test"][TARGET_BINARY].to_numpy(dtype=int)
    tr_c = transform_features(splits["train"], prep_base)
    te_c_b = transform_features(splits["test"], prep_base)

    bg_25_idx = select_stratified_indices(y_train, 25, seed=SEED)
    bg_100_idx = select_stratified_indices(y_train, 100, seed=SEED + 1)
    samp_100_idx = select_stratified_indices(y_test, 100, seed=SEED)

    bg_25 = np.hstack([tr_c["x_cat"][bg_25_idx], tr_c["x_num"][bg_25_idx]])
    bg_100 = np.hstack([tr_c["x_cat"][bg_100_idx], tr_c["x_num"][bg_100_idx]])
    samp_100 = np.hstack([te_c_b["x_cat"][samp_100_idx], te_c_b["x_num"][samp_100_idx]])
    samp_m20 = build_corrupted_inputs(splits["test"].iloc[samp_100_idx], prep_base, mcar_rate=0.20, seed=SEED)
    samp_100_m20 = np.hstack([samp_m20["x_cat"], samp_m20["x_num"]])

    pfn = build_tabtransformer_predict_fn(model_a0, len(PAPER_CAT_COLS), len(PAPER_NUM_COLS), device, batch_size=2048)
    fnames = PAPER_CAT_COLS + PAPER_NUM_COLS

    sv_fast_clean = compute_shap_values(pfn, bg_25, samp_100, fnames, seed=SEED, max_evals=110)
    sv_fast_m20 = compute_shap_values(pfn, bg_25, samp_100_m20, fnames, seed=SEED, max_evals=110)
    sv_hi_clean = compute_shap_values(pfn, bg_100, samp_100, fnames, seed=SEED, max_evals=250)
    sv_hi_m20 = compute_shap_values(pfn, bg_100, samp_100_m20, fnames, seed=SEED, max_evals=250)

    # Local accuracy (additivity) check: sum(shap) + E[f(bg)] vs f(x)
    preds_100 = pfn(samp_100)
    base_val_25 = float(np.mean(pfn(bg_25)))
    add_err_25 = np.abs((sv_fast_clean.sum(axis=1) + base_val_25) - preds_100)
    base_val_100 = float(np.mean(pfn(bg_100)))
    add_err_100 = np.abs((sv_hi_clean.sum(axis=1) + base_val_100) - preds_100)

    stab_fast = compute_shap_stability_metrics(sv_fast_clean, sv_fast_m20, top_k_list=[3, 5])
    stab_hi = compute_shap_stability_metrics(sv_hi_clean, sv_hi_m20, top_k_list=[3, 5])
    # Also cross-compare clean SHAP vectors between (bg=25, eval=110) vs (bg=100, eval=250)
    cross_clean = compute_shap_stability_metrics(sv_hi_clean, sv_fast_clean, top_k_list=[3, 5])

    report["shap_sensitivity_audit"] = {
        "additivity_max_abs_err_bg25_eval110": float(np.max(add_err_25)),
        "additivity_mean_abs_err_bg25_eval110": float(np.mean(add_err_25)),
        "additivity_max_abs_err_bg100_eval250": float(np.max(add_err_100)),
        "clean_cross_agreement_top3_jaccard": cross_clean["local_jaccard_top3_mean"],
        "clean_cross_agreement_cosine": cross_clean["local_cosine_sim_mean"],
        "clean_cross_agreement_global_tau": cross_clean["global_kendall_tau"],
        "mcar20_stability_bg25_eval110": {
            "top3_jaccard": stab_fast["local_jaccard_top3_mean"],
            "local_tau": stab_fast["local_kendall_tau_mean"],
            "cosine": stab_fast["local_cosine_sim_mean"],
        },
        "mcar20_stability_bg100_eval250": {
            "top3_jaccard": stab_hi["local_jaccard_top3_mean"],
            "local_tau": stab_hi["local_kendall_tau_mean"],
            "cosine": stab_hi["local_cosine_sim_mean"],
        },
    }

    out_path = os.path.join(EXT_METRICS_DIR, "audit_forensics_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

