"""Stage 2: Complete Baseline Audit (Calibration, Thresholding, Corruptions, and SHAP Stability)."""
import json
import os
import sys
import time
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

# Allow importing baseline TabTransformer cleanly without modifying src/
EXTENSIONS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EXTENSIONS_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))

from src.model import TabTransformer
from src.utils import set_seed

from research_extensions.config import (
    BASELINE_CHECKPOINT_PATH,
    COMBINED_SCENARIOS,
    CORRUPTION_SEEDS,
    DATA_PATH,
    EXT_FIGURES_DIR,
    EXT_METRICS_DIR,
    MCAR_RATES,
    NOISE_SIGMAS,
    PAPER_CAT_COLS,
    PAPER_NUM_COLS,
    SEED,
    SHAP_BACKGROUND_SIZE,
    SHAP_MAX_EVALS,
    SHAP_SAMPLE_SIZE,
    SHAP_TOP_K_LIST,
    TARGETED_DROP_SETS,
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
    compute_decision_curve,
    compute_full_evaluation,
    find_prc_optimal_threshold,
    mcnemar_test,
    sigmoid,
)
from research_extensions.plotting import (
    plot_corruption_degradation,
    plot_decision_curves,
    plot_reliability_diagrams,
    plot_shap_global_bar,
    plot_shap_stability_bars,
)


def infer_logits(
    model: torch.nn.Module,
    x_num: np.ndarray,
    x_cat: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """Runs batched forward inference returning raw logits."""
    model.eval()
    logits_list = []
    n = len(x_num)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            t_num = torch.tensor(x_num[start:end], dtype=torch.float32, device=device)
            t_cat = torch.tensor(x_cat[start:end], dtype=torch.long, device=device)
            logits = model(t_num, t_cat)
            logits_list.append(logits.detach().cpu().numpy())
    return np.concatenate(logits_list, axis=0)


def main():
    os.makedirs(EXT_FIGURES_DIR, exist_ok=True)
    os.makedirs(EXT_METRICS_DIR, exist_ok=True)
    set_seed(SEED)
    device = torch.device("cpu")

    print("=" * 80)
    print(" STAGE 2: BASELINE TABTRANSFORMER AUDIT (CALIBRATION, CORRUPTION, SHAP STABILITY)")
    print("=" * 80)

    # 1. Load data & identical 70/15/15 split
    df_raw = load_raw_dataset(DATA_PATH)
    splits = split_dataset_identical(df_raw, target_col=TARGET_BINARY, random_state=SEED)
    prep = fit_preprocessor(
        splits["train"],
        cat_cols=PAPER_CAT_COLS,
        num_cols=PAPER_NUM_COLS,
        add_missing_cat_token=False,
    )

    y_train = splits["train"][TARGET_BINARY].to_numpy(dtype=int)
    y_val = splits["val"][TARGET_BINARY].to_numpy(dtype=int)
    y_test = splits["test"][TARGET_BINARY].to_numpy(dtype=int)

    train_clean = transform_features(splits["train"], prep)
    val_clean = transform_features(splits["val"], prep)
    test_clean = transform_features(splits["test"], prep)

    # 2. Load preserved baseline checkpoint
    model = TabTransformer(
        cat_cardinalities=prep["cat_cardinalities"],
        num_numerical=len(PAPER_NUM_COLS),
        embed_dim=32,
        num_heads=8,
        num_layers=4,
        ff_dim=128,
        transformer_dropout=0.1,
        mlp_hidden_dims=[64, 32],
        mlp_dropout=0.2,
        num_classes=1,
    ).to(device)
    ckpt = torch.load(BASELINE_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[Verified Checkpoint] Loaded epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")

    # 3. Clean Validation & Test inference + Baseline Parity Check
    val_logits = infer_logits(model, val_clean["x_num"], val_clean["x_cat"], device)
    test_logits = infer_logits(model, test_clean["x_num"], test_clean["x_cat"], device)

    raw_test_eval = compute_full_evaluation(y_test, sigmoid(test_logits), threshold=0.50)
    print(
        f"[Baseline Parity] Acc={raw_test_eval['accuracy']:.4f}, "
        f"AUC={raw_test_eval['roc_auc']:.4f}, "
        f"ECE={raw_test_eval['ece']:.4f}, Brier={raw_test_eval['brier_score']:.4f}, "
        f"CM=[[{raw_test_eval['tn']}, {raw_test_eval['fp']}], [{raw_test_eval['fn']}, {raw_test_eval['tp']}]]"
    )

    # 4. Fit Post-Hoc Calibrators & PRC Threshold STRICTLY on Validation Data
    calibrators: Dict[str, PostHocCalibrator] = {}
    cal_evals: Dict[str, Dict[str, Any]] = {}
    dca_curves: Dict[str, Dict[str, List[float]]] = {}
    cal_rows = []

    for method in ["none", "temperature", "platt", "isotonic"]:
        cal = PostHocCalibrator(method=method).fit(val_logits, y_val)
        calibrators[method] = cal
        val_probs = cal.predict_proba(val_logits)
        test_probs = cal.predict_proba(test_logits)

        # Standard 0.50 threshold
        ev_05 = compute_full_evaluation(y_test, test_probs, threshold=0.50)
        label_05 = f"Baseline ({method.capitalize()} @ 0.50)"
        cal_evals[label_05] = ev_05
        dca_curves[f"{method.capitalize()} Calibrated"] = compute_decision_curve(y_test, test_probs)
        cal_rows.append({"variant": label_05, "calibration": method, **{k: v for k, v in ev_05.items() if k != "reliability_diagram"}})

        # Validation PRC optimal threshold
        prc_info = find_prc_optimal_threshold(y_val, val_probs)
        tau_star = prc_info["optimal_threshold"]
        ev_prc = compute_full_evaluation(y_test, test_probs, threshold=tau_star)
        label_prc = f"Baseline ({method.capitalize()} @ PRC {tau_star:.3f})"
        cal_rows.append({"variant": label_prc, "calibration": f"{method}+PRC", **{k: v for k, v in ev_prc.items() if k != "reliability_diagram"}})

    # Select best calibrator by Validation Brier / NLL (and verify ECE)
    best_cal_name = min(
        ["temperature", "platt", "isotonic"],
        key=lambda m: compute_full_evaluation(y_val, calibrators[m].predict_proba(val_logits))["brier_score"],
    )
    best_cal = calibrators[best_cal_name]
    best_prc = find_prc_optimal_threshold(y_val, best_cal.predict_proba(val_logits))
    print(
        f"[Validation Calibration Selection] Best calibrator={best_cal_name} "
        f"(Temp={getattr(best_cal, 'temperature', 1.0):.4f}), "
        f"PRC optimal threshold={best_prc['optimal_threshold']:.4f}"
    )

    df_cal = pd.DataFrame(cal_rows)
    df_cal.to_csv(os.path.join(EXT_METRICS_DIR, "baseline_calibration_comparison.csv"), index=False)

    # Reliability diagram & DCA plots
    plot_reliability_diagrams(
        cal_evals,
        os.path.join(EXT_FIGURES_DIR, "baseline_reliability_diagrams.png"),
        title="Baseline TabTransformer Reliability Diagrams (Uncalibrated vs. Post-Hoc)",
    )
    plot_decision_curves(
        dca_curves,
        os.path.join(EXT_FIGURES_DIR, "baseline_decision_curve_analysis.png"),
        title="Baseline TabTransformer Decision Curve Analysis (Test Set)",
    )

    # McNemar test comparing Uncalibrated@0.50 vs BestCal@PRC
    pred_uncal = (sigmoid(test_logits) >= 0.50).astype(int)
    pred_cal_prc = (best_cal.predict_proba(test_logits) >= best_prc["optimal_threshold"]).astype(int)
    mcn = mcnemar_test(y_test, pred_uncal, pred_cal_prc)
    with open(os.path.join(EXT_METRICS_DIR, "baseline_calibration_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_calibrator": best_cal_name,
                "temperature_fitted": float(calibrators["temperature"].temperature),
                "prc_optimal_threshold": best_prc,
                "mcnemar_uncal05_vs_bestcal_prc": mcn,
            },
            f,
            indent=2,
        )

    # 5. Controlled Corruption Stress-Tests on Test Set
    print("\n--- Running Controlled Corruption Stress-Tests (MCAR, Targeted, Noise, Combined) ---")
    scenarios = [{"name": "clean", "regime": "clean", "level": 0.0, "mcar_rate": 0.0, "drop_cols": None, "noise_sigma": 0.0, "seeds": [SEED]}]
    for r in MCAR_RATES:
        scenarios.append({"name": f"mcar_{int(r*100)}", "regime": "mcar", "level": r, "mcar_rate": r, "drop_cols": None, "noise_sigma": 0.0, "seeds": CORRUPTION_SEEDS})
    for t_name, cols in TARGETED_DROP_SETS.items():
        scenarios.append({"name": t_name, "regime": "targeted", "level": len(cols), "mcar_rate": 0.0, "drop_cols": cols, "noise_sigma": 0.0, "seeds": [SEED]})
    for s in NOISE_SIGMAS:
        scenarios.append({"name": f"noise_{s:.2f}", "regime": "noise", "level": s, "mcar_rate": 0.0, "drop_cols": None, "noise_sigma": s, "seeds": CORRUPTION_SEEDS})
    for comb in COMBINED_SCENARIOS:
        scenarios.append({"name": comb["name"], "regime": "combined", "level": comb["mcar_rate"], "mcar_rate": comb["mcar_rate"], "drop_cols": None, "noise_sigma": comb["noise_sigma"], "seeds": CORRUPTION_SEEDS})

    corr_rows = []
    eval_modes = [
        ("A0_Baseline_Uncal_0.50", calibrators["none"], 0.50),
        (f"A1_Baseline_{best_cal_name.capitalize()}_0.50", best_cal, 0.50),
        (f"A1b_Baseline_{best_cal_name.capitalize()}_PRC", best_cal, best_prc["optimal_threshold"]),
    ]

    metric_keys = ["accuracy", "precision_1", "recall_1", "f1_1", "macro_f1", "roc_auc", "pr_auc", "brier_score", "ece", "nll", "fn", "fp"]

    for sc in scenarios:
        per_mode_runs: Dict[str, List[Dict[str, Any]]] = {m[0]: [] for m in eval_modes}
        for c_seed in sc["seeds"]:
            corr_inputs = build_corrupted_inputs(
                splits["test"],
                prep,
                mcar_rate=sc["mcar_rate"],
                drop_cols=sc["drop_cols"],
                noise_sigma=sc["noise_sigma"],
                seed=c_seed,
            )
            c_logits = infer_logits(model, corr_inputs["x_num"], corr_inputs["x_cat"], device)
            for m_name, cal_obj, thr in eval_modes:
                probs = cal_obj.predict_proba(c_logits)
                ev = compute_full_evaluation(y_test, probs, threshold=thr)
                per_mode_runs[m_name].append(ev)

        for m_name, _, thr in eval_modes:
            runs = per_mode_runs[m_name]
            row = {
                "model": m_name,
                "scenario": sc["name"],
                "regime": sc["regime"],
                "level": sc["level"],
                "threshold": thr,
                "n_seeds": len(runs),
            }
            for mk in metric_keys:
                vals = [r[mk] for r in runs]
                row[mk] = float(np.mean(vals))
                row[f"{mk}_std"] = float(np.std(vals))
            corr_rows.append(row)

    df_corr = pd.DataFrame(corr_rows)
    df_corr.to_csv(os.path.join(EXT_METRICS_DIR, "baseline_corruption_results.csv"), index=False)
    print("[Saved] baseline_corruption_results.csv")

    # Plot MCAR & Noise degradation curves
    df_mcar = df_corr[df_corr["regime"].isin(["clean", "mcar"])].copy()
    plot_corruption_degradation(
        df_mcar,
        x_col="level",
        x_label="MCAR Missingness Rate",
        group_col="model",
        metrics=["accuracy", "recall_1", "roc_auc", "ece", "brier_score", "pr_auc"],
        metric_labels=["Accuracy", "Diabetic Recall (Sensitivity)", "ROC-AUC", "Expected Calibration Error (ECE ↓)", "Brier Score (↓)", "PR-AUC"],
        save_path=os.path.join(EXT_FIGURES_DIR, "baseline_mcar_degradation.png"),
        title="Baseline TabTransformer Degradation Under MCAR Missingness (5%–30%)",
    )

    df_noise = df_corr[df_corr["regime"].isin(["clean", "noise"])].copy()
    plot_corruption_degradation(
        df_noise,
        x_col="level",
        x_label="Gaussian Measurement Noise (Std Dev σ)",
        group_col="model",
        metrics=["accuracy", "recall_1", "roc_auc", "ece", "brier_score", "pr_auc"],
        metric_labels=["Accuracy", "Diabetic Recall (Sensitivity)", "ROC-AUC", "Expected Calibration Error (ECE ↓)", "Brier Score (↓)", "PR-AUC"],
        save_path=os.path.join(EXT_FIGURES_DIR, "baseline_noise_degradation.png"),
        title="Baseline TabTransformer Degradation Under Gaussian Measurement Noise",
    )

    # Targeted drop plot for A0 vs A1b
    df_tgt = df_corr[df_corr["regime"].isin(["clean", "targeted"])]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), dpi=300)
    tgt_scenarios = ["clean", "drop_glucose_fasting", "drop_hba1c", "drop_glucose_and_hba1c"]
    tgt_labels = ["Clean", "Drop Fasting Glucose", "Drop HbA1c", "Drop Both"]
    x_pos = np.arange(len(tgt_scenarios))
    models_plot = df_tgt["model"].unique()
    w = 0.25
    colors = ["#d62728", "#1f77b4", "#2ca02c"]
    for ax, (mk, ml) in zip(axes, [("roc_auc", "ROC-AUC"), ("recall_1", "Diabetic Recall"), ("ece", "ECE (↓)")]):
        for i, m_name in enumerate(models_plot):
            sub = df_tgt[df_tgt["model"] == m_name].set_index("scenario")
            vals = [sub.loc[s, mk] for s in tgt_scenarios]
            ax.bar(x_pos + (i - 1) * w, vals, width=w, color=colors[i], alpha=0.88, label=m_name)
        ax.set_title(ml, fontsize=11, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(tgt_labels, rotation=18, ha="right", fontsize=8.5)
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        ax.legend(fontsize=7.5)
    plt.suptitle("Baseline TabTransformer Impact of Targeted Glycemic Lab Missingness", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(EXT_FIGURES_DIR, "baseline_targeted_missingness.png"))
    plt.close()

    # 6. SHAP Explainability & Explanation-Stability Audit (Stratified n=500 Test Patients)
    print(f"\n--- Running SHAP Explainability & Stability Audit (n={SHAP_SAMPLE_SIZE} Test Patients) ---")
    t_shap0 = time.time()
    feature_names = PAPER_CAT_COLS + PAPER_NUM_COLS
    bg_idx = select_stratified_indices(y_train, SHAP_BACKGROUND_SIZE, seed=SEED)
    test_idx = select_stratified_indices(y_test, SHAP_SAMPLE_SIZE, seed=SEED)

    bg_x = np.hstack([train_clean["x_cat"][bg_idx], train_clean["x_num"][bg_idx]])
    clean_x = np.hstack([test_clean["x_cat"][test_idx], test_clean["x_num"][test_idx]])

    predict_fn_raw = build_tabtransformer_predict_fn(
        model=model,
        n_cat=len(PAPER_CAT_COLS),
        n_num=len(PAPER_NUM_COLS),
        device=device,
        calibrator=None,
        use_missing_mask=False,
        batch_size=2048,
    )

    shap_clean = compute_shap_values(
        predict_fn_raw,
        background=bg_x,
        samples=clean_x,
        feature_names=feature_names,
        seed=SEED,
        max_evals=SHAP_MAX_EVALS,
    )
    print(f"[SHAP Clean Done] shape={shap_clean.shape} in {time.time() - t_shap0:.1f}s")

    # Save clean beeswarm plot
    plt.figure(figsize=(9.5, 5.8), dpi=300)
    shap.summary_plot(shap_clean, clean_x, feature_names=feature_names, show=False)
    plt.title("Baseline TabTransformer — Clean Test SHAP Summary (n=500)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(EXT_FIGURES_DIR, "baseline_shap_beeswarm_clean.png"), dpi=300, bbox_inches="tight")
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

    stab_rows = []
    global_imp_dict = {"Clean": np.mean(np.abs(shap_clean), axis=0)}
    df_test_sub = splits["test"].iloc[test_idx].copy()

    for sc in shap_scenarios:
        t_sc = time.time()
        corr_sub = build_corrupted_inputs(
            df_test_sub,
            prep,
            mcar_rate=sc["mcar_rate"],
            drop_cols=sc["drop_cols"],
            noise_sigma=sc["noise_sigma"],
            seed=SEED,
        )
        corrupt_x = np.hstack([corr_sub["x_cat"], corr_sub["x_num"]])
        shap_corrupt = compute_shap_values(
            predict_fn_raw,
            background=bg_x,
            samples=corrupt_x,
            feature_names=feature_names,
            seed=SEED,
            max_evals=SHAP_MAX_EVALS,
        )
        stab = compute_shap_stability_metrics(shap_clean, shap_corrupt, top_k_list=SHAP_TOP_K_LIST)
        global_imp_dict[sc["name"]] = np.array(stab["global_mean_abs_shap_corrupt"])
        stab_row = {
            "model": "A0_Baseline_TabTransformer",
            "scenario": sc["name"],
            **{k: v for k, v in stab.items() if not isinstance(v, list)},
        }
        stab_rows.append(stab_row)
        print(
            f"  [{sc['name']}] Top3-Jaccard={stab['local_jaccard_top3_mean']:.4f}, "
            f"LocalTau={stab['local_kendall_tau_mean']:.4f}, "
            f"Cosine={stab['local_cosine_sim_mean']:.4f}, "
            f"RelL2Drift={stab['local_rel_l2_drift_mean']:.4f} ({time.time() - t_sc:.1f}s)"
        )

    df_stab = pd.DataFrame(stab_rows)
    df_stab.to_csv(os.path.join(EXT_METRICS_DIR, "baseline_shap_stability.csv"), index=False)

    df_global_imp = pd.DataFrame({"feature": feature_names, **{k: v for k, v in global_imp_dict.items()}})
    df_global_imp.to_csv(os.path.join(EXT_METRICS_DIR, "baseline_shap_global_importance.csv"), index=False)

    # Plot SHAP global importance comparison & stability bars
    plot_shap_global_bar(
        {k: global_imp_dict[k] for k in ["Clean", "mcar_20", "drop_hba1c", "noise_0.25", "mcar_20_noise_0.25"]},
        feature_names=feature_names,
        save_path=os.path.join(EXT_FIGURES_DIR, "baseline_shap_global_comparison.png"),
        title="Baseline TabTransformer — Global SHAP Importance Shift Under Corruptions",
    )
    plot_shap_stability_bars(
        df_stab,
        scenario_col="scenario",
        group_col="model",
        save_path=os.path.join(EXT_FIGURES_DIR, "baseline_shap_stability.png"),
        title="Baseline TabTransformer — SHAP Explanation Stability Across Corruptions",
    )

    # Save raw clean & corrupted shap arrays for later Stage 3 comparison
    np.savez_compressed(
        os.path.join(EXT_METRICS_DIR, "baseline_shap_arrays.npz"),
        test_idx=test_idx,
        bg_idx=bg_idx,
        shap_clean=shap_clean,
        feature_names=np.array(feature_names),
    )

    print("\n" + "=" * 80)
    print(" STAGE 2 BASELINE AUDIT COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()

