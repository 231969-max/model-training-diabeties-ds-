"""Stage 3B: Train & Evaluate Ablations (A2, A3, A4, A5) Against Preserved Baseline (A0, A1)."""
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
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

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
    EXT_MODELS_DIR,
    MCAR_RATES,
    NOISE_SIGMAS,
    PAPER_CAT_COLS,
    PAPER_NUM_COLS,
    SEED,
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
)
from research_extensions.robust_model import (
    RobustTabTransformer,
    infer_robust_logits,
    train_robust_tabtransformer,
)


def infer_baseline_logits(model, x_num, x_cat, device, batch_size=1024) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(x_num), batch_size):
            e = min(s + batch_size, len(x_num))
            tn = torch.tensor(x_num[s:e], dtype=torch.float32, device=device)
            tc = torch.tensor(x_cat[s:e], dtype=torch.long, device=device)
            out.append(model(tn, tc).detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def fit_classical_pipeline(
    train_df: pd.DataFrame,
    cat_cols: List[str],
    num_cols: List[str],
    target_col: str,
    model_type: str,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Fits OHE + StandardScaler + Train-Only Imputer + RF/LR on exact same split."""
    prep = fit_preprocessor(train_df, cat_cols=cat_cols, num_cols=num_cols, add_missing_cat_token=False)
    tr_t = transform_features(train_df, prep)
    ohe = OneHotEncoder(
        categories=[list(range(c)) for c in prep["cat_cardinalities"]],
        sparse_output=False,
        handle_unknown="ignore",
    )
    x_cat_ohe = ohe.fit_transform(tr_t["x_cat"]) if len(cat_cols) > 0 else np.zeros((len(train_df), 0))
    X_tr = np.hstack([x_cat_ohe, tr_t["x_num"]])
    y_tr = train_df[target_col].to_numpy(dtype=int)

    if model_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
    elif model_type == "lr":
        clf = LogisticRegression(max_iter=1000, random_state=seed)
    else:
        raise ValueError(model_type)

    clf.fit(X_tr, y_tr)
    return {"prep": prep, "ohe": ohe, "clf": clf}


def predict_classical_logits(bundle: Dict[str, Any], inputs_dict: Dict[str, np.ndarray]) -> np.ndarray:
    ohe = bundle["ohe"]
    clf = bundle["clf"]
    x_cat_ohe = ohe.transform(inputs_dict["x_cat"]) if inputs_dict["x_cat"].shape[1] > 0 else np.zeros((len(inputs_dict["x_num"]), 0))
    X = np.hstack([x_cat_ohe, inputs_dict["x_num"]])
    probs = np.clip(clf.predict_proba(X)[:, 1], 1e-6, 1.0 - 1e-6)
    return np.log(probs / (1.0 - probs))


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_boot: int = 500,
    seed: int = SEED,
) -> Dict[str, List[float]]:
    from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
    from research_extensions.metrics_calibration import compute_ece

    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs, aucs, briers, eces = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        accs.append(accuracy_score(yt, (yp >= threshold).astype(int)))
        aucs.append(roc_auc_score(yt, yp))
        briers.append(brier_score_loss(yt, yp))
        ece_val, _ = compute_ece(yt, yp, n_bins=15)
        eces.append(ece_val)

    def _ci(arr):
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    return {"accuracy_ci95": _ci(accs), "roc_auc_ci95": _ci(aucs), "brier_ci95": _ci(briers), "ece_ci95": _ci(eces)}


def main():
    os.makedirs(EXT_FIGURES_DIR, exist_ok=True)
    os.makedirs(EXT_METRICS_DIR, exist_ok=True)
    os.makedirs(EXT_MODELS_DIR, exist_ok=True)
    set_seed(SEED)
    device = torch.device("cpu")

    print("=" * 85)
    print(" STAGE 3B: ROBUST TABTRANSFORMER TRAINING, ABLATIONS & CORRUPTION BENCHMARK")
    print("=" * 85)

    # 1. Load data, identical split, and consensus feature list
    df_raw = load_raw_dataset(DATA_PATH)
    splits = split_dataset_identical(df_raw, target_col=TARGET_BINARY, random_state=SEED)
    y_train = splits["train"][TARGET_BINARY].to_numpy(dtype=int)
    y_val = splits["val"][TARGET_BINARY].to_numpy(dtype=int)
    y_test = splits["test"][TARGET_BINARY].to_numpy(dtype=int)

    df_fs = pd.read_csv(os.path.join(EXT_METRICS_DIR, "consensus_feature_selection.csv"))
    cons_rows = df_fs[df_fs["selected_consensus"] == 1]
    CONS_CAT_COLS = cons_rows[cons_rows["type"] == "categorical"]["feature"].tolist()
    CONS_NUM_COLS = cons_rows[cons_rows["type"] == "numerical"]["feature"].tolist()
    print(f"[Consensus Features ({len(CONS_CAT_COLS)+len(CONS_NUM_COLS)})] Cat={CONS_CAT_COLS}, Num={CONS_NUM_COLS}")

    # 2. Preprocessors (Baseline 9-feat, Robust 9-feat, Robust Consensus-8)
    prep_base = fit_preprocessor(splits["train"], PAPER_CAT_COLS, PAPER_NUM_COLS, add_missing_cat_token=False)
    prep_rob9 = fit_preprocessor(splits["train"], PAPER_CAT_COLS, PAPER_NUM_COLS, add_missing_cat_token=True)
    prep_cons = fit_preprocessor(splits["train"], CONS_CAT_COLS, CONS_NUM_COLS, add_missing_cat_token=True)

    # Compute standardized training medians for imputed numerical fallback
    imp_scaled_rob9 = prep_rob9["scaler"].transform(
        pd.DataFrame([prep_rob9["num_medians"]])[PAPER_NUM_COLS]
    )[0].astype(np.float32)
    imp_scaled_cons = prep_cons["scaler"].transform(
        pd.DataFrame([prep_cons["num_medians"]])[CONS_NUM_COLS]
    )[0].astype(np.float32)

    # Key glycemic indices for targeted batch dropout during stochastic training
    key_idx_9 = [PAPER_NUM_COLS.index(c) for c in ["hba1c", "glucose_fasting"] if c in PAPER_NUM_COLS]
    key_idx_cons = [CONS_NUM_COLS.index(c) for c in ["hba1c", "glucose_fasting", "glucose_postprandial"] if c in CONS_NUM_COLS]

    # Precompute clean splits
    val_base = transform_features(splits["val"], prep_base)
    test_base = transform_features(splits["test"], prep_base)

    tr_rob9 = transform_features(splits["train"], prep_rob9)
    val_rob9 = transform_features(splits["val"], prep_rob9)
    test_rob9 = transform_features(splits["test"], prep_rob9)

    tr_cons = transform_features(splits["train"], prep_cons)
    val_cons = transform_features(splits["val"], prep_cons)
    test_cons = transform_features(splits["test"], prep_cons)

    # 3. Load Preserved Baseline (A0 / A1)
    model_a0 = TabTransformer(
        cat_cardinalities=prep_base["cat_cardinalities"],
        num_numerical=len(PAPER_NUM_COLS),
        embed_dim=32, num_heads=8, num_layers=4, ff_dim=128,
        transformer_dropout=0.1, mlp_hidden_dims=[64, 32], mlp_dropout=0.2, num_classes=1,
    ).to(device)
    model_a0.load_state_dict(torch.load(BASELINE_CHECKPOINT_PATH, map_location=device)["model_state_dict"])
    model_a0.eval()

    # 4. Train A2 (Missingness-Aware 9-feat, Clean-trained)
    print("\n--- Training A2: Missingness-Aware TabTransformer (Clean Training Only, 9 Features) ---")
    set_seed(SEED)
    model_a2 = RobustTabTransformer(
        cat_cardinalities=prep_rob9["cat_cardinalities"],
        num_numerical=len(PAPER_NUM_COLS),
        use_missing_mask=True,
    ).to(device)
    hist_a2 = train_robust_tabtransformer(
        model_a2, tr_rob9, y_train, val_rob9, y_val,
        missing_cat_ids=[prep_rob9["missing_cat_ids"][c] for c in PAPER_CAT_COLS],
        imputed_num_values=imp_scaled_rob9,
        key_num_indices=key_idx_9,
        stochastic_corruption=False,
        save_path=os.path.join(EXT_MODELS_DIR, "a2_mask_clean.pt"),
        num_epochs=15, device=device,
    )

    # 5. Train A3 (RobustTabTransformer 9-feat, Mask + Stochastic Corruption Training)
    print("\n--- Training A3: RobustTabTransformer (Mask + Stochastic Masking/Noise, 9 Features) ---")
    set_seed(SEED)
    model_a3 = RobustTabTransformer(
        cat_cardinalities=prep_rob9["cat_cardinalities"],
        num_numerical=len(PAPER_NUM_COLS),
        use_missing_mask=True,
    ).to(device)
    hist_a3 = train_robust_tabtransformer(
        model_a3, tr_rob9, y_train, val_rob9, y_val,
        missing_cat_ids=[prep_rob9["missing_cat_ids"][c] for c in PAPER_CAT_COLS],
        imputed_num_values=imp_scaled_rob9,
        key_num_indices=key_idx_9,
        stochastic_corruption=True,
        save_path=os.path.join(EXT_MODELS_DIR, "a3_robust_9feat.pt"),
        num_epochs=15, device=device,
    )

    # 6. Train A4 (RobustTabTransformer Consensus-8)
    print("\n--- Training A4: RobustTabTransformer (Mask + Stochastic Masking/Noise, Consensus-8) ---")
    set_seed(SEED)
    model_a4 = RobustTabTransformer(
        cat_cardinalities=prep_cons["cat_cardinalities"],
        num_numerical=len(CONS_NUM_COLS),
        use_missing_mask=True,
    ).to(device)
    hist_a4 = train_robust_tabtransformer(
        model_a4, tr_cons, y_train, val_cons, y_val,
        missing_cat_ids=[prep_cons["missing_cat_ids"][c] for c in CONS_CAT_COLS],
        imputed_num_values=imp_scaled_cons,
        key_num_indices=key_idx_cons,
        stochastic_corruption=True,
        save_path=os.path.join(EXT_MODELS_DIR, "a4_robust_consensus8.pt"),
        num_epochs=15, device=device,
    )

    # Save training curves plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    for lbl, h, col in [("A2 (Mask Only, 9-feat)", hist_a2, "#ff7f0e"), ("A3 (Robust, 9-feat)", hist_a3, "#1f77b4"), ("A4 (Robust, Consensus-8)", hist_a4, "#2ca02c")]:
        axes[0].plot(h["epoch"], h["val_loss"], "o-", color=col, lw=2, label=lbl)
        axes[1].plot(h["epoch"], [a * 100 for a in h["val_acc"]], "o-", color=col, lw=2, label=lbl)
        axes[2].plot(h["epoch"], h["val_auc"], "o-", color=col, lw=2, label=lbl)
    axes[0].set_title("Validation NLL Loss"); axes[0].set_xlabel("Epoch"); axes[0].grid(True, linestyle=":", alpha=0.6); axes[0].legend(fontsize=8)
    axes[1].set_title("Validation Accuracy (%)"); axes[1].set_xlabel("Epoch"); axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[2].set_title("Validation ROC-AUC"); axes[2].set_xlabel("Epoch"); axes[2].grid(True, linestyle=":", alpha=0.6)
    plt.suptitle("Ablation Variants — Validation Convergence (Clean Validation Set)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(EXT_FIGURES_DIR, "ablation_training_curves.png"))
    plt.close()

    # 7. Train A5 Classical ML Baselines (RF & LR on 9-feat and Consensus-8)
    print("\n--- Training A5 Classical Baselines (RF & LR on 9-feat & Consensus-8) ---")
    rf_9 = fit_classical_pipeline(splits["train"], PAPER_CAT_COLS, PAPER_NUM_COLS, TARGET_BINARY, "rf")
    lr_9 = fit_classical_pipeline(splits["train"], PAPER_CAT_COLS, PAPER_NUM_COLS, TARGET_BINARY, "lr")
    rf_c = fit_classical_pipeline(splits["train"], CONS_CAT_COLS, CONS_NUM_COLS, TARGET_BINARY, "rf")
    lr_c = fit_classical_pipeline(splits["train"], CONS_CAT_COLS, CONS_NUM_COLS, TARGET_BINARY, "lr")

    # 8. Validation-Only Calibration Fitting (Compare None, Temp, Platt, Isotonic on Val)
    model_registry = [
        {
            "id": "A0_Baseline_Uncal",
            "prep": prep_base,
            "infer_fn": lambda inp: infer_baseline_logits(model_a0, inp["x_num"], inp["x_cat"], device),
            "cal_method": "none",
        },
        {
            "id": "A1_Baseline_TempCal",
            "prep": prep_base,
            "infer_fn": lambda inp: infer_baseline_logits(model_a0, inp["x_num"], inp["x_cat"], device),
            "cal_method": "temperature",
        },
        {
            "id": "A2_MaskClean_TempCal",
            "prep": prep_rob9,
            "infer_fn": lambda inp: infer_robust_logits(model_a2, inp["x_num"], inp["x_cat"], inp["m_num"], device),
            "cal_method": "temperature",
        },
        {
            "id": "A3_Robust9_TempCal",
            "prep": prep_rob9,
            "infer_fn": lambda inp: infer_robust_logits(model_a3, inp["x_num"], inp["x_cat"], inp["m_num"], device),
            "cal_method": "temperature",
        },
        {
            "id": "A3p_Robust9_PlattCal",
            "prep": prep_rob9,
            "infer_fn": lambda inp: infer_robust_logits(model_a3, inp["x_num"], inp["x_cat"], inp["m_num"], device),
            "cal_method": "platt",
        },
        {
            "id": "A4_RobustCons8_TempCal",
            "prep": prep_cons,
            "infer_fn": lambda inp: infer_robust_logits(model_a4, inp["x_num"], inp["x_cat"], inp["m_num"], device),
            "cal_method": "temperature",
        },
        {
            "id": "A5_RF_9feat_PlattCal",
            "prep": rf_9["prep"],
            "infer_fn": lambda inp, b=rf_9: predict_classical_logits(b, inp),
            "cal_method": "platt",
        },
        {
            "id": "A5_LR_9feat",
            "prep": lr_9["prep"],
            "infer_fn": lambda inp, b=lr_9: predict_classical_logits(b, inp),
            "cal_method": "none",
        },
        {
            "id": "A5_RF_Cons8_PlattCal",
            "prep": rf_c["prep"],
            "infer_fn": lambda inp, b=rf_c: predict_classical_logits(b, inp),
            "cal_method": "platt",
        },
        {
            "id": "A5_LR_Cons8",
            "prep": lr_c["prep"],
            "infer_fn": lambda inp, b=lr_c: predict_classical_logits(b, inp),
            "cal_method": "none",
        },
    ]

    clean_rows = []
    rel_plot_dict = {}
    dca_plot_dict = {}
    clean_probs_store = {}

    for entry in model_registry:
        val_inp = transform_features(splits["val"], entry["prep"])
        test_inp = transform_features(splits["test"], entry["prep"])
        val_logits = entry["infer_fn"](val_inp)
        test_logits = entry["infer_fn"](test_inp)

        cal = PostHocCalibrator(method=entry["cal_method"]).fit(val_logits, y_val)
        entry["calibrator"] = cal

        val_probs = cal.predict_proba(val_logits)
        test_probs = cal.predict_proba(test_logits)
        prc_info = find_prc_optimal_threshold(y_val, val_probs)
        entry["prc_threshold"] = prc_info["optimal_threshold"]

        ev_05 = compute_full_evaluation(y_test, test_probs, threshold=0.50)
        ci_05 = bootstrap_metric_ci(y_test, test_probs, threshold=0.50, n_boot=300)
        clean_probs_store[entry["id"]] = test_probs

        clean_rows.append(
            {
                "model": entry["id"],
                "cal_method": entry["cal_method"],
                "temperature": getattr(cal, "temperature", 1.0),
                "prc_threshold": entry["prc_threshold"],
                **{k: v for k, v in ev_05.items() if k != "reliability_diagram"},
                "acc_ci95_lo": ci_05["accuracy_ci95"][0],
                "acc_ci95_hi": ci_05["accuracy_ci95"][1],
                "auc_ci95_lo": ci_05["roc_auc_ci95"][0],
                "auc_ci95_hi": ci_05["roc_auc_ci95"][1],
            }
        )
        if entry["id"] in ["A0_Baseline_Uncal", "A3_Robust9_TempCal", "A4_RobustCons8_TempCal", "A5_RF_9feat_PlattCal", "A5_LR_9feat"]:
            rel_plot_dict[entry["id"]] = ev_05
            dca_plot_dict[entry["id"]] = compute_decision_curve(y_test, test_probs)

    df_clean = pd.DataFrame(clean_rows)
    df_clean.to_csv(os.path.join(EXT_METRICS_DIR, "ablation_clean_comparison.csv"), index=False)
    print("\n[Clean Test Comparison]")
    print(df_clean[["model", "accuracy", "precision_1", "recall_1", "f1_1", "roc_auc", "pr_auc", "brier_score", "ece", "fn", "fp"]].to_string(index=False))

    plot_reliability_diagrams(
        rel_plot_dict,
        os.path.join(EXT_FIGURES_DIR, "ablation_clean_reliability_diagrams.png"),
        title="Clean Test Reliability Diagrams — Baseline vs. RobustTabTransformer vs. Classical ML",
    )
    plot_decision_curves(
        dca_plot_dict,
        os.path.join(EXT_FIGURES_DIR, "ablation_clean_dca.png"),
        title="Decision Curve Analysis — Clean Test Clinical Net Benefit",
    )

    # 9. Full Corruption Stress-Test Across All Models
    print("\n--- Running Full Corruption Stress-Tests Across All 10 Model Variants ---")
    scenarios = [{"name": "clean", "regime": "clean", "level": 0.0, "mcar_rate": 0.0, "drop_cols": None, "noise_sigma": 0.0, "seeds": [SEED]}]
    for r in MCAR_RATES:
        scenarios.append({"name": f"mcar_{int(r*100)}", "regime": "mcar", "level": r, "mcar_rate": r, "drop_cols": None, "noise_sigma": 0.0, "seeds": CORRUPTION_SEEDS})
    for t_name, cols in TARGETED_DROP_SETS.items():
        scenarios.append({"name": t_name, "regime": "targeted", "level": len(cols), "mcar_rate": 0.0, "drop_cols": cols, "noise_sigma": 0.0, "seeds": [SEED]})
    for s in NOISE_SIGMAS:
        scenarios.append({"name": f"noise_{s:.2f}", "regime": "noise", "level": s, "mcar_rate": 0.0, "drop_cols": None, "noise_sigma": s, "seeds": CORRUPTION_SEEDS})
    for comb in COMBINED_SCENARIOS:
        scenarios.append({"name": comb["name"], "regime": "combined", "level": comb["mcar_rate"], "mcar_rate": comb["mcar_rate"], "drop_cols": None, "noise_sigma": comb["noise_sigma"], "seeds": CORRUPTION_SEEDS})

    metric_keys = ["accuracy", "precision_1", "recall_1", "f1_1", "macro_f1", "roc_auc", "pr_auc", "brier_score", "ece", "nll", "fn", "fp"]
    all_corr_rows = []
    # Store seed=42 probs on key scenarios for McNemar tests
    key_corrupt_probs: Dict[str, Dict[str, np.ndarray]] = {"mcar_20": {}, "drop_hba1c": {}, "noise_0.25": {}, "mcar_20_noise_0.25": {}}

    for sc in scenarios:
        per_model_runs: Dict[str, List[Dict[str, Any]]] = {e["id"]: [] for e in model_registry}
        for c_seed in sc["seeds"]:
            for e in model_registry:
                c_inp = build_corrupted_inputs(
                    splits["test"],
                    e["prep"],
                    mcar_rate=sc["mcar_rate"],
                    drop_cols=sc["drop_cols"],
                    noise_sigma=sc["noise_sigma"],
                    seed=c_seed,
                )
                logits = e["infer_fn"](c_inp)
                probs = e["calibrator"].predict_proba(logits)
                ev = compute_full_evaluation(y_test, probs, threshold=0.50)
                per_model_runs[e["id"]].append(ev)
                if c_seed == SEED and sc["name"] in key_corrupt_probs:
                    key_corrupt_probs[sc["name"]][e["id"]] = probs

        for e in model_registry:
            runs = per_model_runs[e["id"]]
            row = {
                "model": e["id"],
                "scenario": sc["name"],
                "regime": sc["regime"],
                "level": sc["level"],
                "n_seeds": len(runs),
            }
            for mk in metric_keys:
                vals = [r[mk] for r in runs]
                row[mk] = float(np.mean(vals))
                row[f"{mk}_std"] = float(np.std(vals))
            all_corr_rows.append(row)

    df_all_corr = pd.DataFrame(all_corr_rows)
    df_all_corr.to_csv(os.path.join(EXT_METRICS_DIR, "ablation_corruption_results.csv"), index=False)
    print("[Saved] ablation_corruption_results.csv")

    # Statistical significance tests (McNemar's test)
    stat_tests = {}
    pairs = [
        ("A0_Baseline_Uncal", "A3_Robust9_TempCal"),
        ("A0_Baseline_Uncal", "A4_RobustCons8_TempCal"),
        ("A3_Robust9_TempCal", "A4_RobustCons8_TempCal"),
        ("A3_Robust9_TempCal", "A5_RF_9feat_PlattCal"),
        ("A4_RobustCons8_TempCal", "A5_RF_Cons8_PlattCal"),
    ]
    stat_tests["clean"] = {
        f"{a}_vs_{b}": mcnemar_test(y_test, (clean_probs_store[a] >= 0.5).astype(int), (clean_probs_store[b] >= 0.5).astype(int))
        for a, b in pairs
    }
    for sc_name, p_map in key_corrupt_probs.items():
        stat_tests[sc_name] = {
            f"{a}_vs_{b}": mcnemar_test(y_test, (p_map[a] >= 0.5).astype(int), (p_map[b] >= 0.5).astype(int))
            for a, b in pairs
        }
    with open(os.path.join(EXT_METRICS_DIR, "ablation_statistical_tests.json"), "w", encoding="utf-8") as f:
        json.dump(stat_tests, f, indent=2)

    # 10. Generate Key Comparison Plots (Focus on clean set of 6 main models for readability)
    plot_models = [
        "A0_Baseline_Uncal",
        "A2_MaskClean_TempCal",
        "A3_Robust9_TempCal",
        "A4_RobustCons8_TempCal",
        "A5_RF_9feat_PlattCal",
        "A5_RF_Cons8_PlattCal",
    ]
    df_sub = df_all_corr[df_all_corr["model"].isin(plot_models)].copy()

    plot_corruption_degradation(
        df_sub[df_sub["regime"].isin(["clean", "mcar"])],
        x_col="level",
        x_label="MCAR Missingness Rate",
        group_col="model",
        metrics=["accuracy", "recall_1", "roc_auc", "ece", "brier_score", "pr_auc"],
        metric_labels=["Accuracy", "Diabetic Recall", "ROC-AUC", "ECE (↓)", "Brier Score (↓)", "PR-AUC"],
        save_path=os.path.join(EXT_FIGURES_DIR, "ablation_mcar_degradation.png"),
        title="Model Comparison Under MCAR Missingness (5%–30%)",
    )

    plot_corruption_degradation(
        df_sub[df_sub["regime"].isin(["clean", "noise"])],
        x_col="level",
        x_label="Gaussian Noise Std Dev (σ)",
        group_col="model",
        metrics=["accuracy", "recall_1", "roc_auc", "ece", "brier_score", "pr_auc"],
        metric_labels=["Accuracy", "Diabetic Recall", "ROC-AUC", "ECE (↓)", "Brier Score (↓)", "PR-AUC"],
        save_path=os.path.join(EXT_FIGURES_DIR, "ablation_noise_degradation.png"),
        title="Model Comparison Under Gaussian Measurement Noise",
    )

    # Targeted missingness bar comparison
    df_tgt = df_sub[df_sub["regime"].isin(["clean", "targeted"])]
    tgt_scenarios = ["clean", "drop_glucose_fasting", "drop_hba1c", "drop_glucose_and_hba1c"]
    tgt_labels = ["Clean", "Drop Fasting Glucose", "Drop HbA1c", "Drop Both"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=300)
    x_pos = np.arange(len(tgt_scenarios))
    w = 0.13
    colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b"]
    for ax, (mk, ml) in zip(axes, [("accuracy", "Accuracy (↑)"), ("roc_auc", "ROC-AUC (↑)"), ("ece", "ECE (↓)")]):
        for i, m_name in enumerate(plot_models):
            sub = df_tgt[df_tgt["model"] == m_name].set_index("scenario")
            vals = [sub.loc[s, mk] for s in tgt_scenarios]
            ax.bar(x_pos + (i - (len(plot_models) - 1) / 2) * w, vals, width=w, color=colors[i], alpha=0.88, label=m_name)
        ax.set_title(ml, fontsize=11, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(tgt_labels, rotation=18, ha="right", fontsize=8.5)
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        ax.legend(fontsize=7)
    plt.suptitle("Targeted Glycemic Lab Missingness — Baseline vs. RobustTabTransformer vs. RF", fontsize=13.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(EXT_FIGURES_DIR, "ablation_targeted_missingness.png"))
    plt.close()

    print("\n" + "=" * 85)
    print(" STAGE 3B COMPLETED SUCCESSFULLY!")
    print("=" * 85)


if __name__ == "__main__":
    main()

