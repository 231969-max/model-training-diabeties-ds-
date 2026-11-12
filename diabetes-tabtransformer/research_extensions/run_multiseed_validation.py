"""Rigorous 3-Seed Reproducibility & Untouched-Test Validation Suite (A0 vs. A4)."""
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
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

EXTENSIONS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EXTENSIONS_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))

from src.dataset import create_dataloaders
from src.model import TabTransformer
from src.train import train_pipeline
from src.utils import set_seed

from research_extensions.config import (
    BASELINE_CHECKPOINT_PATH,
    DATA_PATH,
    EXT_FIGURES_DIR,
    EXT_METRICS_DIR,
    EXT_MODELS_DIR,
    PAPER_CAT_COLS,
    PAPER_NUM_COLS,
    TARGET_BINARY,
)
from research_extensions.consensus_fs import run_consensus_feature_selection
from research_extensions.corruptions import build_corrupted_inputs
from research_extensions.data_pipeline import (
    fit_preprocessor,
    load_raw_dataset,
    split_dataset_identical,
    transform_features,
)
from research_extensions.metrics_calibration import (
    PostHocCalibrator,
    compute_ece,
    compute_full_evaluation,
    find_prc_optimal_threshold,
    mcnemar_test,
    sigmoid,
)
from research_extensions.plotting import plot_corruption_degradation
from research_extensions.robust_model import (
    RobustTabTransformer,
    infer_robust_logits,
    train_robust_tabtransformer,
)

AUDIT_SEEDS = [42, 101, 202]


def infer_base_logits(model: nn.Module, inp: Dict[str, np.ndarray], device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(inp["x_num"]), 2048):
            e = min(s + 2048, len(inp["x_num"]))
            tn = torch.tensor(inp["x_num"][s:e], dtype=torch.float32, device=device)
            tc = torch.tensor(inp["x_cat"][s:e], dtype=torch.long, device=device)
            out.append(model(tn, tc).detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def train_baseline_tabtransformer_for_seed(
    splits: Dict[str, pd.DataFrame],
    prep_base: Dict[str, Any],
    seed: int,
    save_path: str,
    device: torch.device,
) -> TabTransformer:
    """Trains an exact baseline TabTransformer using src/model.py & src/train.py on a given split seed."""
    tr_b = transform_features(splits["train"], prep_base)
    va_b = transform_features(splits["val"], prep_base)

    def _to_df(inp: Dict[str, np.ndarray], raw_df: pd.DataFrame) -> pd.DataFrame:
        d = {}
        for j, c in enumerate(PAPER_CAT_COLS):
            d[c] = inp["x_cat"][:, j]
        for j, c in enumerate(PAPER_NUM_COLS):
            d[c] = inp["x_num"][:, j]
        d[TARGET_BINARY] = raw_df[TARGET_BINARY].to_numpy(dtype=int)
        return pd.DataFrame(d)

    # Dummy test split for create_dataloaders (test is evaluated separately)
    dl_splits = {
        "train": _to_df(tr_b, splits["train"]),
        "val": _to_df(va_b, splits["val"]),
        "test": _to_df(va_b, splits["val"]),
    }
    train_loader, val_loader, _ = create_dataloaders(
        dl_splits, PAPER_CAT_COLS, PAPER_NUM_COLS, TARGET_BINARY, batch_size=256, eval_batch_size=512
    )
    model = TabTransformer(
        cat_cardinalities=prep_base["cat_cardinalities"],
        num_numerical=len(PAPER_NUM_COLS),
        embed_dim=32, num_heads=8, num_layers=4, ff_dim=128,
        transformer_dropout=0.1, mlp_hidden_dims=[64, 32], mlp_dropout=0.2, num_classes=1,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    train_pipeline(model, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs=15, device=device, save_model_path=save_path)
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def bootstrap_1000_ci(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50, seed: int = 42) -> Dict[str, List[float]]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs, aucs, praucs, f1s, briers, eces = [], [], [], [], [], []
    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        yhat = (yp >= threshold).astype(int)
        accs.append(accuracy_score(yt, yhat))
        aucs.append(roc_auc_score(yt, yp))
        praucs.append(average_precision_score(yt, yp))
        f1s.append(f1_score(yt, yhat, pos_label=1, zero_division=0))
        briers.append(brier_score_loss(yt, yp))
        ece_v, _ = compute_ece(yt, yp, n_bins=15)
        eces.append(ece_v)

    def _ci(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    return {
        "accuracy_ci95": _ci(accs),
        "roc_auc_ci95": _ci(aucs),
        "pr_auc_ci95": _ci(praucs),
        "f1_1_ci95": _ci(f1s),
        "brier_ci95": _ci(briers),
        "ece_ci95": _ci(eces),
    }


def main():
    os.makedirs(EXT_FIGURES_DIR, exist_ok=True)
    os.makedirs(EXT_METRICS_DIR, exist_ok=True)
    os.makedirs(EXT_MODELS_DIR, exist_ok=True)
    device = torch.device("cpu")

    print("=" * 85)
    print(" RIGOROUS 3-SEED VALIDATION & UNTOUCHED-TEST AUDIT (A0 VS. A4)")
    print("=" * 85)

    df_raw = load_raw_dataset(DATA_PATH)

    # Gate check: verify categorical missingness fix (mcar20_cat_missing_token_rate ~ 0.20, observed match = 1.0)
    s0 = split_dataset_identical(df_raw, TARGET_BINARY, random_state=42)
    pt = fit_preprocessor(s0["train"], PAPER_CAT_COLS, PAPER_NUM_COLS, add_missing_cat_token=True)
    pf = fit_preprocessor(s0["train"], PAPER_CAT_COLS, PAPER_NUM_COLS, add_missing_cat_token=False)
    m20_t = build_corrupted_inputs(s0["test"], pt, mcar_rate=0.20, seed=42)
    m20_f = build_corrupted_inputs(s0["test"], pf, mcar_rate=0.20, seed=42)
    cl_f = transform_features(s0["test"], pf)
    cat_tok_rate = float(np.mean([np.mean(m20_t["x_cat"][:, j] == pt["missing_cat_ids"][c]) for j, c in enumerate(PAPER_CAT_COLS)]))
    assert 0.19 <= cat_tok_rate <= 0.21, f"Unexpected cat missing token rate: {cat_tok_rate}"
    for j, c in enumerate(PAPER_CAT_COLS):
        obs = m20_t["m_cat"][:, j] == 0
        assert np.all(m20_t["x_cat"][obs, j] == cl_f["x_cat"][obs, j]), f"Mismatch on observed {c} (True)"
        assert np.all(m20_f["x_cat"][obs, j] == cl_f["x_cat"][obs, j]), f"Mismatch on observed {c} (False)"
    print(f"[Gate Check PASSED] mcar20_cat_missing_token_rate={cat_tok_rate:.4f}, all observed categoricals 100% intact.")

    scenarios = [
        {"name": "clean", "regime": "clean", "level": 0.0, "mcar_rate": 0.0, "drop_cols": None, "noise_sigma": 0.0},
        {"name": "mcar_10", "regime": "mcar", "level": 0.10, "mcar_rate": 0.10, "drop_cols": None, "noise_sigma": 0.0},
        {"name": "mcar_20", "regime": "mcar", "level": 0.20, "mcar_rate": 0.20, "drop_cols": None, "noise_sigma": 0.0},
        {"name": "mcar_30", "regime": "mcar", "level": 0.30, "mcar_rate": 0.30, "drop_cols": None, "noise_sigma": 0.0},
        {"name": "drop_glucose_fasting", "regime": "targeted", "level": 1.0, "mcar_rate": 0.0, "drop_cols": ["glucose_fasting"], "noise_sigma": 0.0},
        {"name": "drop_hba1c", "regime": "targeted", "level": 1.0, "mcar_rate": 0.0, "drop_cols": ["hba1c"], "noise_sigma": 0.0},
        {"name": "drop_glucose_and_hba1c", "regime": "targeted", "level": 2.0, "mcar_rate": 0.0, "drop_cols": ["glucose_fasting", "hba1c"], "noise_sigma": 0.0},
        {"name": "noise_0.10", "regime": "noise", "level": 0.10, "mcar_rate": 0.0, "drop_cols": None, "noise_sigma": 0.10},
        {"name": "noise_0.25", "regime": "noise", "level": 0.25, "mcar_rate": 0.0, "drop_cols": None, "noise_sigma": 0.25},
        {"name": "noise_0.50", "regime": "noise", "level": 0.50, "mcar_rate": 0.0, "drop_cols": None, "noise_sigma": 0.50},
        {"name": "mcar_20_noise_0.25", "regime": "combined", "level": 0.20, "mcar_rate": 0.20, "drop_cols": None, "noise_sigma": 0.25},
    ]

    per_seed_rows = []
    fs_stability_rows = []
    stat_summary: Dict[str, Any] = {}

    for seed in AUDIT_SEEDS:
        print(f"\n==================== INDEPENDENT SEED = {seed} ====================")
        set_seed(seed)
        splits = split_dataset_identical(df_raw, target_col=TARGET_BINARY, random_state=seed)
        y_train = splits["train"][TARGET_BINARY].to_numpy(dtype=int)
        y_val = splits["val"][TARGET_BINARY].to_numpy(dtype=int)
        y_test = splits["test"][TARGET_BINARY].to_numpy(dtype=int)

        # 1. Train-only Consensus Feature Selection for this seed
        df_fs, cons_cat, cons_num = run_consensus_feature_selection(splits["train"], top_k_per_selector=10, min_votes=3, seed=seed)
        fs_stability_rows.append({"seed": seed, "cons_cat": ",".join(cons_cat), "cons_num": ",".join(cons_num), "n_features": len(cons_cat) + len(cons_num)})
        print(f"[Seed {seed} Train-Only Consensus FS ({len(cons_cat)+len(cons_num)})] Cat={cons_cat}, Num={cons_num}")

        # 2. Preprocessors fitted strictly on Train_s
        prep_base = fit_preprocessor(splits["train"], PAPER_CAT_COLS, PAPER_NUM_COLS, add_missing_cat_token=False)
        prep_cons = fit_preprocessor(splits["train"], cons_cat, cons_num, add_missing_cat_token=True)

        # 3. Baseline A0_seed{seed}
        if seed == 42:
            model_a0 = TabTransformer(prep_base["cat_cardinalities"], len(PAPER_NUM_COLS), 32, 8, 4, 128, 0.1, [64, 32], 0.2, 1).to(device)
            model_a0.load_state_dict(torch.load(BASELINE_CHECKPOINT_PATH, map_location=device)["model_state_dict"])
            model_a0.eval()
            print("[Seed 42 A0] Loaded preserved baseline checkpoint.")
        else:
            a0_path = os.path.join(EXT_MODELS_DIR, f"audit_a0_seed{seed}.pt")
            if os.path.exists(a0_path):
                model_a0 = TabTransformer(prep_base["cat_cardinalities"], len(PAPER_NUM_COLS), 32, 8, 4, 128, 0.1, [64, 32], 0.2, 1).to(device)
                model_a0.load_state_dict(torch.load(a0_path, map_location=device)["model_state_dict"])
                model_a0.eval()
            else:
                print(f"[Seed {seed} A0] Training fresh baseline TabTransformer...")
                model_a0 = train_baseline_tabtransformer_for_seed(splits, prep_base, seed, a0_path, device)

        # 4. Improved A4_seed{seed}
        a4_path = os.path.join(EXT_MODELS_DIR, f"audit_a4_seed{seed}.pt")
        tr_cons = transform_features(splits["train"], prep_cons)
        val_cons = transform_features(splits["val"], prep_cons)
        imp_scaled_cons = prep_cons["scaler"].transform(pd.DataFrame([prep_cons["num_medians"]])[cons_num])[0].astype(np.float32)
        key_idx_cons = [cons_num.index(c) for c in ["hba1c", "glucose_fasting", "glucose_postprandial"] if c in cons_num]

        set_seed(seed)
        model_a4 = RobustTabTransformer(prep_cons["cat_cardinalities"], len(cons_num), use_missing_mask=True).to(device)
        if os.path.exists(a4_path):
            model_a4.load_state_dict(torch.load(a4_path, map_location=device)["model_state_dict"])
            model_a4.eval()
        else:
            print(f"[Seed {seed} A4] Training fresh RobustTabTransformer (Consensus)...")
            train_robust_tabtransformer(
                model_a4, tr_cons, y_train, val_cons, y_val,
                missing_cat_ids=[prep_cons["missing_cat_ids"][c] for c in cons_cat],
                imputed_num_values=imp_scaled_cons,
                key_num_indices=key_idx_cons,
                stochastic_corruption=True,
                save_path=a4_path,
                num_epochs=15,
                device=device,
            )

        # 5. Untouched Validation-Only Calibrator Selection (Clean Val + MCAR-20% Val)
        val_c = transform_features(splits["val"], prep_cons)
        val_m20 = build_corrupted_inputs(splits["val"], prep_cons, mcar_rate=0.20, seed=seed)
        vl_c = infer_robust_logits(model_a4, val_c["x_num"], val_c["x_cat"], val_c["m_num"], device)
        vl_m20 = infer_robust_logits(model_a4, val_m20["x_num"], val_m20["x_cat"], val_m20["m_num"], device)

        cand_scores = {}
        cals = {}
        for m in ["none", "temperature", "platt", "isotonic"]:
            cal = PostHocCalibrator(m).fit(vl_c, y_val)
            cals[m] = cal
            ev_vc = compute_full_evaluation(y_val, cal.predict_proba(vl_c))
            ev_vm = compute_full_evaluation(y_val, cal.predict_proba(vl_m20))
            cand_scores[m] = 0.5 * (ev_vc["nll"] + ev_vm["nll"])
        best_cal_name = min(["temperature", "platt", "isotonic"], key=lambda k: cand_scores[k])
        best_cal = cals[best_cal_name]
        prc_info = find_prc_optimal_threshold(y_val, best_cal.predict_proba(vl_c))
        print(f"[Seed {seed} Val-Only Calibrator] Selected={best_cal_name} (scores={ {k: round(v, 4) for k, v in cand_scores.items()} }), PRC_thr={prc_info['optimal_threshold']:.4f}")

        # 6. Evaluate A0 vs A4 across all scenarios on untouched Test_s
        seed_tests = {}
        for sc in scenarios:
            inp_a0 = build_corrupted_inputs(splits["test"], prep_base, mcar_rate=sc["mcar_rate"], drop_cols=sc["drop_cols"], noise_sigma=sc["noise_sigma"], seed=seed)
            probs_a0 = sigmoid(infer_base_logits(model_a0, inp_a0, device))
            ev_a0 = compute_full_evaluation(y_test, probs_a0, threshold=0.50)

            inp_a4 = build_corrupted_inputs(splits["test"], prep_cons, mcar_rate=sc["mcar_rate"], drop_cols=sc["drop_cols"], noise_sigma=sc["noise_sigma"], seed=seed)
            logits_a4 = infer_robust_logits(model_a4, inp_a4["x_num"], inp_a4["x_cat"], inp_a4["m_num"], device)
            probs_a4 = best_cal.predict_proba(logits_a4)
            ev_a4 = compute_full_evaluation(y_test, probs_a4, threshold=0.50)
            ev_a4_prc = compute_full_evaluation(y_test, probs_a4, threshold=prc_info["optimal_threshold"])

            for m_label, ev_obj in [
                ("A0_Baseline_Uncal", ev_a0),
                ("A4_RobustCons_ValCal_0.50", ev_a4),
                ("A4_RobustCons_ValCal_PRC", ev_a4_prc),
            ]:
                per_seed_rows.append(
                    {
                        "seed": seed,
                        "model": m_label,
                        "calibrator_selected": "none" if "A0" in m_label else best_cal_name,
                        "scenario": sc["name"],
                        "regime": sc["regime"],
                        "level": sc["level"],
                        **{k: v for k, v in ev_obj.items() if k != "reliability_diagram"},
                    }
                )

            if sc["name"] in ["clean", "mcar_20", "drop_hba1c", "noise_0.25", "mcar_20_noise_0.25"]:
                mcn = mcnemar_test(y_test, (probs_a0 >= 0.50).astype(int), (probs_a4 >= 0.50).astype(int))
                ci_a0 = bootstrap_1000_ci(y_test, probs_a0, threshold=0.50, seed=seed)
                ci_a4 = bootstrap_1000_ci(y_test, probs_a4, threshold=0.50, seed=seed)
                seed_tests[sc["name"]] = {"mcnemar_a0_vs_a4": mcn, "ci95_a0": ci_a0, "ci95_a4": ci_a4}

        stat_summary[f"seed_{seed}"] = {
            "val_calibrator_scores": cand_scores,
            "val_selected_calibrator": best_cal_name,
            "val_prc_threshold": prc_info,
            "key_scenarios_ci_mcnemar": seed_tests,
        }

    df_per_seed = pd.DataFrame(per_seed_rows)
    df_per_seed.to_csv(os.path.join(EXT_METRICS_DIR, "multiseed_a0_vs_a4_per_seed.csv"), index=False)

    df_fs_stab = pd.DataFrame(fs_stability_rows)
    df_fs_stab.to_csv(os.path.join(EXT_METRICS_DIR, "multiseed_consensus_fs_stability.csv"), index=False)

    # Aggregate Mean ± SD across 3 independent seeds
    agg_metrics = ["accuracy", "roc_auc", "pr_auc", "f1_1", "precision_1", "recall_1", "macro_f1", "brier_score", "ece", "nll", "fn", "fp"]
    summary_rows = []
    for (m_name, sc_name, regime, level), grp in df_per_seed.groupby(["model", "scenario", "regime", "level"], sort=False):
        row = {"model": m_name, "scenario": sc_name, "regime": regime, "level": level, "n_seeds": len(grp)}
        for mk in agg_metrics:
            row[mk] = float(grp[mk].mean())
            row[f"{mk}_std"] = float(grp[mk].std(ddof=1))
        summary_rows.append(row)

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(os.path.join(EXT_METRICS_DIR, "multiseed_a0_vs_a4_summary.csv"), index=False)

    with open(os.path.join(EXT_METRICS_DIR, "multiseed_statistical_tests_ci.json"), "w", encoding="utf-8") as f:
        json.dump(stat_summary, f, indent=2)

    # Plots: MCAR & Noise 3-seed Mean ± SD comparison
    df_plot = df_sum[df_sum["model"].isin(["A0_Baseline_Uncal", "A4_RobustCons_ValCal_0.50"])].copy()
    plot_corruption_degradation(
        df_plot[df_plot["regime"].isin(["clean", "mcar"])],
        x_col="level",
        x_label="MCAR Missingness Rate",
        group_col="model",
        metrics=["accuracy", "roc_auc", "pr_auc", "f1_1", "brier_score", "ece"],
        metric_labels=["Accuracy", "ROC-AUC", "PR-AUC", "F1-Score (Diabetic)", "Brier Score (↓)", "ECE (↓)"],
        save_path=os.path.join(EXT_FIGURES_DIR, "multiseed_mcar_comparison.png"),
        title="3-Seed Reproducibility Audit (Mean ± SD): A0 Baseline vs. A4 RobustTabTransformer Under MCAR",
    )
    plot_corruption_degradation(
        df_plot[df_plot["regime"].isin(["clean", "noise"])],
        x_col="level",
        x_label="Gaussian Noise Std Dev (σ)",
        group_col="model",
        metrics=["accuracy", "roc_auc", "pr_auc", "f1_1", "brier_score", "ece"],
        metric_labels=["Accuracy", "ROC-AUC", "PR-AUC", "F1-Score (Diabetic)", "Brier Score (↓)", "ECE (↓)"],
        save_path=os.path.join(EXT_FIGURES_DIR, "multiseed_noise_comparison.png"),
        title="3-Seed Reproducibility Audit (Mean ± SD): A0 Baseline vs. A4 RobustTabTransformer Under Noise",
    )

    print("\n=== 3-SEED SUMMARY (A0 vs A4 @ 0.50) ===")
    print(
        df_plot[["model", "scenario", "accuracy", "accuracy_std", "roc_auc", "roc_auc_std", "pr_auc", "f1_1", "brier_score", "ece", "ece_std"]].to_string(
            index=False
        )
    )
    print("\n[3-SEED VALIDATION AUDIT COMPLETED SUCCESSFULLY!]")


if __name__ == "__main__":
    main()

