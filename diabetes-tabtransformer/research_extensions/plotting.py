"""Publication-quality matplotlib/seaborn plotting utilities for all research stages."""
import os
from typing import Any, Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_reliability_diagrams(
    curves: Dict[str, Dict[str, Any]],
    save_path: str,
    title: str = "Reliability Diagrams — TabTransformer Calibration",
):
    """Plots calibration reliability curves for multiple calibration variants."""
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=300)
    ax.plot([0, 1], [0, 1], "k--", lw=1.6, label="Perfect Calibration")

    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
    markers = ["o", "s", "^", "D", "v"]
    for idx, (name, eval_dict) in enumerate(curves.items()):
        rel = eval_dict["reliability_diagram"]
        confs, accs = [], []
        for c, a in zip(rel["bin_confs"], rel["bin_accs"]):
            if a is not None:
                confs.append(c)
                accs.append(a)
        label = f"{name} (ECE={eval_dict['ece']:.4f}, Brier={eval_dict['brier_score']:.4f})"
        ax.plot(
            confs,
            accs,
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            lw=2.0,
            markersize=5.5,
            label=label,
        )

    ax.set_xlabel("Mean Predicted Probability", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Empirical Positive Fraction", fontsize=11, fontweight="semibold")
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_decision_curves(
    dca_curves: Dict[str, Dict[str, List[float]]],
    save_path: str,
    title: str = "Decision Curve Analysis (Clinical Net Benefit)",
):
    """Plots Decision Curve Analysis comparing model variants against Treat-All and Treat-None."""
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(8, 5.8), dpi=300)

    first_key = next(iter(dca_curves))
    th = dca_curves[first_key]["thresholds"]
    ax.plot(th, dca_curves[first_key]["net_benefit_all"], "k:", lw=1.6, label="Treat All")
    ax.plot(th, dca_curves[first_key]["net_benefit_none"], "k--", lw=1.4, label="Treat None")

    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
    for idx, (name, dca) in enumerate(dca_curves.items()):
        ax.plot(
            dca["thresholds"],
            dca["net_benefit_model"],
            color=colors[idx % len(colors)],
            lw=2.2,
            label=name,
        )

    ax.set_xlabel("Decision Probability Threshold ($p_t$)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Net Clinical Benefit", fontsize=11, fontweight="semibold")
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=10)
    ax.set_ylim(-0.05, max(dca_curves[first_key]["net_benefit_all"]) + 0.08)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9.5, frameon=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_corruption_degradation(
    df_results: pd.DataFrame,
    x_col: str,
    x_label: str,
    group_col: str,
    metrics: List[str],
    metric_labels: List[str],
    save_path: str,
    title: str,
):
    """Multi-panel degradation curve comparing models across corruption intensity."""
    _ensure_dir(save_path)
    n_panels = len(metrics)
    ncols = 2
    nrows = (n_panels + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.5 * nrows), dpi=300)
    axes = np.atleast_1d(axes).flatten()

    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    markers = ["o", "s", "^", "D", "v", "P"]

    groups = list(df_results[group_col].unique())
    for p_idx, (m_col, m_lbl) in enumerate(zip(metrics, metric_labels)):
        ax = axes[p_idx]
        for g_idx, g_name in enumerate(groups):
            sub = df_results[df_results[group_col] == g_name].sort_values(x_col)
            y_vals = sub[m_col].to_numpy()
            std_col = f"{m_col}_std"
            ax.plot(
                sub[x_col],
                y_vals,
                marker=markers[g_idx % len(markers)],
                color=colors[g_idx % len(colors)],
                lw=2.0,
                label=g_name,
            )
            if std_col in sub.columns:
                y_std = sub[std_col].to_numpy()
                ax.fill_between(
                    sub[x_col],
                    y_vals - y_std,
                    y_vals + y_std,
                    color=colors[g_idx % len(colors)],
                    alpha=0.14,
                )
        ax.set_title(m_lbl, fontsize=11, fontweight="bold")
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(m_lbl, fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=8.5, frameon=True)

    for j in range(n_panels, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(title, fontsize=13.5, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_shap_global_bar(
    mean_abs_dict: Dict[str, np.ndarray],
    feature_names: List[str],
    save_path: str,
    title: str = "Global SHAP Feature Importance (Mean |SHAP|)",
):
    """Grouped horizontal bar plot comparing global SHAP importances across conditions/models."""
    _ensure_dir(save_path)
    first_key = next(iter(mean_abs_dict))
    order = np.argsort(mean_abs_dict[first_key])  # ascending for barh
    ordered_names = [feature_names[i] for i in order]

    n_groups = len(mean_abs_dict)
    y_pos = np.arange(len(feature_names))
    height = 0.78 / n_groups
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, ax = plt.subplots(figsize=(9.5, max(5.2, 0.52 * len(feature_names))), dpi=300)
    for g_idx, (label, vals) in enumerate(mean_abs_dict.items()):
        offset = (g_idx - (n_groups - 1) / 2.0) * height
        ax.barh(
            y_pos + offset,
            np.asarray(vals)[order],
            height=height,
            color=colors[g_idx % len(colors)],
            alpha=0.88,
            label=label,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered_names, fontsize=10)
    ax.set_xlabel("Mean |SHAP Value| (Impact on Predicted Probability)", fontsize=10.5, fontweight="semibold")
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=10)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=9.5, frameon=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_shap_stability_bars(
    df_stab: pd.DataFrame,
    scenario_col: str,
    group_col: str,
    save_path: str,
    title: str = "SHAP Explanation Stability Under Clinical Data Corruptions",
):
    """Plots Top-3 Jaccard, Kendall's Tau, Cosine Similarity, and Relative L2 Drift across scenarios."""
    _ensure_dir(save_path)
    metrics = [
        ("local_jaccard_top3_mean", "Local Top-3 Jaccard Similarity (↑)"),
        ("local_kendall_tau_mean", "Local Kendall's Tau (↑)"),
        ("local_cosine_sim_mean", "Local Cosine Similarity (↑)"),
        ("local_rel_l2_drift_mean", "Local Relative L2 Attribution Drift (↓)"),
    ]
    scenarios = list(df_stab[scenario_col].unique())
    groups = list(df_stab[group_col].unique())
    x = np.arange(len(scenarios))
    width = 0.78 / max(1, len(groups))
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=300)
    axes = axes.flatten()
    for ax, (m_col, m_title) in zip(axes, metrics):
        for g_idx, g_name in enumerate(groups):
            sub = df_stab[df_stab[group_col] == g_name].set_index(scenario_col)
            vals = [sub.loc[s, m_col] if s in sub.index else np.nan for s in scenarios]
            offset = (g_idx - (len(groups) - 1) / 2.0) * width
            ax.bar(x + offset, vals, width=width, color=colors[g_idx % len(colors)], alpha=0.88, label=g_name)
        ax.set_title(m_title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=22, ha="right", fontsize=8.5)
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        ax.legend(fontsize=8.5, frameon=True)

    plt.suptitle(title, fontsize=13.5, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

