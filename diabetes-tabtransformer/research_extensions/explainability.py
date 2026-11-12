"""SHAP explainability and quantitative explanation-stability metrics under clinical corruption."""
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import shap
import torch
from scipy.stats import kendalltau, spearmanr

from .metrics_calibration import PostHocCalibrator, sigmoid


def build_tabtransformer_predict_fn(
    model: torch.nn.Module,
    n_cat: int,
    n_num: int,
    device: torch.device,
    calibrator: Optional[PostHocCalibrator] = None,
    use_missing_mask: bool = False,
    imputed_scaled_medians: Optional[np.ndarray] = None,
    batch_size: int = 1024,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Creates a vectorized prediction function `f(Z) -> probs` over the unified feature array
    Z = [x_cat (n_cat cols), x_num_or_nan (n_num cols)].
    When use_missing_mask=True, NaN entries in numerical columns set m_num=1 and fill
    x_num with standardized training medians.
    """
    model.eval()
    if imputed_scaled_medians is None:
        imputed_scaled_medians = np.zeros(n_num, dtype=np.float32)

    def predict_fn(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        if z.ndim == 1:
            z = z.reshape(1, -1)
        n = z.shape[0]

        if n_cat > 0:
            x_cat_np = np.rint(z[:, :n_cat]).astype(np.int64)
        else:
            x_cat_np = np.zeros((n, 0), dtype=np.int64)

        if n_num > 0:
            z_num = z[:, n_cat : n_cat + n_num].copy()
            nan_mask = np.isnan(z_num)
            m_num_np = nan_mask.astype(np.float32)
            if nan_mask.any():
                for j in range(n_num):
                    if nan_mask[:, j].any():
                        z_num[nan_mask[:, j], j] = imputed_scaled_medians[j]
            x_num_np = z_num.astype(np.float32)
        else:
            x_num_np = np.zeros((n, 0), dtype=np.float32)
            m_num_np = np.zeros((n, 0), dtype=np.float32)

        logits_all = []
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                t_num = torch.tensor(x_num_np[start:end], dtype=torch.float32, device=device)
                t_cat = torch.tensor(x_cat_np[start:end], dtype=torch.long, device=device)
                if use_missing_mask:
                    t_mask = torch.tensor(m_num_np[start:end], dtype=torch.float32, device=device)
                    logits = model(t_num, t_cat, t_mask)
                else:
                    logits = model(t_num, t_cat)
                logits_all.append(logits.detach().cpu().numpy())

        logits_cat = np.concatenate(logits_all, axis=0)
        if calibrator is not None:
            return calibrator.predict_proba(logits_cat)
        return sigmoid(logits_cat)

    return predict_fn


def select_stratified_indices(y: np.ndarray, n_samples: int, seed: int = 42) -> np.ndarray:
    """Selects deterministic stratified sample indices from y."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    frac_1 = len(idx_1) / len(y)
    n_1 = int(round(n_samples * frac_1))
    n_0 = n_samples - n_1
    chosen_0 = rng.choice(idx_0, size=min(n_0, len(idx_0)), replace=False)
    chosen_1 = rng.choice(idx_1, size=min(n_1, len(idx_1)), replace=False)
    combined = np.sort(np.concatenate([chosen_0, chosen_1]))
    return combined


def compute_shap_values(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    background: np.ndarray,
    samples: np.ndarray,
    feature_names: List[str],
    seed: int = 42,
    max_evals: int = 200,
) -> np.ndarray:
    """
    Computes Shapley attributions using shap.PermutationExplainer (efficient, model-agnostic,
    additive, deterministic, supports NaN sentinel when handled by predict_fn).
    """
    explainer = shap.PermutationExplainer(
        predict_fn,
        background,
        feature_names=feature_names,
        seed=seed,
    )
    explanation = explainer(samples, max_evals=max(max_evals, 2 * len(feature_names) + 1), silent=True)
    return np.asarray(explanation.values, dtype=np.float64)


def compute_shap_stability_metrics(
    shap_clean: np.ndarray,
    shap_corrupt: np.ndarray,
    top_k_list: List[int] = [3, 5],
) -> Dict[str, Any]:
    """
    Quantifies global and local SHAP explanation consistency between clean and corrupted inputs
    evaluated on the exact same patients.
    """
    # 1. Global importance vectors (mean |SHAP|)
    g_clean = np.mean(np.abs(shap_clean), axis=0)
    g_corrupt = np.mean(np.abs(shap_corrupt), axis=0)

    global_tau, global_tau_p = kendalltau(g_clean, g_corrupt)
    global_rho, global_rho_p = spearmanr(g_clean, g_corrupt)

    # Top-K global Jaccard
    global_rank_clean = np.argsort(-g_clean)
    global_rank_corrupt = np.argsort(-g_corrupt)
    global_jaccard = {}
    for k in top_k_list:
        s1 = set(global_rank_clean[:k].tolist())
        s2 = set(global_rank_corrupt[:k].tolist())
        global_jaccard[f"global_jaccard_top{k}"] = float(len(s1 & s2) / len(s1 | s2))

    # 2. Local per-patient metrics
    n = shap_clean.shape[0]
    local_taus = []
    local_cosines = []
    local_rel_l2 = []
    local_jaccards = {k: [] for k in top_k_list}

    for i in range(n):
        vc = shap_clean[i]
        vn = shap_corrupt[i]
        abs_c = np.abs(vc)
        abs_n = np.abs(vn)

        tau_i, _ = kendalltau(abs_c, abs_n)
        if not np.isnan(tau_i):
            local_taus.append(float(tau_i))

        norm_c = float(np.linalg.norm(vc))
        norm_n = float(np.linalg.norm(vn))
        cos_i = float(np.dot(vc, vn) / max(norm_c * norm_n, 1e-12))
        local_cosines.append(cos_i)

        rel_l2_i = float(np.linalg.norm(vn - vc) / max(norm_c, 1e-12))
        local_rel_l2.append(rel_l2_i)

        rc = np.argsort(-abs_c)
        rn = np.argsort(-abs_n)
        for k in top_k_list:
            sc = set(rc[:k].tolist())
            sn = set(rn[:k].tolist())
            local_jaccards[k].append(float(len(sc & sn) / len(sc | sn)))

    result: Dict[str, Any] = {
        "global_kendall_tau": float(global_tau) if not np.isnan(global_tau) else 0.0,
        "global_kendall_p": float(global_tau_p) if not np.isnan(global_tau_p) else 1.0,
        "global_spearman_rho": float(global_rho) if not np.isnan(global_rho) else 0.0,
        "global_spearman_p": float(global_rho_p) if not np.isnan(global_rho_p) else 1.0,
        "local_kendall_tau_mean": float(np.mean(local_taus)) if local_taus else 0.0,
        "local_kendall_tau_std": float(np.std(local_taus)) if local_taus else 0.0,
        "local_cosine_sim_mean": float(np.mean(local_cosines)),
        "local_cosine_sim_std": float(np.std(local_cosines)),
        "local_rel_l2_drift_mean": float(np.mean(local_rel_l2)),
        "local_rel_l2_drift_std": float(np.std(local_rel_l2)),
        "global_mean_abs_shap_clean": g_clean.tolist(),
        "global_mean_abs_shap_corrupt": g_corrupt.tolist(),
    }
    result.update(global_jaccard)
    for k in top_k_list:
        result[f"local_jaccard_top{k}_mean"] = float(np.mean(local_jaccards[k]))
        result[f"local_jaccard_top{k}_std"] = float(np.std(local_jaccards[k]))

    return result

