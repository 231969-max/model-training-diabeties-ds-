"""Comprehensive discrimination, probability calibration, DCA, and statistical testing utilities."""
from typing import Any, Dict, List, Tuple
import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    z_clip = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z_clip))


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> Tuple[float, Dict[str, Any]]:
    """
    Computes Expected Calibration Error (ECE) and bin statistics for reliability diagrams.
    Uses uniform probability bins in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confs, bin_accs, bin_counts = [], [], []
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == 0:
            in_bin = (y_prob >= lo) & (y_prob <= hi)
        else:
            in_bin = (y_prob > lo) & (y_prob <= hi)
        count = int(np.sum(in_bin))
        bin_counts.append(count)
        if count > 0:
            conf = float(np.mean(y_prob[in_bin]))
            acc = float(np.mean(y_true[in_bin]))
            ece += (count / n) * abs(acc - conf)
            bin_confs.append(conf)
            bin_accs.append(acc)
        else:
            bin_confs.append(float((lo + hi) / 2.0))
            bin_accs.append(np.nan)

    return float(ece), {
        "bin_edges": bin_edges.tolist(),
        "bin_confs": bin_confs,
        "bin_accs": [None if np.isnan(a) else a for a in bin_accs],
        "bin_counts": bin_counts,
    }


def compute_decision_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray = np.linspace(0.05, 0.85, 81),
) -> Dict[str, List[float]]:
    """
    Computes Decision Curve Analysis (DCA) Net Benefit across probability thresholds,
    compared against 'Treat All' and 'Treat None' strategies.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n = len(y_true)
    prevalence = float(np.mean(y_true))

    nb_model, nb_all, nb_none = [], [], []
    for pt in thresholds:
        preds = (y_prob >= pt).astype(int)
        tp = float(np.sum((preds == 1) & (y_true == 1)))
        fp = float(np.sum((preds == 1) & (y_true == 0)))
        weight = pt / (1.0 - pt)
        nb_model.append((tp / n) - (fp / n) * weight)
        nb_all.append(prevalence - (1.0 - prevalence) * weight)
        nb_none.append(0.0)

    return {
        "thresholds": thresholds.tolist(),
        "net_benefit_model": nb_model,
        "net_benefit_all": nb_all,
        "net_benefit_none": nb_none,
    }


def find_prc_optimal_threshold(y_val: np.ndarray, val_probs: np.ndarray) -> Dict[str, float]:
    """Finds the F1-maximizing probability threshold strictly on validation Precision-Recall curve."""
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    # thresholds has length len(precisions) - 1
    f1s = (2.0 * precisions[:-1] * recalls[:-1]) / np.maximum(precisions[:-1] + recalls[:-1], 1e-12)
    best_idx = int(np.argmax(f1s))
    return {
        "optimal_threshold": float(thresholds[best_idx]),
        "val_f1_at_threshold": float(f1s[best_idx]),
        "val_precision_at_threshold": float(precisions[best_idx]),
        "val_recall_at_threshold": float(recalls[best_idx]),
    }


class PostHocCalibrator:
    """Validation-fitted post-hoc calibrator supporting Temperature, Platt, and Isotonic calibration."""

    def __init__(self, method: str = "temperature"):
        assert method in {"none", "temperature", "platt", "isotonic"}
        self.method = method
        self.temperature = 1.0
        self.platt_lr = None
        self.isotonic = None

    def fit(self, val_logits: np.ndarray, y_val: np.ndarray) -> "PostHocCalibrator":
        val_logits = np.asarray(val_logits, dtype=np.float64)
        y_val = np.asarray(y_val, dtype=np.float64)
        if self.method == "none":
            return self
        if self.method == "temperature":
            def nll_loss(log_t: np.ndarray) -> float:
                t = np.exp(log_t[0])
                p = np.clip(sigmoid(val_logits / t), 1e-7, 1.0 - 1e-7)
                return float(-np.mean(y_val * np.log(p) + (1.0 - y_val) * np.log(1.0 - p)))

            res = minimize(nll_loss, x0=np.array([0.0]), method="BFGS")
            self.temperature = float(np.exp(res.x[0]))
        elif self.method == "platt":
            self.platt_lr = LogisticRegression(C=1e6, solver="lbfgs")
            self.platt_lr.fit(val_logits.reshape(-1, 1), y_val.astype(int))
        elif self.method == "isotonic":
            val_probs = sigmoid(val_logits)
            self.isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self.isotonic.fit(val_probs, y_val)
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64)
        if self.method == "none":
            return sigmoid(logits)
        if self.method == "temperature":
            return sigmoid(logits / self.temperature)
        if self.method == "platt":
            return self.platt_lr.predict_proba(logits.reshape(-1, 1))[:, 1]
        if self.method == "isotonic":
            return self.isotonic.predict(sigmoid(logits))
        raise ValueError(f"Unknown method: {self.method}")


def compute_full_evaluation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
    n_bins: int = 15,
) -> Dict[str, Any]:
    """Computes discrimination, calibration, confusion matrix, and summary metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    ece, rel_data = compute_ece(y_true, y_prob, n_bins=n_bins)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_1": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_1": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_0": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_0": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_0": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "ece": float(ece),
        "nll": float(log_loss(y_true, y_prob)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "reliability_diagram": rel_data,
    }


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict[str, float]:
    """Exact/asymptotic McNemar's test comparing two models' binary predictions on the same test set."""
    from scipy.stats import binomtest, chi2

    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    n01 = int(np.sum((~correct_a) & correct_b))  # A wrong, B right
    n10 = int(np.sum(correct_a & (~correct_b)))  # A right, B wrong
    n_disc = n01 + n10
    if n_disc == 0:
        return {"n01": n01, "n10": n10, "statistic": 0.0, "p_value": 1.0}
    if n_disc < 25:
        p_val = float(binomtest(min(n01, n10), n=n_disc, p=0.5).pvalue)
        stat = float((abs(n01 - n10) - 1) ** 2 / n_disc)
    else:
        stat = float((abs(n01 - n10) - 1) ** 2 / n_disc)
        p_val = float(chi2.sf(stat, df=1))
    return {"n01": n01, "n10": n10, "statistic": stat, "p_value": p_val}
