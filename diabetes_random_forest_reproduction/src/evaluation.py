import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, Any, List, Tuple

CLASS_ORDER = ['Gestational', 'No Diabetes', 'Pre-Diabetes', 'Type 1', 'Type 2']

def evaluate_model_cv(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes Stratified K-Fold cross-validation to reproduce Orange Data Mining's Test & Score evaluation.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Generate cross-validated predictions and probability estimates
    preds = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    
    # Try getting predicted probabilities for ROC analysis
    try:
        probs = cross_val_predict(model, X, y, cv=cv, method='predict_proba', n_jobs=-1)
        classes = getattr(model, 'classes_', np.unique(y))
        
        # Multiclass OvR ROC-AUC
        ovr_auc_weighted = roc_auc_score(y, probs, multi_class='ovr', average='weighted')
        ovr_auc_macro = roc_auc_score(y, probs, multi_class='ovr', average='macro')
    except Exception:
        probs = None
        classes = np.unique(y)
        ovr_auc_weighted = 0.5
        ovr_auc_macro = 0.5

    ca = accuracy_score(y, preds)
    prec_weighted = precision_score(y, preds, average='weighted', zero_division=0)
    rec_weighted = recall_score(y, preds, average='weighted', zero_division=0)
    f1_weighted = f1_score(y, preds, average='weighted', zero_division=0)
    
    prec_macro = precision_score(y, preds, average='macro', zero_division=0)
    rec_macro = recall_score(y, preds, average='macro', zero_division=0)
    f1_macro = f1_score(y, preds, average='macro', zero_division=0)
    
    mcc = matthews_corrcoef(y, preds)
    
    cm = confusion_matrix(y, preds, labels=CLASS_ORDER)
    report_dict = classification_report(y, preds, labels=CLASS_ORDER, output_dict=True, zero_division=0)
    report_str = classification_report(y, preds, labels=CLASS_ORDER, digits=4, zero_division=0)
    
    return {
        'CA': float(ca),
        'Precision_weighted': float(prec_weighted),
        'Recall_weighted': float(rec_weighted),
        'F1_weighted': float(f1_weighted),
        'Precision_macro': float(prec_macro),
        'Recall_macro': float(rec_macro),
        'F1_macro': float(f1_macro),
        'MCC': float(mcc),
        'AUC_ovr_weighted': float(ovr_auc_weighted),
        'AUC_ovr_macro': float(ovr_auc_macro),
        'confusion_matrix': cm.tolist(),
        'classification_report_str': report_str,
        'classification_report_dict': report_dict,
        'predictions': preds.tolist(),
        'probabilities': probs.tolist() if probs is not None else None,
        'classes': CLASS_ORDER
    }

def generate_paper_comparison_table(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Constructs a complete comparison table comparing paper reported metrics with reproduced results
    for Random Forest, Decision Tree, and Constant Model.
    """
    paper_reference = {
        'Random Forest': {'AUC': -1.692, 'CA': 0.996, 'F1': 0.994, 'Prec': 0.992, 'Recall': 0.996, 'MCC': 0.992},
        'Decision Tree': {'AUC': -1.697, 'CA': 0.996, 'F1': 0.994, 'Prec': 0.992, 'Recall': 0.996, 'MCC': 0.992},
        'Constant':      {'AUC': -0.843, 'CA': 0.598, 'F1': 0.448, 'Prec': 0.358, 'Recall': 0.598, 'MCC': 0.000}
    }
    
    rows = []
    for model_name, paper_vals in paper_reference.items():
        if model_name in results_dict:
            rep = results_dict[model_name]
            rows.append({
                'Model': model_name,
                'Paper CA (Acc)': f"{paper_vals['CA']:.3f}",
                'Reprod CA': f"{rep['CA']:.4f}",
                'Diff CA': f"{(rep['CA'] - paper_vals['CA']):+.4f}",
                'Paper F1': f"{paper_vals['F1']:.3f}",
                'Reprod F1 (Weighted)': f"{rep['F1_weighted']:.4f}",
                'Diff F1': f"{(rep['F1_weighted'] - paper_vals['F1']):+.4f}",
                'Paper Prec': f"{paper_vals['Prec']:.3f}",
                'Reprod Prec': f"{rep['Precision_weighted']:.4f}",
                'Paper Recall': f"{paper_vals['Recall']:.3f}",
                'Reprod Recall': f"{rep['Recall_weighted']:.4f}",
                'Paper MCC': f"{paper_vals['MCC']:.3f}",
                'Reprod MCC': f"{rep['MCC']:.4f}",
                'Paper AUC (Orange)': f"{paper_vals['AUC']:.3f}",
                'Reprod OvR AUC': f"{rep['AUC_ovr_weighted']:.4f}"
            })
            
    return pd.DataFrame(rows)
