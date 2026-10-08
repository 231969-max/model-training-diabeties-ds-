import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from typing import List, Dict, Any

CLASS_ORDER = ['Gestational', 'No Diabetes', 'Pre-Diabetes', 'Type 1', 'Type 2']

def plot_class_distribution(y: pd.Series, save_path: str = "results/figures/class_distribution.png"):
    """
    Plots the target class distribution (diabetes_stage).
    """
    counts = y.value_counts()[CLASS_ORDER]
    proportions = (counts / len(y)) * 100
    
    plt.figure(figsize=(8, 5), dpi=300)
    colors = ['#f59e0b', '#10b981', '#3b82f6', '#ec4899', '#8b5cf6']
    bars = plt.bar(CLASS_ORDER, counts, color=colors, edgecolor='#1f2937', lw=1.2, alpha=0.85)
    
    for bar, count, prop in zip(bars, counts, proportions):
        plt.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1000,
            f"{count:,}\n({prop:.2f}%)",
            ha='center', va='bottom', fontsize=9.5, fontweight='bold'
        )
        
    plt.title('Target Distribution: diabetes_stage (100,000 Clinical Profiles)', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Diabetes Stage', fontsize=11, fontweight='semibold')
    plt.ylabel('Number of Patients', fontsize=11, fontweight='semibold')
    plt.ylim(0, max(counts) * 1.18)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[Figure Saved] Class Distribution: {save_path}")

def plot_confusion_matrix_custom(cm: np.ndarray, model_name: str, save_path: str):
    """
    Plots annotated confusion matrix for a specific model matching Orange Data Mining layout.
    """
    plt.figure(figsize=(7, 6), dpi=300)
    
    # Calculate row totals and column totals
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    
    # Annotations
    annot_matrix = []
    for i in range(len(CLASS_ORDER)):
        row = []
        for j in range(len(CLASS_ORDER)):
            val = cm[i, j]
            row.append(f"{val:,}")
        annot_matrix.append(row)
        
    sns.heatmap(
        cm, annot=np.array(annot_matrix), fmt="", cmap='Blues',
        xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER,
        cbar=True, annot_kws={"size": 10, "weight": "bold"}
    )
    
    plt.title(f'Confusion Matrix — {model_name}\n(Orange Data Mining Replication)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Predicted Class', fontsize=11, fontweight='semibold')
    plt.ylabel('Actual Class', fontsize=11, fontweight='semibold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[Figure Saved] Confusion Matrix ({model_name}): {save_path}")

def plot_multiclass_roc(y_true: pd.Series, y_prob: np.ndarray, model_name: str, save_path: str):
    """
    Plots One-vs-Rest multiclass ROC curves for each target class.
    """
    plt.figure(figsize=(7.5, 6), dpi=300)
    
    # Binarize labels
    y_bin = pd.get_dummies(y_true)[CLASS_ORDER].values
    colors = ['#f59e0b', '#10b981', '#3b82f6', '#ec4899', '#8b5cf6']
    
    if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] == len(CLASS_ORDER):
        for i, (cls_name, color) in enumerate(zip(CLASS_ORDER, colors)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2, label=f'{cls_name} (AUC = {roc_auc:.4f})')
    else:
        # Constant model dummy flat diagonal
        plt.plot([0, 1], [0, 1], color='#6b7280', lw=2, linestyle=':', label='Constant Baseline (AUC = 0.5000)')

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='semibold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='semibold')
    plt.title(f'Multiclass ROC Analysis — {model_name}', fontsize=12, fontweight='bold', pad=12)
    plt.legend(loc="lower right", fontsize=9, frameon=True)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[Figure Saved] ROC Analysis ({model_name}): {save_path}")

def plot_performance_lift_curve(y_true: pd.Series, y_prob: np.ndarray, model_name: str, target_class: str = 'Gestational', save_path: str = "results/figures/performance_curve.png"):
    """
    Plots cumulative gain / lift curve for target class (matching Gambar 13, 16, 19).
    """
    plt.figure(figsize=(7, 5.5), dpi=300)
    
    cls_idx = CLASS_ORDER.index(target_class)
    y_binary = (y_true == target_class).astype(int).values
    
    if y_prob is not None and y_prob.ndim == 2:
        probs = y_prob[:, cls_idx]
        sorted_indices = np.argsort(probs)[::-1]
        sorted_y = y_binary[sorted_indices]
        
        cum_positives = np.cumsum(sorted_y)
        total_positives = np.sum(y_binary)
        
        fraction_data = np.arange(1, len(y_binary) + 1) / len(y_binary)
        cumulative_gain = cum_positives / total_positives if total_positives > 0 else fraction_data
        
        # Lift calculation: cumulative_gain / fraction_data
        lift = np.where(fraction_data > 0, cumulative_gain / fraction_data, 1.0)
        
        plt.plot(fraction_data, cumulative_gain, color='#2563eb', lw=2.2, label=f'{model_name} Cumulative Gain')
        plt.plot(fraction_data, fraction_data, 'r--', lw=1.5, label='Random Baseline')
    else:
        fraction_data = np.linspace(0, 1, 100)
        plt.plot(fraction_data, fraction_data, 'r--', lw=1.5, label='Constant Baseline')
        
    plt.xlabel('Fraction of Population Sampled', fontsize=11, fontweight='semibold')
    plt.ylabel('Cumulative True Positive Rate (Gain)', fontsize=11, fontweight='semibold')
    plt.title(f'Performance / Lift Curve ({target_class}) — {model_name}', fontsize=12, fontweight='bold', pad=12)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[Figure Saved] Performance Curve ({model_name}): {save_path}")

def plot_prediction_examples(df_orig: pd.DataFrame, preds_dict: Dict[str, List[str]], save_path: str = "results/figures/prediction_examples.png"):
    """
    Renders prediction output inspection widget equivalent to Orange Predictions widget (Gambar 20).
    """
    sample_df = df_orig.head(15).copy()
    for model_name, preds in preds_dict.items():
        sample_df[f"Pred_{model_name}"] = preds[:15]
        
    display_cols = ['age', 'gender', 'glucose_fasting', 'hba1c', 'diagnosed_diabetes', 'diabetes_stage'] + [f"Pred_{m}" for m in preds_dict.keys()]
    sub_df = sample_df[[c for c in display_cols if c in sample_df.columns]]
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.axis('off')
    
    table = ax.table(
        cellText=sub_df.values,
        colLabels=sub_df.columns,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.2, 1.4)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#dbeafe')
            cell.set_text_props(weight='bold')
        elif row % 2 == 1:
            cell.set_facecolor('#f9fafb')
            
    plt.title('Sample Output Predictions (Orange Data Mining Predictions Widget Recreation)', fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[Figure Saved] Prediction Examples: {save_path}")
