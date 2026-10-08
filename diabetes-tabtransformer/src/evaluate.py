import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import torch
import torch.nn as nn
from typing import Dict, Any, List

def compute_all_metrics(y_true: List[int], y_pred: List[int], y_prob: List[float]) -> Dict[str, Any]:
    """
    Computes all standard classification performance metrics and per-class breakdowns.
    """
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    
    # Class 0 metrics (Non-diabetic)
    prec_0 = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    rec_0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_0 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    
    # Class 1 metrics (Diabetic)
    prec_1 = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_1 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    # Macro and Weighted metrics
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    report_str = classification_report(
        y_true, y_pred,
        target_names=['Non-Diabetic (0)', 'Diabetic (1)'],
        digits=4
    )
    
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': float(acc),
        'roc_auc': float(auc),
        'class_0': {
            'precision': float(prec_0),
            'recall': float(rec_0),
            'f1_score': float(f1_0)
        },
        'class_1': {
            'precision': float(prec_1),
            'recall': float(rec_1),
            'f1_score': float(f1_1)
        },
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'classification_report': report_str,
        'confusion_matrix': cm.tolist()
    }

def generate_comparison_table(metrics: Dict[str, Any]) -> pd.DataFrame:
    """
    Constructs a comparison table comparing paper reported values with reproduced results.
    """
    rows = [
        {
            "Metric / Target": "Overall Accuracy",
            "Paper Reported": "82.55%",
            "Reproduced Result": f"{metrics['accuracy']*100:.2f}%",
            "Difference": f"{(metrics['accuracy']*100 - 82.55):+.2f}%"
        },
        {
            "Metric / Target": "Class 0 (Non-Diabetic) Precision",
            "Paper Reported": "0.7716",
            "Reproduced Result": f"{metrics['class_0']['precision']:.4f}",
            "Difference": f"{(metrics['class_0']['precision'] - 0.7716):+.4f}"
        },
        {
            "Metric / Target": "Class 0 (Non-Diabetic) Recall",
            "Paper Reported": "0.8006",
            "Reproduced Result": f"{metrics['class_0']['recall']:.4f}",
            "Difference": f"{(metrics['class_0']['recall'] - 0.8006):+.4f}"
        },
        {
            "Metric / Target": "Class 0 (Non-Diabetic) F1-score",
            "Paper Reported": "0.7858",
            "Reproduced Result": f"{metrics['class_0']['f1_score']:.4f}",
            "Difference": f"{(metrics['class_0']['f1_score'] - 0.7858):+.4f}"
        },
        {
            "Metric / Target": "Class 1 (Diabetic) Precision",
            "Paper Reported": "0.8637",
            "Reproduced Result": f"{metrics['class_1']['precision']:.4f}",
            "Difference": f"{(metrics['class_1']['precision'] - 0.8637):+.4f}"
        },
        {
            "Metric / Target": "Class 1 (Diabetic) Recall",
            "Paper Reported": "0.8420",
            "Reproduced Result": f"{metrics['class_1']['recall']:.4f}",
            "Difference": f"{(metrics['class_1']['recall'] - 0.8420):+.4f}"
        },
        {
            "Metric / Target": "Class 1 (Diabetic) F1-score",
            "Paper Reported": "0.8527",
            "Reproduced Result": f"{metrics['class_1']['f1_score']:.4f}",
            "Difference": f"{(metrics['class_1']['f1_score'] - 0.8527):+.4f}"
        },
        {
            "Metric / Target": "ROC-AUC Score",
            "Paper Reported": "0.9009",
            "Reproduced Result": f"{metrics['roc_auc']:.4f}",
            "Difference": f"{(metrics['roc_auc'] - 0.9009):+.4f}"
        }
    ]
    return pd.DataFrame(rows)

def plot_roc_curve(y_true: List[int], y_prob: List[float], save_path: str = "outputs/figures/roc_curve.png"):
    """
    Plots and saves Receiver Operating Characteristic (ROC) curve similar to Figure 3.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(7, 6), dpi=300)
    plt.plot(fpr, tpr, color='#1f77b4', lw=2.2, label=f'ROC Curve (AUC = {auc_val:.4f})')
    plt.plot([0, 1], [0, 1], color='#d62728', lw=1.8, linestyle='--', label='Random Classifier')
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='semibold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='semibold')
    plt.title('Receiver Operating Characteristic Curve (TabTransformer)', fontsize=13, fontweight='bold', pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[Figure Saved] ROC Curve: {save_path}")

def plot_confusion_matrix(y_true: List[int], y_pred: List[int], save_path: str = "outputs/figures/confusion_matrix.png"):
    """
    Plots annotated confusion matrix with True Negatives, False Positives, False Negatives, and True Positives.
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = [
        [f"TN\n{cm[0,0]:,}\n({cm[0,0]/cm.sum()*100:.1f}%)", f"FP\n{cm[0,1]:,}\n({cm[0,1]/cm.sum()*100:.1f}%)"],
        [f"FN\n{cm[1,0]:,}\n({cm[1,0]/cm.sum()*100:.1f}%)", f"TP\n{cm[1,1]:,}\n({cm[1,1]/cm.sum()*100:.1f}%)"]
    ]
    
    plt.figure(figsize=(6.5, 5.5), dpi=300)
    sns.heatmap(
        cm, annot=labels, fmt="", cmap='Blues', cbar=True,
        xticklabels=['Non-Diabetic (0)', 'Diabetic (1)'],
        yticklabels=['Non-Diabetic (0)', 'Diabetic (1)'],
        annot_kws={"size": 12, "weight": "bold"}
    )
    plt.xlabel('Predicted Diagnosis', fontsize=12, fontweight='semibold')
    plt.ylabel('Actual Diagnosis', fontsize=12, fontweight='semibold')
    plt.title('Confusion Matrix — TabTransformer Diabetes Classification', fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[Figure Saved] Confusion Matrix: {save_path}")

def plot_training_history(history: Dict[str, List[float]], save_path: str = "outputs/figures/training_history.png"):
    """
    Plots multi-metric training curves across epochs: Loss, Accuracy, F1-Score, ROC-AUC.
    """
    epochs = history['epoch']
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=300)
    
    # Loss Curve
    axes[0, 0].plot(epochs, history['train_loss'], 'o-', color='#1f77b4', label='Train Loss', lw=2)
    axes[0, 0].plot(epochs, history['val_loss'], 's--', color='#ff7f0e', label='Validation Loss', lw=2)
    axes[0, 0].set_title('Binary Cross-Entropy Loss Progression', fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (BCEWithLogits)')
    axes[0, 0].legend(frameon=True)
    axes[0, 0].grid(True, linestyle=':', alpha=0.6)
    
    # Accuracy Curve
    axes[0, 1].plot(epochs, [a * 100 for a in history['train_acc']], 'o-', color='#2ca02c', label='Train Accuracy', lw=2)
    axes[0, 1].plot(epochs, [a * 100 for a in history['val_acc']], 's--', color='#9467bd', label='Val Accuracy', lw=2)
    axes[0, 1].set_title('Classification Accuracy Progression (%)', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend(frameon=True)
    axes[0, 1].grid(True, linestyle=':', alpha=0.6)
    
    # Validation F1 Progression
    axes[1, 0].plot(epochs, history['val_f1'], 'd-', color='#d62728', label='Validation F1 (Diabetic Class)', lw=2)
    axes[1, 0].plot(epochs, history['val_precision'], '^--', color='#8c564b', label='Val Precision', lw=1.8)
    axes[1, 0].plot(epochs, history['val_recall'], 'v:', color='#e377c2', label='Val Recall', lw=1.8)
    axes[1, 0].set_title('Validation F1, Precision & Recall', fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend(frameon=True)
    axes[1, 0].grid(True, linestyle=':', alpha=0.6)
    
    # Validation ROC-AUC & LR
    axes[1, 1].plot(epochs, history['val_auc'], 'o-', color='#17becf', label='Validation ROC-AUC', lw=2)
    axes[1, 1].set_title('Validation ROC-AUC Progression', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('ROC-AUC Score')
    axes[1, 1].legend(frameon=True)
    axes[1, 1].grid(True, linestyle=':', alpha=0.6)
    
    plt.suptitle('TabTransformer Training and Validation Convergence History', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[Figure Saved] Training History: {save_path}")

def plot_dataset_distribution(df: pd.DataFrame, save_path: str = "outputs/figures/dataset_distribution.png"):
    """
    Recreates comprehensive health indicators distribution visualization similar to Figure 1.
    """
    key_indicators = [
        'age', 'bmi', 'glucose_fasting', 'glucose_postprandial',
        'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides',
        'hba1c', 'insulin_level', 'systolic_bp', 'diastolic_bp',
        'physical_activity_minutes_per_week', 'sleep_hours_per_day', 'diet_score', 'alcohol_consumption_per_week'
    ]
    present_cols = [col for col in key_indicators if col in df.columns]
    
    n_cols = 4
    n_rows = (len(present_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.2 * n_rows), dpi=300)
    axes = axes.flatten()
    
    palette = sns.color_palette('tab10', len(present_cols))
    
    for i, col in enumerate(present_cols):
        ax = axes[i]
        sns.histplot(df[col], kde=True, ax=ax, color=palette[i % 10], bins=30, edgecolor=None, alpha=0.6)
        ax.set_title(col.replace('_', ' ').title(), fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Frequency')
        ax.grid(True, linestyle=':', alpha=0.5)
        
    for j in range(len(present_cols), len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle('Clinical & Lifestyle Health Indicators Feature Distributions (Kaggle Diabetes Dataset)', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[Figure Saved] Dataset Distribution: {save_path}")

def plot_architecture_diagram(save_path: str = "outputs/figures/tabtransformer_architecture.png"):
    """
    Generates high-resolution TabTransformer architectural schematic matching Figure 2.
    """
    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=300)
    ax.axis('off')
    
    # Custom colored boxes representing pipeline components
    boxes = [
        {"title": "1. Input CSV\n(100k records, 31 cols)", "pos": (0.12, 0.82), "color": "#ffddc1", "ec": "#d97706"},
        {"title": "2. Preprocessing\nDrop Leakage & Target\nStandardScaler + LabelEnc", "pos": (0.12, 0.50), "color": "#ffddc1", "ec": "#d97706"},
        {"title": "3. TabTransformer Input\nCategorical (4) + Numeric (5)", "pos": (0.42, 0.82), "color": "#fef3c7", "ec": "#d97706"},
        {"title": "4. Categorical Embeddings\n(4 x 32-dim Vectors)", "pos": (0.42, 0.50), "color": "#dbeafe", "ec": "#2563eb"},
        {"title": "5. Transformer Blocks\n(4 Layers, 8 Heads, GELU)\nContextual Self-Attention", "pos": (0.72, 0.82), "color": "#fed7aa", "ec": "#ea580c"},
        {"title": "6. MLP Classifier\n[64 -> 32], ReLU, Dropout\nCombined Latent Vectors", "pos": (0.72, 0.50), "color": "#fbcfe8", "ec": "#db2777"},
        {"title": "7. Prediction Output\nBinary diagnosed_diabetes\n(Logits / Sigmoid >= 0.5)", "pos": (0.42, 0.15), "color": "#fde047", "ec": "#ca8a04"},
    ]
    
    for b in boxes:
        ax.text(
            b["pos"][0], b["pos"][1], b["title"],
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.6", facecolor=b["color"], edgecolor=b["ec"], lw=2)
        )
        
    # Draw connecting arrows
    arrow_props = dict(facecolor='#374151', edgecolor='#374151', width=1.5, headwidth=8, headlength=8, shrink=0.1)
    
    # 1 -> 2
    ax.annotate('', xy=(0.12, 0.59), xytext=(0.12, 0.74), arrowprops=arrow_props)
    # 2 -> 3
    ax.annotate('', xy=(0.32, 0.82), xytext=(0.22, 0.60), arrowprops=arrow_props)
    # 3 -> 4
    ax.annotate('', xy=(0.42, 0.59), xytext=(0.42, 0.74), arrowprops=arrow_props)
    # 4 -> 5
    ax.annotate('', xy=(0.63, 0.82), xytext=(0.52, 0.58), arrowprops=arrow_props)
    # 5 -> 6
    ax.annotate('', xy=(0.72, 0.59), xytext=(0.72, 0.74), arrowprops=arrow_props)
    # 6 -> 7
    ax.annotate('', xy=(0.53, 0.22), xytext=(0.63, 0.42), arrowprops=arrow_props)
    
    plt.title('TabTransformer Model End-to-End Pipeline & Architecture Diagram (JoCoSiR Re-creation)', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[Figure Saved] Architecture Diagram: {save_path}")
