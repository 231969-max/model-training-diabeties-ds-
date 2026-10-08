import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add current project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import set_seed, get_device, save_json
from src.preprocessing import (
    load_and_inspect_data, preprocess_pipeline,
    generate_preprocessing_summary_table,
    PAPER_CAT_COLS, PAPER_NUM_COLS, TARGET_BINARY
)
from src.dataset import create_dataloaders
from src.model import TabTransformer
from src.train import train_pipeline, eval_epoch
from src.evaluate import (
    compute_all_metrics, generate_comparison_table,
    plot_roc_curve, plot_confusion_matrix,
    plot_training_history, plot_dataset_distribution,
    plot_architecture_diagram
)

def run_experiment(
    data_path: str = "data/diabetes_dataset.csv",
    num_epochs: int = 15,
    batch_size: int = 256,
    eval_batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    seed: int = 42
):
    print("=" * 80)
    print(" REPRODUCING JOCOSIR RESEARCH PAPER: TABTRANSFORMER DIABETES PREDICTION")
    print("=" * 80)
    
    # 1. Reproducibility & Device Setup
    set_seed(seed)
    device = get_device()
    
    # 2. Step 1: Load Data & Inspect
    print("\n--- STEP 1 & 2: DATA LOADING & PREPROCESSING ---")
    df_raw, raw_info = load_and_inspect_data(data_path)
    print(f"Loaded raw dataset from: {data_path}")
    print(f"Raw shape: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")
    print(f"Categorical features: {PAPER_CAT_COLS}")
    print(f"Numerical features: {PAPER_NUM_COLS}")
    print(f"Target column: {TARGET_BINARY}")
    
    # 3. Preprocessing Pipeline
    splits, meta = preprocess_pipeline(
        df_raw,
        cat_cols=PAPER_CAT_COLS,
        num_cols=PAPER_NUM_COLS,
        target_col=TARGET_BINARY,
        test_size=0.15,
        val_size=0.15,
        random_state=seed
    )
    
    print(f"Train split: {len(splits['train']):,} rows (70.0%)")
    print(f"Val split:   {len(splits['val']):,} rows (15.0%)")
    print(f"Test split:  {len(splits['test']):,} rows (15.0%)")
    print(f"Categorical cardinalities: {meta['cat_cardinalities']}")
    
    # Generate Preprocessing Summary Table
    prep_table = generate_preprocessing_summary_table(df_raw, splits['train'])
    prep_table.to_csv("outputs/metrics/preprocessing_summary_table.csv", index=False)
    print("\n[Preprocessing Summary Table]")
    print(prep_table.to_string(index=False))
    
    # 4. Generate Dataset Distribution Plot (Figure 1 recreation)
    print("\n--- STEP 15: GENERATING DATASET DISTRIBUTION FIGURE ---")
    plot_dataset_distribution(df_raw, "outputs/figures/dataset_distribution.png")
    
    # 5. Generate Architecture Diagram (Figure 2 recreation)
    print("\n--- STEP 16: GENERATING ARCHITECTURE DIAGRAM ---")
    plot_architecture_diagram("outputs/figures/tabtransformer_architecture.png")
    
    # 6. Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        splits,
        cat_cols=PAPER_CAT_COLS,
        num_cols=PAPER_NUM_COLS,
        target_col=TARGET_BINARY,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size
    )
    
    # 7. Model Initialization (Steps 6-10)
    print("\n--- STEP 6 - 10: MODEL INITIALIZATION & HYPERPARAMETERS ---")
    model = TabTransformer(
        cat_cardinalities=meta['cat_cardinalities'],
        num_numerical=len(PAPER_NUM_COLS),
        embed_dim=32,
        num_heads=8,
        num_layers=4,
        ff_dim=128,
        transformer_dropout=0.1,
        mlp_hidden_dims=[64, 32],
        mlp_dropout=0.2,
        num_classes=1
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TabTransformer initialized with {total_params:,} trainable parameters.")
    print("Architecture Summary:")
    print(f"  - Embedding dimension: 32 per categorical feature (4 features)")
    print(f"  - Transformer: 4 layers, 8 heads, feedforward dim 128, GELU activation, dropout 0.1")
    print(f"  - Concatenation dimension: (4 * 32) + 5 = 133 features")
    print(f"  - MLP Classifier: 133 -> 64 (ReLU, 0.2 dropout) -> 32 (ReLU, 0.2 dropout) -> 1 (Logits)")
    print(f"  - Loss function: BCEWithLogitsLoss")
    print(f"  - Optimizer: AdamW(lr={lr}, weight_decay={weight_decay})")
    print(f"  - Scheduler: ReduceLROnPlateau(mode='min', factor=0.5, patience=1)")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    
    # 8. Train Model (Step 11)
    checkpoint_path = "outputs/models/best_tabtransformer.pt"
    history, best_val_metrics = train_pipeline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        num_epochs=num_epochs,
        device=device,
        save_model_path=checkpoint_path
    )
    
    # Save training history
    save_json(history, "outputs/metrics/training_history.json")
    
    # Plot Training History Curves (Step 17)
    plot_training_history(history, "outputs/figures/training_history.png")
    
    # 9. Load Best Model Checkpoint for Final Test Evaluation (Steps 12 - 14, 18)
    print("\n--- STEP 12 - 14, 18: FINAL TEST EVALUATION ---")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model weights from epoch {checkpoint['epoch']} (Val Loss: {checkpoint['val_loss']:.4f})")
    
    test_eval = eval_epoch(model, test_loader, criterion, device)
    test_metrics = compute_all_metrics(test_eval['trues'], test_eval['preds'], test_eval['probs'])
    
    # Save Test Metrics JSON
    save_json(test_metrics, "outputs/metrics/test_metrics.json")
    
    print("\n[Test Classification Report]")
    print(test_metrics['classification_report'])
    print(f"Test ROC-AUC Score: {test_metrics['roc_auc']:.4f}")
    
    # Generate Comparison Table with Paper Reference Results
    comp_table = generate_comparison_table(test_metrics)
    comp_table.to_csv("outputs/metrics/paper_comparison_table.csv", index=False)
    print("\n[Comparison Table: Paper Reported vs Reproduced Results]")
    print(comp_table.to_string(index=False))
    
    # Plot ROC Curve (Figure 3 recreation)
    plot_roc_curve(test_eval['trues'], test_eval['probs'], "outputs/figures/roc_curve.png")
    
    # Plot Confusion Matrix
    plot_confusion_matrix(test_eval['trues'], test_eval['preds'], "outputs/figures/confusion_matrix.png")
    
    print("\n" + "=" * 80)
    print(" REPRODUCTION EXPERIMENT SUCCESSFULLY COMPLETED!")
    print(" All figures saved to: outputs/figures/")
    print(" All metrics saved to: outputs/metrics/")
    print(" Best model saved to:  outputs/models/")
    print("=" * 80)
    return test_metrics, comp_table

if __name__ == "__main__":
    run_experiment()
