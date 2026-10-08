import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, List, Tuple, Any

def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Executes a single training epoch over all batches.
    """
    model.train()
    total_loss = 0.0
    preds = []
    trues = []
    
    for batch in loader:
        x_num = batch['num'].to(device)
        x_cat = batch['cat'].to(device)
        y = batch['y'].to(device)
        
        optimizer.zero_grad()
        logits = model(x_num, x_cat)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x_num.size(0)
        
        with torch.no_grad():
            probs = torch.sigmoid(logits).cpu().numpy()
            yhat = (probs >= 0.5).astype(int)
            preds.extend(yhat.tolist())
            trues.extend(y.cpu().numpy().tolist())
            
    epoch_loss = total_loss / len(loader.dataset)
    epoch_acc = accuracy_score(trues, preds)
    return epoch_loss, epoch_acc

def eval_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluates model performance across validation or test set without gradient calculation.
    """
    model.eval()
    total_loss = 0.0
    preds = []
    trues = []
    prob_list = []
    
    with torch.no_grad():
        for batch in loader:
            x_num = batch['num'].to(device)
            x_cat = batch['cat'].to(device)
            y = batch['y'].to(device)
            
            logits = model(x_num, x_cat)
            loss = criterion(logits, y)
            total_loss += loss.item() * x_num.size(0)
            
            probs = torch.sigmoid(logits).cpu().numpy()
            yhat = (probs >= threshold).astype(int)
            
            preds.extend(yhat.tolist())
            trues.extend(y.cpu().numpy().tolist())
            prob_list.extend(probs.tolist())
            
    eval_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(trues, preds)
    precision = precision_score(trues, preds, zero_division=0)
    recall = recall_score(trues, preds, zero_division=0)
    f1 = f1_score(trues, preds, zero_division=0)
    
    try:
        auc = roc_auc_score(trues, prob_list)
    except Exception:
        auc = 0.5
        
    return {
        'loss': eval_loss,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'preds': preds,
        'trues': trues,
        'probs': prob_list
    }

def train_pipeline(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    num_epochs: int = 15,
    device: torch.device = torch.device('cpu'),
    save_model_path: str = "outputs/models/best_tabtransformer.pt"
) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    """
    Runs the complete multi-epoch training loop with validation evaluation,
    learning-rate scheduling via ReduceLROnPlateau, and best checkpoint retention.
    """
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_auc': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    best_metrics = {}
    
    print(f"\n================ STARTING TABTRANSFORMER TRAINING ================")
    print(f"Total Epochs: {num_epochs} | Device: {device}")
    print(f"{'Epoch':<6} | {'Train Loss':<11} | {'Train Acc':<10} | {'Val Loss':<10} | {'Val Acc':<9} | {'Val F1':<8} | {'Val AUC':<8} | {'Time (s)':<8}")
    print("-" * 85)
    
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        
        # 1. Training pass
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # 2. Validation pass
        val_res = eval_epoch(model, val_loader, criterion, device)
        
        # 3. Learning rate scheduler step based on validation loss
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_res['loss'])
        
        elapsed = time.time() - t0
        
        # 4. Record history
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_res['loss'])
        history['val_acc'].append(val_res['accuracy'])
        history['val_precision'].append(val_res['precision'])
        history['val_recall'].append(val_res['recall'])
        history['val_f1'].append(val_res['f1'])
        history['val_auc'].append(val_res['auc'])
        history['lr'].append(current_lr)
        
        print(f"{epoch:<6} | {train_loss:<11.4f} | {train_acc*100:<9.2f}% | {val_res['loss']:<10.4f} | {val_res['accuracy']*100:<8.2f}% | {val_res['f1']:<8.4f} | {val_res['auc']:<8.4f} | {elapsed:<8.2f}")
        
        # 5. Checkpoint saving
        if val_res['loss'] < best_val_loss:
            best_val_loss = val_res['loss']
            best_metrics = val_res.copy()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_res['loss'],
                'val_acc': val_res['accuracy'],
                'val_auc': val_res['auc']
            }, save_model_path)
            
    print("-" * 85)
    print(f"[Training Complete] Best Checkpoint saved with Val Loss = {best_val_loss:.4f} to {save_model_path}\n")
    return history, best_metrics
