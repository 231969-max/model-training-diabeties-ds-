import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Dict, List, Tuple

class DiabetesTabularDataset(Dataset):
    """
    PyTorch Dataset for tabular clinical data.
    Separates categorical feature indices, normalized numerical feature values, and target labels.
    """
    def __init__(self, df: pd.DataFrame, cat_cols: List[str], num_cols: List[str], target_col: str):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.target_col = target_col
        
        # Convert categorical indices to long tensor: [N, num_categorical]
        self.x_cat = torch.tensor(df[cat_cols].values, dtype=torch.long)
        
        # Convert normalized numerical values to float tensor: [N, num_numerical]
        self.x_num = torch.tensor(df[num_cols].values, dtype=torch.float32)
        
        # Convert target labels to float tensor: [N]
        self.y = torch.tensor(df[target_col].values, dtype=torch.float32)
        
    def __len__(self) -> int:
        return len(self.y)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'cat': self.x_cat[idx],
            'num': self.x_num[idx],
            'y': self.y[idx]
        }

def create_dataloaders(
    splits: Dict[str, pd.DataFrame],
    cat_cols: List[str],
    num_cols: List[str],
    target_col: str,
    batch_size: int = 256,
    eval_batch_size: int = 512,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders for train, validation, and test splits.
    """
    train_ds = DiabetesTabularDataset(splits['train'], cat_cols, num_cols, target_col)
    val_ds = DiabetesTabularDataset(splits['val'], cat_cols, num_cols, target_col)
    test_ds = DiabetesTabularDataset(splits['test'], cat_cols, num_cols, target_col)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
