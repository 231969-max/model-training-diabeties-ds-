import os
import random
import numpy as np
import torch
import json

def set_seed(seed: int = 42):
    """
    Sets random seeds across Python, NumPy, and PyTorch for full reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_device() -> torch.device:
    """
    Automatically detects and returns CUDA GPU device if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("[Device] CUDA not available. Using CPU.")
    return device

def save_json(data: dict, filepath: str):
    """
    Saves dictionary to JSON file with indentation.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"[Saved] {filepath}")

def load_json(filepath: str) -> dict:
    """
    Loads JSON file into dictionary.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
