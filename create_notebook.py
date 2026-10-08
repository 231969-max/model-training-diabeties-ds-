import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = []

# Title & Abstract
cells.append(nbf.v4.new_markdown_cell("""# Classification of Health Indicators for Diabetes Mellitus Prediction Using a TabTransformer Model on Clinical Tabular Data
**Paper Authors**: Al Khaidar, Sri Kurnia (*Journal of Computer Science Research - JoCoSiR*)  
**Reproduction Study & Implementation**: Complete End-to-End Deep Learning Experiment

---
### Research Overview & Objective
This notebook reproduces the complete machine learning and deep learning experimental pipeline from the research paper. The goal is to predict diabetes mellitus (`diagnosed_diabetes`: 0 = Non-diabetic, 1 = Diabetic) from clinical tabular health indicators using a **TabTransformer** architecture with self-attention embeddings.
"""))

# Step 1: Environment & Seeds
cells.append(nbf.v4.new_code_cell("""import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Execution Device: {device}')
"""))

# Step 2: Load Data & Inspect
cells.append(nbf.v4.new_markdown_cell("""## Step 1 & 2: Dataset Loading, Exploratory Inspection & Preprocessing
Load `diabetes_dataset.csv` (100,000 patient records) and inspect attributes, distributions, and missing values.
"""))

cells.append(nbf.v4.new_code_cell("""df_raw = pd.read_csv('../data/diabetes_dataset.csv')
print(f'Dataset Shape: {df_raw.shape}')
print(f'Total Missing Values: {df_raw.isnull().sum().sum()}')
print('\\nTarget (diagnosed_diabetes) Distribution:')
print(df_raw['diagnosed_diabetes'].value_counts(normalize=True))
df_raw.head()
"""))

# Step 3: Feature Engineering & Preprocessing Table
cells.append(nbf.v4.new_code_cell("""# Define Feature Groups as specified in paper Section 3.2
cat_cols = ['gender', 'smoking_status', 'family_history_diabetes', 'employment_status']
num_cols = ['age', 'bmi', 'glucose_fasting', 'cholesterol_total', 'hba1c']
target_col = 'diagnosed_diabetes'

print(f'Categorical Features ({len(cat_cols)}):', cat_cols)
print(f'Numerical Features ({len(num_cols)}):', num_cols)

# Stratified Splitting (70% Train, 15% Val, 15% Test)
train_val_df, test_df = train_test_split(df_raw, test_size=0.15, random_state=SEED, stratify=df_raw[target_col])
train_df, val_df = train_test_split(train_val_df, test_size=0.15/0.85, random_state=SEED, stratify=train_val_df[target_col])

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

print(f'Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}')

# Fit LabelEncoder on Train only
cat_dims = []
encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    train_df[c] = le.fit_transform(train_df[c].astype(str))
    val_df[c] = le.transform(val_df[c].astype(str))
    test_df[c] = le.transform(test_df[c].astype(str))
    encoders[c] = le
    cat_dims.append(len(le.classes_))

# Fit StandardScaler on Train only
scaler = StandardScaler()
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
val_df[num_cols] = scaler.transform(val_df[num_cols])
test_df[num_cols] = scaler.transform(test_df[num_cols])

print('Categorical Cardinalities:', cat_dims)
"""))

# Step 4: Dataset & DataLoader
cells.append(nbf.v4.new_code_cell("""class DiabetesTabularDataset(Dataset):
    def __init__(self, df, cat_cols, num_cols, target_col):
        self.x_cat = torch.tensor(df[cat_cols].values, dtype=torch.long)
        self.x_num = torch.tensor(df[num_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df[target_col].values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return {'cat': self.x_cat[idx], 'num': self.x_num[idx], 'y': self.y[idx]}

train_ds = DiabetesTabularDataset(train_df, cat_cols, num_cols, target_col)
val_ds = DiabetesTabularDataset(val_df, cat_cols, num_cols, target_col)
test_ds = DiabetesTabularDataset(test_df, cat_cols, num_cols, target_col)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
"""))

# Step 5: TabTransformer Model
cells.append(nbf.v4.new_markdown_cell("""## TabTransformer PyTorch Model
- Categorical embedding dimension: 32 per feature
- Multi-Head Self-Attention: 4 layers, 8 heads, Feedforward dim: 128, GELU activation, Dropout: 0.1
- Feature concatenation: Contextual categorical embeddings + Normalized numerical features
- MLP Classifier: Linear(133 -> 64) -> ReLU -> Dropout(0.2) -> Linear(64 -> 32) -> ReLU -> Dropout(0.2) -> Linear(32 -> 1)
"""))

cells.append(nbf.v4.new_code_cell("""class TabTransformer(nn.Module):
    def __init__(self, cat_cardinalities, num_numerical, embed_dim=32, num_heads=8, num_layers=4, ff_dim=128, dropout=0.1, mlp_dropout=0.2):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(card, embed_dim) for card in cat_cardinalities])
        self.num_cat = len(cat_cardinalities)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        in_mlp_dim = (self.num_cat * embed_dim) + num_numerical
        self.mlp = nn.Sequential(
            nn.Linear(in_mlp_dim, 64),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x_num, x_cat):
        cat_embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_stack = torch.stack(cat_embeds, dim=1) # [B, num_cat, 32]
        
        contextual_cat = self.transformer(cat_stack) # [B, num_cat, 32]
        cat_flat = contextual_cat.flatten(start_dim=1) # [B, num_cat * 32]
        
        combined = torch.cat([cat_flat, x_num], dim=1) # [B, 133]
        logits = self.mlp(combined).squeeze(-1) # [B]
        return logits

model = TabTransformer(cat_dims, len(num_cols)).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
print(model)
"""))

# Step 6: Load Checkpoint & Evaluate
cells.append(nbf.v4.new_markdown_cell("""## Step 11 & 12: Load Best Trained Model Checkpoint & Evaluate on Test Set"""))

cells.append(nbf.v4.new_code_cell("""checkpoint_path = '../outputs/models/best_tabtransformer.pt'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best checkpoint from epoch {checkpoint['epoch']} (Val Loss: {checkpoint['val_loss']:.4f})")
    
model.eval()
test_preds, test_trues, test_probs = [], [], []
with torch.no_grad():
    for batch in test_loader:
        x_num = batch['num'].to(device)
        x_cat = batch['cat'].to(device)
        y = batch['y'].to(device)
        logits = model(x_num, x_cat)
        probs = torch.sigmoid(logits).cpu().numpy()
        yhat = (probs >= 0.5).astype(int)
        
        test_preds.extend(yhat.tolist())
        test_trues.extend(y.cpu().numpy().tolist())
        test_probs.extend(probs.tolist())

print(classification_report(test_trues, test_preds, target_names=['Non-Diabetic (0)', 'Diabetic (1)'], digits=4))
print(f'Test ROC-AUC Score: {roc_auc_score(test_trues, test_probs):.4f}')
"""))

# Step 7: Comparison Table
cells.append(nbf.v4.new_markdown_cell("""## Step 13: Comparison Table (Paper Reported vs Reproduced Results)"""))

cells.append(nbf.v4.new_code_cell("""comp_df = pd.read_csv('../outputs/metrics/paper_comparison_table.csv')
comp_df
"""))

nb.cells = cells

target_nb_path = 'diabetes-tabtransformer/notebooks/diabetes_tabtransformer.ipynb'
os.makedirs(os.path.dirname(target_nb_path), exist_ok=True)
with open(target_nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print('Notebook successfully written to:', target_nb_path)
