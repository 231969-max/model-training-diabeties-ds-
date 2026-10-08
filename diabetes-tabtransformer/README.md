# Classification of Health Indicators for Diabetes Mellitus Prediction Using a TabTransformer Model on Clinical Tabular Data

**Paper Reference**: Al Khaidar & Sri Kurnia, *Journal of Computer Science Research (JoCoSiR)*, Vol. 2, No. 2, pp. 9–15. DOI: [10.65126/jocosir.v2i2.52](http://doi.org/10.65126/jocosir.v2i2.52)

---

## 1. Research Objective
This repository contains a full, reproducible implementation of the research paper **"Classification of Health Indicators for Diabetes Mellitus Prediction Using a TabTransformer Model on Clinical Tabular Data"**.

The objective is to establish an artificial intelligence-based clinical decision support model capable of accurately predicting diabetes mellitus status (`diagnosed_diabetes`: `0 = non-diabetic`, `1 = diabetic`) using high-dimensional heterogeneous clinical tabular data. The TabTransformer architecture applies self-attention mechanisms over categorical feature embeddings while fusing normalized continuous indicators into a multi-layer perceptron (MLP) classification network.

---

## 2. Dataset Overview
- **Source**: Kaggle Clinical Tabular Diabetes Dataset.
- **Profiles**: 100,000 unique patient records.
- **Raw Dimension**: 100,000 rows × 31 columns.
- **Target Variable**: `diagnosed_diabetes` (Binary: 0 = Non-Diabetic [40.0%], 1 = Diabetic [60.0%]).
- **Secondary Target**: `diabetes_stage` (Multi-class: Type 2, Pre-Diabetes, No Diabetes, Gestational, Type 1).

---

## 3. Features & Clinical Indicators
The TabTransformer architecture consumes 9 primary clinical indicators:

### Categorical Features (4)
1. `gender` (Male, Female, Other)
2. `smoking_status` (Never, Former, Current)
3. `family_history_diabetes` (0 = No, 1 = Yes)
4. `employment_status` (Employed, Unemployed, Retired, Student)

### Numerical Features (5)
1. `age` (Years)
2. `bmi` (Body Mass Index, kg/m²)
3. `glucose_fasting` (Fasting blood glucose level, mg/dL)
4. `cholesterol_total` (Total serum cholesterol, mg/dL)
5. `hba1c` (Glycated hemoglobin percentage, %)

---

## 4. Preprocessing Pipeline
1. **Leakage Removal**: Explicitly drops `diabetes_risk_score` (calculated diagnostic index) and `diabetes_stage` (disease severity outcome) prior to modeling.
2. **Missing-Value Handling**: Median imputation for numerical attributes, mode imputation for categorical attributes.
3. **Stratified Splitting**: 70% Training (69,999 records), 15% Validation (15,001 records), 15% Testing (15,000 records).
4. **Numerical Standardization**: `StandardScaler` fitted strictly on training data and applied to validation/testing data.
5. **Categorical Integer Encoding**: `LabelEncoder` fitted strictly on training data.

---

## 5. TabTransformer Architecture Pipeline
```
Input CSV (100,000 records)
    ↓
Data Cleaning & Leakage Removal
    ↓
Categorical & Numerical Separation
    ↓
Categorical Embeddings (4 × 32-dim vectors)
    ↓
4-Layer Multi-Head Self-Attention Transformer Blocks (8 heads, GELU, FF=128, Dropout=0.1)
    ↓
Contextual Categorical Representation (Flattened to 128-dim)
    ↓
Concatenation with 5 Normalized Numerical Features (Total = 133-dim vector)
    ↓
2-Layer MLP Classifier [133 → 64 → 32 → 1] (ReLU, Dropout=0.2)
    ↓
Output Logits (BCEWithLogitsLoss / Sigmoid ≥ 0.5)
```

---

## 6. Comprehensive Hyperparameter Fidelity Table

| Hyperparameter | Paper Value | Implementation Value | Status | Evidence / Reference in Paper |
| :--- | :---: | :---: | :---: | :--- |
| **Categorical Embedding Dimension** | `32` | `32` | **EXACT MATCH** | Section 3.2: *"categorical features with a 32-dimensional embedding"* |
| **Transformer Encoder Layers** | `4` | `4` | **EXACT MATCH** | Section 3.2: *"transformer block consists of 4 layers"* |
| **Attention Heads ($n_{\text{head}}$)** | `8` | `8` | **EXACT MATCH** | Section 3.2: *"with 8 attention heads"* |
| **Feedforward Dimension ($d_{\text{ff}}$)** | `128` | `128` | **EXACT MATCH** | Section 3.2: *"a feedforward dimension of 128"* |
| **Transformer Activation** | `GELU` | `GELU` | **EXACT MATCH** | Section 3.2: *"GELU activation"* |
| **Transformer Dropout** | `0.1` | `0.1` | **EXACT MATCH** | Section 3.2: *"and a dropout of 0.1"* |
| **MLP Hidden Dimensions** | `[64, 32]` | `[64, 32]` | **EXACT MATCH** | Section 3.2: *"two hidden layers [64, 32]"* |
| **MLP Activation** | `ReLU` | `ReLU` | **EXACT MATCH** | Section 3.2: *"ReLU activation"* |
| **MLP Dropout** | `0.2` | `0.2` | **EXACT MATCH** | Section 3.2: *"dropout 0.2"* |
| **Loss Function** | `BCEWithLogitsLoss` | `BCEWithLogitsLoss` | **EXACT MATCH** | Section 4.4 & Figure 4: `criterion = nn.BCEWithLogitsLoss()` |
| **Optimizer Type** | `AdamW` | `AdamW` | **EXACT MATCH** | Section 4.4 & Figure 4: `torch.optim.AdamW(...)` |
| **Learning Rate ($\alpha$)** | `1e-3` | `1e-3` | **INFERRED FROM PAPER** | Figure 4 code snippet: `lr=1e-3` |
| **Optimizer Weight Decay** | `1e-5` | `1e-5` | **INFERRED FROM PAPER** | Figure 4 code snippet: `weight_decay=1e-5` |
| **LR Scheduler Type** | `ReduceLROnPlateau` | `ReduceLROnPlateau` | **EXACT MATCH** | Section 4.4 & Figure 4: `torch.optim.lr_scheduler.ReduceLROnPlateau` |
| **Scheduler Factor** | `0.5` | `0.5` | **INFERRED FROM PAPER** | Figure 4 code snippet: `factor=0.5` |
| **Scheduler Patience** | `1` | `1` | **INFERRED FROM PAPER** | Figure 4 code snippet: `patience=1` |
| **Scheduler Mode** | `'min'` | `'min'` | **INFERRED FROM PAPER** | Figure 4 code snippet: `mode='min'` |
| **Classification Threshold** | `0.5` | `0.5` | **EXACT MATCH** | Section 4.4 & Figure 4: `(probs >= 0.5).astype(int)` |
| **Batch Size** | *Not stated* | `256` (Train), `512` (Eval) | **NOT SPECIFIED BY PAPER** | Assumed standard mini-batch size |
| **Number of Epochs** | *Not stated* | `15` | **NOT SPECIFIED BY PAPER** | Assumed sufficient epochs for loss stabilization |
| **Train/Val/Test Split Ratio**| *Not stated* | `70% / 15% / 15%` | **NOT SPECIFIED BY PAPER** | Standard stratified cross-validation partition |
| **Random Seed** | *Not stated* | `42` | **NOT SPECIFIED BY PAPER** | Standard reproducibility seed across PyTorch/NumPy |
| **Weight Initialization** | *Not stated* | PyTorch Default | **NOT SPECIFIED BY PAPER** | Standard PyTorch default initialization |
| **Minimum Learning Rate** | *Not stated* | `0.0` (PyTorch default) | **NOT SPECIFIED BY PAPER** | Default floor in `ReduceLROnPlateau` |
| **Early Stopping** | *Not stated* | Best Val Checkpoint | **NOT SPECIFIED BY PAPER** | Best checkpoint retained on minimum validation loss |

---

## 7. Experimental Results & Paper Comparison

| Metric / Target | Paper Reported Result | Reproduced Result (Test Set) | Difference | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | **82.55%** | **91.93%** | **+9.38%** | Exceeded |
| **Class 0 (Non-Diabetic) Precision** | 0.7716 | **0.8344** | +0.0628 | Exceeded |
| **Class 0 (Non-Diabetic) Recall** | 0.8006 | **0.9960** | +0.1954 | Exceeded |
| **Class 0 (Non-Diabetic) F1-score** | 0.7858 | **0.9081** | +0.1223 | Exceeded |
| **Class 1 (Diabetic) Precision** | 0.8637 | **0.9969** | +0.1332 | Exceeded |
| **Class 1 (Diabetic) Recall** | 0.8420 | **0.8682** | +0.0262 | Exceeded |
| **Class 1 (Diabetic) F1-score** | **0.8527** | **0.9281** | **+0.0754** | Exceeded |
| **ROC-AUC Score** | **0.9009** | **0.9418** | **+0.0409** | Exceeded |

### Test Classification Report
```
                  precision    recall  f1-score   support

Non-Diabetic (0)     0.8344    0.9960    0.9081      6000
    Diabetic (1)     0.9969    0.8682    0.9281      9000

        accuracy                         0.9193     15000
       macro avg     0.9157    0.9321    0.9181     15000
    weighted avg     0.9319    0.9193    0.9201     15000

ROC-AUC Score: 0.9418
```

---

## 8. Exact-Result Discrepancy Investigation
While methodology and architecture are replicated with 100% fidelity, the reproduced numerical metrics show higher performance (91.93% accuracy vs 82.55%). The root causes are:

1. **Number of Training Epochs & Early Stopping (Primary Factor)**:
   - At **Epoch 1**, the validation accuracy is **81.71%** (extremely close to the paper's reported **82.55%**).
   - As training continues through Epochs 5–15 with `ReduceLROnPlateau`, the TabTransformer optimizes further, achieving **91.93%**.
2. **Train/Val/Test Split Ratio (Secondary Factor)**:
   - The paper never specifies its split proportion or cross-validation fold size. A different split alters test class proportions and evaluation metrics.
3. **Feature Selection Nuance in Paper**:
   - Table 2 of the paper lists 26 input features (6 categorical + 20 numerical), while Section 3.2 and Figure 2 specify 9 features (4 categorical + 5 numerical). When tested on all 26 features at early epochs, validation accuracy is ~87.9%–89.4%.
4. **Random Seed & Hardware Initialization**:
   - Random seed, CPU/GPU hardware environment, and weight initialization were omitted in the original publication.

---

## 9. Final Reproduction Test Answers

1. **Does the implementation use every explicitly specified hyperparameter exactly as the paper?**  
   **YES**. Every dimension, head count, layer count, activation function, dropout rate, loss function, optimizer, scheduler, and decision threshold matches the paper.
2. **Does the preprocessing match what can be determined from the paper?**  
   **YES**. Identity columns and data leakage features (`diabetes_risk_score`, `diabetes_stage`) are removed, numericals are scaled via `StandardScaler` fitted on training data, and categoricals are integer-encoded for embeddings.
3. **Does the architecture match the paper?**  
   **YES**. 4 embedding lookup tables ($d=32$) $\to$ 4-layer Transformer Encoder (8 heads, $d_{\text{ff}}=128$, GELU, dropout 0.1) $\to$ contextual categorical representations concatenated with 5 standardized numerical features $\to$ 2-layer MLP (`[64, 32]`, ReLU, dropout 0.2) $\to$ output logits.
4. **Does the training procedure match the paper?**  
   **YES**. `BCEWithLogitsLoss` loss minimization, `AdamW` optimization, and `ReduceLROnPlateau` scheduling responding to validation loss.
5. **Do the results match the paper numerically?**  
   **NO**. The reproduced model achieves higher performance (91.93% accuracy vs. 82.55%, 0.9418 AUC vs. 0.9009 AUC).
6. **What specific undocumented information prevents exact numerical reproduction?**  
   - Exact number of training epochs / early stopping criteria.
   - Exact train/val/test split ratio and cross-validation scheme.
   - Random seed and initialization state.
7. **Is it scientifically honest to call this an "exact reproduction"?**  
   It is scientifically honest to designate this as an **Exact Methodological Reproduction**, but **NOT an Exact Numerical Reproduction**. Faking or forcing the metrics to 82.55% would violate research integrity.

---

## 10. Final Reproduction Verdict

| Evaluation Category | Verdict | Notes |
| :--- | :---: | :--- |
| **Methodology Reproduced** | **YES** | Complete data pipeline, leakage removal, scaling, and embeddings match |
| **Specified Hyperparameters Reproduced** | **YES** | All 18 architectural & optimization hyperparameters match exactly or are faithful inferences |
| **Training Procedure Reproduced** | **YES** | `train_epoch`, `eval_epoch`, `AdamW`, `ReduceLROnPlateau`, `BCEWithLogitsLoss` implemented as published |
| **Numerical Results Reproduced** | **NO** | Reaches 91.93% test accuracy vs 82.55% reported (explained by epoch convergence and split factors) |
| **Exact Numerical Reproduction Possible from Paper** | **NO** | Missing epoch count/stopping criteria, split ratios, and random seed prevent bit-exact numerical replication |

---

## 11. Project Structure
```
diabetes-tabtransformer/
├── data/
│   └── diabetes_dataset.csv
├── src/
│   ├── preprocessing.py             # Leakage removal, encoding & scaling
│   ├── dataset.py                   # PyTorch Dataset & DataLoader
│   ├── model.py                     # TabTransformer architecture
│   ├── train.py                     # Multi-epoch training & validation loop
│   ├── evaluate.py                  # Metrics calculation & plotting functions
│   └── utils.py                     # Seeds, device detection, JSON I/O
├── outputs/
│   ├── figures/
│   │   ├── dataset_distribution.png         # Fig 1 recreation
│   │   ├── tabtransformer_architecture.png # Fig 2 recreation
│   │   ├── training_history.png             # Training/Validation Loss, Acc, F1
│   │   ├── roc_curve.png                    # Fig 3 recreation (AUC = 0.9418)
│   │   └── confusion_matrix.png             # Annotated TN, FP, FN, TP
│   ├── metrics/
│   │   ├── preprocessing_summary_table.csv
│   │   ├── paper_comparison_table.csv
│   │   ├── training_history.json
│   │   └── test_metrics.json
│   └── models/
│       └── best_tabtransformer.pt   # Best checkpoint (Val Loss = 0.2149)
├── notebooks/
│   └── diabetes_tabtransformer.ipynb# Interactive Jupyter Notebook
├── requirements.txt
├── main.py                          # Master experiment runner
└── README.md                        # Documentation
```

---

## 12. How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Master Experiment
```bash
python main.py
```

### 3. Interactive Jupyter Notebook
Open and execute:
```bash
jupyter notebook notebooks/diabetes_tabtransformer.ipynb
```
