import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# Title & Abstract
cells.append(nbf.v4.new_markdown_cell("""# Research Paper Reproduction & Leakage Audit
**Paper Title**: *Klasifikasi Indikator Kesehatan Diabetes Menggunakan Algoritma Random Forest*  
**Authors**: Haura Syahla, Haris Izzudin, Fariz Aditya Pratama, Beni Rahmatullah, Ahmad Jurnaidi Wahidin, Ika Kurniawati  
**Journal**: Jurnal Teknik Informatika dan Teknologi Informasi (JUTITI), Vol. 5, No. 3, Des 2025, pp. 401–416  
**DOI**: [10.55606/jutiti.v5i3.6338](https://doi.org/10.55606/jutiti.v5i3.6338)

---
### Objective
1. **Experiment A (Faithful Reproduction)**: Reproduce the exact Orange Data Mining workflow with Random Forest, Decision Tree, and Constant Model on target `diabetes_stage`.
2. **Reproducibility & Leakage Audit**: Mathematically audit why the paper achieved ~99.6% accuracy.
3. **Experiment B (Leakage-Aware Model)**: Retrain on a clean, leakage-free feature set.
"""))

# Imports & Data Loading
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

df = pd.read_csv('../data/diabetes-health-indicators-dataset.csv')
print(f"Dataset Shape: {df.shape}")
print("\\nTarget 'diabetes_stage' distribution:")
display(df['diabetes_stage'].value_counts())
"""))

# Experiment A Metrics
cells.append(nbf.v4.new_markdown_cell("""## 1. Experiment A: Paper Reported vs Reproduced Results"""))
cells.append(nbf.v4.new_code_cell("""comp_a_df = pd.read_csv('../results/metrics/paper_comparison_table.csv')
display(comp_a_df)
"""))

# Leakage Audit
cells.append(nbf.v4.new_markdown_cell("""## 2. Leakage Audit: Why Did Random Forest Reach 99.6% Accuracy?
We inspect the contingency cross-tabulation of `diagnosed_diabetes` vs `diabetes_stage` and feature importances.
"""))
cells.append(nbf.v4.new_code_cell("""# Crosstab showing diagnosed_diabetes perfectly segregates target classes
ct = pd.crosstab(df['diagnosed_diabetes'], df['diabetes_stage'], margins=True)
display(ct)
"""))

# Experiment B Comparison
cells.append(nbf.v4.new_markdown_cell("""## 3. Experiment B: Clean Leakage-Free Results vs Original Paper"""))
cells.append(nbf.v4.new_code_cell("""comp_b_df = pd.read_csv('../results/metrics/experiment_a_vs_b_comparison.csv')
display(comp_b_df)
"""))

nb.cells = cells

nb_path = 'diabetes_random_forest_reproduction/notebooks/reproduction_analysis.ipynb'
with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print('Notebook successfully written to:', nb_path)
