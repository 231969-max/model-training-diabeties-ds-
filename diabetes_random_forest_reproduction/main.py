import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.paper_reproduction import run_faithful_reproduction
from experiments.improved_model import run_improved_experiment

def main():
    print("=" * 85)
    print(" JUTITI RESEARCH REPRODUCTION & AUDIT PIPELINE")
    print(" Paper: 'Klasifikasi Indikator Kesehatan Diabetes Menggunakan Algoritma Random Forest'")
    print(" Authors: Syahla et al. (JUTITI, Vol 5 No 3, Dec 2025)")
    print("=" * 85)
    
    # 1. Run Faithful Reproduction (Experiment A)
    print("\n>>> STAGE 1: EXECUTING FAITHFUL EXPERIMENT REPRODUCTION (EXPERIMENT A)...")
    faithful_results, faithful_comp_df = run_faithful_reproduction()
    
    # 2. Run Improved / Leakage-Aware Reproduction (Experiment B)
    print("\n>>> STAGE 2: EXECUTING LEAKAGE AUDIT & CORRECTED EXPERIMENT (EXPERIMENT B)...")
    improved_results, improved_comp_df = run_improved_experiment()
    
    print("\n" + "=" * 85)
    print(" ALL EXPERIMENTAL STAGES COMPLETED SUCCESSFULLY!")
    print(" - Figures saved in: results/figures/")
    print(" - Metrics saved in: results/metrics/")
    print(" - Confusion Matrices saved in: results/confusion_matrices/")
    print("=" * 85)

if __name__ == "__main__":
    main()
