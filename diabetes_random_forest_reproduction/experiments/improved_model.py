import os
import sys
import json
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.load_data import load_dataset
from src.preprocessing import preprocess_leakage_aware
from src.models import get_random_forest_model
from src.evaluation import evaluate_model_cv, CLASS_ORDER
from src.leakage_audit import audit_dataset_leakage
from src.visualization import plot_confusion_matrix_custom, plot_multiclass_roc

def run_improved_experiment():
    print("=" * 80)
    print(" EXPERIMENT B: LEAKAGE-AWARE & METHODOLOGICALLY SOUND EXPERIMENT")
    print(" Goal: Evaluate true predictive capability after removing target-derived leakage features")
    print("=" * 80)
    
    # 1. Load Data
    data_path = "data/diabetes-health-indicators-dataset.csv"
    df, _ = load_dataset(data_path)
    
    # 2. Run Diagnostic Leakage Audit
    print("\n--- RUNNING DATA LEAKAGE AUDIT ---")
    audit_results = audit_dataset_leakage(df)
    with open("results/metrics/leakage_audit_summary.json", "w") as f:
        json.dump(audit_results, f, indent=4)
        
    print(f"Top Leakage Feature: 'diagnosed_diabetes' (Feature Importance = {audit_results['diagnosed_diabetes_importance']:.4f})")
    print("Leakage Diagnosis:", audit_results['explanation'])
    
    # 3. Preprocessing: Drop 'diagnosed_diabetes' and 'diabetes_risk_score'
    X, y, feature_names = preprocess_leakage_aware(df)
    print(f"\n[Preprocessing] Cleaned feature set: {len(feature_names)} genuine predictors (dropped leakage columns).")
    
    # 4. Initialize Models: Standard RF vs Balanced RF
    models = {
        'Random Forest (Leakage-Free)': get_random_forest_model(n_estimators=100, class_weight=None, random_state=42),
        'Balanced Random Forest': get_random_forest_model(n_estimators=100, class_weight='balanced', random_state=42)
    }
    
    # 5. Cross-Validation Evaluation
    results = {}
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name} via 5-Fold Stratified CV...")
        res = evaluate_model_cv(model, X, y, n_splits=5, random_state=42)
        results[model_name] = res
        
        print(f"[{model_name}] -> CA: {res['CA']:.4f} | Macro F1: {res['F1_macro']:.4f} | Weighted F1: {res['F1_weighted']:.4f} | MCC: {res['MCC']:.4f} | OvR AUC: {res['AUC_ovr_weighted']:.4f}")
        
        # Save Confusion Matrix
        cm_array = np.array(res['confusion_matrix'])
        cm_df = pd.DataFrame(cm_array, index=CLASS_ORDER, columns=CLASS_ORDER)
        cm_df.to_csv(f"results/confusion_matrices/{model_name.lower().replace(' ', '_').replace('-', '_')}_cm.csv")
        plot_confusion_matrix_custom(cm_array, model_name, f"results/figures/{model_name.lower().replace(' ', '_').replace('-', '_')}_confusion_matrix.png")
        
        # Save Multiclass ROC Analysis
        y_prob = np.array(res['probabilities']) if res['probabilities'] is not None else None
        plot_multiclass_roc(y, y_prob, model_name, f"results/figures/{model_name.lower().replace(' ', '_').replace('-', '_')}_roc_curve.png")
        
    # 6. Build Comparison Table between Experiment A (Original Paper) and Experiment B (Leakage-Free)
    comparison_rows = []
    
    # Load faithful results if available
    faithful_metrics_path = "results/metrics/faithful_reproduction_metrics.json"
    if os.path.exists(faithful_metrics_path):
        with open(faithful_metrics_path, "r") as f:
            faithful_dict = json.load(f)
        if 'Random Forest' in faithful_dict:
            rf_orig = faithful_dict['Random Forest']
            comparison_rows.append({
                'Experiment': 'Experiment A (Faithful Reproduction)',
                'Model': 'Random Forest (with Leakage)',
                'CA (Accuracy)': f"{rf_orig['CA']:.4f}",
                'Weighted F1': f"{rf_orig['F1_weighted']:.4f}",
                'Macro F1': f"{rf_orig['F1_macro']:.4f}",
                'Weighted Prec': f"{rf_orig['Precision_weighted']:.4f}",
                'Weighted Recall': f"{rf_orig['Recall_weighted']:.4f}",
                'MCC': f"{rf_orig['MCC']:.4f}",
                'OvR AUC': f"{rf_orig['AUC_ovr_weighted']:.4f}"
            })
            
    for m_name, res in results.items():
        comparison_rows.append({
            'Experiment': 'Experiment B (Improved / Leakage-Free)',
            'Model': m_name,
            'CA (Accuracy)': f"{res['CA']:.4f}",
            'Weighted F1': f"{res['F1_weighted']:.4f}",
            'Macro F1': f"{res['F1_macro']:.4f}",
            'Weighted Prec': f"{res['Precision_weighted']:.4f}",
            'Weighted Recall': f"{res['Recall_weighted']:.4f}",
            'MCC': f"{res['MCC']:.4f}",
            'OvR AUC': f"{res['AUC_ovr_weighted']:.4f}"
        })
        
    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv("results/metrics/experiment_a_vs_b_comparison.csv", index=False)
    
    print("\n" + "=" * 80)
    print(" EXPERIMENT A (ORIGINAL) VS EXPERIMENT B (CORRECTED) COMPARISON")
    print("=" * 80)
    print(comp_df.to_string(index=False))
    
    with open("results/metrics/improved_model_metrics.json", "w") as f:
        save_metrics = {
            m: {k: v for k, v in res.items() if k not in ['predictions', 'probabilities']}
            for m, res in results.items()
        }
        json.dump(save_metrics, f, indent=4)
        
    print("\n[Done] All Experiment B outputs saved to results/")
    return results, comp_df

if __name__ == "__main__":
    run_improved_experiment()
