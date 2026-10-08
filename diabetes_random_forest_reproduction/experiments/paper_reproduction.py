import os
import sys
import json
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.load_data import load_dataset, print_dataset_summary
from src.preprocessing import preprocess_faithful
from src.models import get_random_forest_model, get_decision_tree_model, get_constant_model
from src.evaluation import evaluate_model_cv, generate_paper_comparison_table, CLASS_ORDER
from src.visualization import (
    plot_class_distribution, plot_confusion_matrix_custom,
    plot_multiclass_roc, plot_performance_lift_curve, plot_prediction_examples
)

def run_faithful_reproduction():
    print("=" * 80)
    print(" EXPERIMENT A: FAITHFUL RESEARCH PAPER REPRODUCTION")
    print(" Paper: 'Klasifikasi Indikator Kesehatan Diabetes Menggunakan Algoritma Random Forest'")
    print(" Platform: Orange Data Mining Workflow Simulation (Python / Scikit-Learn)")
    print("=" * 80)
    
    # 1. Load Data
    data_path = "data/diabetes-health-indicators-dataset.csv"
    df, meta = load_dataset(data_path)
    print_dataset_summary(meta)
    
    # 2. Visualizations: Class Distribution
    plot_class_distribution(df['diabetes_stage'], "results/figures/class_distribution.png")
    
    # 3. Preprocess Faithful (Keep all 30 features as configured in Orange Columns widget)
    X, y, feature_names = preprocess_faithful(df)
    print(f"\n[Preprocessing] Retained all {len(feature_names)} features (including 'diagnosed_diabetes' & 'diabetes_risk_score').")
    
    # 4. Initialize Models
    models = {
        'Random Forest': get_random_forest_model(n_estimators=100, random_state=42),
        'Decision Tree': get_decision_tree_model(random_state=42),
        'Constant': get_constant_model()
    }
    
    # 5. Evaluate all models via 5-Fold Stratified Cross-Validation (Test & Score widget)
    results = {}
    preds_dict = {}
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name} via 5-Fold Cross-Validation...")
        res = evaluate_model_cv(model, X, y, n_splits=5, random_state=42)
        results[model_name] = res
        preds_dict[model_name] = res['predictions']
        
        print(f"[{model_name}] -> CA: {res['CA']:.4f} | Prec: {res['Precision_weighted']:.4f} | Rec: {res['Recall_weighted']:.4f} | F1: {res['F1_weighted']:.4f} | MCC: {res['MCC']:.4f}")
        
        # Save Confusion Matrix figure & CSV
        cm_array = np.array(res['confusion_matrix'])
        cm_df = pd.DataFrame(cm_array, index=CLASS_ORDER, columns=CLASS_ORDER)
        cm_df.to_csv(f"results/confusion_matrices/{model_name.lower().replace(' ', '_')}_cm.csv")
        plot_confusion_matrix_custom(cm_array, model_name, f"results/figures/{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
        
        # Save Multiclass ROC Analysis
        y_prob = np.array(res['probabilities']) if res['probabilities'] is not None else None
        plot_multiclass_roc(y, y_prob, model_name, f"results/figures/{model_name.lower().replace(' ', '_')}_roc_curve.png")
        
        # Save Performance / Lift Curve
        plot_performance_lift_curve(y, y_prob, model_name, 'Gestational', f"results/figures/{model_name.lower().replace(' ', '_')}_performance_curve.png")
        
    # 6. Save Predictions Sample
    plot_prediction_examples(df, preds_dict, "results/figures/prediction_examples.png")
    
    # 7. Comparison Table vs Paper Reported Results
    comp_df = generate_paper_comparison_table(results)
    comp_df.to_csv("results/metrics/paper_comparison_table.csv", index=False)
    
    print("\n" + "=" * 80)
    print(" PAPER REPORTED VS REPRODUCED COMPARISON TABLE (EXPERIMENT A)")
    print("=" * 80)
    print(comp_df.to_string(index=False))
    
    # 8. Save Complete Metrics JSON
    with open("results/metrics/faithful_reproduction_metrics.json", "w") as f:
        # Save summary without large lists
        save_metrics = {
            m: {k: v for k, v in res.items() if k not in ['predictions', 'probabilities']}
            for m, res in results.items()
        }
        json.dump(save_metrics, f, indent=4)
        
    print("\n[Done] All Experiment A outputs saved to results/")
    return results, comp_df

if __name__ == "__main__":
    run_faithful_reproduction()
