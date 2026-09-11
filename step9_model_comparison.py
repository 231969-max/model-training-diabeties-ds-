"""
=============================================================================
Step 9: Model Comparison — Logistic Regression vs. Random Forest
=============================================================================

Viva & Lab Concepts (Explained in Simple Terms):

1. WHY DO WE COMPARE MODELS?
   No single Machine Learning model is universally superior for every problem.
   Comparing our linear baseline (Logistic Regression) with an ensemble model 
   (Random Forest) allows us to evaluate whether non-linear modeling delivers 
   measurable performance gains and understand the exact clinical trade-offs.

2. WHAT ARE THE KEY TRADE-OFFS IN MEDICAL CLASSIFICATION?
   - High Precision (Random Forest: 99.85%):
     Virtually eliminates False Alarms (only 31 false positives out of 16,001 healthy patients).
     Patients are not subjected to unnecessary anxiety or further invasive testing.
   - High Recall (Logistic Regression: 89.38%):
     Minimizes False Negatives (2,549 missed cases vs. 3,174 for Random Forest).
     In healthcare, missing a sick patient (False Negative) can delay life-saving treatment.
=============================================================================
"""

# -----------------------------------------------------------------------------
# 1. Evaluation Results from Steps 6 and 8 (Exact Measured Values)
# -----------------------------------------------------------------------------
# Logistic Regression (Step 6)
lr_acc = 0.8600
lr_prec = 0.8755
lr_rec = 0.8938
lr_cm = [[12951, 3050], [2549, 21450]]
lr_tn, lr_fp, lr_fn, lr_tp = 12951, 3050, 2549, 21450

# Random Forest (Step 8)
rf_acc = 0.9199
rf_prec = 0.9985
rf_rec = 0.8677
rf_cm = [[15970, 31], [3174, 20825]]
rf_tn, rf_fp, rf_fn, rf_tp = 15970, 31, 3174, 20825

# -----------------------------------------------------------------------------
# 2. Compute Metric Differences (Random Forest - Logistic Regression)
# -----------------------------------------------------------------------------
acc_diff = (rf_acc - lr_acc) * 100
prec_diff = (rf_prec - lr_prec) * 100
rec_diff = (rf_rec - lr_rec) * 100

# -----------------------------------------------------------------------------
# 3. Print Comparison Table
# -----------------------------------------------------------------------------
print("=== STEP 9: MODEL COMPARISON ===")
print()
print(f"{'Metric':<19} {'Logistic Regression':<22} {'Random Forest':<15}")
print("-" * 57)
print(f"{'Accuracy':<19} {lr_acc*100:>6.2f}%                 {rf_acc*100:>6.2f}%")
print(f"{'Precision':<19} {lr_prec*100:>6.2f}%                 {rf_prec*100:>6.2f}%")
print(f"{'Recall':<19} {lr_rec*100:>6.2f}%                 {rf_rec*100:>6.2f}%")
print()

# -----------------------------------------------------------------------------
# 4. Print Differences
# -----------------------------------------------------------------------------
print("Metric Differences (Random Forest vs. Logistic Regression):")
print(f"Accuracy:  Random Forest {acc_diff:+.2f} percentage points")
print(f"Precision: Random Forest {prec_diff:+.2f} percentage points")
print(f"Recall:    Random Forest {rec_diff:+.2f} percentage points")
print()

# -----------------------------------------------------------------------------
# 5. Compare Confusion Matrices
# -----------------------------------------------------------------------------
print("Confusion Matrices Comparison:")
print()
print("Logistic Regression:")
print(f"[[{lr_tn:5d}  {lr_fp:4d}]")
print(f" [ {lr_fn:4d} {lr_tp:5d}]]")
print(f"  TN = {lr_tn:,} | FP = {lr_fp:,} | FN = {lr_fn:,} | TP = {lr_tp:,}")
print()
print("Random Forest:")
print(f"[[{rf_tn:5d}    {rf_fp:2d}]")
print(f" [ {rf_fn:4d} {rf_tp:5d}]]")
print(f"  TN = {rf_tn:,} | FP = {rf_fp:,}    | FN = {rf_fn:,} | TP = {rf_tp:,}")
print()

# -----------------------------------------------------------------------------
# 6. Detailed Trade-off and Clinical Analysis
# -----------------------------------------------------------------------------
print("Analysis of Trade-offs:")
print(f"- Accuracy:        Random Forest is higher by {acc_diff:+.2f} percentage points ({rf_acc*100:.2f}% vs {lr_acc*100:.2f}%).")
print(f"- Precision:       Random Forest is higher by {prec_diff:+.2f} percentage points ({rf_prec*100:.2f}% vs {lr_prec*100:.2f}%).")
print(f"- False Positives: Random Forest dramatically reduces false alarms (only {rf_fp} vs {lr_fp:,}).")
print(f"- Recall:          Logistic Regression is higher by {abs(rec_diff):.2f} percentage points ({lr_rec*100:.2f}% vs {rf_rec*100:.2f}%).")
print(f"- False Negatives: Random Forest has more missed diagnoses ({rf_fn:,} vs {lr_fn:,}).")
print()
print("Clinical Significance:")
print("In diabetes classification, Recall is critical because False Negatives represent")
print("patients with undiagnosed diabetes who may miss vital medical intervention.")
print("Conversely, Precision is critical to avoid unnecessary treatments and medical costs.")
print()

# -----------------------------------------------------------------------------
# 7. Final Model-Selection Conclusion
# -----------------------------------------------------------------------------
print("Final Model Selection Conclusion:")
conclusion = (
    "\"Based on the evaluation results, Random Forest is the stronger overall model "
    "because it achieves higher accuracy and precision and produces dramatically fewer "
    "false positives. However, Logistic Regression has slightly higher recall, meaning "
    "it detects a slightly larger proportion of actual diabetes cases. Therefore, Random Forest "
    "is the better overall-performing model based on the measured metrics, while Logistic Regression "
    "may be preferable if maximizing recall is the primary priority.\""
)
print(conclusion)
print()

# -----------------------------------------------------------------------------
# 8. Viva-Friendly Summary
# -----------------------------------------------------------------------------
print("VIVA ANSWER:")
print()
print("Which model performed better?")
print()
print("Random Forest performed better overall because:")
print(f"1. Its accuracy was {rf_acc*100:.2f}% compared with {lr_acc*100:.2f}%.")
print(f"2. Its precision was {rf_prec*100:.2f}% compared with {lr_prec*100:.2f}%.")
print(f"3. It produced only {rf_fp} false positives compared with {lr_fp:,}.")
print(f"4. Logistic Regression had slightly better recall:")
print(f"   {lr_rec*100:.2f}% compared with Random Forest's {rf_rec*100:.2f}%.")
print()
print("Therefore, Random Forest is the stronger overall model,")
print("while Logistic Regression has the advantage in recall.")
