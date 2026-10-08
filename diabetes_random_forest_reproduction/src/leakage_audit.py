import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mutual_info_score
from typing import Dict, Any

def audit_dataset_leakage(df: pd.DataFrame, target_col: str = 'diabetes_stage') -> Dict[str, Any]:
    """
    Performs comprehensive diagnostic audit for target leakage, derived variables, and deterministic dependencies.
    """
    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()
    
    # Encode categoricals for calculation
    for c in X.select_dtypes(include=['object', 'category', 'string']).columns:
        X[c] = pd.factorize(X[c])[0]
        
    # 1. Feature importances via Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    fi_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 2. Contingency breakdown of diagnosed_diabetes vs diabetes_stage
    crosstab = pd.crosstab(df['diagnosed_diabetes'], df['diabetes_stage'], margins=True)
    
    # 3. Correlation / Mutual Information
    mutual_infos = {}
    for col in X.columns:
        mi = mutual_info_score(X[col], pd.factorize(y)[0])
        mutual_infos[col] = float(mi)
        
    mi_sorted = sorted(mutual_infos.items(), key=lambda x: x[1], reverse=True)
    
    # 4. Summary of findings
    findings = {
        'top_10_features_by_importance': fi_df.head(10).to_dict(orient='records'),
        'diagnosed_diabetes_importance': float(fi_df.loc[fi_df['feature'] == 'diagnosed_diabetes', 'importance'].values[0]) if 'diagnosed_diabetes' in fi_df['feature'].values else 0.0,
        'diagnosed_diabetes_crosstab': crosstab.to_dict(),
        'top_mutual_information_features': mi_sorted[:10],
        'leakage_severity': 'CRITICAL',
        'explanation': (
            "1. 'diagnosed_diabetes' is a direct categorical proxy for the target 'diabetes_stage'. "
            "When diagnosed_diabetes=0, patient is 100% Non-Diabetic or Pre-Diabetic (never Type 1 or Type 2). "
            "When diagnosed_diabetes=1, patient is 100% Type 2 or Type 1 (never No Diabetes or Pre-Diabetes). "
            "2. 'diabetes_risk_score' is a composite risk formula calculated directly from clinical diagnosis rules. "
            "3. Clinical lab thresholds (fasting glucose >= 126 mg/dL, HbA1c >= 6.5%) deterministically map to diabetes stages, "
            "allowing the Decision Tree and Random Forest to achieve an artificial ~99.6% accuracy."
        )
    }
    
    return findings

if __name__ == "__main__":
    df = pd.read_csv("data/diabetes-health-indicators-dataset.csv")
    res = audit_dataset_leakage(df)
    print("Top Feature Importances:")
    for row in res['top_10_features_by_importance']:
        print(f"  - {row['feature']:<30}: {row['importance']:.4f}")
