"""
=============================================================================
CLINICAL DIABETES PREDICTION DASHBOARD (DUAL MODEL SELECTOR)
=============================================================================
Flask application supporting model selection toggle between:
1. Random Forest Classifier (100 Ensemble Trees - High Precision 99.83%)
2. Logistic Regression (Linear Baseline - High Recall 89.38%)
=============================================================================
"""

import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# -----------------------------------------------------------------------------
# 1. Load Pre-Trained Pipeline Models at Startup
# -----------------------------------------------------------------------------
RF_MODEL_PATH = 'random_forest_diabetes_model.pkl'
LR_MODEL_PATH = 'logistic_regression_diabetes_model.pkl'

if not os.path.exists(RF_MODEL_PATH):
    raise FileNotFoundError(f"Model file '{RF_MODEL_PATH}' not found.")
if not os.path.exists(LR_MODEL_PATH):
    raise FileNotFoundError(f"Model file '{LR_MODEL_PATH}' not found.")

rf_model = joblib.load(RF_MODEL_PATH)
lr_model = joblib.load(LR_MODEL_PATH)
print("Both Random Forest and Logistic Regression pipelines loaded successfully.")

# Feature column definitions
NUMERICAL_COLS = [
    'age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week',
    'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day',
    'family_history_diabetes', 'hypertension_history', 'cardiovascular_history',
    'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
    'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides',
    'glucose_fasting', 'glucose_postprandial', 'insulin_level', 'hba1c',
    'diabetes_risk_score', 'daily_water_intake_liters', 'fruit_vegetable_servings_per_day',
    'annual_health_checkups', 'waist_circumference_cm', 'daily_steps',
    'resting_respiratory_rate', 'vitamin_d_level_ng_ml', 'medication_adherence_score'
]

CATEGORICAL_COLS = [
    'gender', 'ethnicity', 'education_level', 'income_level',
    'employment_status', 'smoking_status', 'stress_level', 'salt_intake_level'
]

CATEGORICAL_OPTIONS = {
    'gender': ['Male', 'Female', 'Other'],
    'ethnicity': ['Asian', 'White', 'Hispanic', 'Black', 'Other'],
    'education_level': ['Highschool', 'Graduate', 'Postgraduate', 'No formal'],
    'income_level': ['Lower-Middle', 'Middle', 'Low', 'Upper-Middle', 'High'],
    'employment_status': ['Employed', 'Unemployed', 'Retired', 'Student'],
    'smoking_status': ['Never', 'Former', 'Current'],
    'stress_level': ['Low', 'Moderate', 'High'],
    'salt_intake_level': ['Low', 'Moderate', 'High']
}

DEFAULT_FORM_VALUES = {
    'age': 50,
    'gender': 'Male',
    'ethnicity': 'Asian',
    'education_level': 'Graduate',
    'income_level': 'Middle',
    'employment_status': 'Employed',
    'smoking_status': 'Never',
    'alcohol_consumption_per_week': 2,
    'physical_activity_minutes_per_week': 120,
    'diet_score': 6.0,
    'sleep_hours_per_day': 7.0,
    'screen_time_hours_per_day': 6.0,
    'family_history_diabetes': 0,
    'hypertension_history': 0,
    'cardiovascular_history': 0,
    'bmi': 25.5,
    'waist_to_hip_ratio': 0.85,
    'systolic_bp': 118,
    'diastolic_bp': 76,
    'heart_rate': 72,
    'cholesterol_total': 185,
    'hdl_cholesterol': 55,
    'ldl_cholesterol': 100,
    'triglycerides': 120,
    'glucose_fasting': 105,
    'glucose_postprandial': 140,
    'insulin_level': 8.5,
    'hba1c': 6.1,
    'diabetes_risk_score': 28.0,
    'daily_water_intake_liters': 2.3,
    'stress_level': 'Moderate',
    'fruit_vegetable_servings_per_day': 3,
    'annual_health_checkups': 1,
    'waist_circumference_cm': 88.0,
    'daily_steps': 6800,
    'resting_respiratory_rate': 16,
    'vitamin_d_level_ng_ml': 28.5,
    'salt_intake_level': 'Moderate',
    'medication_adherence_score': 7.5
}

# Empirical evaluation metrics for both models on 40,000 test cases
MODEL_SPECS = {
    'random_forest': {
        'id': 'random_forest',
        'name': 'Random Forest Classifier',
        'badge': '100 Decision Trees (Ensemble)',
        'type': 'Non-Linear Ensemble',
        'accuracy': '91.97%',
        'precision': '99.83%',
        'recall': '86.77%',
        'f1_score': '92.84%',
        'roc_auc': '0.9426',
        'strength': 'Exceptional Precision (99.83%) & minimal false alarms (only 36 FP in 40k cases)',
        'tn': '15,965',
        'fp': '36',
        'fn': '3,175',
        'tp': '20,824'
    },
    'logistic_regression': {
        'id': 'logistic_regression',
        'name': 'Logistic Regression',
        'badge': 'Linear Probability Model',
        'type': 'Linear Baseline',
        'accuracy': '86.00%',
        'precision': '87.55%',
        'recall': '89.38%',
        'f1_score': '88.43%',
        'roc_auc': '0.9341',
        'strength': 'Higher Recall / Sensitivity (89.38%) for broader population screening',
        'tn': '12,951',
        'fp': '3,050',
        'fn': '2,549',
        'tp': '21,450'
    }
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    form_values = DEFAULT_FORM_VALUES.copy()
    selected_model_key = 'random_forest'
    
    if request.method == 'POST':
        selected_model_key = request.form.get('model_choice', 'random_forest')
        if selected_model_key not in MODEL_SPECS:
            selected_model_key = 'random_forest'
            
        active_model = rf_model if selected_model_key == 'random_forest' else lr_model
        
        try:
            # 1. Collect and cast form inputs
            input_dict = {}
            for col in NUMERICAL_COLS:
                raw_val = request.form.get(col, '')
                if '.' in str(raw_val):
                    input_dict[col] = float(raw_val)
                else:
                    input_dict[col] = int(raw_val) if raw_val.isdigit() or (raw_val.startswith('-') and raw_val[1:].isdigit()) else float(raw_val)
                form_values[col] = raw_val
                
            for col in CATEGORICAL_COLS:
                val = request.form.get(col, CATEGORICAL_OPTIONS[col][0])
                input_dict[col] = str(val)
                form_values[col] = val
                
            # 2. Build input DataFrame
            input_df = pd.DataFrame([input_dict])
            
            # 3. Generate prediction with active selected model
            pred_class = int(active_model.predict(input_df)[0])
            pred_proba = active_model.predict_proba(input_df)[0]
            
            prob_0 = pred_proba[0] * 100
            prob_1 = pred_proba[1] * 100
            
            status_message = "Model prediction: diabetes detected." if pred_class == 1 else "Model prediction: diabetes not detected."
            status_color = "danger" if pred_class == 1 else "success"
            
            # Also calculate shadow prediction from other model for comparative clinical insight
            other_model = lr_model if selected_model_key == 'random_forest' else rf_model
            other_class = int(other_model.predict(input_df)[0])
            other_proba = other_model.predict_proba(input_df)[0]
            other_model_name = "Logistic Regression" if selected_model_key == 'random_forest' else "Random Forest"
            
            prediction_result = {
                'selected_model_name': MODEL_SPECS[selected_model_key]['name'],
                'selected_model_badge': MODEL_SPECS[selected_model_key]['badge'],
                'predicted_class': pred_class,
                'class_label': 'Diabetic (Class 1)' if pred_class == 1 else 'Healthy / Non-Diabetic (Class 0)',
                'prob_0': f"{prob_0:.2f}",
                'prob_1': f"{prob_1:.2f}",
                'status_message': status_message,
                'status_color': status_color,
                'other_model_name': other_model_name,
                'other_class': other_class,
                'other_prob_1': f"{other_proba[1]*100:.2f}"
            }
            
        except Exception as e:
            prediction_result = {
                'error': f"Error processing input values: {str(e)}"
            }

    return render_template(
        'index.html',
        options=CATEGORICAL_OPTIONS,
        form_values=form_values,
        result=prediction_result,
        models=MODEL_SPECS,
        selected_model=selected_model_key
    )

if __name__ == '__main__':
    print("Starting Diabetes Risk Prediction Dual-Model Flask App on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)
