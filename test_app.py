"""
Automated unit test for Flask web application with Dual Model Selection
"""
import app as flask_app

client = flask_app.app.test_client()

print("1. Testing GET / route...")
response_get = client.get('/')
assert response_get.status_code == 200
assert b'Diabetes Risk Prediction' in response_get.data
assert b'Random Forest Classifier' in response_get.data
assert b'Logistic Regression' in response_get.data
print("GET / passed successfully.")

print("2. Testing POST / with Random Forest (Healthy Profile)...")
rf_payload = flask_app.DEFAULT_FORM_VALUES.copy()
rf_payload['model_choice'] = 'random_forest'
response_rf = client.post('/', data=rf_payload)
assert response_rf.status_code == 200
assert b'Prediction Result' in response_rf.data
assert b'Random Forest Classifier' in response_rf.data
print("POST / (Random Forest) passed successfully.")

print("3. Testing POST / with Logistic Regression (Diabetic Profile)...")
lr_payload = flask_app.DEFAULT_FORM_VALUES.copy()
lr_payload.update({
    'model_choice': 'logistic_regression',
    'age': 62,
    'glucose_fasting': 155,
    'glucose_postprandial': 230,
    'hba1c': 8.8,
    'insulin_level': 18.0,
    'bmi': 34.2,
    'waist_circumference_cm': 108.0,
    'family_history_diabetes': 1,
    'hypertension_history': 1,
    'diabetes_risk_score': 55.0
})
response_lr = client.post('/', data=lr_payload)
assert response_lr.status_code == 200
assert b'Prediction Result' in response_lr.data
assert b'Logistic Regression' in response_lr.data
print("POST / (Logistic Regression) passed successfully.")

print("4. Testing GET /api/health...")
res_health = client.get('/api/health')
assert res_health.status_code == 200
assert res_health.json['status'] == 'healthy'
print("GET /api/health passed successfully.")

print("5. Testing GET /api/models...")
res_models = client.get('/api/models')
assert res_models.status_code == 200
assert 'random_forest' in res_models.json['models']
assert 'logistic_regression' in res_models.json['models']
print("GET /api/models passed successfully.")

print("6. Testing POST /api/predict with JSON payload (Random Forest)...")
res_api_pred = client.post('/api/predict', json={
    **flask_app.DEFAULT_FORM_VALUES,
    'model_choice': 'random_forest'
})
assert res_api_pred.status_code == 200
assert res_api_pred.json['success'] is True
assert 'prediction' in res_api_pred.json
assert 'probability_diabetic' in res_api_pred.json
print("POST /api/predict passed successfully.")

print("\nALL FLASK APP & API TESTS PASSED!")
