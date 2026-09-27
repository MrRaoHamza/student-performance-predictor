"""
Student Performance Prediction — Flask API
==========================================
Routes:
  GET  /              → main UI
  POST /predict       → returns prediction + probability + insights
  GET  /model-stats   → returns model metrics + feature importances
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ─────────────────────────────────────────
# Auto-train if model files are missing
# (happens on first Render deploy)
# ─────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, 'models')

if not os.path.exists(os.path.join(MODELS_DIR, 'rf_model.pkl')):
    print("Model not found — running train_model.py ...")
    import subprocess, sys
    subprocess.run(
        [sys.executable, os.path.join(BASE_DIR, 'train_model.py')],
        check=True
    )
    print("Training complete.")

# ─────────────────────────────────────────
# Load model artifacts
# ─────────────────────────────────────────
model          = joblib.load(os.path.join(MODELS_DIR, 'rf_model.pkl'))
label_encoders = joblib.load(os.path.join(MODELS_DIR, 'label_encoders.pkl'))

with open(os.path.join(MODELS_DIR, 'metadata.json')) as f:
    metadata = json.load(f)

FEATURES         = metadata['features']
CATEGORICAL_COLS = metadata['categorical_cols']
NUMERIC_COLS     = metadata['numeric_cols']
ENCODER_MAPS     = metadata['encoder_maps']


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def engineer_features(d: dict) -> dict:
    """Add engineered features to the raw input dict."""
    famsup_val  = 1 if d.get('famsup')   == 'yes' else 0
    schoolsup_v = 1 if d.get('schoolsup') == 'yes' else 0
    higher_v    = 1 if d.get('higher')   == 'yes' else 0
    romantic_v  = 1 if d.get('romantic') == 'yes' else 0

    d['study_freetime_ratio']  = float(d['studytime']) / (float(d['freetime']) + 1)
    d['parental_support_score']= float(d['Medu']) + float(d['Fedu']) + famsup_val + schoolsup_v
    d['alcohol_index']         = (float(d['Dalc']) * 2 + float(d['Walc'])) / 3
    d['social_risk']           = float(d['goout']) + romantic_v * 2
    d['prior_failure']         = 1 if int(d['failures']) > 0 else 0
    d['high_aspiration']       = higher_v
    return d


def encode_input(d: dict) -> pd.DataFrame:
    """Encode categorical fields and return a one-row DataFrame."""
    row = {}
    for feat in FEATURES:
        val = d.get(feat)
        if val is None:
            raise ValueError(f"Missing feature: {feat}")
        if feat in CATEGORICAL_COLS:
            le = label_encoders[feat]
            if str(val) not in le.classes_:
                raise ValueError(f"Unknown value '{val}' for feature '{feat}'. "
                                 f"Valid: {list(le.classes_)}")
            row[feat] = le.transform([str(val)])[0]
        else:
            row[feat] = float(val)
    return pd.DataFrame([row])[FEATURES]


def build_insights(d: dict, probability: float) -> list:
    """
    Generate plain-English risk/strength factors based on input values.
    Returns a list of dicts: {type: 'risk'|'strength', text: str}
    """
    insights = []
    failures = int(d.get('failures', 0))
    studytime = int(d.get('studytime', 1))
    absences  = int(d.get('absences', 0))
    G1 = float(d.get('G1', 0))
    G2 = float(d.get('G2', 0))
    Medu = int(d.get('Medu', 0))
    Fedu = int(d.get('Fedu', 0))
    higher = d.get('higher', 'no')
    internet = d.get('internet', 'no')
    Dalc = int(d.get('Dalc', 1))
    Walc = int(d.get('Walc', 1))
    goout = int(d.get('goout', 1))
    schoolsup = d.get('schoolsup', 'no')
    famsup = d.get('famsup', 'no')

    # ── Risk factors
    if failures >= 2:
        insights.append({'type': 'risk', 'text': f'{failures} past course failure(s) — strong negative predictor'})
    elif failures == 1:
        insights.append({'type': 'risk', 'text': '1 past failure — increases risk of not passing'})

    if G1 < 10 or G2 < 10:
        insights.append({'type': 'risk', 'text': f'Low term grades (G1={int(G1)}, G2={int(G2)}) — grade trend is critical'})

    if absences > 15:
        insights.append({'type': 'risk', 'text': f'High absences ({absences} days) — significantly lowers pass rate'})
    elif absences > 8:
        insights.append({'type': 'risk', 'text': f'Above-average absences ({absences} days) — worth monitoring'})

    if studytime == 1:
        insights.append({'type': 'risk', 'text': 'Very low study time (< 2 hrs/week) — highest risk factor'})

    if Dalc >= 4 or Walc >= 4:
        insights.append({'type': 'risk', 'text': 'High alcohol consumption — correlated with lower performance'})

    if goout >= 4 and studytime <= 2:
        insights.append({'type': 'risk', 'text': 'High social activity combined with low study time'})

    if higher == 'no':
        insights.append({'type': 'risk', 'text': 'No plans for higher education — lower academic motivation'})

    # ── Strength factors
    if G1 >= 14 and G2 >= 14:
        insights.append({'type': 'strength', 'text': f'Strong term grades (G1={int(G1)}, G2={int(G2)}) — best predictor of passing'})
    elif G1 >= 10 and G2 >= 10:
        insights.append({'type': 'strength', 'text': f'Passing term grades (G1={int(G1)}, G2={int(G2)}) — on track'})

    if studytime >= 3:
        insights.append({'type': 'strength', 'text': 'Good study habits (3–4+ hrs/week) — positive factor'})

    if failures == 0:
        insights.append({'type': 'strength', 'text': 'No prior course failures — clean academic record'})

    if Medu >= 3 or Fedu >= 3:
        insights.append({'type': 'strength', 'text': 'Educated parents — strong support environment'})

    if famsup == 'yes' or schoolsup == 'yes':
        insights.append({'type': 'strength', 'text': 'Has extra educational support (family or school)'})

    if internet == 'yes':
        insights.append({'type': 'strength', 'text': 'Internet access at home — better study resources'})

    if higher == 'yes':
        insights.append({'type': 'strength', 'text': 'Plans to pursue higher education — strong academic motivation'})

    if absences == 0:
        insights.append({'type': 'strength', 'text': 'Perfect attendance — no absences'})

    # Limit to top 5 most relevant
    risks = [i for i in insights if i['type'] == 'risk'][:3]
    strengths = [i for i in insights if i['type'] == 'strength'][:3]
    return risks + strengths


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', encoder_maps=ENCODER_MAPS)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Engineer features — wrap in try so KeyErrors become 422, not 500
        try:
            data = engineer_features(data)
        except (KeyError, TypeError, ValueError) as e:
            return jsonify({'error': f'Missing or invalid field: {e}'}), 422

        # Encode & build DataFrame
        X = encode_input(data)

        # Predict
        prob        = float(model.predict_proba(X)[0][1])   # P(pass)
        prediction  = int(prob >= 0.5)
        label       = 'Pass' if prediction == 1 else 'Fail'

        # Risk band
        if prob >= 0.80:
            risk_band = 'Low Risk'
        elif prob >= 0.60:
            risk_band = 'Moderate Risk'
        elif prob >= 0.40:
            risk_band = 'High Risk'
        else:
            risk_band = 'Very High Risk'

        # Insights
        insights = build_insights(data, prob)

        # Top feature contributions (manual approximation using importances)
        importances = metadata['feature_importances']
        top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:6]

        return jsonify({
            'prediction': label,
            'pass': prediction,
            'probability': round(prob * 100, 1),
            'risk_band': risk_band,
            'insights': insights,
            'top_features': [{'name': f.replace('_', ' ').title(), 'importance': round(v * 100, 1)}
                             for f, v in top_features]
        })

    except ValueError as e:
        return jsonify({'error': str(e)}), 422
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/model-stats')
def model_stats():
    return jsonify({
        'metrics': metadata['metrics'],
        'best_params': metadata['best_params'],
        'feature_importances': metadata['feature_importances'],
        'dataset_info': {
            'rows': 649,
            'features': len(FEATURES),
            'algorithm': 'Random Forest (GridSearchCV tuned)',
            'resampling': 'SMOTE'
        }
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
