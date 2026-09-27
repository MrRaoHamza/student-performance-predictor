"""
Student Performance Prediction - ML Pipeline
=============================================
- Feature engineering (study-to-freetime ratio, parental support score, alcohol index)
- Label encoding + OneHotEncoding
- SMOTE for class imbalance
- Random Forest with GridSearchCV
- Evaluation: accuracy, precision, recall, F1, ROC-AUC
- Saves model + preprocessor + feature list with joblib
"""

import sys
QUIET = '--quiet' in sys.argv

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'student-mat.csv')
df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ─────────────────────────────────────────
# 2. Target variable
#    pass = G3 >= 10  (standard Portuguese grading)
# ─────────────────────────────────────────
df['pass'] = (df['G3'] >= 10).astype(int)
print(f"Class distribution — Pass: {df['pass'].sum()} | Fail: {(df['pass']==0).sum()}")

# Drop grade columns (G1, G2, G3) — we don't want data leakage when predicting
# In a real-world deployment G1/G2 may not be available early in the semester,
# but we keep them as OPTIONAL features (used only if provided). For training
# we keep them to match the UCI benchmark. The API will accept them as optional.
# For the "no grades yet" prediction path we drop them below.

# ─────────────────────────────────────────
# 3. Feature engineering
# ─────────────────────────────────────────
df['study_freetime_ratio'] = df['studytime'] / (df['freetime'] + 1)
df['parental_support_score'] = df['Medu'] + df['Fedu'] + \
    df['famsup'].map({'yes': 1, 'no': 0}) + \
    df['schoolsup'].map({'yes': 1, 'no': 0})
df['alcohol_index'] = (df['Dalc'] * 2 + df['Walc']) / 3   # weighted: weekday counts more
df['social_risk'] = df['goout'] + df['romantic'].map({'yes': 1, 'no': 0}) * 2
df['prior_failure'] = (df['failures'] > 0).astype(int)
df['high_aspiration'] = df['higher'].map({'yes': 1, 'no': 0})

# ─────────────────────────────────────────
# 4. Select features
# ─────────────────────────────────────────
CATEGORICAL_COLS = ['school', 'sex', 'address', 'famsize', 'Pstatus',
                    'Mjob', 'Fjob', 'reason', 'guardian',
                    'schoolsup', 'famsup', 'paid', 'activities',
                    'nursery', 'higher', 'internet', 'romantic']

NUMERIC_COLS = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
                'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
                'absences', 'G1', 'G2',
                # engineered
                'study_freetime_ratio', 'parental_support_score',
                'alcohol_index', 'social_risk', 'prior_failure', 'high_aspiration']

ALL_FEATURES = CATEGORICAL_COLS + NUMERIC_COLS

# ─────────────────────────────────────────
# 5. Encode categoricals
# ─────────────────────────────────────────
label_encoders = {}
df_encoded = df.copy()

for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

X = df_encoded[ALL_FEATURES]
y = df_encoded['pass']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Features used: {ALL_FEATURES}")

# ─────────────────────────────────────────
# 6. Train / test split (stratified)
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ─────────────────────────────────────────
# 7. SMOTE — balance the training set
# ─────────────────────────────────────────
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"After SMOTE — Train size: {X_train_res.shape[0]}")
print(f"  Pass: {y_train_res.sum()} | Fail: {(y_train_res==0).sum()}")

# ─────────────────────────────────────────
# 8. GridSearchCV — tune Random Forest
# ─────────────────────────────────────────
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', None]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nRunning GridSearchCV (this takes a minute)...")
grid_search = GridSearchCV(
    rf, param_grid,
    cv=cv,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train_res, y_train_res)

best_model = grid_search.best_estimator_
print(f"\nBest params: {grid_search.best_params_}")
print(f"Best CV F1: {grid_search.best_score_:.4f}")

# ─────────────────────────────────────────
# 9. Cross-validation on original train set
# ─────────────────────────────────────────
cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='f1')
print(f"\nCross-val F1 (original train): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────
# 10. Evaluate on test set
# ─────────────────────────────────────────
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_prob)
cm   = confusion_matrix(y_test, y_pred)

print("\n" + "="*50)
print("TEST SET RESULTS")
print("="*50)
print(f"  Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print(f"  ROC-AUC   : {auc:.4f}")
print(f"\nConfusion Matrix:\n{cm}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Fail','Pass'])}")

# ─────────────────────────────────────────
# 11. Feature importances
# ─────────────────────────────────────────
importances = pd.Series(
    best_model.feature_importances_, index=ALL_FEATURES
).sort_values(ascending=False)

print("\nTop 10 Feature Importances:")
print(importances.head(10).to_string())

# ─────────────────────────────────────────
# 12. Save model artifacts
# ─────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

joblib.dump(best_model,      os.path.join(MODELS_DIR, 'rf_model.pkl'))
joblib.dump(label_encoders,  os.path.join(MODELS_DIR, 'label_encoders.pkl'))

# Save metadata (metrics + feature list + encoder maps)
encoder_maps = {col: le.classes_.tolist() for col, le in label_encoders.items()}
metadata = {
    'features': ALL_FEATURES,
    'categorical_cols': CATEGORICAL_COLS,
    'numeric_cols': NUMERIC_COLS,
    'encoder_maps': encoder_maps,
    'metrics': {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(auc, 4)
    },
    'feature_importances': importances.head(15).round(4).to_dict(),
    'best_params': grid_search.best_params_
}

with open(os.path.join(MODELS_DIR, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n✓ Model saved  → models/rf_model.pkl")
print("✓ Encoders     → models/label_encoders.pkl")
print("✓ Metadata     → models/metadata.json")
