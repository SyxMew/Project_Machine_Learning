# complete_heart_pipeline.py
# ===========================================================
#  PROGRAM: Prediksi Risiko Penyakit Jantung (lengkap)
#  METODE: Decision Tree + KNN (GridSearch + Voting Ensemble)
#  OUTPUT : model (joblib), figures (ROC, CM, feature importance), csv hasil prediksi
# ===========================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, roc_auc_score, ConfusionMatrixDisplay
)

# -------------------------
# CONFIG
# -------------------------
FILE_NAME = 'heart.csv'
RANDOM_STATE = 42
TEST_SIZE = 0.2
FIG_DIR = 'figures'
MODEL_FILE = 'ensemble_model.joblib'
RESULTS_CSV = 'predictions_results.csv'
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(FILE_NAME)
print(f"Loaded data: {FILE_NAME}  — shape: {df.shape}\n")

if 'target' not in df.columns:
    raise ValueError("Kolom 'target' tidak ditemukan di dataset.")

print("== INFO ==")
print(df.info(), "\n")
print("== MISSING VALUES ==")
print(df.isnull().sum(), "\n")
print("== TARGET VALUE COUNTS ==")
print(df['target'].value_counts(), "\n")
print("== DESCRIPTIVE SAMPLE ==")
print(df.head(), "\n")


cat_cols = ['sex', 'cp', 'restecg', 'exang', 'slope', 'thal', 'ca', 'fbs']
cat_cols = [c for c in cat_cols if c in df.columns]
num_cols = [c for c in df.columns if c not in cat_cols + ['target']]

print(f"Numeric cols ({len(num_cols)}): {num_cols}")
print(f"Categorical cols ({len(cat_cols)}): {cat_cols}\n")

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
print(f"Split: train={X_train.shape[0]}, test={X_test.shape[0]}\n")

num_imputer = SimpleImputer(strategy='median')
num_scaler = StandardScaler()

num_pipeline_for_knn = Pipeline([
    ('imputer', num_imputer),
    ('scaler', num_scaler)
])

num_pipeline_for_dt = Pipeline([
    ('imputer', num_imputer)  
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

from sklearn.compose import ColumnTransformer
preproc_for_knn = ColumnTransformer([
    ('num', num_pipeline_for_knn, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

preproc_for_dt = ColumnTransformer([
    ('num', num_pipeline_for_dt, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

pipe_dt = Pipeline([
    ('pre', preproc_for_dt),
    ('clf', DecisionTreeClassifier(random_state=RANDOM_STATE))
])

pipe_knn = Pipeline([
    ('pre', preproc_for_knn),
    ('clf', KNeighborsClassifier())
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

param_grid_dt = {
    'clf__max_depth': [None, 3, 5, 7, 9],
    'clf__min_samples_split': [2, 5, 10],
    'clf__criterion': ['gini', 'entropy']
}

param_grid_knn = {
    'clf__n_neighbors': [3, 5, 7, 9],
    'clf__weights': ['uniform', 'distance'],
    'clf__p': [1, 2]
}

print("Starting GridSearchCV for Decision Tree...")
grid_dt = GridSearchCV(pipe_dt, param_grid_dt, cv=cv, scoring='f1', n_jobs=-1, verbose=0)
grid_dt.fit(X_train, y_train)
print("Best DT params:", grid_dt.best_params_, " Best CV f1-score:", grid_dt.best_score_)

print("\nStarting GridSearchCV for KNN...")
grid_knn = GridSearchCV(pipe_knn, param_grid_knn, cv=cv, scoring='f1', n_jobs=-1, verbose=0)
grid_knn.fit(X_train, y_train)
print("Best KNN params:", grid_knn.best_params_, " Best CV f1-score:", grid_knn.best_score_, "\n")

best_dt = grid_dt.best_estimator_
best_knn = grid_knn.best_estimator_

ensemble = VotingClassifier(
    estimators=[('dt', best_dt), ('knn', best_knn)],
    voting='soft',
    n_jobs=-1
)

print("Fitting ensemble on training data...")
ensemble.fit(X_train, y_train)
print("Ensemble fitted.\n")

y_pred_dt = best_dt.predict(X_test)
y_pred_knn = best_knn.predict(X_test)
y_pred_ensemble = ensemble.predict(X_test)

acc_dt = accuracy_score(y_test, y_pred_dt)
acc_knn = accuracy_score(y_test, y_pred_knn)
acc_ensemble = accuracy_score(y_test, y_pred_ensemble)

print("=== ACCURACY (test) ===")
print(f"Decision Tree : {acc_dt*100:.2f}%")
print(f"KNN           : {acc_knn*100:.2f}%")
print(f"Ensemble      : {acc_ensemble*100:.2f}%\n")

print("=== CLASSIFICATION REPORT (Ensemble) ===")
print(classification_report(y_test, y_pred_ensemble, target_names=['Tidak Berisiko','Beresiko']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_ensemble)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred_Tidak','Pred_Berisi'], yticklabels=['True_Tidak','True_Berisi'])
plt.title('Confusion Matrix (Ensemble)')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'confusion_matrix_ensemble.png'), dpi=200)
plt.show()

# ROC & AUC
if hasattr(ensemble, "predict_proba"):
    y_proba = ensemble.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f'Ensemble ROC (AUC = {roc_auc:.3f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve (Ensemble)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'roc_curve_ensemble.png'), dpi=200)
    plt.show()
    print(f"Ensemble ROC-AUC: {roc_auc:.4f}\n")
else:
    print("Ensemble has no predict_proba — ROC-AUC not available.\n")


try:
    preproc = best_dt.named_steps['pre']
    num_names = num_cols
    ohe = preproc.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = list(ohe.get_feature_names_out(cat_cols)) if hasattr(ohe, 'get_feature_names_out') else []
    feature_names = num_names + cat_feature_names

    dt_model = best_dt.named_steps['clf']
    importances = dt_model.feature_importances_
    if len(importances) == len(feature_names):
        fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
        plt.figure(figsize=(8,6))
        sns.barplot(x='importance', y='feature', data=fi_df.head(20))
        plt.title('Feature Importance (Decision Tree)')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'feature_importance_dt.png'), dpi=200)
        plt.show()
    else:
        print("Warning: jumlah feature_importances_ tidak sama dengan jumlah nama fitur. Skipping feature importance plot.")
except Exception as e:
    print("Error saat men-generate feature importance:", e)

X_test_reset = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)
results_df = X_test_reset.copy()
results_df['Actual_Target'] = y_test_reset
results_df['DT_Pred'] = y_pred_dt
results_df['KNN_Pred'] = y_pred_knn
results_df['Ensemble_Pred'] = y_pred_ensemble
results_df['Ensemble_Correct'] = results_df['Actual_Target'] == results_df['Ensemble_Pred']
results_df.to_csv(RESULTS_CSV, index=False)
print(f"Saved prediction results -> {RESULTS_CSV}")

joblib.dump(ensemble, MODEL_FILE)
print(f"Saved ensemble model -> {MODEL_FILE}")

with open(os.path.join(FIG_DIR, 'best_params.txt'), 'w') as f:
    f.write("Best DT params:\n")
    f.write(str(grid_dt.best_params_) + "\n")
    f.write("Best DT CV score (f1): " + str(grid_dt.best_score_) + "\n\n")
    f.write("Best KNN params:\n")
    f.write(str(grid_knn.best_params_) + "\n")
    f.write("Best KNN CV score (f1): " + str(grid_knn.best_score_) + "\n")

print("Done. Figures and model saved in the working directory.")
