# ================================================================
# Mini Project : AI-Based Fertilizer and Irrigation Recommendation
# File         : train_irrigation.py
# Team         : Md Shoaib (23EG107D42), C. Jasvitha Reddy (23EG107D10)
#                G. Sumanth Goud (23EG107D24)
# College      : Anurag University, Hyderabad — AIML-D
#
# Base Paper   : "Soil Smart: AI-Powered Soil Testing and Fertilizer
#                Recommendation System"
#                Velagaleti et al., IEEE ICSCCC 2026
#                DOI: 10.1109/ICSCCC69031.2026.11600089
#
# 6 Algorithms (same set as fertilizer for consistency):
#   1. Logistic Regression  — linear baseline
#   2. KNN (K=7)            — distance-based (NEW)
#   3. Decision Tree        — tree-based
#   4. Gradient Boosting    — sequential boosting (NEW)
#   5. Random Forest        — ensemble
#   6. XGBoost              — optimized boosting
#
# No SMOTE needed — irrigation dataset is balanced:
#   Class 0: 55%  Class 1: 38%  Class 2: 7%
#
# Features: MOI (soil moisture), Temperature, Humidity, Soil Type
# Target  : 0=No Irrigation, 1=Irrigation Needed, 2=Excess Water
#
# Run: python src/train_irrigation.py
# ================================================================

import os, warnings, time
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model    import LogisticRegression
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.tree             import DecisionTreeClassifier
from sklearn.ensemble         import (RandomForestClassifier,
                                       GradientBoostingClassifier)
from xgboost                  import XGBClassifier
from sklearn.model_selection  import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics          import (accuracy_score, precision_score,
                                       recall_score, f1_score,
                                       classification_report,
                                       confusion_matrix,
                                       ConfusionMatrixDisplay)

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC    = os.path.join(BASE, 'data', 'processed')
MODELS  = os.path.join(BASE, 'models')
RESULTS = os.path.join(BASE, 'results')

CLASS_NAMES = ['No Irrigation', 'Irrigation Needed', 'Excess Water']

# ── Load data ────────────────────────────────────────────────
train_df = pd.read_csv(os.path.join(PROC, 'irr_train.csv'))
test_df  = pd.read_csv(os.path.join(PROC, 'irr_test.csv'))

FEAT   = ['MOI', 'temp', 'humidity', 'soil_enc']
TARGET = 'result'

X_train = train_df[FEAT].values
y_train = train_df[TARGET].values
X_test  = test_df[FEAT].values
y_test  = test_df[TARGET].values

print(f'Train: {len(X_train)} rows  |  Test: {len(X_test)} rows')
print(f'Features: MOI, Temperature, Humidity, Soil Type')
print(f'Class distribution: '
      f'{dict(zip(*np.unique(y_test, return_counts=True)))}')

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Tuning ────────────────────────────────────────────────────
print('\nTuning Random Forest...')
rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    {'n_estimators':[300,500,700],'max_depth':[10,15,20,None],
     'min_samples_split':[2,5],'min_samples_leaf':[1,2],
     'max_features':['sqrt','log2']},
    n_iter=20, cv=cv, scoring='accuracy',
    n_jobs=-1, random_state=42, verbose=0
)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_
print(f'  Best RF CV acc: {rf_search.best_score_:.4f}')

print('Tuning XGBoost...')
xgb_search = RandomizedSearchCV(
    XGBClassifier(eval_metric='mlogloss', random_state=42,
                  verbosity=0, n_jobs=-1),
    {'n_estimators':[300,500,700],'learning_rate':[0.03,0.05,0.1],
     'max_depth':[5,7,9],'subsample':[0.7,0.8,0.9,1.0],
     'colsample_bytree':[0.7,0.8,0.9,1.0]},
    n_iter=20, cv=cv, scoring='accuracy',
    n_jobs=-1, random_state=42, verbose=0
)
xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_
print(f'  Best XGB CV acc: {xgb_search.best_score_:.4f}')

# ── 6 Algorithms ─────────────────────────────────────────────
models = {
    'Logistic Regression' : LogisticRegression(
        max_iter=1000, C=1.0, random_state=42),
    'KNN (K=7)'           : KNeighborsClassifier(
        n_neighbors=7, metric='euclidean', n_jobs=-1),
    'Decision Tree'       : DecisionTreeClassifier(
        max_depth=12, min_samples_split=5,
        min_samples_leaf=2, random_state=42),
    'Gradient Boosting'   : GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1,
        max_depth=5, random_state=42),
    'Random Forest'       : best_rf,
    'XGBoost'             : best_xgb,
}

# ── Train and evaluate ────────────────────────────────────────
results = {}
print('\n' + '='*65)
print('  IRRIGATION — 6 ALGORITHMS COMPARISON')
print('  (MOI + Temperature + Humidity + Soil Type)')
print('='*65)
print(f'{"Algorithm":<22}  {"Accuracy":>9}  {"Precision":>10}  '
      f'{"Recall":>8}  {"F1":>8}')
print('-'*63)

for name, model in models.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    prec   = precision_score(y_test, y_pred, average='weighted',
                              zero_division=0)
    rec    = recall_score(y_test, y_pred, average='weighted',
                          zero_division=0)
    f1     = f1_score(y_test, y_pred, average='weighted',
                      zero_division=0)
    results[name] = {
        'acc':acc, 'prec':prec, 'rec':rec, 'f1':f1,
        'model':model, 'pred':y_pred,
        'time':round(time.time()-t0, 1)
    }
    print(f'{name:<22}  {acc*100:>8.2f}%  {prec:>10.4f}  '
          f'{rec:>8.4f}  {f1:>8.4f}')

# ── Save comparison CSV ───────────────────────────────────────
summary = pd.DataFrame({
    'Algorithm' : list(results.keys()),
    'Accuracy'  : [f'{v["acc"]*100:.2f}%'  for v in results.values()],
    'Precision' : [f'{v["prec"]:.4f}'       for v in results.values()],
    'Recall'    : [f'{v["rec"]:.4f}'        for v in results.values()],
    'F1 Score'  : [f'{v["f1"]:.4f}'         for v in results.values()],
    'Time(s)'   : [v['time']                 for v in results.values()],
})
summary.to_csv(os.path.join(RESULTS, 'irr_comparison.csv'), index=False)

# ── Comparison bar chart ──────────────────────────────────────
names  = list(results.keys())
accs   = [results[n]['acc']*100 for n in names]
colors = ['#4472C4','#9966FF','#ED7D31','#FF9900','#70AD47','#E53935']

fig, ax = plt.subplots(figsize=(13, 6))
bars = ax.bar(names, accs, color=colors, edgecolor='black', width=0.55)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.4,
            f'{acc:.2f}%', ha='center', va='bottom',
            fontsize=10, fontweight='bold')
ax.set_ylim(50, 108)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Irrigation Recommendation — 6 Algorithms Comparison\n'
             '(MOI + Temperature + Humidity + Soil Type)',
             fontsize=13, fontweight='bold')
ax.set_facecolor('#FAFAFA')
ax.tick_params(axis='x', rotation=15)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, 'irr_comparison.png'), dpi=120)
plt.close()
print('\nSaved: irr_comparison.png')

# ── Best model ────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['acc'])
best      = results[best_name]
print(f'\nBest Algorithm : {best_name}  —  {best["acc"]*100:.2f}%')

# ── Confusion Matrix ──────────────────────────────────────────
cm   = confusion_matrix(y_test, best['pred'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=CLASS_NAMES)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='Blues', colorbar=True, xticks_rotation=10)
ax.set_title(f'Confusion Matrix — {best_name} (Irrigation)\n'
             f'Accuracy: {best["acc"]*100:.2f}%',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, 'irr_confusion.png'), dpi=120)
plt.close()

# ── Feature importance ────────────────────────────────────────
rf_model   = models['Random Forest']
feat_names = ['MOI (Moisture)', 'Temperature', 'Humidity', 'Soil Type']
imp = pd.Series(rf_model.feature_importances_,
                index=feat_names).sort_values()
fig, ax = plt.subplots(figsize=(7, 4))
clrs_bar = ['#E53935' if v > imp.mean() else '#4472C4'
            for v in imp.values]
imp.plot(kind='barh', color=clrs_bar, edgecolor='black', ax=ax)
ax.axvline(imp.mean(), color='black', linestyle='--',
           alpha=0.5, label='Mean')
ax.set_title('Feature Importance — Random Forest (Irrigation)\n'
             'MOI dominates irrigation prediction',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Importance Score')
ax.legend(); ax.set_facecolor('#FAFAFA')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, 'irr_importance.png'), dpi=120)
plt.close()

# ── Classification report ─────────────────────────────────────
print('\n--- Per-Class Report ---')
print(classification_report(y_test, best['pred'],
                             target_names=CLASS_NAMES))

# ── Save model ────────────────────────────────────────────────
joblib.dump(best['model'], os.path.join(MODELS, 'model_irrigation.pkl'))
joblib.dump(best_name,     os.path.join(MODELS, 'best_irr_name.pkl'))
print(f'\nModel saved: {best_name}  —  {best["acc"]*100:.2f}%')
print('Irrigation training COMPLETE.')
