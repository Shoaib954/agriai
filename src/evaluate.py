# ================================================================
# Mini Project : AI-Based Fertilizer and Irrigation Recommendation
# File         : evaluate.py
# Team         : Md Shoaib (23EG107D42), C. Jasvitha Reddy (23EG107D10)
#                G. Sumanth Goud (23EG107D24)
# College      : Anurag University, Hyderabad — AIML-D
#
# Base Paper   : "Soil Smart: AI-Powered Soil Testing and Fertilizer
#                Recommendation System"
#                Velagaleti et al., IEEE ICSCCC 2026
#                DOI: 10.1109/ICSCCC69031.2026.11600089
#
# What this file does:
#   Loads both saved best models, runs them on test data,
#   prints final accuracy table, saves comparison charts.
#
# Evaluation metrics used (same as base paper):
#   Accuracy, Precision, Recall, F1-Score, Confusion Matrix
#
# Run after both training files:
#   python src/evaluate.py
# ================================================================

import os, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score,
                              classification_report)

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC    = os.path.join(BASE, 'data', 'processed')
MODELS  = os.path.join(BASE, 'models')
RESULTS = os.path.join(BASE, 'results')

# ── Load saved models ─────────────────────────────────────────
print('Loading saved models...')
fert_model = joblib.load(os.path.join(MODELS, 'model_fertilizer.pkl'))
irr_model  = joblib.load(os.path.join(MODELS, 'model_irrigation.pkl'))
le_fert    = joblib.load(os.path.join(MODELS, 'le_fert_target.pkl'))
fert_name  = joblib.load(os.path.join(MODELS, 'best_fert_name.pkl'))
irr_name   = joblib.load(os.path.join(MODELS, 'best_irr_name.pkl'))
print(f'Fertilizer model : {fert_name}')
print(f'Irrigation model : {irr_name}')

# ── Load test data ────────────────────────────────────────────
fert_test = pd.read_csv(os.path.join(PROC, 'fert_test.csv'))
irr_test  = pd.read_csv(os.path.join(PROC, 'irr_test.csv'))

FERT_FEAT = ['Soil_Type_enc','Soil_pH','Soil_Moisture',
             'Organic_Carbon','Nitrogen_Level','Phosphorus_Level',
             'Potassium_Level','Temperature','Humidity','Rainfall',
             'Crop_Type_enc','Season_enc']
IRR_FEAT  = ['MOI','temp','humidity','soil_enc']

X_f = fert_test[FERT_FEAT].values; y_f = fert_test['Target'].values
X_i = irr_test[IRR_FEAT].values;   y_i = irr_test['result'].values

# ── Predictions ───────────────────────────────────────────────
pred_f = fert_model.predict(X_f)
pred_i = irr_model.predict(X_i)

# ── Scores ────────────────────────────────────────────────────
def get_scores(y_true, y_pred):
    return {
        'Accuracy' : accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred,
                                      average='weighted',
                                      zero_division=0),
        'Recall'   : recall_score(y_true, y_pred,
                                   average='weighted',
                                   zero_division=0),
        'F1 Score' : f1_score(y_true, y_pred,
                               average='weighted',
                               zero_division=0),
    }

fs  = get_scores(y_f, pred_f)
is_ = get_scores(y_i, pred_i)

# ── Print final table ─────────────────────────────────────────
print('\n' + '='*70)
print('  FINAL RESULTS — FERTILIZER & IRRIGATION MODULES')
print('  (Performance metrics as per IEEE ICSCCC 2026 base paper)')
print('='*70)
print(f'{"Module":<30} {"Algorithm":<22} {"Accuracy":>9}')
print('-'*70)
print(f'{"Fertilizer Recommendation":<30} '
      f'{fert_name:<22} {fs["Accuracy"]*100:>8.2f}%')
print(f'{"Irrigation Recommendation":<30} '
      f'{irr_name:<22} {is_["Accuracy"]*100:>8.2f}%')
print('='*70)
print(f'\nBase paper (IEEE ICSCCC 2026): RF achieved 85% for fertilizer')
print(f'Our result                    : RF achieved {fs["Accuracy"]*100:.2f}% for fertilizer')

# ── Save CSV ──────────────────────────────────────────────────
final = pd.DataFrame([
    {'Module': 'Fertilizer Recommendation',
     'Algorithm': fert_name,
     **{k: round(v, 4) for k, v in fs.items()}},
    {'Module': 'Irrigation Recommendation',
     'Algorithm': irr_name,
     **{k: round(v, 4) for k, v in is_.items()}},
])
final.to_csv(os.path.join(RESULTS, 'final_results.csv'), index=False)
print('\nSaved: final_results.csv')

# ── Final grouped bar chart ────────────────────────────────────
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
fv      = [fs[m]  for m in metrics]
iv      = [is_[m] for m in metrics]
colors4 = ['#4472C4','#ED7D31','#70AD47','#E53935']

x = np.arange(2)
w = 0.18
fig, ax = plt.subplots(figsize=(11, 6))
for i, (metric, color) in enumerate(zip(metrics, colors4)):
    vals = [fs[metric], is_[metric]]
    bars = ax.bar(x + (i-1.5)*w, vals, w,
                  label=metric, color=color, edgecolor='black')
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.005,
                f'{bar.get_height():.3f}',
                ha='center', va='bottom',
                fontsize=8, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Fertilizer\nRecommendation',
                    'Irrigation\nRecommendation'], fontsize=12)
ax.set_ylim(0.5, 1.12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Final Evaluation — Both Modules\n'
             'Base Paper (IEEE ICSCCC 2026): RF 85% | Ours: RF '
             f'{fs["Accuracy"]*100:.2f}%',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_facecolor('#FAFAFA')
fig.patch.set_facecolor('white')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, 'final_evaluation.png'), dpi=120)
plt.close()
print('Saved: final_evaluation.png')

# ── Radar chart ───────────────────────────────────────────────
angles = np.linspace(0, 2*np.pi, len(metrics),
                     endpoint=False).tolist()
angles += angles[:1]
fig, ax = plt.subplots(figsize=(6, 6),
                        subplot_kw=dict(polar=True))
for vals, name, color in [
    (fv,  'Fertilizer', '#4472C4'),
    (iv,  'Irrigation',  '#E53935'),
]:
    v = vals + vals[:1]
    ax.plot(angles, v, 'o-', lw=2, label=name, color=color)
    ax.fill(angles, v, alpha=0.1, color=color)
ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=11)
ax.set_ylim(0.5, 1.0)
ax.set_title('Performance Radar Chart\n'
             'Fertilizer vs Irrigation', fontsize=12,
             fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, 'radar.png'), dpi=120)
plt.close()
print('Saved: radar.png')

# ── Detailed per-class reports ────────────────────────────────
print('\n--- Fertilizer Per-Class Report ---')
print(classification_report(y_f, pred_f,
                             target_names=le_fert.classes_))
print('\n--- Irrigation Per-Class Report ---')
print(classification_report(
    y_i, pred_i,
    target_names=['No Irrigation', 'Irrigation Needed', 'Excess Water']
))

print('\nEvaluation COMPLETE.')
print(f'Fertilizer: {fert_name} — {fs["Accuracy"]*100:.2f}%')
print(f'Irrigation: {irr_name}  — {is_["Accuracy"]*100:.2f}%')
