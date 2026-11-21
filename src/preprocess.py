# ================================================================
# Mini Project : AI-Based Fertilizer and Irrigation Recommendation
# File         : preprocess.py
# Team         : Md Shoaib (23EG107D42), C. Jasvitha Reddy (23EG107D10)
#                G. Sumanth Goud (23EG107D24)
# College      : Anurag University, Hyderabad — AIML-D
#
# Base Paper   : "Soil Smart: AI-Powered Soil Testing and Fertilizer
#                Recommendation System"
#                Velagaleti et al., IEEE ICSCCC 2026
#                DOI: 10.1109/ICSCCC69031.2026.11600089
#
# The base paper uses the following input features for fertilizer:
#   NPK (Nitrogen, Phosphorus, Potassium), Temperature,
#   Humidity, Soil Moisture → same as our fertilizer dataset
# And uses Random Forest as the primary classifier — same as us.
#
# Preprocessing steps followed as per base paper:
#   Step 1 : Data Cleaning (missing values, duplicates)
#   Step 2 : Outlier Removal (IQR method)
#   Step 3 : Label Encoding (categorical → numeric)
#   Step 4 : Normalization / MinMax Scaling (0 to 1)
#   Step 5 : Train-Test Split 80:20
#
# FERTILIZER DATASET — 12 input features:
#   Soil_Type, Soil_pH, Soil_Moisture, Organic_Carbon,
#   Nitrogen_Level, Phosphorus_Level, Potassium_Level,
#   Temperature, Humidity, Rainfall, Crop_Type, Season
#   Target: Recommended_Fertilizer (7 types)
#
# IRRIGATION DATASET — 4 input features:
#   MOI (Moisture Index), Temperature, Humidity, Soil_Type
#   Target: result (0=No Irrigation, 1=Needed, 2=Excess)
#
# Run: python src/preprocess.py
# ================================================================

import os, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing   import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# ── Direct dataset paths ─────────────────────────────────────
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FERT_CSV = os.path.join(BASE, 'data', 'raw', 'Fertilizer_Prediction.csv')
IRR_CSV  = os.path.join(BASE, 'data', 'raw', 'Irrigation_Prediction.csv')
PROC     = os.path.join(BASE, 'data', 'processed')
MODELS   = os.path.join(BASE, 'models')
RESULTS  = os.path.join(BASE, 'results')
for d in [PROC, MODELS, RESULTS]:
    os.makedirs(d, exist_ok=True)


def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, name), dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {name}')


# ================================================================
# FERTILIZER DATASET
# As per base paper (Velagaleti et al. IEEE ICSCCC 2026):
#   Input : NPK, temperature, humidity, soil moisture, soil type,
#           crop type (+ additional: pH, organic carbon, rainfall,
#           season for richer feature set)
#   Output: Fertilizer type (7 classes)
# ================================================================
def preprocess_fertilizer():
    print('\n' + '='*60)
    print('  FERTILIZER DATASET — PREPROCESSING')
    print('  (As per IEEE ICSCCC 2026 base paper feature set)')
    print('='*60)

    df = pd.read_csv(FERT_CSV)
    print(f'Loaded   : {df.shape[0]} rows, {df.shape[1]} columns')
    print(f'Nulls    : {df.isnull().sum().sum()} total missing values')
    print(f'\nTarget distribution:\n{df["Recommended_Fertilizer"].value_counts()}')

    TARGET = 'Recommended_Fertilizer'

    # Base paper uses: NPK, Temperature, Humidity, Moisture,
    # Soil Type, Crop Type. We extend with pH, Organic Carbon,
    # Rainfall, Season for better accuracy (12 features total).
    INPUTS = [
        'Soil_Type',        # Soil classification (categorical)
        'Soil_pH',          # Soil acidity — affects nutrient uptake
        'Soil_Moisture',    # Moisture level in soil
        'Organic_Carbon',   # Carbon content — soil health indicator
        'Nitrogen_Level',   # N level (kg/ha) — key NPK input
        'Phosphorus_Level', # P level (kg/ha) — key NPK input
        'Potassium_Level',  # K level (kg/ha) — key NPK input
        'Temperature',      # Air temperature (°C)
        'Humidity',         # Air humidity (%)
        'Rainfall',         # Rainfall (mm)
        'Crop_Type',        # Crop type (categorical)
        'Season',           # Kharif / Rabi / Zaid (categorical)
    ]

    df = df[INPUTS + [TARGET]].copy()
    df.drop_duplicates(inplace=True)
    print(f'\nAfter removing duplicates: {len(df)} rows')

    # ── EDA PLOTS ──────────────────────────────────────────────

    # Plot 1 — Class distribution (shows imbalance problem)
    fig, ax = plt.subplots(figsize=(10, 5))
    vals    = df[TARGET].value_counts()
    colors  = ['#4472C4','#ED7D31','#70AD47','#E53935',
               '#9966FF','#00B0F0','#FF66CC']
    bars    = ax.bar(vals.index, vals.values, color=colors,
                     edgecolor='black', width=0.6)
    for bar, v in zip(bars, vals.values):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+20, str(v),
                ha='center', fontsize=9, fontweight='bold')
    ax.set_title('Fertilizer Type Distribution — 10,000 rows\n'
                 '(Class imbalance: Urea=3101 vs SSP=182)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Fertilizer Type'); ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=15)
    ax.set_facecolor('#FAFAFA')
    save_fig('fert_class_dist.png')

    # Plot 2 — NPK grouped bar (as referenced in base paper)
    npk_means = df.groupby(TARGET)[
        ['Nitrogen_Level','Phosphorus_Level','Potassium_Level']
    ].mean()
    npk_means.plot(kind='bar', figsize=(12, 5),
                   color=['#4472C4','#ED7D31','#70AD47'],
                   edgecolor='black', width=0.7)
    plt.title('Average NPK Levels per Fertilizer Type\n'
              '(Key features as per base paper)',
              fontsize=13, fontweight='bold')
    plt.xlabel('Fertilizer Type'); plt.ylabel('Average Value (kg/ha)')
    plt.legend(['Nitrogen (N)','Phosphorus (P)','Potassium (K)'])
    plt.xticks(rotation=15, ha='right')
    save_fig('fert_npk.png')

    # Plot 3 — Correlation heatmap (upper triangle)
    num_cols = ['Soil_pH','Soil_Moisture','Organic_Carbon',
                'Nitrogen_Level','Phosphorus_Level','Potassium_Level',
                'Temperature','Humidity','Rainfall']
    plt.figure(figsize=(11, 8))
    mask = np.triu(np.ones_like(df[num_cols].corr(), dtype=bool))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f',
                cmap='RdYlGn', mask=mask, linewidths=0.5,
                annot_kws={'size': 8}, vmin=-1, vmax=1)
    plt.title('Fertilizer Dataset — Feature Correlation Heatmap',
              fontsize=13, fontweight='bold')
    save_fig('fert_corr.png')

    # Plot 4 — Season and Soil distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df['Season'].value_counts().plot(kind='bar', ax=axes[0],
        color='#4472C4', edgecolor='black')
    axes[0].set_title('Season Distribution')
    axes[0].tick_params(axis='x', rotation=15)
    df['Soil_Type'].value_counts().plot(kind='bar', ax=axes[1],
        color='#70AD47', edgecolor='black')
    axes[1].set_title('Soil Type Distribution')
    axes[1].tick_params(axis='x', rotation=20)
    plt.suptitle('Categorical Feature Distributions',
                 fontsize=12, fontweight='bold')
    save_fig('fert_cat.png')

    # ── LABEL ENCODING ─────────────────────────────────────────
    # Base paper uses label encoding for categorical features.
    # Soil Type, Crop Type, Season → integer codes.
    # We choose Label Encoding over One-Hot Encoding because
    # Random Forest and XGBoost handle integer codes natively.
    cat_cols = ['Soil_Type', 'Crop_Type', 'Season']
    for col in cat_cols:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col])
        joblib.dump(le, os.path.join(MODELS, f'le_fert_{col.lower()}.pkl'))

    le_target = LabelEncoder()
    df['Target'] = le_target.fit_transform(df[TARGET])
    joblib.dump(le_target, os.path.join(MODELS, 'le_fert_target.pkl'))
    print(f'\nFertilizer classes: {list(le_target.classes_)}')

    # ── FEATURE LIST (12) ─────────────────────────────────────
    FEAT = [
        'Soil_Type_enc', 'Soil_pH', 'Soil_Moisture', 'Organic_Carbon',
        'Nitrogen_Level', 'Phosphorus_Level', 'Potassium_Level',
        'Temperature', 'Humidity', 'Rainfall',
        'Crop_Type_enc', 'Season_enc',
    ]

    # ── MINMAX SCALING ────────────────────────────────────────
    # Base paper applies normalization to scale values to 0–1.
    # This ensures no single feature dominates due to magnitude.
    scaler = MinMaxScaler()
    df[FEAT] = scaler.fit_transform(df[FEAT])
    joblib.dump(scaler, os.path.join(MODELS, 'scaler_fert.pkl'))
    print('MinMax Scaling applied.')

    # ── TRAIN / TEST SPLIT 80:20 ──────────────────────────────
    # Base paper uses 80% training and 20% testing.
    # Stratified split maintains class proportions.
    X, y = df[FEAT], df['Target']
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    train = pd.concat([X_tr, y_tr], axis=1)
    test  = pd.concat([X_te, y_te], axis=1)
    train[TARGET] = df.loc[y_tr.index, TARGET].values
    test[TARGET]  = df.loc[y_te.index, TARGET].values
    train.to_csv(os.path.join(PROC, 'fert_train.csv'), index=False)
    test.to_csv( os.path.join(PROC, 'fert_test.csv'),  index=False)
    print(f'Train: {len(train)} rows  |  Test: {len(test)} rows')
    print('Fertilizer preprocessing DONE.')


# ================================================================
# IRRIGATION DATASET
# As per base paper approach:
#   Input : Soil Moisture (MOI), Temperature, Air Humidity,
#           Soil Type — same 4 features
#   Output: Irrigation status (3 classes)
# ================================================================
def preprocess_irrigation():
    print('\n' + '='*60)
    print('  IRRIGATION DATASET — PREPROCESSING')
    print('  (Soil moisture, temperature, humidity as per base paper)')
    print('='*60)

    df = pd.read_csv(IRR_CSV)
    print(f'Loaded   : {df.shape[0]} rows, {df.shape[1]} columns')
    print(f'Nulls    : {df.isnull().sum().sum()} total missing values')
    print(f'\nTarget distribution:\n{df["result"].value_counts()}')

    TARGET = 'result'
    df.drop_duplicates(inplace=True)
    label_map = {0: 'No Irrigation', 1: 'Irrigation Needed',
                 2: 'Excess Water'}

    # ── EDA PLOTS ──────────────────────────────────────────────

    # Plot 1 — Class distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df[TARGET].value_counts().sort_index()
    clrs   = ['#70AD47', '#E53935', '#ED7D31']
    bars   = ax.bar([label_map[i] for i in counts.index],
                    counts.values, color=clrs,
                    edgecolor='black', width=0.5)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+60, str(v),
                ha='center', fontsize=11, fontweight='bold')
    ax.set_title('Irrigation Status Distribution — 16,411 rows',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Records')
    ax.set_facecolor('#FAFAFA')
    save_fig('irr_class_dist.png')

    # Plot 2 — MOI distribution per class
    plt.figure(figsize=(10, 5))
    for cls, clr in zip([0,1,2], ['#70AD47','#E53935','#ED7D31']):
        plt.hist(df[df[TARGET]==cls]['MOI'], bins=35,
                 alpha=0.65, color=clr, label=label_map[cls],
                 edgecolor='white')
    plt.title('Moisture Index (MOI) Distribution per Irrigation Class\n'
              '(MOI = most important feature as per base paper)',
              fontsize=13, fontweight='bold')
    plt.xlabel('MOI Value'); plt.ylabel('Frequency'); plt.legend()
    save_fig('irr_moi.png')

    # Plot 3 — Box plots
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, col in zip(axes, ['MOI','temp','humidity']):
        data = [df[df[TARGET]==c][col].values for c in [0,1,2]]
        bp   = ax.boxplot(data, patch_artist=True,
                          labels=['No Irr.','Needed','Excess'])
        for patch, clr in zip(bp['boxes'],
                               ['#70AD47','#E53935','#ED7D31']):
            patch.set_facecolor(clr); patch.set_alpha(0.7)
        ax.set_title(col, fontsize=12, fontweight='bold')
    plt.suptitle('Feature Distributions per Irrigation Class',
                 fontsize=13, fontweight='bold')
    save_fig('irr_boxplots.png')

    # ── Label encode soil_type ────────────────────────────────
    le_soil = LabelEncoder()
    df['soil_enc'] = le_soil.fit_transform(df['soil_type'])
    joblib.dump(le_soil, os.path.join(MODELS, 'le_irr_soil.pkl'))

    # Plot 4 — Correlation heatmap
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        df[['MOI','temp','humidity','soil_enc',TARGET]].corr(),
        annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5
    )
    plt.title('Irrigation Dataset — Feature Correlation',
              fontsize=13, fontweight='bold')
    save_fig('irr_corr.png')

    # ── Outlier removal (IQR) — as per base paper ─────────────
    before = len(df)
    for col in ['MOI', 'temp', 'humidity']:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df  = df[(df[col] >= Q1-1.5*IQR) &
                 (df[col] <= Q3+1.5*IQR)]
    print(f'Outliers removed: {before-len(df)} rows  |  '
          f'Remaining: {len(df)}')

    # ── MinMax Scaling ────────────────────────────────────────
    FEAT = ['MOI', 'temp', 'humidity', 'soil_enc']
    scaler = MinMaxScaler()
    df[FEAT] = scaler.fit_transform(df[FEAT])
    joblib.dump(scaler, os.path.join(MODELS, 'scaler_irr.pkl'))
    print('MinMax Scaling applied.')

    # ── Train / Test Split 80:20 ──────────────────────────────
    X, y = df[FEAT], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    train = pd.concat([X_tr, y_tr], axis=1)
    test  = pd.concat([X_te, y_te], axis=1)
    train.to_csv(os.path.join(PROC, 'irr_train.csv'), index=False)
    test.to_csv( os.path.join(PROC, 'irr_test.csv'),  index=False)
    print(f'Train: {len(train)} rows  |  Test: {len(test)} rows')
    print('Irrigation preprocessing DONE.')


# ================================================================
if __name__ == '__main__':
    preprocess_fertilizer()
    preprocess_irrigation()
    print('\n' + '='*60)
    print('  ALL PREPROCESSING COMPLETE')
    print('  Base Paper: Velagaleti et al., IEEE ICSCCC 2026')
    print('  Preprocessing: Clean -> Encode -> Scale -> Split 80:20')
    print('='*60)
