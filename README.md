# AgriAI — Smart Farming Recommendation System

> AI-Based Fertilizer & Irrigation Recommendation Using Machine Learning  
> Mini Project · Anurag University, Hyderabad · AIML-D

---

## Team

| Name | HT Number |
|---|---|
| C. Jasvitha Reddy | 23EG107D10 |
| Md Shoaib | 23EG107D42 |
| G. Sumanth Goud | 23EG107D24 |

---

## Overview

AgriAI is a Streamlit web application that provides real-time fertilizer and irrigation recommendations based on soil, weather, and crop inputs. It trains and compares six machine learning algorithms across two independent modules and presents results through an interactive dark-themed dashboard.

**Base Paper:** Velagaleti et al., *"Soil Smart: AI-Powered Soil Testing and Fertilizer Recommendation System"*, IEEE ICSCCC 2026. DOI: `10.1109/ICSCCC69031.2026.11600089`

---

## Project Structure

```
project/
├── app.py                        # Streamlit dashboard
├── requirements.txt
├── src/
│   ├── preprocess.py             # Data cleaning, encoding, scaling, train/test split
│   ├── train_fertilizer.py       # Train 6 algorithms on fertilizer dataset
│   ├── train_irrigation.py       # Train 6 algorithms on irrigation dataset
│   └── evaluate.py               # Final evaluation, charts, and reports
├── data/
│   ├── raw/
│   │   ├── Fertilizer_Prediction.csv
│   │   └── Irrigation_Prediction.csv
│   └── processed/                # Generated after preprocessing
│       ├── fert_train.csv / fert_test.csv
│       └── irr_train.csv / irr_test.csv
├── models/                       # Saved .pkl files (models + encoders + scalers)
└── results/                      # Generated charts and CSVs
```

---

## Datasets

| Module | Dataset | Rows | Features | Classes |
|---|---|---|---|---|
| Fertilizer | Fertilizer_Prediction.csv | 10,000 | 12 | 7 fertilizer types |
| Irrigation | Irrigation_Prediction.csv | 16,411 | 4 | 3 irrigation statuses |

**Fertilizer features:** Soil Type, Soil pH, Soil Moisture, Organic Carbon, Nitrogen, Phosphorus, Potassium, Temperature, Humidity, Rainfall, Crop Type, Season

**Irrigation features:** MOI (Moisture Index), Temperature, Humidity, Soil Type

**Irrigation classes:** `0` = No Irrigation Needed · `1` = Irrigation Needed · `2` = Excess Water

---

## Algorithms Compared

Six algorithms are evaluated for both modules:

1. **Logistic Regression** — linear baseline
2. **K-Nearest Neighbors (K=7)** — distance-based
3. **Decision Tree** — interpretable tree model
4. **Gradient Boosting** — sequential ensemble
5. **Random Forest** — primary model from base paper (tuned with RandomizedSearchCV)
6. **XGBoost** — optimized gradient boosting (tuned with RandomizedSearchCV)

---

## Our Contributions vs Base Paper

| Item | Base Paper | This Project |
|---|---|---|
| Fertilizer accuracy | 85% (Random Forest) | **86.05%** (Random Forest) |
| Class imbalance | Not addressed | **SMOTE applied** |
| Hyperparameter tuning | None | **RandomizedSearchCV** |
| Algorithms compared | 4 | **6** |

**Research gaps addressed:**
- The base paper noted that "accuracy diminished with imbalanced datasets" — we fix this with SMOTE, raising SSP class F1-score from 0.06 to a competitive level and improving macro F1 from 0.58 to 0.76.
- Systematic hyperparameter tuning via RandomizedSearchCV with 5-fold StratifiedKFold cross-validation.
- Two additional algorithms (KNN, Gradient Boosting) added for a more comprehensive comparison.

---

## Preprocessing Pipeline

Steps follow the base paper methodology:

1. **Data cleaning** — remove missing values and duplicates
2. **Outlier removal** — IQR method (irrigation dataset)
3. **Label encoding** — categorical features (Soil Type, Crop Type, Season, Soil Type for irrigation)
4. **MinMax scaling** — normalize all features to [0, 1]
5. **Train/test split** — 80:20 stratified split

---

## Installation

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
streamlit==1.36.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
matplotlib==3.9.1
seaborn==0.13.2
joblib==1.4.2
xgboost==2.1.1
```

> `imbalanced-learn` is also required for SMOTE. Install it with:
> ```bash
> pip install imbalanced-learn
> ```

---

## Running the Project

Run the scripts in order from the project root:

```bash
# Step 1 — Preprocess both datasets
python src/preprocess.py

# Step 2 — Train fertilizer models (saves best model + charts)
python src/train_fertilizer.py

# Step 3 — Train irrigation models (saves best model + charts)
python src/train_irrigation.py

# Step 4 — Final evaluation and report generation
python src/evaluate.py

# Step 5 — Launch the web app
streamlit run app.py
```

---

## App Features

The Streamlit dashboard has five tabs:

| Tab | Description |
|---|---|
| **Get Recommendations** | Enter soil/weather inputs and receive fertilizer + irrigation predictions with confidence scores, NPK comparison chart, soil health radar, and downloadable report |
| **Data Explorer** | Interactive charts for both datasets — pie charts, box plots, scatter plots, 3D visualization |
| **Model Results** | Algorithm comparison bar charts, confusion matrices, feature importance plots, SMOTE effect charts |
| **History** | Logs all predictions made in the current session with export to CSV |
| **About** | Project details, algorithm justification table, SMOTE effect table, and research references |

---

## Saved Model Files

After training, the `models/` directory contains:

| File | Description |
|---|---|
| `model_fertilizer.pkl` | Best fertilizer model |
| `model_irrigation.pkl` | Best irrigation model |
| `best_fert_name.pkl` | Name of best fertilizer algorithm |
| `best_irr_name.pkl` | Name of best irrigation algorithm |
| `scaler_fert.pkl` / `scaler_irr.pkl` | MinMax scalers |
| `le_fert_soil_type.pkl` | Label encoder — fertilizer soil type |
| `le_fert_crop_type.pkl` | Label encoder — crop type |
| `le_fert_season.pkl` | Label encoder — season |
| `le_fert_target.pkl` | Label encoder — fertilizer target classes |
| `le_irr_soil.pkl` | Label encoder — irrigation soil type |

---

## Research References

| # | Paper | Venue | Year |
|---|---|---|---|
| **[1]** | Velagaleti et al. — Soil Smart: AI-Powered Soil Testing & Fertilizer Recommendation | IEEE ICSCCC | 2026 |
| [2] | Sunilkumar et al. — Smart Irrigation System Recommending Crop and Fertilizers | IEEE ICNWC | 2024 |
| [3] | Khaliq et al. — AI-Driven Smart Agriculture (Integrated Approach) | IEEE Access | 2025 |
| [4] | Musanase et al. — Soil Fertility Prediction Using ML | MDPI Sustainability | 2023 |
| [5] | Tanaka et al. — Gradient Boosting for Fertilizer Recommendation | Springer | 2024 |
| [6] | Chawla et al. — SMOTE: Synthetic Minority Over-sampling Technique | JAIR | 2002 |
