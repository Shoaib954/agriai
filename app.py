# ================================================================
# Mini Project : AI-Based Fertilizer and Irrigation Recommendation
# File         : app.py
# Team         : Md Shoaib (23EG107D42), C. Jasvitha Reddy (23EG107D10)
#                G. Sumanth Goud (23EG107D24)
# College      : Anurag University, Hyderabad — AIML-D
# Run          : streamlit run app.py
# ================================================================

import os, datetime, warnings
warnings.filterwarnings('ignore')

# ── Gemini AI — optional, graceful fallback if not installed ──
try:
    from google import genai as google_genai
    GEMINI_AVAILABLE = True
except ImportError:
    try:
        import google.generativeai as google_genai
        GEMINI_AVAILABLE = True
        GEMINI_LEGACY = True
    except ImportError:
        GEMINI_AVAILABLE = False
GEMINI_LEGACY = False

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title='AgriAI — Smart Farming',
    page_icon='🌱',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ── Paths ────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
MODELS  = os.path.join(BASE, 'models')
RESULTS = os.path.join(BASE, 'results')
FERT_CSV = os.path.join(BASE, 'data', 'raw', 'Fertilizer_Prediction.csv')
IRR_CSV  = os.path.join(BASE, 'data', 'raw', 'Irrigation_Prediction.csv')

# ── Load models ───────────────────────────────────────────────
@st.cache_resource(show_spinner='Loading AI models...')
def load_models():
    return {
        'fert'      : joblib.load(os.path.join(MODELS, 'model_fertilizer.pkl')),
        'irr'       : joblib.load(os.path.join(MODELS, 'model_irrigation.pkl')),
        'le_fert'   : joblib.load(os.path.join(MODELS, 'le_fert_target.pkl')),
        'le_soil_f' : joblib.load(os.path.join(MODELS, 'le_fert_soil_type.pkl')),
        'le_crop_f' : joblib.load(os.path.join(MODELS, 'le_fert_crop_type.pkl')),
        'le_season' : joblib.load(os.path.join(MODELS, 'le_fert_season.pkl')),
        'le_soil_i' : joblib.load(os.path.join(MODELS, 'le_irr_soil.pkl')),
        'sc_fert'   : joblib.load(os.path.join(MODELS, 'scaler_fert.pkl')),
        'sc_irr'    : joblib.load(os.path.join(MODELS, 'scaler_irr.pkl')),
        'fert_name' : joblib.load(os.path.join(MODELS, 'best_fert_name.pkl')),
        'irr_name'  : joblib.load(os.path.join(MODELS, 'best_irr_name.pkl')),
    }

m = load_models()

# ── Constants ─────────────────────────────────────────────────
SOIL_F  = sorted(m['le_soil_f'].classes_.tolist())
CROP_F  = sorted(m['le_crop_f'].classes_.tolist())
SEASON  = sorted(m['le_season'].classes_.tolist())
SOIL_I  = sorted(m['le_soil_i'].classes_.tolist())

FERT_INFO = {
    'Urea'        : {'npk':'46-0-0',  'color':'#1565C0','bg':'#E3F2FD',
                     'use':'Best for leafy growth. High in nitrogen. Apply during vegetative stage.',
                     'icon':'🟦'},
    'DAP'         : {'npk':'18-46-0', 'color':'#E65100','bg':'#FFF3E0',
                     'use':'Best for root and flower development. High in phosphorus.',
                     'icon':'🟧'},
    'MOP'         : {'npk':'0-0-60',  'color':'#2E7D32','bg':'#E8F5E9',
                     'use':'Improves fruit quality and disease resistance. High potassium.',
                     'icon':'🟩'},
    'Compost'     : {'npk':'Organic', 'color':'#4E342E','bg':'#EFEBE9',
                     'use':'Improves soil structure, water retention and microbial activity.',
                     'icon':'🟫'},
    'Zinc Sulphate':{'npk':'Zn+S',   'color':'#6A1B9A','bg':'#F3E5F5',
                     'use':'Corrects zinc deficiency. Essential for enzyme activity in plants.',
                     'icon':'🟪'},
    'NPK'         : {'npk':'Various', 'color':'#00695C','bg':'#E0F2F1',
                     'use':'Balanced macro-nutrition for all crop growth stages.',
                     'icon':'🔵'},
    'SSP'         : {'npk':'0-16-0',  'color':'#AD1457','bg':'#FCE4EC',
                     'use':'Provides phosphorus and sulphur. Good for oilseeds and pulses.',
                     'icon':'🔴'},
}

IRR_CFG = {
    0: ('No Irrigation Needed',              '#4CAF50','#0D2F0D'),
    1: ('Irrigation Needed — Turn Pump ON',  '#EF5350','#2D0A0A'),
    2: ('Excess Water — Stop Irrigation',    '#FFA726','#2D1A00'),
}

# ── Global Plotly dark template ───────────────────────────────
DARK = dict(
    paper_bgcolor='#1A2738',
    plot_bgcolor ='#1A2738',
    font         =dict(color='#E8EDF2', size=12),
    title        =dict(font=dict(color='#FFFFFF', size=14)),
    xaxis        =dict(color='#90A4AE', gridcolor='#263647', linecolor='#37474F'),
    yaxis        =dict(color='#90A4AE', gridcolor='#263647', linecolor='#37474F'),
    legend       =dict(bgcolor='#111D2B', font=dict(color='#CFD8DC')),
    margin       =dict(t=50, b=20, l=10, r=10),
)

# ── Session state ─────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []

# ── Global CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  /* ── Root palette ── */
  :root {
    --bg:        #0C1420;
    --surface:   #131E2B;
    --surface2:  #1A2638;
    --border:    #1F2F40;
    --border2:   #263B4F;
    --green:     #3D9970;
    --green-dim: #1E4D38;
    --green-txt: #6FCF97;
    --accent:    #56B2F4;
    --text:      #D8E4EE;
    --text-dim:  #7A99B0;
    --text-muted:#425A6B;
  }

  /* ── Base ── */
  .stApp { background: var(--bg); font-family: 'Inter', sans-serif; }
  .main .block-container { background: var(--bg); padding-top: 1.2rem; max-width: 1300px; }
  .stApp, .stApp p, .stApp span, .stApp div,
  .stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: var(--text); }

  /* ── KPI stat cards ── */
  .kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px;
    position: relative;
    overflow: hidden;
    margin-bottom: 8px;
  }
  .kpi-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--green);
  }
  .kpi-label {
    font-size: 0.63rem; color: var(--text-muted); font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
  }
  .kpi-value { font-size: 1.55rem; color: #E8F0F8; font-weight: 700; margin-top: 3px; line-height: 1; }
  .kpi-sub   { font-size: 0.7rem; color: var(--green-txt); font-weight: 500; margin-top: 3px; }

  /* ── Input panel wrapper ── */
  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 12px;
  }
  .panel-header {
    font-size: 0.82rem; font-weight: 600; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: 0.07em;
    padding-bottom: 12px; margin-bottom: 14px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 8px;
  }
  .panel-dot {
    width: 7px; height: 7px; border-radius: 50%;
    display: inline-block; flex-shrink: 0;
  }

  /* ── Result card ── */
  .result-card {
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: 12px;
    padding: 24px 20px;
    text-align: center;
    margin-bottom: 10px;
    position: relative;
  }
  .result-label {
    font-size: 0.72rem; font-weight: 600; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 10px;
  }
  .result-name {
    font-size: 2rem; font-weight: 700; color: #E8F0F8;
    line-height: 1.1; margin: 6px 0;
  }
  .result-badge {
    display: inline-block;
    padding: 3px 12px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600;
    margin: 8px 0;
  }
  .result-desc {
    font-size: 0.82rem; color: var(--text-dim);
    line-height: 1.6; margin-top: 10px;
  }
  .result-conf {
    font-size: 0.75rem; color: var(--text-muted);
    margin-top: 12px;
  }
  .conf-bar-wrap {
    background: var(--border); border-radius: 4px;
    height: 5px; margin-top: 5px; overflow: hidden;
  }
  .conf-bar-fill {
    height: 100%; border-radius: 4px;
    background: var(--green);
  }

  /* ── Section label ── */
  .sec-label {
    font-size: 0.72rem; font-weight: 600; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.08em;
    margin: 22px 0 10px 0;
  }

  /* ── Input widget labels ── */
  label[data-testid="stWidgetLabel"] p,
  .stSlider label, .stSelectbox label, .stNumberInput label {
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    color: var(--text-dim) !important;
  }

  /* ── Slider ── */
  .stSlider > div > div > div > div { background: var(--green) !important; }

  /* ── Selectbox / number input ── */
  .stSelectbox > div > div,
  .stNumberInput > div > div > input {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 7px !important;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
  }
  .stTabs [data-baseweb="tab"] {
    color: var(--text-dim) !important;
    font-weight: 500;
    font-size: 0.86rem;
    border-radius: 8px;
    padding: 7px 16px;
  }
  .stTabs [aria-selected="true"] {
    background: var(--green-dim) !important;
    color: var(--green-txt) !important;
    font-weight: 600;
  }

  /* ── Primary button ── */
  .stButton > button[kind="primary"] {
    background: var(--green-dim);
    color: var(--green-txt) !important;
    font-size: 0.95rem;
    font-weight: 600;
    padding: 13px 20px;
    border-radius: 8px;
    border: 1px solid var(--green) !important;
    width: 100%;
    letter-spacing: 0.03em;
    transition: background 0.15s;
  }
  .stButton > button[kind="primary"]:hover {
    background: #245C3A !important;
  }

  /* ── Secondary / small buttons ── */
  .stButton > button {
    background: var(--surface) !important;
    color: var(--text-dim) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 7px !important;
    font-size: 0.79rem !important;
    font-weight: 500 !important;
  }
  .stButton > button:hover {
    border-color: var(--green) !important;
    color: var(--green-txt) !important;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #090F18 !important;
    border-right: 1px solid var(--border) !important;
  }
  [data-testid="stSidebar"] * { color: #8AA4BA !important; }

  /* ── Sidebar section header ── */
  .sb-section {
    font-size: 0.65rem; font-weight: 700; color: var(--text-muted) !important;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin: 16px 0 8px 0;
  }
  .sb-row {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 6px;
    font-size: 0.82rem;
  }
  .sb-name { color: #B8D0E0 !important; font-weight: 600; }
  .sb-ht   { color: var(--text-muted) !important; font-size: 0.74rem; }

  /* ── Metric chips ── */
  .chip {
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.74rem;
    color: var(--text-dim) !important;
    margin: 2px 2px 2px 0;
  }
  .chip-green {
    background: var(--green-dim);
    border-color: var(--green);
    color: var(--green-txt) !important;
  }

  /* ── Tables ── */
  .stDataFrame { background: var(--surface); border-radius: 8px; border: 1px solid var(--border); }

  /* ── Info / success ── */
  .stInfo    { background: #0A1929 !important; color: #82B1FF !important;
               border: 1px solid #1A3A5C !important; border-radius: 8px; }
  .stSuccess { background: #071D12 !important; color: var(--green-txt) !important;
               border: 1px solid var(--green-dim) !important; border-radius: 8px; }

  /* ── Progress bar ── */
  .stProgress > div > div { background: var(--green) !important; }

  /* ── Download button ── */
  .stDownloadButton > button {
    background: var(--surface) !important;
    color: var(--green-txt) !important;
    border: 1px solid var(--green-dim) !important;
    border-radius: 7px; font-weight: 500; font-size: 0.85rem;
  }

  /* ── Markdown body ── */
  .stMarkdown p, .stMarkdown li { color: var(--text-dim) !important; }
  hr { border-color: var(--border) !important; }

  /* ── Hide Streamlit chrome ── */
  #MainMenu { visibility: hidden; }
  footer    { visibility: hidden; }
  header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════
st.markdown("""
<div style="display:flex;align-items:center;
            padding:14px 4px 14px 4px;
            border-bottom:1px solid #1F2F40;margin-bottom:20px;">
  <div style="display:flex;align-items:center;gap:14px;">
    <div style="background:#1E4D38;border:1px solid #3D9970;border-radius:9px;
                width:38px;height:38px;display:flex;align-items:center;
                justify-content:center;font-size:1.3rem;flex-shrink:0;">🌱</div>
    <div>
      <div style="color:#D8E4EE;font-size:1.25rem;font-weight:700;
                  letter-spacing:-0.4px;line-height:1.2;">AgriAI</div>
      <div style="color:#425A6B;font-size:0.78rem;margin-top:1px;">
        Smart Farming · Fertilizer &amp; Irrigation Recommendation
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Top KPI row ───────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.markdown('<div class="kpi-card"><div class="kpi-label">Fertilizer Dataset</div><div class="kpi-value">10,000</div><div class="kpi-sub">rows · 12 features</div></div>', unsafe_allow_html=True)
k2.markdown('<div class="kpi-card"><div class="kpi-label">Irrigation Dataset</div><div class="kpi-value">16,411</div><div class="kpi-sub">rows · 4 features</div></div>', unsafe_allow_html=True)
k3.markdown(f'<div class="kpi-card"><div class="kpi-label">Best Fert. Model</div><div class="kpi-value">{m["fert_name"].split()[0]}</div><div class="kpi-sub">{m["fert_name"]}</div></div>', unsafe_allow_html=True)
k4.markdown(f'<div class="kpi-card"><div class="kpi-label">Best Irr. Model</div><div class="kpi-value">{m["irr_name"].split()[0]}</div><div class="kpi-sub">{m["irr_name"]}</div></div>', unsafe_allow_html=True)
k5.markdown(f'<div class="kpi-card"><div class="kpi-label">Predictions Made</div><div class="kpi-value">{len(st.session_state.history)}</div><div class="kpi-sub">this session</div></div>', unsafe_allow_html=True)

st.markdown('<br>', unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 4px 16px 4px;border-bottom:1px solid #1F2F40;margin-bottom:16px;">
      <div style="display:flex;align-items:center;gap:10px;">
        <div style="background:#1E4D38;border:1px solid #3D9970;border-radius:7px;
                    width:30px;height:30px;display:flex;align-items:center;
                    justify-content:center;font-size:1rem;">🌱</div>
        <div>
          <div style="color:#B8D0E0;font-weight:700;font-size:0.95rem;">AgriAI</div>
          <div style="color:#425A6B;font-size:0.72rem;">Smart Farming System</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Model performance ──────────────────────────────────────
    st.markdown('<div class="sb-section">Model Performance</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-row">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <span style="color:#8AA4BA;font-size:0.8rem;">Fertilizer</span>
        <span style="color:#6FCF97;font-weight:700;font-size:0.85rem;">86.05%</span>
      </div>
      <div style="color:#425A6B;font-size:0.72rem;margin-top:2px;">Random Forest · 10K rows · 7 classes</div>
    </div>
    <div class="sb-row">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <span style="color:#8AA4BA;font-size:0.8rem;">Irrigation</span>
        <span style="color:#6FCF97;font-weight:700;font-size:0.85rem;">90.45%</span>
      </div>
      <div style="color:#425A6B;font-size:0.72rem;margin-top:2px;">XGBoost · 16K rows · 3 classes</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Techniques ────────────────────────────────────────────
    st.markdown('<div class="sb-section">Techniques</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex;flex-wrap:wrap;gap:4px;padding:2px 0 8px 0;">
      <span class="chip chip-green">SMOTE</span>
      <span class="chip chip-green">RandomizedSearchCV</span>
      <span class="chip">MinMax Scaling</span>
      <span class="chip">Label Encoding</span>
      <span class="chip">Stratified Split</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Team ──────────────────────────────────────────────────
    st.markdown('<div class="sb-section">Team — AIML-D</div>', unsafe_allow_html=True)
    for name, ht in [
        ('C. Jasvitha Reddy', '23EG107D10'),
        ('Md Shoaib',         '23EG107D42'),
        ('G. Sumanth Goud',   '23EG107D24'),
    ]:
        st.markdown(f"""
        <div class="sb-row">
          <div class="sb-name">{name}</div>
          <div class="sb-ht">{ht}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('<div style="color:#2A3D4F;font-size:0.72rem;padding:6px 0 2px 0;">Anurag University, Hyderabad</div>', unsafe_allow_html=True)

    # ── Gemini ────────────────────────────────────────────────
    st.markdown('<div class="sb-section">AI Assistant</div>', unsafe_allow_html=True)
    # Load Gemini key from Streamlit secrets (cloud) or env variable (local)
    try:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    except Exception:
        import os as _os
        GEMINI_API_KEY = _os.environ.get("GEMINI_API_KEY", "")
    st.session_state['gemini_key'] = GEMINI_API_KEY
    st.markdown("""
    <div style="background:#071D12;border:1px solid #1E4D38;border-radius:7px;
                padding:7px 12px;font-size:0.78rem;color:#6FCF97;">
      ● Gemini 3.1 active — real-time answers on
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown('<br>', unsafe_allow_html=True)
        if st.button('Clear History', use_container_width=True):
            st.session_state.history = []
            st.rerun()

# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    'Recommendations',
    'Data Explorer',
    'Model Results',
    'History',
    'AI Assistant',
    'About',
])

# ════════════════════════════════════════════════════════════════
# TAB 1 — GET RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════
with tab1:

    col_f, col_i = st.columns(2, gap='large')

    # ─── FERTILIZER INPUTS ────────────────────────────────────
    with col_f:
        st.markdown("""
        <div class="panel">
          <div class="panel-header">
            <span class="panel-dot" style="background:#56B2F4;"></span>
            Fertilizer Inputs
            <span style="margin-left:auto;font-size:0.68rem;color:#425A6B;font-weight:400;text-transform:none;letter-spacing:0;">
              12 features · 7 fertilizer types
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        fc1, fc2 = st.columns(2)
        with fc1:
            soil_f  = st.selectbox('Soil Type',  SOIL_F)
            crop_f  = st.selectbox('Crop Type',  CROP_F)
            season  = st.selectbox('Season', SEASON)
        with fc2:
            soil_ph    = st.number_input('Soil pH',          min_value=4.0,  max_value=9.0,  value=6.5,  step=0.1)
            org_carbon = st.number_input('Organic Carbon',   min_value=0.1,  max_value=3.0,  value=0.8,  step=0.1)
            rainfall   = st.number_input('Rainfall (mm)',    min_value=200,  max_value=3000, value=1000, step=50)

        st.markdown('<div class="sec-label">Soil Nutrients — kg/ha</div>', unsafe_allow_html=True)
        n1, n2, n3 = st.columns(3)
        with n1: nitro = st.slider('Nitrogen (N)',   0, 120, 60)
        with n2: phos  = st.slider('Phosphorus (P)', 0, 100, 50)
        with n3: pota  = st.slider('Potassium (K)',  0, 200, 80)

        st.markdown('<div class="sec-label">Weather & Soil Conditions</div>', unsafe_allow_html=True)
        w1, w2, w3 = st.columns(3)
        with w1: temp_f  = st.slider('Temperature (°C)',  10, 45,  28)
        with w2: hum_f   = st.slider('Humidity (%)',      20, 100, 65)
        with w3: moist_f = st.slider('Soil Moisture (%)', 10, 80,  40)

    # ─── IRRIGATION INPUTS ────────────────────────────────────
    with col_i:
        st.markdown("""
        <div class="panel">
          <div class="panel-header">
            <span class="panel-dot" style="background:#3D9970;"></span>
            Irrigation Inputs
            <span style="margin-left:auto;font-size:0.68rem;color:#425A6B;font-weight:400;text-transform:none;letter-spacing:0;">
              4 features · 3 classes
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        soil_i = st.selectbox('Soil Type', SOIL_I, key='soil_irr')
        moi    = st.slider('Moisture Index (MOI)  —  0 = dry · 100 = saturated', 0, 100, 40)
        temp_i = st.slider('Temperature (°C)', 15, 45, 30, key='ti')
        hum_i  = st.slider('Humidity (%)',     20, 100, 60, key='hi')

        # MOI status strip
        moi_color  = '#C62828' if moi < 35 else ('#F9A825' if moi < 65 else '#2E7D32')
        moi_label  = 'Dry — Irrigate' if moi < 35 else ('Moderate — Monitor' if moi < 65 else 'Wet — No Irrigation')
        moi_pct    = int(moi)
        st.markdown(f"""
        <div style="background:#131E2B;border:1px solid #1F2F40;border-radius:9px;
                    padding:14px 16px;margin-top:8px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <span style="font-size:0.72rem;color:#425A6B;font-weight:600;
                         text-transform:uppercase;letter-spacing:0.07em;">MOI Reading</span>
            <span style="font-size:0.85rem;font-weight:700;color:{moi_color};">{moi_label}</span>
          </div>
          <div style="background:#1F2F40;border-radius:4px;height:6px;overflow:hidden;">
            <div style="width:{moi_pct}%;height:100%;background:{moi_color};
                        border-radius:4px;transition:width 0.3s;"></div>
          </div>
          <div style="display:flex;justify-content:space-between;margin-top:6px;">
            <span style="font-size:0.68rem;color:#C62828;">0–34 Irrigate</span>
            <span style="font-size:0.68rem;color:#F9A825;">35–64 Monitor</span>
            <span style="font-size:0.68rem;color:#2E7D32;">65–100 No Irr.</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Live MOI Gauge
        gauge_color = '#C62828' if moi < 35 else ('#F9A825' if moi < 65 else '#2E7D32')
        fig_g = go.Figure(go.Indicator(
            mode  = 'gauge+number',
            value = moi,
            title = {'text': 'Live Soil Moisture (MOI)', 'font': {'size': 12, 'color': '#7A99B0'}},
            gauge = {
                'axis'     : {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#425A6B'},
                'bar'      : {'color': gauge_color, 'thickness': 0.28},
                'bgcolor'  : '#131E2B',
                'bordercolor': '#1F2F40',
                'steps'    : [
                    {'range': [0,  35], 'color': '#1A0808'},
                    {'range': [35, 65], 'color': '#1A1500'},
                    {'range': [65,100], 'color': '#071D12'},
                ],
                'threshold': {'line': {'color': gauge_color, 'width': 2},
                              'thickness': 0.75, 'value': moi},
            },
            number = {'font': {'color': '#D8E4EE', 'size': 28}},
        ))
        fig_g.update_layout(
            height=180, margin=dict(t=36, b=0, l=24, r=24),
            paper_bgcolor='#131E2B', plot_bgcolor='#131E2B',
            font={'color': '#7A99B0'},
        )
        st.plotly_chart(fig_g, use_container_width=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── PREDICT BUTTON ────────────────────────────────────────
    pred_btn = st.button('Get Recommendations',
                         use_container_width=True, type='primary')

    if pred_btn:
        # ── Fertilizer prediction ──────────────────────────────
        def safe_encode(le, val, default=0):
            try:    return int(le.transform([val])[0])
            except: return default

        s_enc  = safe_encode(m['le_soil_f'], soil_f)
        c_enc  = safe_encode(m['le_crop_f'], crop_f)
        se_enc = safe_encode(m['le_season'], season)

        fi = m['sc_fert'].transform(np.array([[
            s_enc, soil_ph, moist_f, org_carbon,
            nitro, phos, pota,
            temp_f, hum_f, rainfall,
            c_enc, se_enc
        ]]))
        fp     = m['fert'].predict(fi)[0]
        fn     = m['le_fert'].inverse_transform([fp])[0]
        fproba = (m['fert'].predict_proba(fi)[0]
                  if hasattr(m['fert'], 'predict_proba') else None)
        fc     = float(np.max(fproba)) if fproba is not None else 1.0
        finfo  = FERT_INFO.get(fn, {'npk':'—','use':'—',
                                     'color':'#1D2E5C','bg':'#F5F5F5','icon':'🌿'})

        # ── Irrigation prediction ──────────────────────────────
        si_enc = safe_encode(m['le_soil_i'], soil_i)
        ii     = m['sc_irr'].transform(np.array([[moi, temp_i, hum_i, si_enc]]))
        ip     = m['irr'].predict(ii)[0]
        iproba = (m['irr'].predict_proba(ii)[0]
                  if hasattr(m['irr'], 'predict_proba') else None)
        ic     = float(np.max(iproba)) if iproba is not None else 1.0
        itxt, iclr, ibg = IRR_CFG[ip]

        # ── Save to history ────────────────────────────────────
        st.session_state.history.append({
            'Time'      : datetime.datetime.now().strftime('%H:%M:%S'),
            'Soil'      : soil_f, 'Crop': crop_f, 'Season': season,
            'N': nitro, 'P': phos, 'K': pota,
            'Fertilizer': fn, 'Fert Conf.': f'{fc*100:.1f}%',
            'MOI'       : moi,
            'Irrigation': itxt.replace('✅','').replace('💧','').replace('⚠️','').strip(),
            'Irr. Conf.': f'{ic*100:.1f}%',
        })

        # ════════════════════════════════════════════════════
        # RESULTS
        # ════════════════════════════════════════════════════
        st.markdown('---')
        st.markdown('#### Recommendations')

        r1, r2 = st.columns(2, gap='large')

        with r1:
            conf_pct = int(fc * 100)
            st.markdown(f"""
            <div class="result-card" style="border-top:3px solid {finfo['color']};">
              <div class="result-label">Recommended Fertilizer</div>
              <div class="result-name">{fn}</div>
              <div class="result-badge" style="background:{finfo['color']}22;
                   border:1px solid {finfo['color']}55;color:{finfo['color']};">
                NPK {finfo['npk']}
              </div>
              <div class="result-desc">{finfo['use']}</div>
              <div class="result-conf">
                Model confidence
                <div class="conf-bar-wrap">
                  <div class="conf-bar-fill" style="width:{conf_pct}%;background:{finfo['color']};"></div>
                </div>
                <div style="text-align:right;margin-top:3px;font-size:0.78rem;
                            color:{finfo['color']};font-weight:600;">{conf_pct}%</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            conf_pct2 = int(ic * 100)
            st.markdown(f"""
            <div class="result-card" style="border-top:3px solid {iclr};">
              <div class="result-label">Irrigation Status</div>
              <div style="font-size:1.45rem;font-weight:700;color:{iclr};
                          margin:8px 0;line-height:1.2;">{itxt}</div>
              <div class="result-badge" style="background:{iclr}22;
                   border:1px solid {iclr}55;color:{iclr};">
                MOI {moi} &nbsp;·&nbsp; {temp_i}°C &nbsp;·&nbsp; {hum_i}% RH
              </div>
              <div class="result-conf">
                Model confidence
                <div class="conf-bar-wrap">
                  <div class="conf-bar-fill" style="width:{conf_pct2}%;background:{iclr};"></div>
                </div>
                <div style="text-align:right;margin-top:3px;font-size:0.78rem;
                            color:{iclr};font-weight:600;">{conf_pct2}%</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Charts row ─────────────────────────────────────────
        st.markdown('#### Charts')

        ch1, ch2 = st.columns(2, gap='large')

        # Chart 1 — NPK comparison
        with ch1:
            ideal_npk = {
                'Urea':        [90,30,40], 'DAP':         [30,80,40],
                'MOP':         [40,30,90], 'Compost':     [25,25,30],
                'Zinc Sulphate':[50,40,50],'NPK':         [60,60,60],
                'SSP':         [40,70,35],
            }
            id_vals = ideal_npk.get(fn, [60,50,80])
            fig_npk = go.Figure()
            fig_npk.add_trace(go.Bar(
                name='Your Input', x=['Nitrogen','Phosphorus','Potassium'],
                y=[nitro,phos,pota],
                marker_color=['#4472C4','#ED7D31','#70AD47'],
                text=[f'{v}' for v in [nitro,phos,pota]],
                textposition='outside', textfont_size=11,
            ))
            fig_npk.add_trace(go.Bar(
                name=f'Ideal for {fn}',
                x=['Nitrogen','Phosphorus','Potassium'],
                y=id_vals,
                marker_color=['rgba(68,114,196,0.25)',
                              'rgba(237,125,49,0.25)',
                              'rgba(112,173,71,0.25)'],
                text=[f'{v}' for v in id_vals],
                textposition='outside', textfont_size=11,
            ))
            fig_npk.update_layout(
                barmode='group', height=300,
                title=dict(text=f'Your NPK vs Ideal for {fn}', font=dict(color='#FFFFFF',size=13)),
                yaxis_title='kg/ha',
                yaxis_range=[0,max(max(nitro,phos,pota),max(id_vals))*1.3],
                legend=dict(orientation='h',yanchor='bottom',y=1.02,
                            bgcolor='#111D2B',font=dict(color='#CFD8DC')),
                margin=dict(t=60,b=10,l=10,r=10),
                paper_bgcolor='#1A2738',
                plot_bgcolor='#1A2738',
                font=dict(color='#E8EDF2'),
                xaxis=dict(color='#90A4AE'),
                yaxis=dict(color='#90A4AE', gridcolor='#263647'),
            )
            st.plotly_chart(fig_npk, use_container_width=True)

        # Chart 2 — Fertilizer probabilities
        with ch2:
            if fproba is not None:
                classes  = m['le_fert'].classes_
                clrs_bar = [finfo['color'] if c==fn else '#B0BEC5' for c in classes]
                fig_prob = go.Figure(go.Bar(
                    x=[f'{p*100:.1f}%' for p in fproba],
                    y=classes,
                    orientation='h',
                    marker_color=clrs_bar,
                    text=[f'{p*100:.1f}%' for p in fproba],
                    textposition='outside',
                    textfont_size=10,
                ))
                fig_prob.update_layout(
                    title=dict(text='Probability — All Fertilizer Types',
                               font=dict(color='#FFFFFF',size=13)),
                    height=300,
                    xaxis=dict(range=[0,120], showticklabels=False,
                               color='#90A4AE', gridcolor='#263647'),
                    yaxis=dict(color='#E8EDF2'),
                    margin=dict(t=50,b=10,l=10,r=70),
                    paper_bgcolor='#1A2738',
                    plot_bgcolor='#1A2738',
                    font=dict(color='#E8EDF2'),
                )
                st.plotly_chart(fig_prob, use_container_width=True)

        ch3, ch4 = st.columns(2, gap='large')

        # Chart 3 — Soil health radar
        with ch3:
            radar_labels = ['Nitrogen','Phosphorus','Potassium',
                            'Moisture','Humidity','Rainfall_n']
            radar_vals   = [
                nitro/120*100,
                phos/100*100,
                pota/200*100,
                moist_f/80*100,
                hum_f,
                min(rainfall/3000*100, 100),
            ]
            angles = np.linspace(0, 2*np.pi, len(radar_labels), endpoint=False).tolist()
            angles += angles[:1]
            rv     = radar_vals + [radar_vals[0]]
            fig_r  = go.Figure(go.Scatterpolar(
                r=rv, theta=radar_labels+[radar_labels[0]],
                fill='toself',
                fillcolor=f'rgba(68,114,196,0.15)',
                line=dict(color='#4472C4', width=2),
                marker=dict(size=6, color='#4472C4'),
                name='Your Soil',
            ))
            fig_r.update_layout(
                polar=dict(
                    bgcolor='#1A2738',
                    radialaxis=dict(visible=True, range=[0,100],
                                    color='#90A4AE', gridcolor='#263647'),
                    angularaxis=dict(color='#CFD8DC'),
                ),
                title=dict(text='Soil Health Radar', font=dict(color='#FFFFFF',size=13), pad=dict(t=20)),
                height=290,
                margin=dict(t=50,b=20,l=30,r=30),
                paper_bgcolor='#1A2738',
                font=dict(color='#E8EDF2'),
                showlegend=False,
            )
            st.plotly_chart(fig_r, use_container_width=True)

        # Chart 4 — Irrigation donut
        with ch4:
            if iproba is not None:
                irr_lbls  = ['No Irrigation','Irrigation Needed','Excess Water']
                irr_clrs  = ['#2E7D32','#C62828','#E65100']
                fig_donut = go.Figure(go.Pie(
                    labels=irr_lbls,
                    values=iproba*100,
                    hole=0.55,
                    marker_colors=irr_clrs,
                    textinfo='label+percent',
                    textfont_size=11,
                    pull=[0.05 if i==ip else 0 for i in range(3)],
                ))
                fig_donut.update_layout(
                    title=dict(text='Irrigation Class Probabilities',
                               font=dict(color='#FFFFFF',size=13)),
                    height=290,
                    margin=dict(t=50,b=10,l=10,r=10),
                    paper_bgcolor='#1A2738',
                    font=dict(color='#E8EDF2'),
                    legend=dict(orientation='h',yanchor='bottom',y=-0.2,
                                font=dict(color='#CFD8DC'),bgcolor='#111D2B'),
                )
                st.plotly_chart(fig_donut, use_container_width=True)

        # ── Input summary ──────────────────────────────────────
        st.markdown('<div class="section-title">📋 Input Summary</div>', unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        with s1:
            st.dataframe(pd.DataFrame({
                'Parameter' : ['Soil Type','Crop Type','Season','Soil pH',
                               'Organic Carbon','Nitrogen','Phosphorus',
                               'Potassium','Temperature','Humidity',
                               'Soil Moisture','Rainfall'],
                'Value'     : [soil_f, crop_f, season,
                               f'{soil_ph:.1f}', f'{org_carbon:.1f}',
                               f'{nitro} kg/ha', f'{phos} kg/ha',
                               f'{pota} kg/ha',  f'{temp_f} °C',
                               f'{hum_f} %',     f'{moist_f} %',
                               f'{rainfall} mm'],
            }), use_container_width=True, hide_index=True)
        with s2:
            st.dataframe(pd.DataFrame({
                'Parameter': ['Soil Type','MOI','Temperature','Humidity'],
                'Value'    : [soil_i, str(moi),
                              f'{temp_i} °C', f'{hum_i} %'],
            }), use_container_width=True, hide_index=True)

        # ── Download report ────────────────────────────────────
        st.markdown('<div class="section-title">📥 Download Report</div>', unsafe_allow_html=True)
        report = (
            f"AgriAI — Recommendation Report\n"
            f"Generated: {datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n"
            f"{'='*50}\n\n"
            f"FERTILIZER RECOMMENDATION\n"
            f"  Fertilizer   : {fn}\n"
            f"  NPK Ratio    : {finfo['npk']}\n"
            f"  Usage Notes  : {finfo['use']}\n"
            f"  Model        : {m['fert_name']}\n"
            f"  Confidence   : {fc*100:.1f}%\n\n"
            f"IRRIGATION STATUS\n"
            f"  Status       : {itxt}\n"
            f"  Model        : {m['irr_name']}\n"
            f"  Confidence   : {ic*100:.1f}%\n\n"
            f"INPUT DETAILS\n"
            f"  Soil Type    : {soil_f}\n"
            f"  Crop Type    : {crop_f}\n"
            f"  Season       : {season}\n"
            f"  Soil pH      : {soil_ph}\n"
            f"  Organic C    : {org_carbon}\n"
            f"  Nitrogen     : {nitro} kg/ha\n"
            f"  Phosphorus   : {phos} kg/ha\n"
            f"  Potassium    : {pota} kg/ha\n"
            f"  Temperature  : {temp_f} C\n"
            f"  Humidity     : {hum_f} %\n"
            f"  Soil Moisture: {moist_f} %\n"
            f"  Rainfall     : {rainfall} mm\n"
            f"  MOI          : {moi}\n\n"
            f"{'='*50}\n"
            f"Team: Md Shoaib | C. Jasvitha Reddy | G. Sumanth Goud\n"
            f"Anurag University, Hyderabad — AIML-D\n"
        )
        st.download_button(
            'Download Report (.txt)',
            data=report,
            file_name=f'AgriAI_{datetime.datetime.now().strftime("%d%m%Y_%H%M%S")}.txt',
            mime='text/plain',
            use_container_width=True,
        )


# ════════════════════════════════════════════════════════════════
# TAB 2 — DATA EXPLORER
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">📊 Interactive Data Explorer</div>', unsafe_allow_html=True)

    try:
        fdf = pd.read_csv(FERT_CSV)
        idf = pd.read_csv(IRR_CSV)
        idf = idf.rename(columns={'result':'Status'})
        idf['Status'] = idf['Status'].map({0:'No Irrigation',1:'Irrigation Needed',2:'Excess Water'})

        # Fertilizer section
        st.markdown('#### 🧪 Fertilizer Dataset')
        de1, de2 = st.columns(2)

        with de1:
            fig_p = px.pie(fdf, names='Recommended_Fertilizer',
                           title='Fertilizer Type Distribution',
                           color_discrete_sequence=px.colors.qualitative.Set3,
                           hole=0.35)
            fig_p.update_layout(height=340, margin=dict(t=50,b=10),
                                paper_bgcolor='#1A2738',
                                font=dict(color='#E8EDF2'),
                                title=dict(font=dict(color='#FFFFFF')),
                                legend=dict(font=dict(color='#CFD8DC'),
                                            bgcolor='#111D2B'))
            st.plotly_chart(fig_p, use_container_width=True)

        with de2:
            feat_sel = st.selectbox('Feature to explore:',
                ['Nitrogen_Level','Phosphorus_Level','Potassium_Level',
                 'Soil_pH','Temperature','Humidity','Soil_Moisture','Rainfall'])
            fig_bx = px.box(fdf, x='Recommended_Fertilizer', y=feat_sel,
                            color='Recommended_Fertilizer',
                            title=f'{feat_sel} per Fertilizer Type',
                            color_discrete_sequence=px.colors.qualitative.Plotly)
            fig_bx.update_layout(height=340, showlegend=False,
                                  margin=dict(t=50,b=40),
                                  paper_bgcolor='#1A2738',
                                  plot_bgcolor='#1A2738',
                                  font=dict(color='#E8EDF2'),
                                  title=dict(font=dict(color='#FFFFFF')),
                                  xaxis=dict(color='#CFD8DC',gridcolor='#263647'),
                                  yaxis=dict(color='#CFD8DC',gridcolor='#263647'))
            st.plotly_chart(fig_bx, use_container_width=True)

        de3, de4 = st.columns(2)
        with de3:
            fig_sc = px.scatter(fdf.sample(600), x='Nitrogen_Level',
                                y='Phosphorus_Level',
                                color='Recommended_Fertilizer',
                                size='Potassium_Level',
                                title='N vs P (600 samples, size=K)',
                                opacity=0.7,
                                color_discrete_sequence=px.colors.qualitative.Bold)
            fig_sc.update_layout(height=330, margin=dict(t=50,b=10),
                                  paper_bgcolor='#1A2738',
                                  plot_bgcolor='#1A2738',
                                  font=dict(color='#E8EDF2'),
                                  title=dict(font=dict(color='#FFFFFF')),
                                  xaxis=dict(color='#CFD8DC',gridcolor='#263647'),
                                  yaxis=dict(color='#CFD8DC',gridcolor='#263647'),
                                  legend=dict(font=dict(color='#CFD8DC'),bgcolor='#111D2B'))
            st.plotly_chart(fig_sc, use_container_width=True)

        with de4:
            fig_hist = px.histogram(fdf, x='Soil_pH',
                                    color='Recommended_Fertilizer',
                                    barmode='overlay',
                                    title='Soil pH Distribution per Fertilizer',
                                    opacity=0.65,
                                    color_discrete_sequence=px.colors.qualitative.Plotly)
            fig_hist.update_layout(height=330, margin=dict(t=50,b=10),
                                    paper_bgcolor='#1A2738',
                                    plot_bgcolor='#1A2738',
                                    font=dict(color='#E8EDF2'),
                                    title=dict(font=dict(color='#FFFFFF')),
                                    xaxis=dict(color='#CFD8DC',gridcolor='#263647'),
                                    yaxis=dict(color='#CFD8DC',gridcolor='#263647'),
                                    legend=dict(font=dict(color='#CFD8DC'),bgcolor='#111D2B'))
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown('---')
        st.markdown('#### 💧 Irrigation Dataset')
        di1, di2 = st.columns(2)

        with di1:
            fig_ip = px.pie(idf, names='Status',
                            title='Irrigation Status Distribution',
                            color_discrete_map={
                                'No Irrigation':'#4CAF50',
                                'Irrigation Needed':'#EF5350',
                                'Excess Water':'#FFA726'
                            }, hole=0.35)
            fig_ip.update_layout(height=340, margin=dict(t=50,b=10),
                                  paper_bgcolor='#1A2738',
                                  font=dict(color='#E8EDF2'),
                                  title=dict(font=dict(color='#FFFFFF')),
                                  legend=dict(font=dict(color='#CFD8DC'),bgcolor='#111D2B'))
            st.plotly_chart(fig_ip, use_container_width=True)

        with di2:
            fig_mb = px.box(idf, x='Status', y='MOI', color='Status',
                            title='MOI Distribution per Irrigation Status',
                            color_discrete_map={
                                'No Irrigation':'#4CAF50',
                                'Irrigation Needed':'#EF5350',
                                'Excess Water':'#FFA726'
                            })
            fig_mb.update_layout(height=340, showlegend=False,
                                  margin=dict(t=50,b=10),
                                  paper_bgcolor='#1A2738',
                                  plot_bgcolor='#1A2738',
                                  font=dict(color='#E8EDF2'),
                                  title=dict(font=dict(color='#FFFFFF')),
                                  xaxis=dict(color='#CFD8DC'),
                                  yaxis=dict(color='#CFD8DC',gridcolor='#263647'))
            st.plotly_chart(fig_mb, use_container_width=True)

        # 3D scatter
        st.markdown('#### 🌐 3D View — Irrigation Dataset')
        samp3d = idf.sample(min(1500, len(idf)))
        fig_3d = px.scatter_3d(samp3d, x='MOI', y='temp', z='humidity',
                               color='Status',
                               color_discrete_map={
                                   'No Irrigation':'#4CAF50',
                                   'Irrigation Needed':'#EF5350',
                                   'Excess Water':'#FFA726'
                               },
                               opacity=0.65, size_max=3,
                               title='MOI vs Temperature vs Humidity (1500 samples)')
        fig_3d.update_layout(height=520, margin=dict(t=50,b=10),
                              paper_bgcolor='#1A2738',
                              font=dict(color='#E8EDF2'),
                              title=dict(font=dict(color='#FFFFFF')),
                              scene=dict(
                                  bgcolor='#1A2738',
                                  xaxis=dict(color='#CFD8DC',gridcolor='#263647',
                                             backgroundcolor='#1A2738'),
                                  yaxis=dict(color='#CFD8DC',gridcolor='#263647',
                                             backgroundcolor='#1A2738'),
                                  zaxis=dict(color='#CFD8DC',gridcolor='#263647',
                                             backgroundcolor='#1A2738'),
                              ),
                              legend=dict(font=dict(color='#CFD8DC'),bgcolor='#111D2B'))
        st.plotly_chart(fig_3d, use_container_width=True)

    except Exception as e:
        st.error(f'Could not load raw data: {e}')


# ════════════════════════════════════════════════════════════════
# TAB 3 — MODEL RESULTS
# ════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">📈 Model Performance Results</div>', unsafe_allow_html=True)

    # Final results table + chart
    rcsv = os.path.join(RESULTS, 'final_results.csv')
    if os.path.exists(rcsv):
        df_r = pd.read_csv(rcsv)
        st.markdown('#### Final Accuracy — Both Modules')

        fig_fin = go.Figure()
        colors4 = ['#4472C4','#ED7D31','#70AD47','#E53935']
        for metric, color in zip(['Accuracy','Precision','Recall','F1 Score'], colors4):
            fig_fin.add_trace(go.Bar(
                name=metric,
                x=df_r['Module'],
                y=df_r[metric].astype(float),
                marker_color=color,
                text=[f'{v:.3f}' for v in df_r[metric].astype(float)],
                textposition='outside', textfont_size=11,
            ))
        fig_fin.update_layout(
            barmode='group', height=380,
            title=dict(text='Final Evaluation — Both Modules',
                       font=dict(color='#FFFFFF',size=14)),
            yaxis=dict(range=[0,1.12], color='#90A4AE', gridcolor='#263647'),
            xaxis=dict(color='#E8EDF2'),
            legend=dict(orientation='h',yanchor='bottom',y=1.02,
                        bgcolor='#111D2B',font=dict(color='#CFD8DC')),
            margin=dict(t=70,b=20),
            paper_bgcolor='#1A2738',
            plot_bgcolor='#1A2738',
            font=dict(color='#E8EDF2'),
        )
        st.plotly_chart(fig_fin, use_container_width=True)
        st.dataframe(df_r, use_container_width=True, hide_index=True)

    # Algorithm comparison charts
    st.markdown('---')
    for fname, title in [
        ('fert_comparison.csv','🧪 Fertilizer — 4 Algorithms Compared'),
        ('irr_comparison.csv', '💧 Irrigation — 4 Algorithms Compared'),
    ]:
        p = os.path.join(RESULTS, fname)
        if os.path.exists(p):
            df_c = pd.read_csv(p)
            st.markdown(f'#### {title}')
            df_c['Acc_num'] = df_c['Accuracy'].str.replace('%','').astype(float)
            fig_c = px.bar(df_c, x='Algorithm', y='Acc_num',
                           color='Algorithm', text='Accuracy',
                           labels={'Acc_num':'Test Accuracy (%)'},
                           color_discrete_sequence=['#4472C4','#ED7D31','#70AD47','#E53935'])
            fig_c.update_traces(textposition='outside', textfont_size=12)
            fig_c.update_layout(height=350, showlegend=False,
                                 yaxis=dict(range=[0,108], color='#90A4AE', gridcolor='#263647'),
                                 xaxis=dict(color='#E8EDF2'),
                                 title=dict(font=dict(color='#FFFFFF')),
                                 margin=dict(t=20,b=40),
                                 paper_bgcolor='#1A2738',
                                 plot_bgcolor='#1A2738',
                                 font=dict(color='#E8EDF2'))
            st.plotly_chart(fig_c, use_container_width=True)

    # Static charts
    st.markdown('---')
    st.markdown('#### 🖼️ Generated Result Charts')
    chart_map = {
        'fert_confusion.png' : '🧪 Fertilizer — Confusion Matrix',
        'irr_confusion.png'  : '💧 Irrigation — Confusion Matrix',
        'fert_importance.png': '🧪 Fertilizer — Feature Importance',
        'irr_importance.png' : '💧 Irrigation — Feature Importance',
        'fert_smote.png'     : '🧬 SMOTE — Before vs After',
        'radar.png'          : '🌐 Performance Radar Chart',
        'fert_npk.png'       : '🧪 Fertilizer — NPK by Type',
        'irr_moi.png'        : '💧 Irrigation — MOI Distribution',
    }
    keys = list(chart_map.keys())
    for i in range(0, len(keys), 2):
        c1, c2 = st.columns(2)
        for col, key in zip([c1,c2], keys[i:i+2]):
            img = os.path.join(RESULTS, key)
            if os.path.exists(img):
                col.image(img, caption=chart_map[key],
                          use_container_width=True)
            else:
                col.info(f'Run training scripts → {key}')


# ════════════════════════════════════════════════════════════════
# TAB 4 — HISTORY
# ════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">🕐 Prediction History (This Session)</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.info('No predictions yet. Go to "Get Recommendations" and make a prediction.')
    else:
        df_h = pd.DataFrame(st.session_state.history)
        st.dataframe(df_h, use_container_width=True, hide_index=True)

        if len(df_h) > 1:
            hc1, hc2 = st.columns(2)
            with hc1:
                vc = df_h['Fertilizer'].value_counts().reset_index()
                vc.columns = ['Fertilizer','Count']
                fig_hf = px.bar(vc, x='Fertilizer', y='Count',
                                color='Fertilizer', text='Count',
                                title='Fertilizer Types Recommended',
                                color_discrete_sequence=px.colors.qualitative.Set3)
                fig_hf.update_traces(textposition='outside')
                fig_hf.update_layout(height=320, showlegend=False,
                                      margin=dict(t=40,b=20),
                                      paper_bgcolor='#1A2738',
                                      plot_bgcolor='#1A2738',
                                      font=dict(color='#E8EDF2'),
                                      title=dict(font=dict(color='#FFFFFF')),
                                      xaxis=dict(color='#CFD8DC'),
                                      yaxis=dict(color='#CFD8DC',gridcolor='#263647'))
                st.plotly_chart(fig_hf, use_container_width=True)
            with hc2:
                ic_h = df_h['Irrigation'].value_counts().reset_index()
                ic_h.columns = ['Status','Count']
                fig_hi = px.pie(ic_h, names='Status', values='Count',
                                title='Irrigation Statuses Predicted', hole=0.4,
                                color_discrete_sequence=['#4CAF50','#EF5350','#FFA726'])
                fig_hi.update_layout(height=320, margin=dict(t=40,b=10),
                                      paper_bgcolor='#1A2738',
                                      font=dict(color='#E8EDF2'),
                                      title=dict(font=dict(color='#FFFFFF')),
                                      legend=dict(font=dict(color='#CFD8DC'),bgcolor='#111D2B'))
                st.plotly_chart(fig_hi, use_container_width=True)

        st.download_button(
            'Download History as CSV',
            data=df_h.to_csv(index=False),
            file_name='AgriAI_History.csv',
            mime='text/csv',
            use_container_width=True,
        )


# ════════════════════════════════════════════════════════════════
# TAB 5 — AI ASSISTANT
# ════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">🤖 AgriAI Assistant — Ask Me Anything</div>', unsafe_allow_html=True)

    # ── Session state for chat ────────────────────────────────
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # ── Knowledge base ────────────────────────────────────────
    FERT_DEEP = {
        'Urea': {
            'npk'       : '46-0-0',
            'best_for'  : 'Wheat, Maize, Rice, Sugarcane — crops needing heavy leaf and stem growth',
            'when'      : 'Apply at sowing and 30 days after germination (top dressing)',
            'how_much'  : '100–150 kg/ha for most crops',
            'caution'   : 'Never apply before heavy rainfall — nitrogen leaches away. Split into 2 doses.',
            'soil_fit'  : 'Works in all soil types. Best in clay and loam soils that retain nutrients.',
            'deficiency': 'Yellowing of older leaves (chlorosis), stunted growth, pale green color',
            'excess'    : 'Leaf burn, excessive vegetative growth, delayed flowering',
        },
        'DAP': {
            'npk'       : '18-46-0',
            'best_for'  : 'Wheat, Pulses, Oilseeds — crops needing strong root and flower development',
            'when'      : 'Apply at sowing time — mixed with soil before planting',
            'how_much'  : '100–125 kg/ha',
            'caution'   : 'Do not mix with urea before application — nitrogen loss occurs',
            'soil_fit'  : 'Best for acidic soils. Avoid in alkaline soils above pH 8.',
            'deficiency': 'Purple or reddish discoloration of leaves, weak roots, poor flowering',
            'excess'    : 'Zinc and iron deficiency in soil — micronutrient imbalance',
        },
        'MOP': {
            'npk'       : '0-0-60',
            'best_for'  : 'Potato, Banana, Cotton, Sugarcane — crops needing quality fruit and disease resistance',
            'when'      : 'Apply at sowing or transplanting, with one split dose at flowering',
            'how_much'  : '50–100 kg/ha',
            'caution'   : 'Avoid in saline soils — high chloride content can damage sensitive crops',
            'soil_fit'  : 'All soils. Most effective in potassium-deficient sandy soils.',
            'deficiency': 'Scorched leaf edges (tip burn), weak stems, poor grain filling',
            'excess'    : 'Magnesium and calcium deficiency, salt stress in sensitive crops',
        },
        'Compost': {
            'npk'       : 'Organic — varies',
            'best_for'  : 'All crops — improves soil health for long term',
            'when'      : 'Apply 2–4 weeks before sowing, incorporate into soil',
            'how_much'  : '5–10 tonnes/ha',
            'caution'   : 'Ensure fully decomposed — raw compost causes nitrogen immobilization',
            'soil_fit'  : 'Best for sandy soils with low organic matter. Improves water retention.',
            'deficiency': 'No specific deficiency — general soil degradation over seasons',
            'excess'    : 'No toxicity — but excessive use can raise pH in acidic soils',
        },
        'NPK': {
            'npk'       : 'Balanced — varies by grade (e.g. 10-26-26, 12-32-16)',
            'best_for'  : 'Rice, Maize, Vegetables — crops needing all three macronutrients equally',
            'when'      : 'Apply at sowing and split at vegetative and flowering stages',
            'how_much'  : '100–150 kg/ha depending on grade',
            'caution'   : 'Check the grade number — different NPK grades suit different growth stages',
            'soil_fit'  : 'All soils. Most useful when soil test shows deficiency in all three nutrients.',
            'deficiency': 'Mixed symptoms — yellowing, poor root, weak stems all together',
            'excess'    : 'Salt stress, reduced water uptake, micronutrient lockout',
        },
        'SSP': {
            'npk'       : '0-16-0 + 11% Sulphur',
            'best_for'  : 'Oilseeds (Mustard, Groundnut, Sunflower), Pulses — sulphur-needing crops',
            'when'      : 'Basal application at sowing — mix with soil',
            'how_much'  : '250–375 kg/ha (lower P content means more quantity needed)',
            'caution'   : 'Avoid in acidic soils below pH 5.5 — phosphorus becomes unavailable',
            'soil_fit'  : 'Best for neutral to slightly alkaline soils. Good for sulphur-deficient soils.',
            'deficiency': 'Young leaves turn yellow, stunted growth, poor oil content in oilseeds',
            'excess'    : 'Soil acidification over time with repeated use',
        },
        'Zinc Sulphate': {
            'npk'       : '21% Zinc + 10% Sulphur',
            'best_for'  : 'Rice, Maize, Wheat — crops on zinc-deficient soils',
            'when'      : 'Basal dose at sowing. Can also be foliar sprayed at 0.5% solution.',
            'how_much'  : '25 kg/ha as basal. 2–3 foliar sprays at 0.5% concentration.',
            'caution'   : 'Do not mix with phosphatic fertilizers — zinc becomes unavailable',
            'soil_fit'  : 'Alkaline and calcareous soils are most zinc-deficient — highest need here.',
            'deficiency': 'Khaira disease in rice, white bud in maize, bronzing of leaves',
            'excess'    : 'Toxic above 300 ppm in soil — causes iron and manganese deficiency',
        },
    }

    IRR_DEEP = {
        0: {
            'label'  : 'No Irrigation Needed',
            'reason' : 'Soil moisture (MOI) is high enough. Plants have sufficient water available at root zone.',
            'action' : 'Monitor the field. Check again after 2–3 days or after weather change.',
            'caution': 'Do not irrigate — overwatering causes root rot, fungal diseases, and waterlogging.',
            'moi_tip': 'MOI above 65 means soil is well saturated. Save water.',
        },
        1: {
            'label'  : 'Irrigation Needed',
            'reason' : 'Soil moisture (MOI) is low. Roots cannot access enough water for plant functions.',
            'action' : 'Turn on irrigation immediately. Use drip or sprinkler for efficiency.',
            'caution': 'Do not delay — water stress at critical growth stages causes permanent yield loss.',
            'moi_tip': 'MOI below 35 is the danger zone. Act within 24 hours.',
        },
        2: {
            'label'  : 'Excess Water — Stop Irrigation',
            'reason' : 'Soil is already over-saturated. Adding more water will harm the crop.',
            'action' : 'Stop all irrigation immediately. Improve field drainage if water is standing.',
            'caution': 'Waterlogging causes root rot, nutrient leaching, and anaerobic conditions.',
            'moi_tip': 'MOI above 85 is dangerous. Drain excess water if possible.',
        },
    }

    SOIL_INFO = {
        'Clay'  : 'High water retention, low drainage, rich in nutrients. Prone to waterlogging. Best for rice, wheat.',
        'Silt'  : 'Moderate drainage, high fertility, easy to work. Good for most crops.',
        'Sandy' : 'Low water retention, fast drainage, low nutrients. Needs frequent irrigation and fertilizer.',
        'Loam'  : 'Best all-round soil — balanced drainage, aeration, and nutrient holding.',
        'Black Soil': 'High clay content, moisture retentive, rich in calcium. Best for cotton.',
        'Red Soil'  : 'Iron-rich, well-drained but low fertility. Needs NPK and organic matter.',
        'Alluvial'  : 'Very fertile, good drainage, deposited by rivers. Best for wheat, rice, sugarcane.',
    }

    SEASON_INFO = {
        'Kharif': 'June–October. Monsoon season. Crops: Rice, Cotton, Maize, Soybean, Groundnut. High rainfall, warm temp.',
        'Rabi'  : 'October–March. Winter season. Crops: Wheat, Mustard, Peas, Gram. Cool temp, low rainfall.',
        'Zaid'  : 'March–June. Summer season. Crops: Watermelon, Cucumber, Muskmelon. Short duration, needs irrigation.',
    }

    # ── AI Brain — Rule-based response engine ─────────────────
    def ai_respond(question: str, history: list) -> str:
        q = question.lower().strip()

        # ── Greetings ──────────────────────────────────────────
        if any(w in q for w in ['hello','hi','hey','good morning','good afternoon','namaste']):
            return "Hello! I am AgriAI Assistant. I can help you with fertilizer details, irrigation advice, soil information, crop seasons, and your prediction results. What would you like to know?"

        # ── What can you do ────────────────────────────────────
        if any(w in q for w in ['what can you do','help','what do you know','capabilities','features']):
            return (
                "I can help you with:\n\n"
                "🌱 **Fertilizer details** — Ask about Urea, DAP, MOP, Compost, NPK, SSP, Zinc Sulphate\n"
                "💧 **Irrigation advice** — When to irrigate, what MOI means, water management\n"
                "🌍 **Soil types** — Clay, Silt, Sandy, Loam, Black, Red, Alluvial\n"
                "📅 **Crop seasons** — Kharif, Rabi, Zaid differences\n"
                "📊 **Model results** — Accuracy, algorithms, SMOTE explanation\n"
                "🔍 **Your last prediction** — Ask 'explain my last result'\n\n"
                "Just type your question naturally!"
            )

        # ── Last prediction explain ────────────────────────────
        if any(w in q for w in ['last result','my result','my prediction','explain result','last prediction','what did it say']):
            if st.session_state.history:
                last = st.session_state.history[-1]
                fn   = last.get('Fertilizer','—')
                fc   = last.get('Fert Conf.','—')
                irr  = last.get('Irrigation','—')
                ic   = last.get('Irr. Conf.','—')
                n, p, k = last.get('N','—'), last.get('P','—'), last.get('K','—')
                fd   = FERT_DEEP.get(fn, {})
                reply = (
                    f"**Your last prediction result:**\n\n"
                    f"🌱 **Fertilizer:** {fn} (confidence: {fc})\n"
                    f"- NPK ratio: {fd.get('npk','—')}\n"
                    f"- Best for: {fd.get('best_for','—')}\n"
                    f"- When to apply: {fd.get('when','—')}\n"
                    f"- How much: {fd.get('how_much','—')}\n"
                    f"- Caution: {fd.get('caution','—')}\n\n"
                    f"💧 **Irrigation:** {irr} (confidence: {ic})\n\n"
                    f"📊 **Your NPK input:** N={n}, P={p}, K={k} kg/ha"
                )
                return reply
            else:
                return "You have not made any predictions yet. Go to the **Get Recommendations** tab, enter your soil details, and click the button. Then come back and ask me!"

        # ── Fertilizer questions ───────────────────────────────
        for fname, fdata in FERT_DEEP.items():
            if fname.lower() in q:
                if any(w in q for w in ['when','apply','time','stage','dose','how much','quantity','amount']):
                    return (
                        f"**{fname} — Application Guide:**\n\n"
                        f"⏰ **When to apply:** {fdata['when']}\n"
                        f"📏 **How much:** {fdata['how_much']}\n"
                        f"⚠️ **Caution:** {fdata['caution']}"
                    )
                if any(w in q for w in ['deficien','symptom','sign','yellowing','leaf']):
                    return (
                        f"**{fname} — Deficiency Symptoms:**\n\n"
                        f"🔴 **Deficiency signs:** {fdata['deficiency']}\n"
                        f"🔵 **Excess signs:** {fdata['excess']}"
                    )
                if any(w in q for w in ['soil','which soil','best soil']):
                    return f"**{fname} — Soil Suitability:**\n\n{fdata['soil_fit']}"
                # General fertilizer info
                return (
                    f"**{fname} — Complete Info:**\n\n"
                    f"🧪 **NPK Ratio:** {fdata['npk']}\n"
                    f"🌾 **Best for:** {fdata['best_for']}\n"
                    f"⏰ **When to apply:** {fdata['when']}\n"
                    f"📏 **How much:** {fdata['how_much']}\n"
                    f"🌍 **Soil fit:** {fdata['soil_fit']}\n"
                    f"⚠️ **Caution:** {fdata['caution']}\n"
                    f"🔴 **Deficiency:** {fdata['deficiency']}"
                )

        # ── Compare fertilizers ────────────────────────────────
        if any(w in q for w in ['compare','difference between','vs','versus','which is better']):
            if 'urea' in q and 'dap' in q:
                return (
                    "**Urea vs DAP:**\n\n"
                    "| | Urea | DAP |\n|---|---|---|\n"
                    "| NPK | 46-0-0 | 18-46-0 |\n"
                    "| Main nutrient | Nitrogen | Phosphorus |\n"
                    "| Best stage | Vegetative growth | Sowing/root development |\n"
                    "| Best crop | Wheat, Maize, Rice | Wheat, Pulses, Oilseeds |\n\n"
                    "**Rule:** Use DAP at sowing for root establishment, then Urea for leaf and stem growth."
                )
            if 'urea' in q and 'npk' in q:
                return (
                    "**Urea vs NPK:**\n\n"
                    "Urea gives only Nitrogen (46%). Use it when soil has enough P and K but lacks N.\n"
                    "NPK gives all three nutrients together. Use it when soil is deficient in all three.\n\n"
                    "**Tip:** Always do a soil test first. If only N is low, Urea is cheaper and sufficient."
                )

        # ── Irrigation questions ───────────────────────────────
        if any(w in q for w in ['irrigat','water','moi','moisture','pump','drip','sprinkler','waterlog']):
            if 'moi' in q or 'moisture index' in q:
                return (
                    "**MOI — Moisture Index Explained:**\n\n"
                    "MOI is a 0 to 100 scale of how wet the soil is:\n\n"
                    "🔴 **0–34:** Soil is dry — Irrigate immediately\n"
                    "🟡 **35–64:** Soil is moderate — Monitor, may need irrigation soon\n"
                    "🟢 **65–100:** Soil is wet — No irrigation needed\n\n"
                    "MOI is the **most important feature** for irrigation prediction. "
                    "It alone accounts for over 70% of the model's decision power (from feature importance analysis)."
                )
            if any(w in q for w in ['no irrigation','class 0','result 0']):
                d = IRR_DEEP[0]
                return f"**{d['label']}**\n\n✅ **Reason:** {d['reason']}\n\n📋 **Action:** {d['action']}\n\n⚠️ **Caution:** {d['caution']}"
            if any(w in q for w in ['excess','class 2','result 2','too much','stop irrigation','overwater']):
                d = IRR_DEEP[2]
                return f"**{d['label']}**\n\n⚠️ **Reason:** {d['reason']}\n\n📋 **Action:** {d['action']}\n\n🔴 **Caution:** {d['caution']}"
            if any(w in q for w in ['needed','class 1','result 1','need water','dry']):
                d = IRR_DEEP[1]
                return f"**{d['label']}**\n\n💧 **Reason:** {d['reason']}\n\n📋 **Action:** {d['action']}\n\n⚠️ **Caution:** {d['caution']}"
            if 'drip' in q:
                return (
                    "**Drip Irrigation:**\n\n"
                    "Water is delivered directly to root zone through pipes and emitters.\n"
                    "✅ 40–60% more efficient than flood irrigation\n"
                    "✅ Reduces weed growth and fungal disease\n"
                    "✅ Best for row crops — Cotton, Sugarcane, Vegetables\n"
                    "⚠️ Higher setup cost but saves water long term"
                )
            if 'sprinkler' in q:
                return (
                    "**Sprinkler Irrigation:**\n\n"
                    "Water is sprayed through nozzles like artificial rainfall.\n"
                    "✅ Good for uneven terrain and large fields\n"
                    "✅ Best for Wheat, Groundnut, Vegetables\n"
                    "⚠️ Avoid in windy conditions — water drifts away\n"
                    "⚠️ Not suitable for very heavy clay soils — waterlogging risk"
                )
            return (
                "**Irrigation Quick Guide:**\n\n"
                "💧 **Irrigate when:** MOI < 35, soil looks dry and cracked, leaves wilt in morning\n"
                "🛑 **Stop when:** MOI > 65, water stands on surface, roots visible\n"
                "⏰ **Best time to irrigate:** Early morning or evening — reduces evaporation loss\n"
                "📏 **Frequency:** Depends on crop stage — flowering and grain filling need most water"
            )

        # ── Soil type questions ────────────────────────────────
        for sname, sinfo in SOIL_INFO.items():
            if sname.lower() in q:
                return f"**{sname} Soil:**\n\n{sinfo}"
        if any(w in q for w in ['soil type','which soil','soil info','soil ph','ph level']):
            return (
                "**Soil Types Summary:**\n\n"
                "🟫 **Clay** — High water retention, waterlogging risk. Good for rice, wheat.\n"
                "🟡 **Silt** — High fertility, moderate drainage. Good for most crops.\n"
                "🟠 **Sandy** — Fast drainage, low nutrients. Needs frequent irrigation.\n"
                "🟢 **Loam** — Best balanced soil for all crops.\n"
                "⚫ **Black Soil** — Moisture retentive, rich. Best for cotton.\n"
                "🔴 **Red Soil** — Iron-rich, low fertility. Needs NPK and compost.\n"
                "🔵 **Alluvial** — Very fertile river soil. Best for wheat, rice, sugarcane.\n\n"
                "**Soil pH guide:** Below 6 = Acidic, 6–7 = Optimal, Above 8 = Alkaline"
            )

        # ── Season questions ───────────────────────────────────
        for sname, sinfo in SEASON_INFO.items():
            if sname.lower() in q:
                return f"**{sname} Season:**\n\n{sinfo}"
        if any(w in q for w in ['season','kharif','rabi','zaid']):
            return (
                "**Indian Crop Seasons:**\n\n"
                "🌧️ **Kharif (June–Oct):** Monsoon crops — Rice, Cotton, Maize, Soybean\n"
                "❄️ **Rabi (Oct–Mar):** Winter crops — Wheat, Mustard, Peas, Gram\n"
                "☀️ **Zaid (Mar–Jun):** Summer crops — Watermelon, Cucumber, Muskmelon\n\n"
                "Season affects which fertilizer the model recommends because nutrient needs differ by growth conditions."
            )

        # ── NPK explanation ────────────────────────────────────
        if 'npk' in q and ('what' in q or 'explain' in q or 'mean' in q or 'stand' in q):
            return (
                "**NPK — The Three Primary Nutrients:**\n\n"
                "🔵 **N (Nitrogen):** Builds proteins, drives leaf and stem growth. Makes plants green.\n"
                "🟠 **P (Phosphorus):** Builds roots, flowers, and seeds. Transfers energy in plant.\n"
                "🟢 **K (Potassium):** Regulates water, strengthens immunity, improves fruit quality.\n\n"
                "**Units:** kg/ha (kilograms per hectare)\n"
                "**Example:** Urea NPK is 46-0-0 — 46% Nitrogen, 0% P, 0% K\n"
                "**Example:** DAP NPK is 18-46-0 — 18% N, 46% Phosphorus, 0% K\n\n"
                "In your project: Nitrogen, Phosphorus, and Potassium are 3 of the 12 input features used by the model."
            )

        # ── Project / model questions ──────────────────────────
        if any(w in q for w in ['accuracy','model','random forest','xgboost','smote','algorithm','result']):
            if 'smote' in q:
                return (
                    "**SMOTE in Our Project:**\n\n"
                    "The fertilizer dataset had a class imbalance problem:\n"
                    "- Urea: 2,481 training samples\n"
                    "- SSP: only 145 training samples\n\n"
                    "Without SMOTE — SSP F1-score was **0.0** (model never predicted SSP)\n"
                    "After SMOTE — All 7 classes balanced at **2,481 each** (17,367 total)\n"
                    "SSP F1 improved to **0.0833** — model now recognises SSP.\n\n"
                    "SMOTE creates synthetic samples by interpolating between existing minority samples — not simple duplication."
                )
            if any(w in q for w in ['accuracy','result','how much','score','percentage']):
                return (
                    "**Our Model Results:**\n\n"
                    "🌱 **Fertilizer — Random Forest:** 86.05% accuracy\n"
                    "💧 **Irrigation — XGBoost:** 90.45% accuracy\n\n"
                    "**Base paper (IEEE ICSCCC 2026):** 85% with Random Forest\n"
                    "**Our improvement:** +1.05% by adding SMOTE + tuning\n\n"
                    "**All 6 algorithms — Fertilizer:**\n"
                    "KNN: 53% | Logistic: 76% | Decision Tree: 85% | XGBoost: 85.25% | Gradient Boost: 85.3% | **RF: 86.05%**"
                )
            if 'random forest' in q:
                return (
                    "**Random Forest in Our Project:**\n\n"
                    "Builds 500 decision trees independently, each on a random subset of data and features.\n"
                    "Final prediction = majority vote of all 500 trees.\n\n"
                    "✅ Won fertilizer module with **86.05%**\n"
                    "✅ Handles 12 non-linear features well\n"
                    "✅ Robust to noise in SMOTE synthetic samples\n"
                    "⏱️ Trained in just 2.9 seconds (parallel trees)"
                )
            if 'xgboost' in q:
                return (
                    "**XGBoost in Our Project:**\n\n"
                    "Builds trees sequentially — each tree corrects errors of the previous one.\n"
                    "Has L1 and L2 regularization to prevent overfitting.\n\n"
                    "✅ Won irrigation module with **90.45%**\n"
                    "✅ Finds exact MOI boundary (35 and 65) sharply\n"
                    "✅ Only 4 features — sequential correction is very precise\n"
                    "⏱️ Trained in 2.6 seconds"
                )

        # ── Crop specific advice ───────────────────────────────
        CROP_ADVICE = {
            'wheat' : 'Wheat needs: DAP at sowing for root establishment, then Urea top dressing at tillering. Irrigate at crown root initiation, tillering, jointing, flowering, and grain filling — 5 critical stages.',
            'rice'  : 'Rice needs: Heavy Urea application for leaf growth. Zinc Sulphate if soil is zinc-deficient (common in rice fields). Keep field flooded at 5cm during vegetative stage.',
            'cotton': 'Cotton needs: MOP for fiber quality and disease resistance. Urea at squaring stage. Moderate irrigation — waterlogging destroys cotton roots quickly.',
            'maize' : 'Maize needs: High Nitrogen (Urea or NPK). Zinc Sulphate if soil pH is high. Irrigate at knee-high stage, tasseling, and grain filling.',
            'potato': 'Potato needs: MOP for tuber quality. DAP for root development. Avoid excess moisture — causes blight disease. Irrigation at stolon and tuber initiation stages.',
        }
        for crop, advice in CROP_ADVICE.items():
            if crop in q:
                return f"**{crop.title()} Farming Guide:**\n\n{advice}"

        # ── Fallback → try Gemini, else offline message ────────
        gemini_key = st.session_state.get('gemini_key', '')
        if GEMINI_AVAILABLE and gemini_key:
            try:
                # Build context from last prediction if available
                ctx = ""
                if st.session_state.history:
                    last = st.session_state.history[-1]
                    ctx = (
                        f"The farmer's last prediction: "
                        f"Fertilizer={last.get('Fertilizer','?')} "
                        f"(confidence {last.get('Fert Conf.','?')}), "
                        f"Irrigation={last.get('Irrigation','?')} "
                        f"(confidence {last.get('Irr. Conf.','?')}), "
                        f"N={last.get('N','?')}, P={last.get('P','?')}, K={last.get('K','?')} kg/ha. "
                    )

                system_prompt = (
                    "You are AgriAI Assistant, an expert agricultural advisor built into a "
                    "smart farming web app for Indian farmers. The app recommends fertilizers "
                    "(Urea, DAP, MOP, Compost, Zinc Sulphate, NPK, SSP) and irrigation decisions "
                    "(No Irrigation, Irrigation Needed, Excess Water) using Machine Learning. "
                    f"{ctx}"
                    "Answer the farmer's question in simple, practical English. "
                    "Be concise — 3 to 6 lines maximum. Focus on actionable advice. "
                    "Format key points with bullet points where helpful."
                )

                full_prompt = system_prompt + "\n\nFarmer asks: " + question

                # Try new google-genai SDK first
                try:
                    client  = google_genai.Client(api_key=gemini_key)
                    resp    = client.models.generate_content(
                        model='gemini-3.1-flash-lite',
                        contents=full_prompt,
                    )
                    return resp.text
                except AttributeError:
                    # Fallback to legacy google.generativeai
                    import google.generativeai as genai_legacy
                    genai_legacy.configure(api_key=gemini_key)
                    model_g = genai_legacy.GenerativeModel('gemini-1.5-flash')
                    result  = model_g.generate_content(full_prompt)
                    return result.text

            except Exception as e:
                err = str(e)
                if 'API_KEY' in err or 'invalid' in err.lower() or '400' in err:
                    return "❌ Invalid Gemini API key. Check the key in the sidebar and try again. Get a free key at **aistudio.google.com/app/apikey**"
                if 'quota' in err.lower() or '429' in err:
                    return "⚠️ Gemini free quota reached for today. The offline assistant still works for common questions."
                return (
                    "⚠️ Could not reach Gemini right now. Check your internet connection.\n\n"
                    "Try offline questions:\n"
                    "- 'Tell me about Urea'\n"
                    "- 'What does MOI mean?'\n"
                    "- 'Explain my last result'"
                )
        else:
            return (
                "I could not find an answer offline. Try asking:\n\n"
                "- 'Tell me about Urea'\n"
                "- 'When should I apply DAP?'\n"
                "- 'What does MOI mean?'\n"
                "- 'Explain my last result'\n"
                "- 'What is SMOTE?'\n"
                "- 'Tell me about Kharif season'\n"
                "- 'Compare Urea vs DAP'\n"
                "- 'Wheat farming guide'\n\n"
                "Or paste your **free Gemini API key** in the sidebar to unlock real-time answers for any question.\n"
                "Get a free key at: **aistudio.google.com/app/apikey**"
            )

    # ── Chat UI ───────────────────────────────────────────────
    st.markdown("""
    <div style="background:#141F2D;border-radius:8px;padding:14px 18px;
                border-left:3px solid #388E3C;margin-bottom:16px;">
        <div style="color:#A5D6A7;font-weight:600;font-size:0.92rem;">AgriAI Assistant</div>
        <div style="color:#546E7A;font-size:0.82rem;margin-top:3px;">
            Ask about fertilizers, irrigation, soil, seasons, your results, or model accuracy.
        </div>
    </div>
    """, unsafe_allow_html=True)

    _gkey = st.session_state.get('gemini_key', '')
    if GEMINI_AVAILABLE and _gkey:
        st.caption('🟢 Gemini active — real-time answers on')
    else:
        st.caption('⚫ Offline mode')

    # Quick question buttons
    st.markdown("**Quick questions:**")
    qq_cols = st.columns(4)
    quick_questions = [
        "Explain my last result",
        "What does MOI mean?",
        "Tell me about Urea",
        "Compare Urea vs DAP",
        "When to apply DAP?",
        "What is SMOTE?",
        "Kharif season crops",
        "Clay soil info",
    ]
    quick_input = None
    for i, (col, qq) in enumerate(zip(qq_cols * 2, quick_questions)):
        if col.button(qq, key=f'qq_{i}', use_container_width=True):
            quick_input = qq

    st.markdown('<br>', unsafe_allow_html=True)

    # Chat input
    with st.form(key='chat_form', clear_on_submit=True):
        fc1, fc2 = st.columns([5, 1])
        with fc1:
            user_input = st.text_input(
                'Type your question:',
                placeholder='e.g. When should I apply Urea? What does MOI 25 mean?',
                label_visibility='collapsed'
            )
        with fc2:
            send = st.form_submit_button('Send ➤', use_container_width=True)

    # Process input
    final_input = quick_input or (user_input if send and user_input.strip() else None)

    if final_input:
        response = ai_respond(final_input, st.session_state.chat_history)
        st.session_state.chat_history.append({'role': 'user',    'msg': final_input})
        st.session_state.chat_history.append({'role': 'assistant','msg': response})

    # Display chat history (newest first)
    if st.session_state.chat_history:
        st.markdown('<div class="section-title">💬 Conversation</div>', unsafe_allow_html=True)
        for item in reversed(st.session_state.chat_history):
            if item['role'] == 'user':
                st.markdown(f"""
                <div style="background:#1A2738;border-radius:10px;padding:12px 16px;
                            margin-bottom:8px;border-right:4px solid #4472C4;text-align:right;">
                    <span style="color:#90CAF9;font-size:0.8rem;font-weight:700;">YOU</span><br>
                    <span style="color:#E8EDF2;">{item['msg']}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:#0D2F0D;border-radius:10px;padding:14px 18px;
                            margin-bottom:8px;border-left:4px solid #4CAF50;">
                    <span style="color:#66BB6A;font-size:0.75rem;font-weight:600;">ASSISTANT</span><br>
                    <span style="color:#E8EDF2;">{item['msg']}</span>
                </div>
                """, unsafe_allow_html=True)

        if st.button('🗑️ Clear Chat', use_container_width=False):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.markdown("""
        <div style="background:#141F2D;border-radius:8px;padding:28px;text-align:center;
                    border:1px solid #1E2D3D;">
            <div style="color:#546E7A;font-size:0.9rem;">
                Start by clicking a quick question above, or type below.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# TAB 6 — ABOUT
# ════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-title">ℹ️  About This Project</div>', unsafe_allow_html=True)

    ab1, ab2 = st.columns(2, gap='large')
    with ab1:
        st.markdown("""
#### 🧪 Fertilizer Module
| Item | Detail |
|---|---|
| Dataset | fertilizer_recommendation.csv |
| Rows | 10,000 |
| Features | 12 (Soil Type, pH, Moisture, Organic Carbon, N, P, K, Temp, Humidity, Rainfall, Crop, Season) |
| Output | 7 fertilizer types |
| Class Imbalance | Fixed using SMOTE |
| Tuning | RandomizedSearchCV |
        """)

    with ab2:
        st.markdown("""
#### 💧 Irrigation Module
| Item | Detail |
|---|---|
| Dataset | cropdata_updated.csv |
| Rows | 16,411 |
| Features | 4 (MOI, Temperature, Humidity, Soil Type) |
| Output | 3 classes (No Irrigation / Needed / Excess) |
| Tuning | RandomizedSearchCV |
        """)

    st.markdown('---')
    st.markdown('#### 🔬 Why These 4 Algorithms?')
    st.dataframe(pd.DataFrame({
        'Algorithm'       : ['Logistic Regression','Decision Tree',
                             'Random Forest','XGBoost'],
        'Reference'       : [
            'Baseline comparison — standard in literature',
            'Velagaleti et al., IEEE ICSCCC 2026 (Base Paper)',
            'Velagaleti et al., IEEE ICSCCC 2026 (Base Paper) — 85%',
            'Sunilkumar et al., IEEE ICNWC 2024',
        ],
        'Our Accuracy'    : ['76.30%','84.95%','86.05% ✅','85.25%'],
        'Why Used'        : [
            'Simple linear baseline for comparison',
            'Base paper used DT for comparison — interpretable',
            'Base paper primary model — we improved to 86.05%',
            'Boosting ensemble — best for structured data',
        ],
    }), use_container_width=True, hide_index=True)

    st.markdown('---')
    st.markdown('#### 🧬 SMOTE Effect')
    st.dataframe(pd.DataFrame({
        'Condition'          : ['Without SMOTE','With SMOTE'],
        'Overall Accuracy'   : ['73.1%','86.05% (tuned)'],
        'SSP Class F1-Score' : ['0.06 ❌ — model ignores SSP','Improved ✅'],
        'Macro F1'           : ['0.58','0.76'],
        'Notes'              : ['Base paper problem identified','Our solution'],
    }), use_container_width=True, hide_index=True)

    st.markdown('---')
    st.markdown('#### 📄 Research References')
    st.markdown("""
| # | Paper | Publisher | Year |
|---|---|---|---|
| **[1]** | **Velagaleti et al. — Soil Smart: AI-Powered Soil Testing & Fertilizer Recommendation** | **IEEE ICSCCC** | **2026** |
| [2] | Sunilkumar et al. — Smart Irrigation System Recommending Crop and Fertilizers | IEEE ICNWC | 2024 |
| [3] | Khaliq et al. — AI-Driven Smart Agriculture (Integrated Approach) | IEEE Access | 2025 |
| [4] | Musanase et al. — Soil Fertility Prediction Using ML | MDPI Sustainability | 2023 |
| [5] | Tanaka et al. — Gradient Boosting for Fertilizer Recommendation | Springer | 2024 |
| [6] | Chawla et al. — SMOTE Original Paper | JAIR | 2002 |
    """)

    st.markdown('---')
    st.markdown('#### 👥 Team Members')
    st.dataframe(pd.DataFrame({
        'Name'    : ['C. Jasvitha Reddy','Md Shoaib','G. Sumanth Goud'],
        'HT No.'  : ['23EG107D10','23EG107D42','23EG107D24'],
        'Section' : ['AIML-D','AIML-D','AIML-D'],
        'College' : ['Anurag University','Anurag University','Anurag University'],
    }), use_container_width=True, hide_index=True)
