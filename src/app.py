"""
app.py  —  Finexcore AI Lending Intelligence Platform
Loan default prediction powered by LightGBM + SHAP
"""

import json
import warnings
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from pathlib import Path
from scipy.special import expit

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR    = Path("models")
PROCESSED_DIR = Path("data/processed")

# ── Finexcore Brand Colors ────────────────────────────────────────────────────
C_NAVY    = "#1B2B4B"   # Finexcore primary dark navy
C_BLUE    = "#0057C8"   # Finexcore accent blue
C_LBLUE   = "#E8F0FB"   # Light blue background
C_WHITE   = "#FFFFFF"
C_GREEN   = "#1A7F5A"   # Approve / Low risk
C_AMBER   = "#D97706"   # Medium risk
C_RED     = "#C0392B"   # Decline / High risk
C_GRAY    = "#6B7280"   # Muted text
C_BORDER  = "#CBD5E1"

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title  = "Finexcore | AI Lending Intelligence",
    page_icon   = "🏦",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── CSS — Finexcore brand ─────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] {{
      font-family: 'Inter', Arial, sans-serif;
  }}
  .block-container {{ padding-top: 1rem; padding-bottom: 1rem; }}

  /* ── Hero banner ── */
  .hero {{
      background: linear-gradient(135deg, {C_NAVY} 0%, {C_BLUE} 100%);
      color: white;
      padding: 2rem 2.5rem;
      border-radius: 12px;
      margin-bottom: 1.8rem;
      position: relative;
      overflow: hidden;
  }}
  .hero::after {{
      content: "";
      position: absolute;
      right: -60px; top: -60px;
      width: 220px; height: 220px;
      border-radius: 50%;
      background: rgba(255,255,255,0.05);
  }}
  .hero h1 {{ margin: 0; font-size: 1.75rem; font-weight: 700; letter-spacing: -0.3px; }}
  .hero p  {{ margin: 0.4rem 0 0; opacity: 0.85; font-size: 0.95rem; font-weight: 400; }}

  /* ── KPI cards ── */
  .kpi-card {{
      background: {C_WHITE};
      border: 1px solid {C_BORDER};
      border-top: 3px solid {C_BLUE};
      border-radius: 10px;
      padding: 1.1rem 1.3rem;
      text-align: center;
      box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }}
  .kpi-val  {{ font-size: 1.9rem; font-weight: 700; color: {C_NAVY}; line-height: 1.1; }}
  .kpi-lbl  {{ font-size: 0.78rem; color: {C_GRAY}; margin-top: 0.3rem;
               font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }}
  .kpi-sub  {{ font-size: 0.75rem; color: {C_BLUE}; margin-top: 0.15rem; }}

  /* ── Section header ── */
  .section-label {{
      font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
      letter-spacing: 1px; color: {C_BLUE}; margin-bottom: 0.4rem;
  }}

  /* ── Risk pill ── */
  .pill-HIGH   {{ display:inline-block; padding:0.25rem 0.9rem; border-radius:20px;
                  background:#FEE2E2; color:{C_RED}; font-weight:700; font-size:0.85rem; }}
  .pill-MEDIUM {{ display:inline-block; padding:0.25rem 0.9rem; border-radius:20px;
                  background:#FEF3C7; color:{C_AMBER}; font-weight:700; font-size:0.85rem; }}
  .pill-LOW    {{ display:inline-block; padding:0.25rem 0.9rem; border-radius:20px;
                  background:#D1FAE5; color:{C_GREEN}; font-weight:700; font-size:0.85rem; }}

  /* ── Result panel ── */
  .result-HIGH   {{ background:#FEF2F2; border:1.5px solid #FECACA;
                    border-left:5px solid {C_RED}; border-radius:10px; padding:1.2rem 1.5rem; }}
  .result-MEDIUM {{ background:#FFFBEB; border:1.5px solid #FDE68A;
                    border-left:5px solid {C_AMBER}; border-radius:10px; padding:1.2rem 1.5rem; }}
  .result-LOW    {{ background:#F0FDF4; border:1.5px solid #BBF7D0;
                    border-left:5px solid {C_GREEN}; border-radius:10px; padding:1.2rem 1.5rem; }}

  /* ── Input panels ── */
  .input-panel {{
      background: {C_WHITE};
      border: 1px solid {C_BORDER};
      border-radius: 10px;
      padding: 1.3rem 1.5rem;
      margin-bottom: 1rem;
  }}
  .input-panel h4 {{
      margin: 0 0 0.9rem;
      color: {C_NAVY};
      font-size: 0.9rem;
      font-weight: 600;
      border-bottom: 2px solid {C_LBLUE};
      padding-bottom: 0.5rem;
  }}

  /* ── Table ── */
  .styled-table {{ width:100%; border-collapse:collapse; font-size:0.85rem; }}
  .styled-table th {{
      background:{C_NAVY}; color:white; padding:0.6rem 0.8rem;
      text-align:left; font-weight:600; font-size:0.78rem; text-transform:uppercase;
  }}
  .styled-table td {{ padding:0.55rem 0.8rem; border-bottom:1px solid {C_BORDER}; }}
  .styled-table tr:nth-child(even) td {{ background:{C_LBLUE}; }}
  .styled-table tr:hover td {{ background:#dbeafe; }}

  /* ── Upload zone ── */
  [data-testid="stFileUploader"] {{
      border: 2px dashed {C_BLUE} !important;
      border-radius: 10px !important;
      padding: 1rem !important;
  }}

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {{
      background: {C_NAVY} !important;
  }}
  [data-testid="stSidebar"] * {{ color: white !important; }}
  [data-testid="stSidebar"] .stRadio label {{ color: rgba(255,255,255,0.85) !important; }}
  [data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.15) !important; }}

  /* ── Buttons ── */
  .stButton > button {{
      background: {C_BLUE} !important;
      color: white !important;
      border: none !important;
      border-radius: 8px !important;
      font-weight: 600 !important;
      padding: 0.55rem 1.5rem !important;
      font-size: 0.9rem !important;
      transition: background 0.2s;
  }}
  .stButton > button:hover {{ background: {C_NAVY} !important; }}

  footer {{ visibility: hidden; }}
  #MainMenu {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CACHED LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Initialising AI engine...")
def load_model():
    return joblib.load(MODELS_DIR / "lgbm_best.pkl")

@st.cache_resource(show_spinner="Loading explainer...")
def load_explainer():
    return shap.TreeExplainer(load_model())

@st.cache_data(show_spinner=False)
def load_meta():
    with open(MODELS_DIR / "feature_cols.json")      as f: feature_cols = json.load(f)
    with open(MODELS_DIR / "threshold.json")          as f: threshold    = json.load(f)["threshold"]
    with open(MODELS_DIR / "metrics.json")            as f: metrics      = json.load(f)
    with open(MODELS_DIR / "shap_feature_desc.json")  as f: feat_desc    = json.load(f)
    shap_top = pd.read_csv(MODELS_DIR / "shap_top20.csv")
    return feature_cols, threshold, metrics, feat_desc, shap_top

@st.cache_data(show_spinner=False)
def load_pipeline_meta():
    with open(PROCESSED_DIR / "pipeline_meta.json") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def kpi(val, label, sub=""):
    return f"""<div class="kpi-card">
      <div class="kpi-val">{val}</div>
      <div class="kpi-lbl">{label}</div>
      {"<div class='kpi-sub'>"+sub+"</div>" if sub else ""}
    </div>"""

def risk_tier(prob, threshold):
    if prob >= threshold:          return "HIGH",   C_RED,   "🔴", "DECLINE"
    elif prob >= threshold * 0.6:  return "MEDIUM", C_AMBER, "🟡", "REVIEW"
    else:                          return "LOW",     C_GREEN, "🟢", "APPROVE"

def predict_single(input_dict, feature_cols, pipeline_meta):
    model     = load_model()
    explainer = load_explainer()

    row = {feat: input_dict.get(feat, np.nan) for feat in feature_cols}
    df  = pd.DataFrame([row])

    for col, med in pipeline_meta["medians"].items():
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(med)
    for col, (lo, hi) in pipeline_meta["clip_bounds"].items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)
    df = df.fillna(0)

    X    = df[feature_cols].values.astype(np.float32)
    prob = float(model.predict_proba(X)[0, 1])

    sv = load_explainer().shap_values(X)
    sv = sv[1][0] if isinstance(sv, list) else sv[0]

    shap_df = pd.DataFrame({
        "feature": feature_cols, "shap": sv, "value": X[0]
    })
    shap_df["abs_shap"] = shap_df["shap"].abs()
    return prob, shap_df.sort_values("abs_shap", ascending=False).reset_index(drop=True)

def gauge_chart(prob):
    colour = C_RED if prob >= 0.65 else (C_AMBER if prob >= 0.39 else C_GREEN)
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = round(prob * 100, 1),
        number= {"suffix": "%", "font": {"size": 42, "color": C_NAVY, "family": "Inter"}},
        delta = {"reference": 50, "valueformat": ".1f",
                 "increasing": {"color": C_RED}, "decreasing": {"color": C_GREEN}},
        gauge = {
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickcolor": C_GRAY, "tickfont": {"size": 10}},
            "bar":  {"color": colour, "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  39], "color": "#D1FAE5"},
                {"range": [39, 65], "color": "#FEF3C7"},
                {"range": [65, 100],"color": "#FEE2E2"},
            ],
            "threshold": {
                "line":  {"color": C_NAVY, "width": 3},
                "thickness": 0.75,
                "value": 65,
            },
        },
        title = {"text": "Default Probability", "font": {"size": 13,
                 "color": C_GRAY, "family": "Inter"}},
    ))
    fig.update_layout(
        height=280, margin=dict(t=40, b=10, l=30, r=30),
        paper_bgcolor="white", font_family="Inter",
    )
    return fig

def waterfall_chart(shap_df, prob, top_n=10):
    top    = shap_df.head(top_n).sort_values("shap")
    colors = [C_RED if s > 0 else C_GREEN for s in top["shap"]]
    labels = [f"{r['feature']}  [{r['value']:.3g}]" for _, r in top.iterrows()]

    fig = go.Figure(go.Bar(
        x           = top["shap"].tolist(),
        y           = labels,
        orientation = "h",
        marker_color= colors,
        text        = [f"{v:+.4f}" for v in top["shap"]],
        textposition= "outside",
        textfont    = {"size": 10, "family": "Inter"},
    ))
    fig.update_layout(
        title       = dict(text=f"Why this score? — Top {top_n} driving factors",
                           font=dict(size=13, color=C_NAVY, family="Inter")),
        xaxis_title = "SHAP Impact on Default Probability",
        yaxis       = dict(tickfont=dict(size=9, family="Inter")),
        height      = 420,
        margin      = dict(t=50, b=40, l=10, r=80),
        paper_bgcolor = "white",
        plot_bgcolor  = "#FAFBFF",
        xaxis       = dict(zeroline=True, zerolinecolor=C_NAVY,
                           zerolinewidth=1.5, gridcolor="#E5EAF3"),
        showlegend  = False,
        font_family = "Inter",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar(metrics, threshold):
    with st.sidebar:
        # Logo
        st.markdown("""
        <div style="text-align:center; padding:1rem 0 0.5rem">
          <img src="https://finexcore.com/wp-content/uploads/2024/07/finexcore-logo-2-1-300x64.png"
               style="max-width:180px; filter:brightness(0) invert(1);" />
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:rgba(255,255,255,0.08); border-radius:8px;
                    padding:0.8rem 1rem; margin:0.8rem 0; font-size:0.82rem;
                    color:rgba(255,255,255,0.8); line-height:1.6;">
          <b style="color:white;">AI Lending Intelligence</b><br>
          Loan default scoring with explainable AI — built for Finexcore's
          lending and collections suite.
        </div>""", unsafe_allow_html=True)

        st.markdown("<hr style='margin:0.8rem 0'>", unsafe_allow_html=True)

        # Live model stats
        st.markdown("""<div style="font-size:0.7rem; font-weight:700;
            text-transform:uppercase; letter-spacing:1px;
            color:rgba(255,255,255,0.5); margin-bottom:0.6rem">
            AI Engine Stats</div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        col1.metric("ROC-AUC",  f"{metrics['oof_roc_auc']:.4f}")
        col2.metric("CV Mean",  f"{metrics['cv_mean_auc']:.4f}")
        col1.metric("Avg Prec", f"{metrics['avg_precision']:.4f}")
        col2.metric("Threshold",f"{threshold:.2f}")

        st.markdown("<hr style='margin:0.8rem 0'>", unsafe_allow_html=True)

        st.markdown("""<div style="font-size:0.7rem; font-weight:700;
            text-transform:uppercase; letter-spacing:1px;
            color:rgba(255,255,255,0.5); margin-bottom:0.6rem">
            Navigate</div>""", unsafe_allow_html=True)

        page = st.radio("", [
            "🔍  Applicant Scoring",
            "📋  Portfolio Batch",
            "📊  AI Engine",
        ], label_visibility="collapsed")

        st.markdown("<hr style='margin:0.8rem 0'>", unsafe_allow_html=True)
        st.markdown("""<div style="font-size:0.75rem; color:rgba(255,255,255,0.45);
            text-align:center; padding-bottom:0.5rem; line-height:1.8">
            Finexcore AI Lending Intelligence<br>
            Powered by LightGBM + SHAP
        </div>""", unsafe_allow_html=True)

    return page


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — APPLICANT SCORING
# ══════════════════════════════════════════════════════════════════════════════

def page_single(feature_cols, threshold, feat_desc, pipeline_meta):
    st.markdown("""
    <div class="hero">
      <h1>🔍 Applicant Risk Scoring</h1>
      <p>Enter applicant details to receive an instant AI-generated default
         probability with full explainability</p>
    </div>""", unsafe_allow_html=True)

    left, right = st.columns([1, 1.1], gap="large")

    # ── LEFT — input form ────────────────────────────────────────────────
    with left:
        tab1, tab2, tab3 = st.tabs(["👤 Personal", "💰 Financial", "📜 Credit History"])

        with tab1:
            st.markdown('<div class="section-label">Personal Details</div>',
                        unsafe_allow_html=True)
            age         = st.slider("Age (years)", 20, 70, 35)
            gender      = st.selectbox("Gender", ["Male", "Female"])
            children    = st.number_input("Number of children", 0, 10, 0)
            fam_members = st.number_input("Family members", 1, 15, 2)
            c1, c2 = st.columns(2)
            own_car    = c1.selectbox("Owns car",    ["No", "Yes"])
            own_realty = c2.selectbox("Owns realty", ["No", "Yes"])

        with tab2:
            st.markdown('<div class="section-label">Financial Profile</div>',
                        unsafe_allow_html=True)
            income      = st.number_input("Annual income (₹)", 10_000, 10_000_000,
                                           200_000, step=10_000,
                                           format="%d")
            credit      = st.number_input("Loan amount (₹)", 10_000, 10_000_000,
                                           500_000, step=10_000, format="%d")
            annuity     = st.number_input("Monthly annuity (₹)", 1_000, 500_000,
                                           25_000, step=1_000, format="%d")
            goods_price = st.number_input("Goods price (₹)", 0, 10_000_000,
                                           450_000, step=10_000, format="%d")
            c1, c2 = st.columns(2)
            employed_yrs  = c1.slider("Years employed", 0, 40, 5)
            is_unemployed = c2.checkbox("Currently unemployed",
                                         value=False)

        with tab3:
            st.markdown('<div class="section-label">Credit Bureau & History</div>',
                        unsafe_allow_html=True)
            st.caption("External credit scores — scale 0.0 (poor) to 1.0 (excellent)")
            c1, c2, c3 = st.columns(3)
            ext1 = c1.slider("Score 1", 0.0, 1.0, 0.50, 0.01)
            ext2 = c2.slider("Score 2", 0.0, 1.0, 0.50, 0.01)
            ext3 = c3.slider("Score 3", 0.0, 1.0, 0.50, 0.01)

            st.divider()
            ins_late_ratio = st.slider(
                "Fraction of installments paid late", 0.0, 1.0, 0.05, 0.01)
            ins_days_late  = st.slider(
                "Maximum days late on any installment", 0, 365, 5)
            c1, c2 = st.columns(2)
            prev_approved  = c1.slider("Previous approval rate", 0.0, 1.0, 0.7, 0.01)
            prev_refused   = c2.number_input("Previous refusals", 0, 20, 0)

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("⚡  Run Risk Assessment",
                            type="primary", use_container_width=True)

    # ── RIGHT — results ──────────────────────────────────────────────────
    with right:
        if not run_btn:
            st.markdown(f"""
            <div style="background:{C_LBLUE}; border:1px dashed {C_BLUE};
                        border-radius:12px; padding:3rem 2rem; text-align:center;
                        margin-top:0.5rem;">
              <div style="font-size:2.5rem">🏦</div>
              <div style="font-size:1.1rem; font-weight:600; color:{C_NAVY};
                          margin:0.8rem 0 0.4rem">
                Ready to Score</div>
              <div style="color:{C_GRAY}; font-size:0.9rem">
                Fill in the applicant details on the left<br>
                and click <b>Run Risk Assessment</b></div>
            </div>""", unsafe_allow_html=True)
            return

        # Build input dict
        input_dict = {
            "AGE_YEARS":             float(age),
            "DAYS_BIRTH":            float(age * 365),
            "CODE_GENDER":           1.0 if gender == "Female" else 0.0,
            "CNT_CHILDREN":          float(children),
            "CNT_FAM_MEMBERS":       float(fam_members),
            "FLAG_OWN_CAR":          1.0 if own_car    == "Yes" else 0.0,
            "FLAG_OWN_REALTY":       1.0 if own_realty == "Yes" else 0.0,
            "AMT_INCOME_TOTAL":      float(income),
            "AMT_CREDIT":            float(credit),
            "AMT_ANNUITY":           float(annuity),
            "AMT_GOODS_PRICE":       float(goods_price),
            "EMPLOYED_YEARS":        0.0 if is_unemployed else float(employed_yrs),
            "DAYS_EMPLOYED":         0.0 if is_unemployed else float(employed_yrs * 365),
            "IS_UNEMPLOYED":         1.0 if is_unemployed else 0.0,
            "CREDIT_INCOME_RATIO":   credit / max(income, 1),
            "ANNUITY_INCOME_RATIO":  annuity / max(income, 1),
            "CREDIT_TERM":           annuity / max(credit, 1),
            "GOODS_CREDIT_RATIO":    goods_price / max(credit, 1),
            "INCOME_PER_PERSON":     income / max(fam_members, 1),
            "EMPLOYED_TO_AGE_RATIO": 0.0 if is_unemployed else employed_yrs / max(age, 1),
            "EXT_SOURCE_1":          ext1,
            "EXT_SOURCE_2":          ext2,
            "EXT_SOURCE_3":          ext3,
            "EXT_SOURCE_MEAN":       (ext1 + ext2 + ext3) / 3,
            "EXT_SOURCE_STD":        float(np.std([ext1, ext2, ext3])),
            "EXT_SOURCE_PRODUCT":    ext1 * ext2 * ext3,
            "EXT_SOURCE_MIN":        min(ext1, ext2, ext3),
            "INS_PAID_LATE_RATIO":   ins_late_ratio,
            "INS_DAYS_LATE_MAX":     float(ins_days_late),
            "INS_DAYS_LATE_MEAN":    float(ins_days_late * 0.3),
            "PREV_APPROVED_RATIO":   prev_approved,
            "PREV_REFUSED_COUNT":    float(prev_refused),
        }

        with st.spinner("Scoring applicant..."):
            prob, shap_df = predict_single(input_dict, feature_cols,
                                            pipeline_meta)

        tier, colour, emoji, decision = risk_tier(prob, threshold)
        score_equiv = max(300, min(850, int(850 - prob * 600)))

        # Gauge
        st.plotly_chart(gauge_chart(prob), use_container_width=True,
                        config={"displayModeBar": False})

        # Risk result banner
        dec_color = C_GREEN if decision == "APPROVE" else (
                    C_AMBER if decision == "REVIEW" else C_RED)
        st.markdown(f"""
        <div class="result-{tier}">
          <div style="display:flex; justify-content:space-between; align-items:center">
            <div>
              <span class="pill-{tier}">{emoji} {tier} RISK</span>
              <span style="margin-left:0.8rem; font-size:1.3rem; font-weight:700;
                           color:{C_NAVY}">{prob:.1%} default probability</span>
            </div>
            <div style="background:{dec_color}; color:white; padding:0.4rem 1.2rem;
                        border-radius:6px; font-weight:700; font-size:0.9rem">
              {decision}
            </div>
          </div>
          <div style="margin-top:0.7rem; display:flex; gap:2rem; font-size:0.85rem;
                      color:{C_GRAY}">
            <span>Threshold: <b>{threshold:.0%}</b></span>
            <span>Equiv. Credit Score: <b>{score_equiv}</b></span>
            <span>Margin: <b>{abs(prob - threshold):.1%}</b></span>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # SHAP waterfall
        st.plotly_chart(waterfall_chart(shap_df, prob),
                        use_container_width=True,
                        config={"displayModeBar": False})

        # Factor table
        st.markdown('<div class="section-label">Factor Detail</div>',
                    unsafe_allow_html=True)
        top10 = shap_df.head(10).copy()
        top10["Effect"]      = top10["shap"].apply(
            lambda x: "↑ Increases risk" if x > 0 else "↓ Reduces risk")
        top10["Description"] = top10["feature"].map(
            lambda f: feat_desc.get(f, f.replace("_", " ").title()))
        top10["SHAP"]  = top10["shap"].round(5)
        top10["Value"] = top10["value"].round(4)

        rows_html = ""
        for _, r in top10.iterrows():
            clr = C_RED if r["SHAP"] > 0 else C_GREEN
            rows_html += f"""<tr>
              <td style="font-weight:500;color:{C_NAVY}">{r['feature']}</td>
              <td style="color:{C_GRAY};font-size:0.8rem">{r['Description']}</td>
              <td style="text-align:center">{r['Value']}</td>
              <td style="text-align:center;color:{clr};font-weight:600">{r['SHAP']:+.5f}</td>
              <td style="font-size:0.8rem;color:{clr}">{r['Effect']}</td>
            </tr>"""

        st.markdown(f"""
        <table class="styled-table">
          <thead><tr>
            <th>Feature</th><th>What it means</th>
            <th>Value</th><th>SHAP</th><th>Effect</th>
          </tr></thead>
          <tbody>{rows_html}</tbody>
        </table>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — BATCH PORTFOLIO
# ══════════════════════════════════════════════════════════════════════════════

def page_batch(feature_cols, threshold, pipeline_meta):
    st.markdown("""
    <div class="hero">
      <h1>📋 Portfolio Batch Scoring</h1>
      <p>Score an entire loan portfolio instantly —
         get risk tiers and approve/decline decisions for every applicant</p>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:{C_LBLUE}; border:1px solid {C_BORDER};
                border-radius:10px; padding:1rem 1.3rem; margin-bottom:1rem;
                font-size:0.88rem; color:{C_NAVY}">
      <b>How to use:</b> Upload a CSV file matching the processed feature schema,
      or click the button below to demo with 500 real applicants from the
      Home Credit test set.
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        uploaded = st.file_uploader("Upload applicant CSV", type=["csv"],
                                     label_visibility="collapsed")
    with c2:
        demo_btn = st.button("📂  Load Demo Portfolio (500 applicants)",
                             use_container_width=True)

    df_batch = None
    if uploaded:
        df_batch = pd.read_csv(uploaded)
        st.success(f"✅  Loaded {len(df_batch):,} applicants from uploaded file.")
    elif demo_btn:
        df_batch = pd.read_csv(PROCESSED_DIR / "test_processed.csv", nrows=500)
        st.success("✅  Loaded 500 applicants from demo portfolio.")

    if df_batch is None:
        return

    model = load_model()
    pmeta = load_pipeline_meta()
    obj_cols   = df_batch.select_dtypes(include=["object","category"]).columns.tolist()
    df_feat    = pd.DataFrame(index=df_batch.index)
    for feat in feature_cols:
        if feat in df_batch.columns and feat not in obj_cols:
            df_feat[feat] = df_batch[feat]
        else:
            df_feat[feat] = pmeta["medians"].get(feat, 0)
    for col, med in pmeta["medians"].items():
        if col in df_feat.columns:
            df_feat[col] = df_feat[col].fillna(med)
    for col, (lo, hi) in pmeta["clip_bounds"].items():
        if col in df_feat.columns:
            df_feat[col] = df_feat[col].clip(lower=lo, upper=hi)
    df_feat = df_feat.fillna(0)

    with st.spinner(f"Scoring {len(df_feat):,} applicants..."):
        probs = model.predict_proba(
            df_feat[feature_cols].values.astype(np.float32))[:, 1]

    if "SK_ID_CURR" not in df_batch.columns:
        df_batch.insert(0, "SK_ID_CURR", range(1, len(df_batch) + 1))

    results             = df_batch[["SK_ID_CURR"]].copy()
    results["DEFAULT_PROBABILITY"] = np.round(probs, 4)
    results["RISK_TIER"] = pd.cut(
        probs,
        bins=[0, threshold * 0.6, threshold, 1.0],
        labels=["LOW", "MEDIUM", "HIGH"],
    ).astype(str)
    results["DECISION"] = np.where(probs >= threshold, "DECLINE", "APPROVE")

    approve_n = (results["DECISION"] == "APPROVE").sum()
    decline_n = (results["DECISION"] == "DECLINE").sum()
    high_n    = (results["RISK_TIER"] == "HIGH").sum()
    med_n     = (results["RISK_TIER"] == "MEDIUM").sum()
    low_n     = (results["RISK_TIER"] == "LOW").sum()

    # KPI row
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Portfolio Summary</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(kpi(f"{len(results):,}", "Total Applications"), unsafe_allow_html=True)
    c2.markdown(kpi(f"{approve_n:,}", "Approved",
                    f"{approve_n/len(results):.1%}"), unsafe_allow_html=True)
    c3.markdown(kpi(f"{decline_n:,}", "Declined",
                    f"{decline_n/len(results):.1%}"), unsafe_allow_html=True)
    c4.markdown(kpi(f"{probs.mean():.3f}", "Mean Probability"), unsafe_allow_html=True)
    c5.markdown(kpi(f"{high_n:,}", "High Risk"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row
    ch1, ch2 = st.columns(2)

    with ch1:
        fig = go.Figure(go.Pie(
            labels = ["Approve", "Decline"],
            values = [approve_n, decline_n],
            marker_colors = [C_GREEN, C_RED],
            hole   = 0.55,
            textinfo = "percent+label",
            textfont = {"size": 13, "family": "Inter"},
        ))
        fig.update_layout(
            title=dict(text="Decision Split",
                       font=dict(size=13, color=C_NAVY, family="Inter")),
            height=280,
            margin=dict(t=50, b=10, l=10, r=10),
            paper_bgcolor="white",
            showlegend=False,
            font_family="Inter",
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})

    with ch2:
        fig = go.Figure()
        for tier, clr, cnt in [("Low",    C_GREEN, low_n),
                                 ("Medium", C_AMBER, med_n),
                                 ("High",   C_RED,   high_n)]:
            fig.add_trace(go.Bar(
                name=tier, x=[cnt], y=["Portfolio"],
                orientation="h", marker_color=clr,
                text=f"{tier}<br>{cnt:,}", textposition="inside",
                textfont={"size": 11, "family": "Inter", "color": "white"},
            ))
        fig.update_layout(
            title=dict(text="Risk Tier Distribution",
                       font=dict(size=13, color=C_NAVY, family="Inter")),
            barmode="stack", height=280,
            margin=dict(t=50, b=10, l=10, r=10),
            paper_bgcolor="white",
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False),
            showlegend=True,
            legend=dict(orientation="h", y=-0.05),
            font_family="Inter",
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})

    # Results table
    st.markdown('<div class="section-label">Scored Applicants</div>',
                unsafe_allow_html=True)

    tier_colors = {"HIGH": "#FEE2E2", "MEDIUM": "#FEF3C7", "LOW": "#D1FAE5"}
    dec_colors  = {"DECLINE": C_RED,  "APPROVE": C_GREEN}

    rows_html = ""
    for _, r in results.head(200).iterrows():
        bg  = tier_colors.get(str(r["RISK_TIER"]), C_WHITE)
        dc  = dec_colors.get(str(r["DECISION"]), C_GRAY)
        bar_w = int(r["DEFAULT_PROBABILITY"] * 120)
        rows_html += f"""<tr style="background:{bg}">
          <td style="font-weight:600;color:{C_NAVY}">{r['SK_ID_CURR']}</td>
          <td>
            <div style="display:flex;align-items:center;gap:6px">
              <div style="width:{bar_w}px;height:8px;background:{dc};
                          border-radius:4px;min-width:2px"></div>
              <span style="font-weight:600;color:{C_NAVY}">
                {r['DEFAULT_PROBABILITY']:.4f}</span>
            </div>
          </td>
          <td><span class="pill-{r['RISK_TIER']}">{r['RISK_TIER']}</span></td>
          <td style="color:{dc};font-weight:700">{r['DECISION']}</td>
        </tr>"""

    st.markdown(f"""
    <table class="styled-table">
      <thead><tr>
        <th>Applicant ID</th>
        <th>Default Probability</th>
        <th>Risk Tier</th>
        <th>Decision</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    csv_out = results[["SK_ID_CURR","DEFAULT_PROBABILITY",
                        "RISK_TIER","DECISION"]].to_csv(index=False)
    st.download_button(
        "⬇️  Download Scored Portfolio CSV",
        data=csv_out, file_name="finexcore_scored_portfolio.csv",
        mime="text/csv", use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — AI ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def page_overview(metrics, shap_top):
    st.markdown("""
    <div class="hero">
      <h1>📊 AI Engine</h1>
      <p>Model performance, validation methodology,
         and feature intelligence powering the platform</p>
    </div>""", unsafe_allow_html=True)

    # Top KPIs
    st.markdown('<div class="section-label">Performance Metrics</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(kpi(metrics["oof_roc_auc"], "OOF ROC-AUC",  "Out-of-fold"),
                unsafe_allow_html=True)
    c2.markdown(kpi(metrics["cv_mean_auc"], "CV Mean AUC",  "5-fold mean"),
                unsafe_allow_html=True)
    c3.markdown(kpi(f"±{metrics['cv_std_auc']}", "CV Std", "Stability"),
                unsafe_allow_html=True)
    c4.markdown(kpi(metrics["avg_precision"], "Avg Precision", "PR-AUC"),
                unsafe_allow_html=True)
    c5.markdown(kpi(metrics["best_threshold"], "Threshold", "F1-tuned"),
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Fold AUC + Confusion Matrix
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="section-label">Cross-Validation Fold AUCs</div>',
                    unsafe_allow_html=True)
        fold_aucs = metrics["fold_aucs"]
        mean_auc  = np.mean(fold_aucs)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"Fold {i+1}" for i in range(len(fold_aucs))],
            y=fold_aucs,
            marker_color=[C_BLUE] * len(fold_aucs),
            text=[f"{a:.5f}" for a in fold_aucs],
            textposition="outside",
            textfont={"size": 10, "family": "Inter"},
        ))
        fig.add_hline(y=mean_auc, line_dash="dash", line_color=C_RED,
                      annotation_text=f"Mean = {mean_auc:.5f}",
                      annotation_font={"color": C_RED, "size": 11})
        fig.update_layout(
            height=300, paper_bgcolor="white", plot_bgcolor="#FAFBFF",
            yaxis=dict(range=[min(fold_aucs)-0.006, max(fold_aucs)+0.008],
                       gridcolor="#E5EAF3"),
            xaxis=dict(showgrid=False),
            margin=dict(t=20, b=30, l=40, r=20),
            font_family="Inter", showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})

    with right:
        st.markdown('<div class="section-label">Confusion Matrix (OOF)</div>',
                    unsafe_allow_html=True)
        cm   = metrics["confusion_matrix"]
        vals = [[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]]
        fig  = go.Figure(go.Heatmap(
            z=vals,
            x=["Predicted: No Default", "Predicted: Default"],
            y=["Actual: No Default", "Actual: Default"],
            colorscale=[[0, "#E8F0FB"], [1, C_NAVY]],
            showscale=False,
            text=[[f"{v:,}" for v in row] for row in vals],
            texttemplate="%{text}",
            textfont={"size": 16, "family": "Inter"},
        ))
        fig.update_layout(
            height=300,
            margin=dict(t=20, b=60, l=130, r=20),
            paper_bgcolor="white",
            font_family="Inter",
            xaxis=dict(tickfont={"size": 10}),
            yaxis=dict(tickfont={"size": 10}),
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)

    # SHAP importance + images
    left2, right2 = st.columns(2, gap="large")

    with left2:
        st.markdown('<div class="section-label">Top 15 SHAP Features</div>',
                    unsafe_allow_html=True)
        top15 = shap_top.head(15)
        fig = go.Figure(go.Bar(
            x=top15["mean_abs_shap"][::-1].tolist(),
            y=top15["feature"][::-1].tolist(),
            orientation="h",
            marker_color=C_BLUE,
            text=[f"{v:.4f}" for v in top15["mean_abs_shap"][::-1]],
            textposition="outside",
            textfont={"size": 9, "family": "Inter"},
        ))
        fig.update_layout(
            height=430, paper_bgcolor="white", plot_bgcolor="#FAFBFF",
            xaxis=dict(showgrid=True, gridcolor="#E5EAF3",
                       title="Mean |SHAP Value|",
                       titlefont={"size": 11}),
            yaxis=dict(tickfont={"size": 9}),
            margin=dict(t=10, b=40, l=10, r=80),
            font_family="Inter", showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})

    with right2:
        st.markdown('<div class="section-label">SHAP Beeswarm</div>',
                    unsafe_allow_html=True)
        st.image(str(MODELS_DIR / "shap_summary.png"), width=520)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Credit Score Dependence — External Bureau</div>',
                unsafe_allow_html=True)
    st.image(str(MODELS_DIR / "shap_dependence_ext_source.png"), width=700)

    # Technical specs table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Technical Specification</div>',
                unsafe_allow_html=True)
    spec = [
        ("Algorithm",        "LightGBM — Gradient Boosted Decision Trees"),
        ("Validation",       "5-Fold Stratified K-Fold Cross Validation"),
        ("Training Data",    "307,511 real loan applicants (Home Credit Group)"),
        ("Features",         "290 engineered features across 8 data sources"),
        ("Class Imbalance",  "scale_pos_weight = 11.4 (11.4 non-defaults per default)"),
        ("Early Stopping",   "100 rounds on validation AUC, max 5,000 trees"),
        ("Explainability",   "SHAP TreeExplainer — exact Shapley values per prediction"),
        ("Threshold Tuning", "F1-maximised on out-of-fold predictions"),
        ("OOF ROC-AUC",      f"{metrics['oof_roc_auc']} — top-tier for this dataset"),
    ]
    rows_html = ""
    for i, (k, v) in enumerate(spec):
        bg = C_LBLUE if i % 2 == 0 else C_WHITE
        rows_html += f"""<tr style="background:{bg}">
          <td style="font-weight:600;color:{C_NAVY};width:220px">{k}</td>
          <td style="color:#374151">{v}</td>
        </tr>"""
    st.markdown(f"""
    <table class="styled-table">
      <thead><tr><th>Component</th><th>Details</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    feature_cols, threshold, metrics, feat_desc, shap_top = load_meta()
    pipeline_meta = load_pipeline_meta()

    page = render_sidebar(metrics, threshold)

    if   page == "🔍  Applicant Scoring":
        page_single(feature_cols, threshold, feat_desc, pipeline_meta)
    elif page == "📋  Portfolio Batch":
        page_batch(feature_cols, threshold, pipeline_meta)
    elif page == "📊  AI Engine":
        page_overview(metrics, shap_top)


if __name__ == "__main__":
    main()