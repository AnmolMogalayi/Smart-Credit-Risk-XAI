import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from backend.predict import predict_single, THRESHOLD, MODEL

st.set_page_config(
    page_title="Finexcore XAI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 💎 PREMIUM GLASSMORPHISM AESTHETICS & CUSTOM CSS
# ==============================================================================
st.markdown("""
<style>
    /* Import Premium Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Animations */
    @keyframes fadeSlideUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulseGlow {
        0% { box-shadow: 0 4px 10px rgba(0, 212, 255, 0.2); }
        50% { box-shadow: 0 4px 25px rgba(0, 212, 255, 0.5); }
        100% { box-shadow: 0 4px 10px rgba(0, 212, 255, 0.2); }
    }
    
    /* Global App Background */
    .stApp {
        background: linear-gradient(135deg, #07090D 0%, #0E1117 100%);
    }

    /* Metric Cards Glassmorphism */
    div[data-testid="stMetric"] {
        background: rgba(30, 42, 56, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-top: 3px solid #00D4FF;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeSlideUp 0.6s ease-out forwards;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 212, 255, 0.2);
    }
    div[data-testid="stMetricValue"] {
        color: #FFFFFF;
        font-weight: 700;
    }

    /* Inputs and Segmented Controls Styling */
    .st-emotion-cache-1n76uvr { /* Primary Button Hover */
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px 0 rgba(0, 212, 255, 0.3) !important;
    }
    
    div.stButton > button {
        border-radius: 12px;
        background: linear-gradient(90deg, #00D4FF, #0088FF);
        border: none;
        color: white;
        font-weight: bold;
        transition: transform 0.2s, box-shadow 0.2s;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        animation: pulseGlow 3s infinite;
    }
    
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 20px rgba(0, 212, 255, 0.4);
    }

    /* Container Borders */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.05) !important;
        background: rgba(14, 17, 23, 0.6);
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        animation: fadeSlideUp 0.8s ease-out forwards;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# HEADER
# ==============================================================================
col_logo, col_title = st.columns([1, 10])
with col_logo:
    st.markdown("<h1 style='color: #00D4FF;'>🏦</h1>", unsafe_allow_html=True)
with col_title:
    st.markdown("<h1 style='margin-bottom: 0px;'>Finexcore AI Lending Intelligence</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8B949E;'>Deep Credit Risk Assessment & Portfolio XAI Analytics</p>", unsafe_allow_html=True)

st.divider()

# ==============================================================================
# DASHBOARD LAYOUT & INPUTS
# ==============================================================================

with st.sidebar:
    st.header("Applicant Configuration")
    
    st.subheader("Demographics")
    gender = st.segmented_control("Gender", options=["Male", "Female"], default="Female")
    code_gender = 1.0 if gender == "Female" else 0.0
    
    age_years = st.slider("Target Age", 18.0, 90.0, 35.0)
    cnt_children = st.number_input("Children", min_value=0, max_value=15, value=0)
    cnt_fam_members = st.number_input("Family Members", min_value=1, max_value=20, value=max(2, int(cnt_children)+1))
    
    car_realty = st.pills("Asset Ownership", options=["Owns Car", "Owns Realty"], selection_mode="multi", default=["Owns Realty"])
    flag_own_car = 1.0 if "Owns Car" in car_realty else 0.0
    flag_own_realty = 1.0 if "Owns Realty" in car_realty else 0.0

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.subheader("Financial Footprint")
        amt_income = st.number_input("Annual Income ($)", 10000.0, 1000000.0, 200000.0, step=10000.0)
        unemp_stat = st.segmented_control("Employment Status", ["Employed", "Unemployed"], default="Employed")
        is_unemployed = 1.0 if unemp_stat == "Unemployed" else 0.0
        employ_years = st.slider("Years Employed", 0.0, 40.0, 5.0) if is_unemployed == 0.0 else 0.0
        
with col2:
    with st.container(border=True):
        st.subheader("Loan Parameters")
        amt_credit = st.number_input("Requested Credit ($)", 10000.0, 5000000.0, 500000.0, step=50000.0)
        amt_goods = st.number_input("Goods Price ($)", 0.0, 5000000.0, 450000.0, step=50000.0)
        amt_annuity = st.number_input("Monthly Annuity ($)", 1000.0, 200000.0, 25000.0, step=1000.0)

with col3:
    with st.container(border=True):
        st.subheader("Credit / External")
        ext1 = st.slider("Ext Source 1", 0.0, 1.0, 0.5)
        ext2 = st.slider("Ext Source 2", 0.0, 1.0, 0.5)
        ext3 = st.slider("Ext Source 3", 0.0, 1.0, 0.5)
        
        ins_paid_late = st.slider("Late Paid Ratio", 0.0, 1.0, 0.05)
        prev_approved = st.slider("Prev Approvals", 0.0, 1.0, 0.70)


# ==============================================================================
# LOGIC & INFERENCE ENGINE
# ==============================================================================

if st.button("🚀 Execute Risk ML Pipeline", use_container_width=True):
    with st.spinner("Scoring profile & calculating SHAP explanations..."):
        
        # Exact logic mapping replicating FastAPI backend mapping
        income = max(np.float64(amt_income), 1.0)
        credit = max(np.float64(amt_credit), 1.0)
        annuity = max(np.float64(amt_annuity), 1.0)
        goods = np.float64(amt_goods)
        age = max(np.float64(age_years), 1.0)
        emp = np.float64(employ_years)
        fam = max(np.float64(cnt_fam_members), 1.0)
        
        input_dict = {
            "AGE_YEARS": age,
            "CODE_GENDER": code_gender,
            "CNT_CHILDREN": np.float64(cnt_children),
            "CNT_FAM_MEMBERS": fam,
            "FLAG_OWN_CAR": flag_own_car,
            "FLAG_OWN_REALTY": flag_own_realty,
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": annuity,
            "AMT_GOODS_PRICE": goods,
            "EMPLOYED_YEARS": emp,
            "IS_UNEMPLOYED": is_unemployed,
            "EXT_SOURCE_1": np.float64(ext1),
            "EXT_SOURCE_2": np.float64(ext2),
            "EXT_SOURCE_3": np.float64(ext3),
            "INS_PAID_LATE_RATIO": np.float64(ins_paid_late),
            "PREV_APPROVED_RATIO": np.float64(prev_approved),
            
            # Derived Logical columns 
            "DAYS_BIRTH": age * 365.0,
            "DAYS_EMPLOYED": 0.0 if is_unemployed else emp * 365.0,
            "CREDIT_INCOME_RATIO": credit / income,
            "ANNUITY_INCOME_RATIO": annuity / income,
            "CREDIT_TERM": annuity / credit,
            "GOODS_CREDIT_RATIO": goods / credit,
            "INCOME_PER_PERSON": income / fam,
            "EMPLOYED_TO_AGE_RATIO": 0.0 if is_unemployed else emp / age,
            "EXT_SOURCE_MEAN": (ext1 + ext2 + ext3) / 3.0,
            "EXT_SOURCE_STD": float(np.std([ext1, ext2, ext3])),
            "EXT_SOURCE_PRODUCT": ext1 * ext2 * ext3,
            "EXT_SOURCE_MIN": min(ext1, ext2, ext3),
            "INS_DAYS_LATE_MEAN": 5.0 * 0.3 # Default from prev interface
        }
        
        # Execute local inference
        results = predict_single(input_dict)
        prob = results["probability"]
        tier = results["risk_tier"]
        dec = results["decision"]
        
        st.divider()
        st.subheader("Model Decision")
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        m_col1.metric("Risk Decision", dec)
        m_col2.metric("Probability of Default", f"{prob*100:.2f}%")
        m_col3.metric("Risk Tier", tier)
        m_col4.metric("Credit Grade", results["credit_score"])
        
        # ── Visualizations ──
        viz1, viz2 = st.columns([1, 1.5])
        
        with viz1:
            # Custom Gauge Chart
            gauge_color = "#00C853" if dec == "APPROVE" else "#FF6D00" if dec == "REVIEW" else "#D50000"
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Default Risk (%)", 'font': {'color': 'white'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "white"},
                    'bar': {'color': gauge_color},
                    'bgcolor': "#1E2A38",
                    'borderwidth': 2,
                    'bordercolor': "#0E1117",
                    'steps': [
                        {'range': [0, THRESHOLD*100], 'color': "#0E1117"},
                        {'range': [THRESHOLD*100, 100], 'color': "rgba(255, 109, 0, 0.1)"}
                    ]
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "white", 'family': "Inter"},
                height=350,
                margin=dict(t=50, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with viz2:
            # SHAP Waterfall/Bar
            shap_data = results["shap_factors"]
            df_shap = pd.DataFrame(shap_data)
            df_shap = df_shap.sort_values(by="shap", ascending=True)

            fig_shap = px.bar(
                df_shap,
                x='shap',
                y='description',
                orientation='h',
                color='direction',
                color_discrete_map={"increases_risk": "#D50000", "reduces_risk": "#00C853"},
                labels={'shap': 'SHAP Impact (Log Odds)', 'description': 'Feature'},
                title="Explainable AI: Feature Risk Contribution"
            )
            fig_shap.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "white", 'family': "Inter"},
                height=350,
                showlegend=False,
                margin=dict(t=50, b=10, l=10, r=10)
            )
            fig_shap.update_yaxes(title="")
            st.plotly_chart(fig_shap, use_container_width=True)

        st.divider()
        st.subheader("Deep Relational Insights")
        viz3, viz4 = st.columns(2)
        
        with viz3:
            # 3. Radar Chart for Trust Dimensions
            categories = ['External Src 1', 'External Src 2', 'External Src 3', 'Prior Approvals', 'Punctuality (1 - Late Ratio)']
            vals = [ext1, ext2, ext3, prev_approved, max(0.0, 1.0 - ins_paid_late)]
            
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=vals + [vals[0]], # Close loop
                theta=categories + [categories[0]],
                fill='toself',
                fillcolor='rgba(0, 212, 255, 0.2)',
                line=dict(color='#00D4FF', width=2),
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255, 255, 255, 0.1)', tickfont=dict(color='white')),
                    angularaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', linecolor='rgba(255, 255, 255, 0.2)', tickfont=dict(color='white'))
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "white", 'family': "Inter"},
                title="Applicant Trust Radar",
                margin=dict(t=50, b=10, l=40, r=40)
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
        with viz4:
            # 4. Donut Chart for Financial Stress Ratio
            annual_debt = annuity * 12.0
            disposable = max(0.0, income - annual_debt)
            
            dti = (annual_debt / income) * 100 if income > 0 else 100
            debt_color = '#D50000' if dti > 60 else '#FF6D00' if dti > 40 else '#34d399'
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Annual Obligation (Debt)', 'Disposable Income Cushion'], 
                values=[annual_debt, disposable], 
                hole=.6,
                marker=dict(colors=[debt_color, '#00C853']),
                hoverinfo="label+percent"
            )])
            fig_pie.update_layout(
                title="Financial Liquidity Stress",
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': "white", 'family': "Inter"},
                margin=dict(t=50, b=10, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
            )
            fig_pie.add_annotation(
                text=f"DTI<br><b style='font-size:24px;'>{dti:.1f}%</b>", 
                x=0.5, y=0.5, showarrow=False, font_color="white"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
