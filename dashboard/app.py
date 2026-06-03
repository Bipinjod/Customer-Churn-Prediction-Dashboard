"""
Customer Churn Prediction Dashboard
------------------------------------
Run from the project root:
    streamlit run dashboard/app.py
"""

import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Page configuration — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS / Theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: sans-serif;
}

/* ── Main background ── */
.stApp {
    background-color: #f8f9fa;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #1e1e2e !important;
}
[data-testid="stSidebar"] * {
    color: #cdd6f4 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: #a6adc8 !important;
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
[data-testid="stSidebar"] hr {
    border-color: #313244 !important;
}

/* ── Metric cards (st.metric) ── */
[data-testid="stMetric"] {
    background: #ffffff;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border-left: 4px solid #6c63ff;
}

/* ── Plotly chart containers ── */
[data-testid="stPlotlyChart"] {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    background: #ffffff;
    padding: 8px;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #ffffff;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    border: 1px solid #e9ecef;
}

/* ── Horizontal rule ── */
hr {
    border-color: #dee2e6;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file, so the app works from any cwd)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
IMPORTANCE_PATH = BASE_DIR / "models" / "feature_importance.csv"
DATA_PATH = BASE_DIR / "data" / "telco_churn.csv"

# ---------------------------------------------------------------------------
# Label-encoding maps — must exactly mirror the LabelEncoder in preprocess.py
# (LabelEncoder sorts unique values alphabetically and assigns 0, 1, 2 …)
# ---------------------------------------------------------------------------
ENCODE = {
    "gender":           {"Female": 0, "Male": 1},
    "Partner":          {"No": 0, "Yes": 1},
    "Dependents":       {"No": 0, "Yes": 1},
    "PhoneService":     {"No": 0, "Yes": 1},
    "MultipleLines":    {"No": 0, "No phone service": 1, "Yes": 2},
    "InternetService":  {"DSL": 0, "Fiber optic": 1, "No": 2},
    "OnlineSecurity":   {"No": 0, "No internet service": 1, "Yes": 2},
    "OnlineBackup":     {"No": 0, "No internet service": 1, "Yes": 2},
    "DeviceProtection": {"No": 0, "No internet service": 1, "Yes": 2},
    "TechSupport":      {"No": 0, "No internet service": 1, "Yes": 2},
    "StreamingTV":      {"No": 0, "No internet service": 1, "Yes": 2},
    "StreamingMovies":  {"No": 0, "No internet service": 1, "Yes": 2},
    "Contract":         {"Month-to-month": 0, "One year": 1, "Two year": 2},
    "PaperlessBilling": {"No": 0, "Yes": 1},
    "PaymentMethod":    {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)":   1,
        "Electronic check":          2,
        "Mailed check":              3,
    },
}

# The exact feature order the model was trained on
FEATURE_ORDER = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)


@st.cache_data
def load_importance():
    return pd.read_csv(IMPORTANCE_PATH)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    return df


model = load_model()
scaler = load_scaler()
importance_df = load_importance()
df_raw = load_data()


# ---------------------------------------------------------------------------
# ── SIDEBAR — Customer Input Form ──────────────────────────────────────────
# ---------------------------------------------------------------------------
st.sidebar.markdown("""
<div style="padding:4px 0 12px 0;">
    <div style="font-size:1.1rem; font-weight:700; color:#cba6f7; letter-spacing:0.02em;">
        Customer Profile
    </div>
    <div style="font-size:0.78rem; color:#a6adc8; margin-top:4px;">
        Adjust the inputs to predict churn probability.
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    gender          = st.selectbox("Gender",           ["Female", "Male"])
    senior          = st.selectbox("Senior Citizen",   ["No", "Yes"])
    partner         = st.selectbox("Partner",          ["No", "Yes"])
    dependents      = st.selectbox("Dependents",       ["No", "Yes"])
    tenure          = st.slider("Tenure (months)",     0, 72, 12)

    st.markdown("---")
    phone_service   = st.selectbox("Phone Service",    ["No", "Yes"])
    multiple_lines  = st.selectbox("Multiple Lines",   ["No", "No phone service", "Yes"])
    internet        = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    st.markdown("---")
    online_security  = st.selectbox("Online Security",    ["No", "No internet service", "Yes"])
    online_backup    = st.selectbox("Online Backup",      ["No", "No internet service", "Yes"])
    device_protect   = st.selectbox("Device Protection",  ["No", "No internet service", "Yes"])
    tech_support     = st.selectbox("Tech Support",       ["No", "No internet service", "Yes"])
    streaming_tv     = st.selectbox("Streaming TV",       ["No", "No internet service", "Yes"])
    streaming_movies = st.selectbox("Streaming Movies",   ["No", "No internet service", "Yes"])

    st.markdown("---")
    contract         = st.selectbox("Contract",        ["Month-to-month", "One year", "Two year"])
    paperless        = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment          = st.selectbox("Payment Method",  [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ])
    monthly_charges  = st.slider("Monthly Charges ($)", 0.0, 150.0, 65.0, step=0.5)
    total_charges    = st.slider("Total Charges ($)",   0.0, 9000.0, 1500.0, step=10.0)


# ---------------------------------------------------------------------------
# Helper — build the input feature vector from sidebar values
# ---------------------------------------------------------------------------
def build_feature_vector() -> np.ndarray:
    raw = {
        "gender":           ENCODE["gender"][gender],
        "SeniorCitizen":    1 if senior == "Yes" else 0,   # already int in dataset
        "Partner":          ENCODE["Partner"][partner],
        "Dependents":       ENCODE["Dependents"][dependents],
        "tenure":           tenure,
        "PhoneService":     ENCODE["PhoneService"][phone_service],
        "MultipleLines":    ENCODE["MultipleLines"][multiple_lines],
        "InternetService":  ENCODE["InternetService"][internet],
        "OnlineSecurity":   ENCODE["OnlineSecurity"][online_security],
        "OnlineBackup":     ENCODE["OnlineBackup"][online_backup],
        "DeviceProtection": ENCODE["DeviceProtection"][device_protect],
        "TechSupport":      ENCODE["TechSupport"][tech_support],
        "StreamingTV":      ENCODE["StreamingTV"][streaming_tv],
        "StreamingMovies":  ENCODE["StreamingMovies"][streaming_movies],
        "Contract":         ENCODE["Contract"][contract],
        "PaperlessBilling": ENCODE["PaperlessBilling"][paperless],
        "PaymentMethod":    ENCODE["PaymentMethod"][payment],
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     total_charges,
    }
    df_input = pd.DataFrame([[raw[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
    return scaler.transform(df_input)


# ---------------------------------------------------------------------------
# ── MAIN PAGE ───────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

# ── Section 1: Gradient Banner Header ───────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 28px;
    box-shadow: 0 6px 30px rgba(0,0,0,0.2);
">
    <h1 style="color:#ffffff; margin:0; font-size:2rem; font-weight:700; letter-spacing:-0.02em;">
        Customer Churn Prediction Dashboard
    </h1>
    <p style="color:#c9d1d9; margin:10px 0 0 0; font-size:1rem; font-weight:300;">
        Use the <strong style='color:#a78bfa;'>sidebar</strong> to enter a customer profile.
        The dashboard predicts churn probability in real-time and highlights
        the key features driving the model's decision.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Section 2: Prediction Result ─────────────────────────────────────────
X_input = build_feature_vector()
churn_prob = float(model.predict_proba(X_input)[0][1]) * 100

# Determine risk tier
if churn_prob < 40:
    risk_label  = "Low Risk"
    risk_icon   = ""
    color       = "#2ecc71"
    border_hex  = "#2ecc71"
    desc        = "Customer is likely to stay."
elif churn_prob < 70:
    risk_label  = "Medium Risk"
    risk_icon   = ""
    color       = "#f39c12"
    border_hex  = "#f39c12"
    desc        = "Moderate churn risk — monitor closely."
else:
    risk_label  = "High Risk"
    risk_icon   = ""
    color       = "#e74c3c"
    border_hex  = "#e74c3c"
    desc        = "Customer is likely to churn — act now."

st.markdown("### Churn Prediction")
col_prob, col_risk, col_spacer = st.columns([2, 2, 4])

with col_prob:
    st.markdown(
        f"""
        <div style="
            text-align:center; padding:28px 20px; border-radius:14px;
            border-left: 6px solid {border_hex};
            background:#ffffff;
            box-shadow: 0 4px 16px rgba(0,0,0,0.08);">
            <div style="font-size:3.2rem; font-weight:700; color:{color};
                        letter-spacing:-0.02em; line-height:1;">
                {churn_prob:.1f}%
            </div>
            <div style="font-size:0.85rem; color:#6c757d; margin-top:8px;
                        font-weight:600; text-transform:uppercase; letter-spacing:0.06em;">
                Churn Probability
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_risk:
    st.markdown(
        f"""
        <div style="
            text-align:center; padding:28px 20px; border-radius:14px;
            border-left: 6px solid {border_hex};
            background:#ffffff;
            box-shadow: 0 4px 16px rgba(0,0,0,0.08);">
            <div style="font-size:1.4rem; font-weight:700; color:{color};
                        margin-top:4px;">{risk_label}</div>
            <div style="font-size:0.82rem; color:#6c757d; margin-top:6px;">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Section 3: Feature Importance Chart ──────────────────────────────────
st.subheader("Top 10 Feature Importances")
st.markdown("Features that most influence the model's predictions (from training).")

top10 = importance_df.head(10).sort_values("Importance")  # ascending for horizontal bar

fig_imp = px.bar(
    top10,
    x="Importance",
    y="Feature",
    orientation="h",
    color="Importance",
    color_continuous_scale="Blues",
    title="Top 10 Most Important Features",
    labels={"Importance": "Importance Score", "Feature": ""},
)
fig_imp.update_layout(
    height=420,
    showlegend=False,
    coloraxis_showscale=False,
    margin=dict(l=10, r=10, t=50, b=10),
)
st.plotly_chart(fig_imp, width='stretch')

st.markdown("---")

# ── Section 4: Data Overview (expander) ──────────────────────────────────
with st.expander("Data Overview — click to expand", expanded=False):
    st.markdown("### Dataset Insights from `telco_churn.csv`")

    ov_col1, ov_col2 = st.columns(2)

    # --- Churn distribution pie chart ---
    with ov_col1:
        churn_counts = df_raw["Churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]

        fig_pie = px.pie(
            churn_counts,
            names="Churn",
            values="Count",
            title="Churn Distribution",
            color="Churn",
            color_discrete_map={"No": "#2ecc71", "Yes": "#e74c3c"},
            hole=0.4,
        )
        fig_pie.update_traces(textinfo="percent+label")
        fig_pie.update_layout(height=380, margin=dict(t=50, b=10))
        st.plotly_chart(fig_pie, width='stretch')

    # --- Churn by contract type bar chart ---
    with ov_col2:
        contract_churn = (
            df_raw.groupby(["Contract", "Churn"])
            .size()
            .reset_index(name="Count")
        )

        fig_bar = px.bar(
            contract_churn,
            x="Contract",
            y="Count",
            color="Churn",
            barmode="group",
            title="Churn by Contract Type",
            color_discrete_map={"No": "#2ecc71", "Yes": "#e74c3c"},
        )
        fig_bar.update_layout(
            height=380,
            margin=dict(t=50, b=10),
            legend_title_text="Churn",
        )
        st.plotly_chart(fig_bar, width='stretch')

    st.markdown(
        f"**Dataset:** {len(df_raw):,} customers &nbsp;|&nbsp; "
        f"**Churn rate:** {df_raw['Churn'].eq('Yes').mean()*100:.1f}%"
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<hr style="margin-top:40px; border-color:#dee2e6;">
<div style="
    text-align:center;
    padding: 16px 0 24px 0;
    color: #6c757d;
    font-size: 0.82rem;
    font-weight: 400;
    letter-spacing: 0.02em;
">
    Built by <strong>Bipin Jod</strong> &nbsp;|&nbsp; BSc Computing Internship Project 2025
</div>
""", unsafe_allow_html=True)
