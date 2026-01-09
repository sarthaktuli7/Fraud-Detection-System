"""
Fraud Detection Dashboard - Streamlit Cloud Version
Author: Sarthak Tuli
Date: January 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import joblib
import os
import time
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main > div {padding-top: 2rem;}
    .stMetric {background-color: #f0f2f6; padding: 15px; border-radius: 5px;}
    div[data-testid="stMetricValue"] {font-size: 28px;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# REQUIRED CLASSES FOR MODEL LOADING
# ============================================================================

class SimpleFeatureEngineering:
    """Feature engineering pipeline - must match training code"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
        self.fitted = False
    
    def fit(self, df):
        df_transformed = self._add_features(df.copy())
        X = df_transformed.drop('Class', axis=1, errors='ignore')
        self.feature_names = list(X.columns)
        self.scaler.fit(X)
        self.fitted = True
        return self
    
    def transform(self, df):
        df_transformed = self._add_features(df.copy())
        for col in self.feature_names:
            if col not in df_transformed.columns:
                df_transformed[col] = 0
        X = df_transformed[self.feature_names]
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=self.feature_names)
    
    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
    
    def _add_features(self, df):
        df['Hour'] = (df['Time'] % 86400) / 3600
        df['Is_Night'] = ((df['Hour'] >= 22) | (df['Hour'] <= 5)).astype(int)
        df['Amount_Log'] = np.log1p(df['Amount'])
        df['V1_V3_product'] = df['V1'] * df['V3']
        df['V4_V12_product'] = df['V4'] * df['V12']
        v_cols = [f'V{i}' for i in range(1, 21)]
        existing_v_cols = [c for c in v_cols if c in df.columns]
        if existing_v_cols:
            df['V_mean'] = df[existing_v_cols].mean(axis=1)
            df['V_std'] = df[existing_v_cols].std(axis=1)
        return df


class EnsembleModel:
    """Ensemble model - must match training code"""
    
    def __init__(self, rf_model=None, xgb_model=None, lr_model=None, scaler=None, feature_names=None, weights=None):
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.lr_model = lr_model
        self.scaler = scaler
        self.feature_names = feature_names
        self.weights = weights or {'rf': 0.3, 'xgb': 0.5, 'lr': 0.2}
    
    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.feature_names]
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        rf_proba = self.rf_model.predict_proba(X_scaled)[:, 1]
        xgb_proba = self.xgb_model.predict_proba(X_scaled)[:, 1]
        lr_proba = self.lr_model.predict_proba(X_scaled)[:, 1]
        
        ensemble_proba = (
            self.weights['rf'] * rf_proba +
            self.weights['xgb'] * xgb_proba +
            self.weights['lr'] * lr_proba
        )
        return np.column_stack([1 - ensemble_proba, ensemble_proba])
    
    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

# ============================================================================
# LOAD MODELS
# ============================================================================

@st.cache_resource
def load_models():
    try:
        models_dir = "models"
        fe_pipeline = joblib.load(os.path.join(models_dir, "feature_engineering_pipeline.pkl"))
        ensemble_model = joblib.load(os.path.join(models_dir, "fraud_detection_ensemble.pkl"))
        return fe_pipeline, ensemble_model, True
    except Exception as e:
        st.error(f"⚠️ Error loading models: {str(e)}")
        return None, None, False

fe_pipeline, ensemble_model, models_loaded = load_models()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_risk_level(prob):
    if prob >= 0.9: return "CRITICAL", "🔴"
    if prob >= 0.7: return "HIGH", "🟠"
    if prob >= 0.3: return "MEDIUM", "🟡"
    return "LOW", "🟢"

def get_risk_color(risk_level):
    colors = {"LOW": "#28a745", "MEDIUM": "#ffc107", "HIGH": "#fd7e14", "CRITICAL": "#dc3545"}
    return colors.get(risk_level, "#6c757d")

def predict_fraud(transaction_df):
    if not models_loaded:
        return 0.0
    try:
        transformed = fe_pipeline.transform(transaction_df)
        prob = ensemble_model.predict_proba(transformed)[0, 1]
        return float(prob)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0.0

def generate_transaction(fraud_bias=0.1):
    """Generate synthetic transaction"""
    is_fraud = np.random.random() < fraud_bias
    
    if is_fraud:
        v_features = {f"V{i}": np.random.normal(0, 2.5) for i in range(1, 21)}
        amount = np.random.lognormal(5, 1.5)
    else:
        v_features = {f"V{i}": np.random.normal(0, 1) for i in range(1, 21)}
        amount = np.random.lognormal(3, 1)
    
    return {
        "Time": np.random.uniform(0, 172800),
        "Amount": round(max(0.01, amount), 2),
        **v_features
    }

def create_gauge_chart(probability):
    """Create gauge chart for fraud probability"""
    risk_level, emoji = get_risk_level(probability)
    color = get_risk_color(risk_level)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{emoji} Fraud Probability", 'font': {'size': 20}},
        number={'suffix': "%", 'font': {'size': 32}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d4edda'},
                {'range': [30, 70], 'color': '#fff3cd'},
                {'range': [70, 90], 'color': '#f8d7da'},
                {'range': [90, 100], 'color': '#f5c6cb'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

if "history" not in st.session_state:
    st.session_state.history = []
if "counter" not in st.session_state:
    st.session_state.counter = 1000
if "total_fraud" not in st.session_state:
    st.session_state.total_fraud = 0
if "total_high_risk" not in st.session_state:
    st.session_state.total_high_risk = 0

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    if models_loaded:
        st.success("✅ Models Loaded Successfully")
    else:
        st.error("❌ Models Not Loaded")
    
    st.divider()
    
    fraud_bias = st.slider(
        "Fraud Simulation Rate",
        min_value=0.0,
        max_value=0.5,
        value=0.18,
        step=0.01,
        help="Probability of generating fraudulent transactions"
    )
    
    auto_refresh = st.checkbox("Auto-refresh Dashboard", value=False)
    if auto_refresh:
        refresh_interval = st.slider("Refresh Interval (seconds)", 1, 30, 5)
    
    st.divider()
    
    st.subheader("📊 Model Info")
    st.markdown("""
    - **Algorithm:** Ensemble (RF + XGB + LR)
    - **ROC-AUC:** 0.968
    - **F1-Score:** 91.9%
    - **Precision:** 94.2%
    - **Recall:** 89.7%
    """)
    
    st.divider()
    
    st.subheader("👨‍💻 About")
    st.markdown("**Sarthak Tuli**")
    st.markdown("Syracuse University")
    st.markdown("Real-time fraud detection using ensemble ML models")
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.counter = 1000
        st.session_state.total_fraud = 0
        st.session_state.total_high_risk = 0
        st.rerun()

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("🛡️ Fraud Detection System")
st.markdown("Real-time ML-powered fraud detection using Ensemble Models")
st.divider()

# Top metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Transactions", len(st.session_state.history))

with col2:
    fraud_pct = f"{(st.session_state.total_fraud/len(st.session_state.history)*100) if st.session_state.history else 0:.1f}%"
    st.metric("Fraud Detected", st.session_state.total_fraud, delta=fraud_pct)

with col3:
    st.metric("High Risk", st.session_state.total_high_risk)

with col4:
    avg_amount = sum(t["Amount"] for t in st.session_state.history) / len(st.session_state.history) if st.session_state.history else 0
    st.metric("Avg Amount", f"${avg_amount:,.2f}")

st.divider()

# ============================================================================
# TRANSACTION MONITORING
# ============================================================================

st.header("📊 Real-Time Transaction Monitoring")

col1, col2 = st.columns([2, 3])

with col1:
    if st.button("🎲 Simulate New Transaction", type="primary", use_container_width=True):
        transaction = generate_transaction(fraud_bias)
        df = pd.DataFrame([transaction])
        
        start_time = time.time()
        fraud_prob = predict_fraud(df)
        processing_time = (time.time() - start_time) * 1000
        
        risk_level, emoji = get_risk_level(fraud_prob)
        is_fraud = fraud_prob > 0.5
        is_high_risk = risk_level in ["HIGH", "CRITICAL"]
        
        record = {
            "transaction_id": f"TXN_{st.session_state.counter}",
            "Amount": transaction["Amount"],
            "fraud_probability": fraud_prob,
            "risk_level": risk_level,
            "is_fraud": is_fraud,
            "is_high_risk": is_high_risk,
            "processing_time_ms": processing_time,
            "timestamp": datetime.now()
        }
        
        st.session_state.history.append(record)
        st.session_state.counter += 1
        if is_fraud:
            st.session_state.total_fraud += 1
        if is_high_risk:
            st.session_state.total_high_risk += 1
        
        st.rerun()

with col2:
    if st.session_state.history:
        latest = st.session_state.history[-1]
        st.info(f"**Latest:** {latest['transaction_id']} | ${latest['Amount']:,.2f} | {latest['risk_level']} | {latest['processing_time_ms']:.1f}ms")

# Display latest transaction
if st.session_state.history:
    latest = st.session_state.history[-1]
    
    st.subheader("Latest Transaction Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Transaction ID", latest['transaction_id'])
    with col2:
        st.metric("Amount", f"${latest['Amount']:,.2f}")
    with col3:
        risk_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}
        st.metric("Risk Level", f"{risk_emoji.get(latest['risk_level'], '⚪')} {latest['risk_level']}")
    with col4:
        st.metric("Processing Time", f"{latest['processing_time_ms']:.1f}ms")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        fig = create_gauge_chart(latest['fraud_probability'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📋 Transaction Details")
        st.markdown(f"""
        - **Fraud Probability:** {latest['fraud_probability']:.2%}
        - **Classification:** {'🚨 FRAUD' if latest['is_fraud'] else '✅ LEGITIMATE'}
        - **Risk Score:** {latest['fraud_probability'] * 100:.1f}/100
        - **Timestamp:** {latest['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
        """)

# ============================================================================
# HISTORICAL ANALYSIS
# ============================================================================

if st.session_state.history:
    st.divider()
    st.header("📈 Historical Analysis")
    
    df = pd.DataFrame(st.session_state.history)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df, 
            x="fraud_probability",
            nbins=30,
            title="Fraud Probability Distribution",
            labels={"fraud_probability": "Fraud Probability"},
            color_discrete_sequence=["#1f77b4"]
        )
        fig.update_layout(showlegend=False, height=400)
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        risk_counts = df["risk_level"].value_counts()
        colors_map = {"LOW": "#28a745", "MEDIUM": "#ffc107", "HIGH": "#fd7e14", "CRITICAL": "#dc3545"}
        
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Risk Level Distribution",
            color=risk_counts.index,
            color_discrete_map=colors_map
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        df_timeline = df.copy()
        fig = px.scatter(
            df_timeline,
            x='timestamp',
            y='fraud_probability',
            color='risk_level',
            size='Amount',
            title="Transaction Timeline",
            labels={"fraud_probability": "Fraud Probability", "timestamp": "Time"},
            color_discrete_map=colors_map,
            hover_data=['transaction_id', 'Amount']
        )
        fig.update_layout(height=400)
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            df,
            x='Amount',
            y='fraud_probability',
            color='risk_level',
            title="Amount vs Fraud Probability",
            labels={"fraud_probability": "Fraud Probability"},
            color_discrete_map=colors_map,
            hover_data=['transaction_id']
        )
        fig.update_layout(height=400)
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📋 Recent Transactions")
    
    display_df = df[["transaction_id", "Amount", "fraud_probability", "risk_level", "is_fraud", "processing_time_ms"]].tail(15).copy()
    display_df["Amount"] = display_df["Amount"].apply(lambda x: f"${x:,.2f}")
    display_df["fraud_probability"] = display_df["fraud_probability"].apply(lambda x: f"{x:.1%}")
    display_df["is_fraud"] = display_df["is_fraud"].apply(lambda x: "🚨 Yes" if x else "✅ No")
    display_df["processing_time_ms"] = display_df["processing_time_ms"].apply(lambda x: f"{x:.1f}ms")
    display_df = display_df.rename(columns={
        "transaction_id": "Transaction ID",
        "Amount": "Amount",
        "fraud_probability": "Fraud Prob",
        "risk_level": "Risk Level",
        "is_fraud": "Fraud?",
        "processing_time_ms": "Processing Time"
    })
    
    st.dataframe(display_df.iloc[::-1], use_container_width=True, hide_index=True)
    
    st.subheader("📊 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Transactions Analyzed", len(df))
    with col2:
        st.metric("Fraud Rate", f"{(df['is_fraud'].sum()/len(df)*100):.1f}%")
    with col3:
        st.metric("Avg Processing Time", f"{df['processing_time_ms'].mean():.1f}ms")
    with col4:
        st.metric("High/Critical Risk", f"{df['is_high_risk'].sum()}")

else:
    st.info("👆 Click 'Simulate New Transaction' to start analyzing transactions!")

st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>🛡️ Fraud Detection System</strong> | Built by <strong>Sarthak Tuli</strong> | Ensemble ML (RF + XGBoost + LR)</p>
    <p>Syracuse University - Martin J. Whitman School of Management</p>
</div>
""", unsafe_allow_html=True)

if auto_refresh and st.session_state.history:
    time.sleep(refresh_interval)
    st.rerun()
