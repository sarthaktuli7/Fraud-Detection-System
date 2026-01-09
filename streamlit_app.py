"""
Fraud Detection Dashboard - Standalone Version for Streamlit Cloud
Author: Sarthak Tuli
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import joblib
import os

# Page config
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
)

# Load models
@st.cache_resource
def load_models():
    try:
        models_dir = "models"
        fe_pipeline = joblib.load(os.path.join(models_dir, "feature_engineering_pipeline.pkl"))
        ensemble_model = joblib.load(os.path.join(models_dir, "fraud_detection_ensemble.pkl"))
        return fe_pipeline, ensemble_model, True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, False

fe_pipeline, ensemble_model, models_loaded = load_models()

# Helper functions
def get_risk_level(prob):
    if prob >= 0.9: return "CRITICAL"
    if prob >= 0.7: return "HIGH"
    if prob >= 0.3: return "MEDIUM"
    return "LOW"

def predict_fraud(transaction_df):
    if not models_loaded:
        return 0.0
    transformed = fe_pipeline.transform(transaction_df)
    prob = ensemble_model.predict_proba(transformed)[0, 1]
    return float(prob)

def generate_transaction(fraud_bias=0.1):
    if np.random.random() < fraud_bias:
        v_features = {f"V{i}": np.random.normal(0, 2) for i in range(1, 21)}
        amount = np.random.lognormal(4, 1.5)
    else:
        v_features = {f"V{i}": np.random.normal(0, 1) for i in range(1, 21)}
        amount = np.random.lognormal(3, 1)
    
    return {
        "Time": np.random.uniform(0, 172800),
        "Amount": round(max(0.01, amount), 2),
        **v_features
    }

# Main UI
st.title("🛡️ Fraud Detection Dashboard")
st.markdown("Real-time ML-powered fraud detection system")

# Sidebar
st.sidebar.header("⚙️ Configuration")
st.sidebar.success("✅ Models Loaded" if models_loaded else "❌ Models Not Loaded")
fraud_bias = st.sidebar.slider("Fraud simulation rate", 0.0, 0.5, 0.1, 0.01)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "counter" not in st.session_state:
    st.session_state.counter = 1000

# Simulate button
st.header("📊 Real-Time Transaction Monitoring")

if st.button("🎲 Simulate New Transaction", type="primary"):
    transaction = generate_transaction(fraud_bias)
    df = pd.DataFrame([transaction])
    
    fraud_prob = predict_fraud(df)
    risk_level = get_risk_level(fraud_prob)
    
    record = {
        "transaction_id": f"TXN_{st.session_state.counter}",
        "Amount": transaction["Amount"],
        "fraud_probability": fraud_prob,
        "risk_level": risk_level,
        "is_fraud": fraud_prob > 0.5,
        "timestamp": datetime.now()
    }
    st.session_state.history.append(record)
    st.session_state.counter += 1
    
    # Display result
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Amount", f"${transaction['Amount']:,.2f}")
    with col2:
        st.metric("Fraud Probability", f"{fraud_prob:.1%}")
    with col3:
        colors = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}
        st.metric("Risk Level", f"{colors.get(risk_level, '⚪')} {risk_level}")

# History
if st.session_state.history:
    st.header("📈 Transaction History")
    df = pd.DataFrame(st.session_state.history)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", len(df))
    col2.metric("Fraud Detected", sum(df["is_fraud"]))
    col3.metric("Avg Amount", f"${df['Amount'].mean():,.2f}")
    col4.metric("High Risk", sum(df["risk_level"].isin(["HIGH", "CRITICAL"])))
    
    # Charts
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="fraud_probability", nbins=20, title="Fraud Probability Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        risk_counts = df["risk_level"].value_counts()
        fig = px.pie(values=risk_counts.values, names=risk_counts.index, title="Risk Level Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Table
    st.subheader("📋 Recent Transactions")
    display_df = df[["transaction_id", "Amount", "fraud_probability", "risk_level", "is_fraud"]].tail(10)
    display_df["Amount"] = display_df["Amount"].apply(lambda x: f"${x:,.2f}")
    display_df["fraud_probability"] = display_df["fraud_probability"].apply(lambda x: f"{x:.1%}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# Footer
st.divider()
st.markdown("**🛡️ Fraud Detection System** | Built by Sarthak Tuli | Syracuse University")
