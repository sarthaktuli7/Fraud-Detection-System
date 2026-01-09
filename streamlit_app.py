"""
Fraud Detection Dashboard - Standalone Version for Streamlit Cloud
Author: Sarthak Tuli
Date: January 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import joblib
import os

# =====================================================================
# Page Configuration
# =====================================================================
st.set_page_config(
    page_title="Fraud Detection System | Sarthak Tuli",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main > div {padding-top: 2rem;}
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .danger-box {
        padding: 1rem;
        background-color: #f8d7da;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# Model Classes (Required for loading pickle files)
# =====================================================================
from sklearn.preprocessing import StandardScaler

class SimpleFeatureEngineering:
    """Feature engineering pipeline"""
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
    """Ensemble model combining RF, XGBoost, and Logistic Regression"""
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


# =====================================================================
# Load Models
# =====================================================================
@st.cache_resource
def load_models():
    """Load trained ML models"""
    try:
        models_dir = "models"
        fe_pipeline = joblib.load(os.path.join(models_dir, "feature_engineering_pipeline.pkl"))
        ensemble_model = joblib.load(os.path.join(models_dir, "fraud_detection_ensemble.pkl"))
        return fe_pipeline, ensemble_model, True
    except Exception as e:
        st.error(f"⚠️ Error loading models: {e}")
        return None, None, False

fe_pipeline, ensemble_model, models_loaded = load_models()


# =====================================================================
# Helper Functions
# =====================================================================
def get_risk_level(prob):
    """Classify risk level based on fraud probability"""
    if prob >= 0.9:
        return "CRITICAL", "🔴"
    if prob >= 0.7:
        return "HIGH", "🟠"
    if prob >= 0.3:
        return "MEDIUM", "🟡"
    return "LOW", "🟢"


def predict_fraud(transaction_df):
    """Make fraud prediction using loaded models"""
    if not models_loaded:
        return np.random.random()  # Fallback for demo
    try:
        transformed = fe_pipeline.transform(transaction_df)
        prob = ensemble_model.predict_proba(transformed)[0, 1]
        return float(prob)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0.0


def generate_transaction(fraud_bias=0.1):
    """Generate a simulated transaction"""
    if np.random.random() < fraud_bias:
        # Fraud-like transaction
        v_features = {f"V{i}": np.random.normal(-1, 2) for i in range(1, 21)}
        amount = np.random.lognormal(4, 1.5)
    else:
        # Normal transaction
        v_features = {f"V{i}": np.random.normal(0, 1) for i in range(1, 21)}
        amount = np.random.lognormal(3, 1)
    
    return {
        "Time": np.random.uniform(0, 172800),
        "Amount": round(max(0.01, min(amount, 25000)), 2),
        **v_features
    }


def create_gauge_chart(value, title):
    """Create a gauge chart for fraud probability"""
    if value < 0.3:
        color = "green"
    elif value < 0.7:
        color = "orange"
    else:
        color = "red"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        number={'suffix': '%', 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d4edda'},
                {'range': [30, 70], 'color': '#fff3cd'},
                {'range': [70, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig


# =====================================================================
# Main Application
# =====================================================================
def main():
    # Header
    st.title("🛡️ Fraud Detection System")
    st.markdown("**Real-time ML-powered fraud detection using Ensemble Models**")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Model status
    if models_loaded:
        st.sidebar.success("✅ Models Loaded Successfully")
    else:
        st.sidebar.error("❌ Models Not Loaded")
        st.sidebar.info("Running in demo mode with random predictions")
    
    # Settings
    fraud_bias = st.sidebar.slider(
        "Fraud Simulation Rate",
        min_value=0.0,
        max_value=0.5,
        value=0.1,
        step=0.01,
        help="Probability of generating a fraudulent transaction"
    )
    
    # Model info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Info")
    st.sidebar.markdown("""
    - **Algorithm**: Ensemble (RF + XGB + LR)
    - **ROC-AUC**: 0.968
    - **F1-Score**: 91.9%
    """)
    
    # Initialize session state
    if "history" not in st.session_state:
        st.session_state.history = []
    if "counter" not in st.session_state:
        st.session_state.counter = 1000
    
    # Main content
    st.markdown("---")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    total_txns = len(st.session_state.history)
    fraud_count = sum(1 for t in st.session_state.history if t.get('is_fraud', False))
    high_risk = sum(1 for t in st.session_state.history if t.get('risk_level') in ['HIGH', 'CRITICAL'])
    avg_amount = np.mean([t['Amount'] for t in st.session_state.history]) if st.session_state.history else 0
    
    col1.metric("Total Transactions", total_txns)
    col2.metric("Fraud Detected", fraud_count, delta=f"{(fraud_count/total_txns*100):.1f}%" if total_txns > 0 else "0%")
    col3.metric("High Risk", high_risk)
    col4.metric("Avg Amount", f"${avg_amount:,.2f}")
    
    st.markdown("---")
    
    # Transaction Simulation Section
    st.header("📊 Real-Time Transaction Monitoring")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("🎲 Simulate New Transaction", type="primary", use_container_width=True):
            # Generate transaction
            transaction = generate_transaction(fraud_bias)
            df = pd.DataFrame([transaction])
            
            # Get prediction
            fraud_prob = predict_fraud(df)
            risk_level, risk_icon = get_risk_level(fraud_prob)
            
            # Create record
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
            
            # Store latest for display
            st.session_state.latest = record
        
        # Quick actions
        st.markdown("##### Quick Actions")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Generate 10", use_container_width=True):
                for _ in range(10):
                    transaction = generate_transaction(fraud_bias)
                    df = pd.DataFrame([transaction])
                    fraud_prob = predict_fraud(df)
                    risk_level, _ = get_risk_level(fraud_prob)
                    
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
                st.rerun()
        
        with col_b:
            if st.button("Clear History", use_container_width=True):
                st.session_state.history = []
                st.session_state.counter = 1000
                if 'latest' in st.session_state:
                    del st.session_state.latest
                st.rerun()
    
    with col2:
        # Display latest transaction
        if 'latest' in st.session_state:
            latest = st.session_state.latest
            risk_level, risk_icon = get_risk_level(latest['fraud_probability'])
            
            st.subheader(f"Latest: {latest['transaction_id']}")
            
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Amount", f"${latest['Amount']:,.2f}")
            mcol2.metric("Fraud Probability", f"{latest['fraud_probability']:.1%}")
            mcol3.metric("Risk Level", f"{risk_icon} {risk_level}")
            
            # Gauge chart
            fig = create_gauge_chart(latest['fraud_probability'], "Fraud Risk Score")
            st.plotly_chart(fig, use_container_width=True)
    
    # History Analysis
    if st.session_state.history:
        st.markdown("---")
        st.header("📈 Transaction History Analysis")
        
        df = pd.DataFrame(st.session_state.history)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df,
                x="fraud_probability",
                nbins=20,
                title="Fraud Probability Distribution",
                color_discrete_sequence=['#1f77b4']
            )
            fig.update_layout(
                xaxis_title="Fraud Probability",
                yaxis_title="Count",
                height=400
            )
            fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Threshold")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            risk_counts = df["risk_level"].value_counts()
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Level Distribution",
                color=risk_counts.index,
                color_discrete_map={
                    "LOW": "#28a745",
                    "MEDIUM": "#ffc107",
                    "HIGH": "#fd7e14",
                    "CRITICAL": "#dc3545"
                }
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Time series (if enough data)
        if len(df) > 1:
            st.subheader("⏱️ Fraud Probability Over Time")
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            fig = go.Figure()
            
            # Add scatter plot with color based on fraud
            colors = ['#dc3545' if f else '#28a745' for f in df['is_fraud']]
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['fraud_probability'],
                mode='markers+lines',
                marker=dict(color=colors, size=10),
                line=dict(color='#6c757d', width=1),
                name='Transactions'
            ))
            
            # Add threshold line
            fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Fraud Threshold")
            
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Fraud Probability",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent transactions table
        st.subheader("📋 Recent Transactions")
        
        display_df = df[["transaction_id", "Amount", "fraud_probability", "risk_level", "is_fraud"]].tail(15)
        display_df = display_df.sort_index(ascending=False)
        display_df["Amount"] = display_df["Amount"].apply(lambda x: f"${x:,.2f}")
        display_df["fraud_probability"] = display_df["fraud_probability"].apply(lambda x: f"{x:.1%}")
        display_df["is_fraud"] = display_df["is_fraud"].apply(lambda x: "🚨 Yes" if x else "✅ No")
        display_df.columns = ["Transaction ID", "Amount", "Fraud Prob", "Risk Level", "Is Fraud"]
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    else:
        st.info("👆 Click 'Simulate New Transaction' to start monitoring")
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🛡️ Fraud Detection System**")
    with col2:
        st.markdown("**👨‍💻 Built by [Sarthak Tuli](https://www.linkedin.com/in/sarthak-tuli/)**")
    with col3:
        st.markdown(f"**📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}**")
    
    # About section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👨‍💻 About")
    st.sidebar.markdown("""
    **Sarthak Tuli**  
    MBA Business Analytics  
    Syracuse University
    
    [LinkedIn](https://www.linkedin.com/in/sarthak-tuli/) | 
    [GitHub](https://github.com/sarthaktuli7)
    """)


if __name__ == "__main__":
    main()
