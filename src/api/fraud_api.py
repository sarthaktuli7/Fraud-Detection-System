"""
Real-Time Fraud Detection API
"""

import logging
import os
import time
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

warnings.filterwarnings("ignore")


# =====================================================================
# REQUIRED: Classes used when saving models (must match training code)
# =====================================================================
class SimpleFeatureEngineering:
    """Simple feature engineering pipeline - required for loading saved model"""
    
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
    """Ensemble model - required for loading saved model"""
    
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection system using ensemble machine learning models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
feature_pipeline = None
ensemble_model = None
model_loaded = False
model_load_time: Optional[str] = None
request_count = 0
total_processing_time = 0.0
start_time = time.time()


# Pydantic Models
class TransactionRequest(BaseModel):
    Time: float = Field(..., ge=0)
    Amount: float = Field(..., ge=0)
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    transaction_id: Optional[str] = None


class FraudPredictionResponse(BaseModel):
    transaction_id: Optional[str]
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    confidence_score: float
    processing_time_ms: float
    model_version: str
    timestamp: str


class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    model_load_time: Optional[str]
    uptime_seconds: float
    version: str


# =====================================================================
# FIXED: Model Loading Function
# =====================================================================
def load_models() -> bool:
    """Load trained models - FIXED PATH RESOLUTION"""
    global feature_pipeline, ensemble_model, model_loaded, model_load_time
    
    try:
        logger.info("Loading fraud detection models...")
        
        # FIXED: Try multiple possible paths
        possible_paths = [
            # If running from project root
            os.path.join(os.getcwd(), "models"),
            # If running from src/api folder
            os.path.join(os.getcwd(), "..", "..", "models"),
            # Absolute path (update this to your actual path)
            r"C:\Users\sarthak\OneDrive\Desktop\fraud-detection-system-main\models",
            # Relative to this file
            os.path.join(os.path.dirname(__file__), "..", "..", "models"),
        ]
        
        models_dir = None
        for path in possible_paths:
            normalized_path = os.path.normpath(path)
            fe_check = os.path.join(normalized_path, "feature_engineering_pipeline.pkl")
            if os.path.exists(fe_check):
                models_dir = normalized_path
                logger.info(f"Found models at: {models_dir}")
                break
        
        if models_dir is None:
            logger.error("Could not find models directory!")
            logger.error(f"Searched paths: {possible_paths}")
            logger.error(f"Current working directory: {os.getcwd()}")
            return False
        
        fe_path = os.path.join(models_dir, "feature_engineering_pipeline.pkl")
        ensemble_path = os.path.join(models_dir, "fraud_detection_ensemble.pkl")
        
        logger.info(f"Loading feature pipeline from: {fe_path}")
        logger.info(f"Loading ensemble model from: {ensemble_path}")
        
        feature_pipeline = joblib.load(fe_path)
        ensemble_model = joblib.load(ensemble_path)
        
        model_loaded = True
        model_load_time = datetime.now().isoformat()
        logger.info("✅ Models loaded successfully!")
        return True
        
    except Exception as e:
        logger.exception(f"❌ Error loading models: {e}")
        model_loaded = False
        return False


def preprocess_transaction(transaction: TransactionRequest) -> pd.DataFrame:
    """Convert transaction to preprocessed DataFrame"""
    try:
        data = transaction.dict(exclude_none=True)
        data.pop("transaction_id", None)
        df = pd.DataFrame([data])
        
        if feature_pipeline is not None and hasattr(feature_pipeline, "transform"):
            return feature_pipeline.transform(df)
        return df
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {e}")


def get_risk_level(probability: float) -> str:
    if probability >= 0.9:
        return "CRITICAL"
    if probability >= 0.7:
        return "HIGH"
    if probability >= 0.3:
        return "MEDIUM"
    return "LOW"


def get_confidence_score(probability: float) -> float:
    return round(abs(probability - 0.5) * 2, 4)


async def check_model_dependency():
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded.")


# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Fraud Detection API...")
    if not load_models():
        logger.warning("⚠️ Models failed to load during startup.")


# Endpoints
@app.get("/")
async def root():
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "model_loaded": model_loaded,
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    return HealthCheckResponse(
        status="healthy" if model_loaded else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model_loaded,
        model_load_time=model_load_time,
        uptime_seconds=time.time() - start_time,
        version="1.0.0",
    )


@app.post("/predict", response_model=FraudPredictionResponse)
async def predict_fraud(
    transaction: TransactionRequest,
    _: None = Depends(check_model_dependency),
):
    global request_count, total_processing_time
    start_ms = time.time() * 1000
    
    try:
        df = preprocess_transaction(transaction)
        fraud_prob = ensemble_model.predict_proba(df)[0, 1]
        risk_level = get_risk_level(fraud_prob)
        confidence = get_confidence_score(fraud_prob)
        processing_time = time.time() * 1000 - start_ms
        
        request_count += 1
        total_processing_time += processing_time
        
        return FraudPredictionResponse(
            transaction_id=transaction.transaction_id,
            fraud_probability=round(float(fraud_prob), 4),
            is_fraud=fraud_prob > 0.5,
            risk_level=risk_level,
            confidence_score=confidence,
            processing_time_ms=round(processing_time, 2),
            model_version="ensemble-v1.0",
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.exception(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@app.get("/metrics")
async def metrics():
    uptime = time.time() - start_time
    avg_time = total_processing_time / request_count if request_count else 0
    return {
        "requests_processed": request_count,
        "average_processing_time_ms": round(avg_time, 2),
        "uptime_seconds": round(uptime, 2),
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/reload-models")
async def reload_models():
    if load_models():
        return {"message": "Models reloaded successfully"}
    raise HTTPException(status_code=500, detail="Failed to reload models")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)