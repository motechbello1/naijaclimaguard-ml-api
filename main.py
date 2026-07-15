"""
NaijaClimaGuard ML API
Serves real flood risk predictions using live Open-Meteo weather data + trained XGBoost model.
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import requests
from datetime import datetime, timedelta
import os

app = FastAPI(
    title="NaijaClimaGuard ML API",
    description="Real-time flood risk predictions for Nigerian locations using NASA/Open-Meteo data + XGBoost",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
MODEL = None
FEATURE_COLS = None

@app.on_event("startup")
def load_model():
    global MODEL, FEATURE_COLS
    model_path = os.path.join(os.path.dirname(__file__), "model", "flood_model.joblib")
    cols_path = os.path.join(os.path.dirname(__file__), "model", "feature_cols.joblib")
    
    if os.path.exists(model_path):
        MODEL = joblib.load(model_path)
        FEATURE_COLS = joblib.load(cols_path)
        print(f"Model loaded: {model_path}")
    else:
        print(f"WARNING: Model not found at {model_path}. Run train_model.py first.")


def fetch_recent_weather(lat: float, lon: float, days: int = 30) -> pd.DataFrame:
    """Fetch recent weather data from Open-Meteo for live predictions."""
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": [
            "precipitation_sum",
            "rain_sum",
            "et0_fao_evapotranspiration",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_hours",
        ],
        "past_days": days,
        "forecast_days": 3,
        "timezone": "Africa/Lagos"
    }
    
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()["daily"]
    
    df = pd.DataFrame({
        "date": pd.to_datetime(data["time"]),
        "precip_sum": data["precipitation_sum"],
        "rain_sum": data["rain_sum"],
        "et0": data["et0_fao_evapotranspiration"],
        "temp_max": data["temperature_2m_max"],
        "temp_min": data["temperature_2m_min"],
        "precip_hours": data["precipitation_hours"],
    })
    return df


def engineer_features(df: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    """Same feature engineering as training — must match exactly."""
    df = df.copy()
    df = df.fillna(0)
    
    df["precip_3d"] = df["precip_sum"].rolling(3, min_periods=1).sum()
    df["precip_7d"] = df["precip_sum"].rolling(7, min_periods=1).sum()
    df["precip_14d"] = df["precip_sum"].rolling(14, min_periods=1).sum()
    df["precip_30d"] = df["precip_sum"].rolling(30, min_periods=1).sum()
    
    df["rain_3d"] = df["rain_sum"].rolling(3, min_periods=1).sum()
    df["rain_7d"] = df["rain_sum"].rolling(7, min_periods=1).sum()
    
    df["rain_intensity"] = np.where(
        df["precip_hours"] > 0,
        df["precip_sum"] / df["precip_hours"],
        0
    )
    
    df["moisture_balance"] = df["precip_sum"] - df["et0"]
    df["moisture_balance_7d"] = df["moisture_balance"].rolling(7, min_periods=1).sum()
    df["moisture_balance_14d"] = df["moisture_balance"].rolling(14, min_periods=1).sum()
    
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_monsoon"] = df["month"].isin([6, 7, 8, 9, 10]).astype(int)
    
    df["temp_range"] = df["temp_max"] - df["temp_min"]
    
    df["latitude"] = lat
    df["longitude"] = lon
    
    return df


def get_risk_level(score: int) -> str:
    if score >= 90: return "EXTREME"
    if score >= 75: return "SEVERE"
    if score >= 60: return "WARNING"
    if score >= 40: return "WATCH"
    return "NORMAL"


@app.get("/")
def root():
    return {
        "service": "NaijaClimaGuard ML API",
        "version": "2.1.0",
        "model": "XGBoost Flood Risk Classifier",
        "data_source": "Open-Meteo (NASA GPM IMERG derived)",
        "status": "operational" if MODEL is not None else "model_not_loaded",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/v1/risk")
def get_risk(
    latitude: float = Query(..., ge=-90, le=90, description="Location latitude"),
    longitude: float = Query(..., ge=-90, le=90, description="Location longitude"),
):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py first.")
    
    try:
        # Fetch REAL current weather data
        raw_df = fetch_recent_weather(latitude, longitude, days=30)
        df = engineer_features(raw_df, latitude, longitude)
        df = df.dropna()
        
        if df.empty:
            raise HTTPException(status_code=422, detail="Insufficient weather data for this location")
        
        # Get the latest row (most recent day + forecast)
        latest = df.iloc[-1:]
        X = latest[FEATURE_COLS]
        
        # Predict flood probability
        flood_prob = float(MODEL.predict_proba(X)[:, 1][0])
        risk_score = int(round(flood_prob * 100))
        risk_level = get_risk_level(risk_score)
        
        # Extract real contributing factors (normalized 0-1)
        precip_7d = float(latest["precip_7d"].values[0])
        moisture = float(latest["moisture_balance_7d"].values[0])
        rain_intensity = float(latest["rain_intensity"].values[0])
        
        # Normalize to 0-1 range for frontend display
        rainfall_norm = min(1.0, precip_7d / 200)  # 200mm/7d = extreme
        discharge_norm = min(1.0, max(0, moisture / 100))  # proxy for river discharge
        soil_norm = min(1.0, max(0, (moisture + 50) / 150))  # proxy for soil saturation
        
        return {
            "risk_assessment": {
                "current_score": risk_score,
                "level": risk_level,
                "flood_probability": round(flood_prob, 4),
            },
            "contributing_factors": {
                "rainfall_intensity": round(rainfall_norm, 2),
                "river_discharge": round(discharge_norm, 2),
                "soil_saturation": round(soil_norm, 2),
            },
            "raw_weather": {
                "precipitation_7d_mm": round(precip_7d, 1),
                "precipitation_today_mm": round(float(latest["precip_sum"].values[0]), 1),
                "temperature_max_c": round(float(latest["temp_max"].values[0]), 1),
                "temperature_min_c": round(float(latest["temp_min"].values[0]), 1),
                "moisture_balance_7d": round(moisture, 1),
            },
            "metadata": {
                "model_version": "2.1.0",
                "data_source": "Open-Meteo (NASA GPM IMERG derived)",
                "latitude": latitude,
                "longitude": longitude,
                "prediction_timestamp": datetime.utcnow().isoformat() + "Z",
                "forecast_window_hours": 72,
            }
        }
        
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Weather data fetch failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
