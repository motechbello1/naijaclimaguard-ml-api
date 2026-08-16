"""
NaijaClimaGuard ML API
Serves real flood risk predictions using live Open-Meteo weather data + trained XGBoost model.
Also exposes a short-duration urban flash-flood nowcast. The urban ML model remains shadow-only
until prospective validation justifies promotion.
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import requests
from datetime import datetime, timedelta
import os

app = FastAPI(
    title="NaijaClimaGuard ML API",
    description="Real-time flood risk predictions for Nigerian locations using Open-Meteo data + validated/shadow ML models",
    version="2.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
FEATURE_COLS = None
URBAN_MODEL = None
URBAN_FEATURE_COLS = None
URBAN_MODEL_STATUS = "not_available"
URBAN_MODEL_CARD = None


@app.on_event("startup")
def load_model():
    global MODEL, FEATURE_COLS, URBAN_MODEL, URBAN_FEATURE_COLS, URBAN_MODEL_STATUS, URBAN_MODEL_CARD
    base = os.path.dirname(__file__)
    model_path = os.path.join(base, "model", "flood_model.joblib")
    cols_path = os.path.join(base, "model", "feature_cols.joblib")

    if os.path.exists(model_path):
        MODEL = joblib.load(model_path)
        FEATURE_COLS = joblib.load(cols_path)
        print(f"Model loaded: {model_path}")
    else:
        print(f"WARNING: Model not found at {model_path}. Run train_model.py first.")

    # The urban model is loaded for SHADOW scoring only. Production decisions remain
    # on the disclosed nowcast heuristic until a separately promoted active artifact exists.
    active_path = os.path.join(base, "model", "urban_flood_model.joblib")
    candidate_path = os.path.join(base, "model", "urban_flood_model_candidate.joblib")
    urban_cols = os.path.join(base, "model", "urban_feature_cols.joblib")
    card_path = os.path.join(base, "model", "urban_model_card.json")

    selected = None
    if os.path.exists(active_path):
        selected = active_path
        URBAN_MODEL_STATUS = "active_validated"
    elif os.path.exists(candidate_path):
        selected = candidate_path
        URBAN_MODEL_STATUS = "shadow_candidate"

    if selected and os.path.exists(urban_cols):
        URBAN_MODEL = joblib.load(selected)
        URBAN_FEATURE_COLS = joblib.load(urban_cols)
        print(f"Urban model loaded ({URBAN_MODEL_STATUS}): {selected}")

    if os.path.exists(card_path):
        try:
            with open(card_path, "r", encoding="utf-8") as handle:
                URBAN_MODEL_CARD = json.load(handle)
        except Exception:
            URBAN_MODEL_CARD = None


def fetch_recent_weather(lat: float, lon: float, days: int = 30) -> pd.DataFrame:
    """Fetch recent daily weather data from Open-Meteo for live predictions."""
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
    df = df.copy().fillna(0)
    df["precip_3d"] = df["precip_sum"].rolling(3, min_periods=1).sum()
    df["precip_7d"] = df["precip_sum"].rolling(7, min_periods=1).sum()
    df["precip_14d"] = df["precip_sum"].rolling(14, min_periods=1).sum()
    df["precip_30d"] = df["precip_sum"].rolling(30, min_periods=1).sum()
    df["rain_3d"] = df["rain_sum"].rolling(3, min_periods=1).sum()
    df["rain_7d"] = df["rain_sum"].rolling(7, min_periods=1).sum()
    df["rain_intensity"] = np.where(df["precip_hours"] > 0, df["precip_sum"] / df["precip_hours"], 0)
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


def fetch_hourly_nowcast(lat: float, lon: float):
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": "precipitation",
            "past_hours": 168,
            "forecast_hours": 6,
            "timezone": "Africa/Lagos",
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json().get("hourly", {})


def _window_sum(values, idx, hours):
    start = max(0, idx - hours + 1)
    return float(np.sum(values[start:idx + 1]))


def _max_rolling(values, idx, lookback, width):
    start = max(0, idx - lookback + 1)
    best = 0.0
    for end in range(start, idx + 1):
        begin = max(start, end - width + 1)
        best = max(best, float(np.sum(values[begin:end + 1])))
    return best


def urban_features(hourly):
    times = hourly.get("time", [])
    values = np.asarray(hourly.get("precipitation", []), dtype=float)
    values = np.nan_to_num(values, nan=0.0)
    values = np.maximum(values, 0.0)
    if not times or len(values) == 0:
        raise ValueError("hourly rainfall unavailable")

    nigeria_now = (datetime.utcnow() + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    current_key = nigeria_now.strftime("%Y-%m-%dT%H:00")
    idx = -1
    for i, value in enumerate(times):
        if value <= current_key:
            idx = i
        else:
            break
    if idx < 0:
        idx = min(len(values) - 1, 167)

    def future(hours):
        return float(np.sum(values[idx + 1:min(len(values), idx + 1 + hours)]))

    return {
        "rain_1h_mm": _window_sum(values, idx, 1),
        "rain_3h_mm": _window_sum(values, idx, 3),
        "rain_6h_mm": _window_sum(values, idx, 6),
        "rain_24h_mm": _window_sum(values, idx, 24),
        "rain_72h_mm": _window_sum(values, idx, 72),
        "rain_168h_mm": _window_sum(values, idx, 168),
        "max_1h_last_6h_mm": _max_rolling(values, idx, 6, 1),
        "max_3h_last_24h_mm": _max_rolling(values, idx, 24, 3),
        "forecast_3h_mm": future(3),
        "forecast_6h_mm": future(6),
        "hour_of_day": nigeria_now.hour,
        "month": nigeria_now.month,
    }


def derived_urban_score(features):
    clamp = lambda x: max(0.0, min(1.0, x))
    burst = max(
        clamp(features["rain_1h_mm"] / 20.0),
        clamp(features["rain_3h_mm"] / 40.0),
        clamp(features["rain_6h_mm"] / 70.0),
        clamp(features["max_1h_last_6h_mm"] / 25.0),
        clamp(features["max_3h_last_24h_mm"] / 55.0),
    )
    antecedent = max(
        clamp(features["rain_24h_mm"] / 100.0),
        clamp(features["rain_72h_mm"] / 180.0),
        clamp(features["rain_168h_mm"] / 300.0),
    )
    near_forecast = max(
        clamp(features["forecast_3h_mm"] / 40.0),
        clamp(features["forecast_6h_mm"] / 70.0),
    )
    return int(round((0.55 * burst + 0.30 * antecedent + 0.15 * near_forecast) * 100))


def urban_level(score: int):
    if score >= 75: return "EMERGENCY"
    if score >= 60: return "WARNING"
    if score >= 40: return "WATCH"
    return "NORMAL"


@app.get("/")
def root():
    return {
        "service": "NaijaClimaGuard ML API",
        "version": "2.2.0",
        "model": "XGBoost Flood Risk Classifier",
        "urban_model_status": URBAN_MODEL_STATUS,
        "data_source": "Open-Meteo",
        "status": "operational" if MODEL is not None else "model_not_loaded",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "urban_model_status": URBAN_MODEL_STATUS,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/v1/urban-model/status")
def urban_model_status():
    return {
        "status": URBAN_MODEL_STATUS,
        "card": URBAN_MODEL_CARD,
        "production_note": "shadow_candidate predictions are evidence only and do not replace the operational nowcast score",
    }


@app.get("/v1/urban-flash-risk")
def get_urban_flash_risk(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
):
    try:
        hourly = fetch_hourly_nowcast(latitude, longitude)
        features = urban_features(hourly)
        score = derived_urban_score(features)
        level = urban_level(score)

        shadow_probability = None
        if URBAN_MODEL is not None and URBAN_FEATURE_COLS:
            row = pd.DataFrame([{name: features.get(name, 0.0) for name in URBAN_FEATURE_COLS}])
            try:
                shadow_probability = float(URBAN_MODEL.predict_proba(row[URBAN_FEATURE_COLS])[:, 1][0])
            except Exception:
                shadow_probability = None

        drivers = []
        if features["rain_1h_mm"] >= 20 or features["max_1h_last_6h_mm"] >= 25:
            drivers.append("very intense rainfall in the latest hours")
        if features["rain_3h_mm"] >= 40 or features["max_3h_last_24h_mm"] >= 55:
            drivers.append("short-duration rainfall burst can overwhelm urban drainage")
        if features["rain_24h_mm"] >= 100 or features["rain_72h_mm"] >= 180:
            drivers.append("antecedent rainfall is already high")
        if features["forecast_3h_mm"] >= 40 or features["forecast_6h_mm"] >= 70:
            drivers.append("more heavy rain is forecast in the next few hours")
        if not drivers:
            drivers.append("no urban rainfall threshold is currently dominant")

        return {
            "risk": {
                "score": score,
                "level": level,
                "decision_model": "urban-flash-v1-derived",
                "shadow_model_probability": round(shadow_probability, 4) if shadow_probability is not None else None,
                "shadow_model_status": URBAN_MODEL_STATUS,
            },
            "features": {key: round(float(value), 2) for key, value in features.items()},
            "drivers": drivers,
            "metadata": {
                "latitude": latitude,
                "longitude": longitude,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "purpose": "short-duration urban/flash flood nowcasting",
                "safety_note": "The shadow ML score is not used operationally until prospective validation and explicit promotion.",
            },
        }
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Hourly weather fetch failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Urban nowcast error: {str(e)}")


@app.get("/v1/risk")
def get_risk(
    latitude: float = Query(..., ge=-90, le=90, description="Location latitude"),
    longitude: float = Query(..., ge=-180, le=180, description="Location longitude"),
):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py first.")

    try:
        raw_df = fetch_recent_weather(latitude, longitude, days=30)
        df = engineer_features(raw_df, latitude, longitude).dropna()
        if df.empty:
            raise HTTPException(status_code=422, detail="Insufficient weather data for this location")

        latest = df.iloc[-1:]
        X = latest[FEATURE_COLS]
        flood_prob = float(MODEL.predict_proba(X)[:, 1][0])
        risk_score = int(round(flood_prob * 100))
        risk_level = get_risk_level(risk_score)

        precip_7d = float(latest["precip_7d"].values[0])
        moisture = float(latest["moisture_balance_7d"].values[0])
        rain_intensity = float(latest["rain_intensity"].values[0])

        rainfall_norm = min(1.0, precip_7d / 200)
        discharge_norm = min(1.0, max(0, moisture / 100))
        soil_norm = min(1.0, max(0, (moisture + 50) / 150))

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
                "data_source": "Open-Meteo",
                "latitude": latitude,
                "longitude": longitude,
                "prediction_timestamp": datetime.utcnow().isoformat() + "Z",
                "forecast_window_hours": 72,
            }
        }

    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Weather data fetch failed: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
