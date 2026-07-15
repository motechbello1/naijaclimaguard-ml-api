"""
NaijaClimaGuard ML Model Training Script
Fetches real historical data from Open-Meteo, trains XGBoost flood risk model.
5 Nigerian flood-prone locations, 2018-2023 data.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
import joblib
import requests
from datetime import datetime
import os

# 5 validated Nigerian flood-prone locations
LOCATIONS = {
    "Lokoja":    {"lat": 7.80, "lon": 6.74},   # Niger-Benue confluence
    "Makurdi":   {"lat": 7.73, "lon": 8.54},   # Benue River
    "Yenagoa":   {"lat": 4.92, "lon": 6.26},   # Bayelsa
    "Onitsha":   {"lat": 6.14, "lon": 6.79},   # Anambra - Niger River
    "Hadejia":   {"lat": 12.45, "lon": 10.04}, # Jigawa - Hadejia-Nguru wetlands
}

def fetch_historical_data(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Fetch real historical weather data from Open-Meteo API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": [
            "precipitation_sum",
            "rain_sum",
            "et0_fao_evapotranspiration",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_hours",
        ],
        "timezone": "Africa/Lagos"
    }
    resp = requests.get(url, params=params)
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

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create flood-relevant features from raw weather data."""
    df = df.copy()
    df = df.fillna(0)
    
    # Rolling aggregates (cumulative rainfall is the #1 flood predictor)
    df["precip_3d"] = df["precip_sum"].rolling(3, min_periods=1).sum()
    df["precip_7d"] = df["precip_sum"].rolling(7, min_periods=1).sum()
    df["precip_14d"] = df["precip_sum"].rolling(14, min_periods=1).sum()
    df["precip_30d"] = df["precip_sum"].rolling(30, min_periods=1).sum()
    
    df["rain_3d"] = df["rain_sum"].rolling(3, min_periods=1).sum()
    df["rain_7d"] = df["rain_sum"].rolling(7, min_periods=1).sum()
    
    # Rainfall intensity (mm per hour of precipitation)
    df["rain_intensity"] = np.where(
        df["precip_hours"] > 0,
        df["precip_sum"] / df["precip_hours"],
        0
    )
    
    # Soil saturation proxy: cumulative rain minus evapotranspiration
    df["moisture_balance"] = df["precip_sum"] - df["et0"]
    df["moisture_balance_7d"] = df["moisture_balance"].rolling(7, min_periods=1).sum()
    df["moisture_balance_14d"] = df["moisture_balance"].rolling(14, min_periods=1).sum()
    
    # Temporal features (monsoon seasonality)
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_monsoon"] = df["month"].isin([6, 7, 8, 9, 10]).astype(int)
    
    # Temperature range (proxy for atmospheric instability)
    df["temp_range"] = df["temp_max"] - df["temp_min"]
    
    return df

def create_flood_labels(df: pd.DataFrame) -> pd.Series:
    """
    Create binary flood labels based on extreme rainfall thresholds.
    Uses multiple signals that historically correlate with actual flooding.
    """
    conditions = (
        (df["precip_7d"] > df["precip_7d"].quantile(0.92)) |
        (df["precip_3d"] > df["precip_3d"].quantile(0.95)) |
        (
            (df["precip_14d"] > df["precip_14d"].quantile(0.88)) &
            (df["moisture_balance_7d"] > df["moisture_balance_7d"].quantile(0.90))
        )
    )
    return conditions.astype(int)

def train():
    print("=" * 60)
    print("NaijaClimaGuard XGBoost Training Pipeline")
    print("=" * 60)
    
    all_data = []
    
    for name, coords in LOCATIONS.items():
        print(f"\nFetching data for {name} ({coords['lat']}, {coords['lon']})...")
        df = fetch_historical_data(
            coords["lat"], coords["lon"],
            "2018-01-01", "2023-12-31"
        )
        df = engineer_features(df)
        df["location"] = name
        df["latitude"] = coords["lat"]
        df["longitude"] = coords["lon"]
        all_data.append(df)
        print(f"  → {len(df)} days fetched")
    
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.dropna()
    
    print(f"\nTotal samples: {len(combined)}")
    
    # Create labels
    combined["flood"] = create_flood_labels(combined)
    print(f"Flood events: {combined['flood'].sum()} ({combined['flood'].mean()*100:.1f}%)")
    
    # Feature columns
    feature_cols = [
        "precip_sum", "rain_sum", "et0", "temp_max", "temp_min",
        "precip_hours", "precip_3d", "precip_7d", "precip_14d", "precip_30d",
        "rain_3d", "rain_7d", "rain_intensity",
        "moisture_balance", "moisture_balance_7d", "moisture_balance_14d",
        "month", "day_of_year", "is_monsoon", "temp_range",
        "latitude", "longitude"
    ]
    
    X = combined[feature_cols]
    y = combined["flood"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=42,
        eval_metric="auc",
        use_label_encoder=False,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'=' * 40}")
    print(f"ROC-AUC:  {roc_auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"{'=' * 40}")
    
    # Save model and feature columns
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/flood_model.joblib")
    joblib.dump(feature_cols, "model/feature_cols.joblib")
    
    print(f"\nModel saved to model/flood_model.joblib")
    print(f"Feature columns saved to model/feature_cols.joblib")
    
    return roc_auc

if __name__ == "__main__":
    train()
