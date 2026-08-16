"""Train a shadow urban flash-flood model from news-labelled flood events.

Multiple newspapers often report the same flood. They must not become multiple
independent positive samples. This trainer clusters articles into state/location
incident-days before collecting weather features, and it excludes weak negative
dates that sit too close to another known flood day.

The model is intentionally shadow-only. Promotion requires prospective validation.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

BASE = os.path.dirname(__file__)
EVENTS = os.path.join(BASE, "data", "urban_flood_events.csv")
MODEL_DIR = os.path.join(BASE, "model")
CANDIDATE = os.path.join(MODEL_DIR, "urban_flood_model_candidate.joblib")
FEATURES_FILE = os.path.join(MODEL_DIR, "urban_feature_cols.joblib")
CARD = os.path.join(MODEL_DIR, "urban_model_card.json")

FEATURES = [
    "rain_1h_mm", "rain_3h_mm", "rain_6h_mm", "rain_24h_mm", "rain_72h_mm", "rain_168h_mm",
    "max_1h_last_6h_mm", "max_3h_last_24h_mm", "hour_of_day", "month",
]
MIN_INCIDENT_DAYS = 20
MIN_STATES = 5


def fetch_history(lat: float, lon: float, start: datetime, end: datetime) -> pd.DataFrame:
    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "hourly": "precipitation",
            "timezone": "Africa/Lagos",
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json().get("hourly", {})
    return pd.DataFrame({
        "time": pd.to_datetime(data.get("time", [])),
        "precipitation": pd.to_numeric(data.get("precipitation", []), errors="coerce").fillna(0.0),
    })


def sum_window(values: np.ndarray, idx: int, hours: int) -> float:
    return float(values[max(0, idx - hours + 1):idx + 1].sum())


def max_rolling(values: np.ndarray, idx: int, lookback: int, width: int) -> float:
    start = max(0, idx - lookback + 1)
    best = 0.0
    for end in range(start, idx + 1):
        begin = max(start, end - width + 1)
        best = max(best, float(values[begin:end + 1].sum()))
    return best


def features_at(frame: pd.DataFrame, when: datetime) -> dict | None:
    if frame.empty:
        return None
    target = pd.Timestamp(when.replace(tzinfo=None)).floor("h")
    eligible = frame.index[frame["time"] <= target]
    if len(eligible) == 0:
        return None
    idx = int(eligible[-1])
    values = frame["precipitation"].to_numpy(dtype=float)
    return {
        "rain_1h_mm": sum_window(values, idx, 1),
        "rain_3h_mm": sum_window(values, idx, 3),
        "rain_6h_mm": sum_window(values, idx, 6),
        "rain_24h_mm": sum_window(values, idx, 24),
        "rain_72h_mm": sum_window(values, idx, 72),
        "rain_168h_mm": sum_window(values, idx, 168),
        "max_1h_last_6h_mm": max_rolling(values, idx, 6, 1),
        "max_3h_last_24h_mm": max_rolling(values, idx, 24, 3),
        "hour_of_day": when.hour,
        "month": when.month,
    }


def cluster_incidents(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    frame["event_dt"] = pd.to_datetime(frame["event_time"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["event_dt", "state", "latitude", "longitude"])
    frame["incident_day"] = frame["event_dt"].dt.strftime("%Y-%m-%d")
    if "location" not in frame.columns:
        frame["location"] = frame["state"]
    frame["article_count"] = 1

    # Keep the earliest discovered report as the event anchor. It is the least
    # likely article time to include rainfall that happened long after onset.
    frame = frame.sort_values("event_dt")
    grouped = frame.groupby(["state", "location", "incident_day"], as_index=False).agg(
        event_time=("event_time", "first"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        article_count=("article_count", "sum"),
        source_count=("source_domain", "nunique"),
    )
    return grouped.sort_values("event_time").reset_index(drop=True)


def build_dataset(incidents: pd.DataFrame) -> pd.DataFrame:
    rows = []
    known_days: dict[str, set[pd.Timestamp]] = {}
    for state, group in incidents.groupby("state"):
        known_days[state] = set(pd.to_datetime(group["incident_day"]).dt.normalize())

    for _, event in incidents.iterrows():
        when = pd.Timestamp(event["event_time"]).to_pydatetime()
        if when.tzinfo is not None:
            when = when.astimezone().replace(tzinfo=None)
        lat, lon = float(event["latitude"]), float(event["longitude"])
        start, end = when - timedelta(days=22), when + timedelta(days=15)
        try:
            history = fetch_history(lat, lon, start, end)
        except Exception as exc:
            print(f"weather history skipped for {event['state']} {event['incident_day']}: {exc}")
            continue

        positive = features_at(history, when)
        if positive:
            rows.append({
                **positive,
                "label": 1,
                "event_time": when.isoformat(),
                "state": event["state"],
                "incident_day": event["incident_day"],
            })

        # Weak negatives are only used if there is no known flood incident in
        # the same state within +/- 1 day of the candidate date.
        for offset in (-14, -10, 10, 14):
            negative_time = when + timedelta(days=offset)
            candidate_day = pd.Timestamp(negative_time.date())
            contaminated = any(abs((candidate_day - positive_day).days) <= 1 for positive_day in known_days.get(event["state"], set()))
            if contaminated:
                continue
            negative = features_at(history, negative_time)
            if negative:
                rows.append({
                    **negative,
                    "label": 0,
                    "event_time": negative_time.isoformat(),
                    "state": event["state"],
                    "incident_day": negative_time.date().isoformat(),
                })

    return pd.DataFrame(rows)


def main():
    if not os.path.exists(EVENTS):
        print("No event store yet; run collect_news_events.py first.")
        return

    raw_events = pd.read_csv(EVENTS)
    raw_events = raw_events[raw_events["label"].astype(str) == "1"].drop_duplicates(subset=["event_time", "state", "title"])
    incidents = cluster_incidents(raw_events)
    state_count = int(incidents["state"].nunique()) if not incidents.empty else 0
    if len(incidents) < MIN_INCIDENT_DAYS or state_count < MIN_STATES:
        print(
            f"Not enough independent national incidents yet: {len(incidents)} incident-days across {state_count} states. "
            f"Raw articles={len(raw_events)}. Need >={MIN_INCIDENT_DAYS} incident-days across >={MIN_STATES} states."
        )
        return

    dataset = build_dataset(incidents)
    if dataset.empty or dataset["label"].nunique() < 2:
        print("Training dataset could not be built.")
        return

    dataset = dataset.sort_values("event_time").reset_index(drop=True)
    split = max(1, int(len(dataset) * 0.8))
    train, test = dataset.iloc[:split], dataset.iloc[split:]
    if test["label"].nunique() < 2:
        split = max(1, int(len(dataset) * 0.7))
        train, test = dataset.iloc[:split], dataset.iloc[split:]

    X_train, y_train = train[FEATURES], train["label"].astype(int)
    X_test, y_test = test[FEATURES], test["label"].astype(int)
    positives = max(1, int(y_train.sum()))
    negatives = max(1, len(y_train) - positives)

    model = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_lambda=2.0,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=negatives / positives,
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probs)) if y_test.nunique() > 1 else None,
        "pr_auc": float(average_precision_score(y_test, probs)) if y_test.nunique() > 1 else None,
        "brier": float(brier_score_loss(y_test, probs)),
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, CANDIDATE)
    joblib.dump(FEATURES, FEATURES_FILE)
    card = {
        "name": "NaijaClimaGuard Urban Flood Model Candidate",
        "status": "shadow_candidate_not_production",
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "raw_positive_news_articles": int(len(raw_events)),
        "independent_incident_days": int(len(incidents)),
        "states_represented": state_count,
        "training_rows": int(len(train)),
        "test_rows": int(len(test)),
        "features": FEATURES,
        "metrics": metrics,
        "label_note": "Positive labels are state/location incident-days clustered from news reports describing observed flooding. Multiple articles about the same event count once. Negative labels are noisy nearby dates with no known flood report and are excluded near known incident days.",
        "known_limitations": [
            "News coverage is geographically and socioeconomically biased.",
            "Event coordinates are currently location centroids rather than exact flooded road coordinates.",
            "Open-Meteo archive rainfall may smooth highly local convective storms.",
            "The dataset has weak negative labels because absence of a news report is not proof that no flood occurred.",
        ],
        "promotion_rule": "Do not promote automatically. Require prospective shadow validation against independently verified flood occurrence and non-occurrence windows, with recall, false-alarm rate and calibration reported by geography.",
    }
    with open(CARD, "w", encoding="utf-8") as handle:
        json.dump(card, handle, indent=2)
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
