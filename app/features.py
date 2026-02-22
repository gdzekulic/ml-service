"""Feature Engineering Pipeline."""

import numpy as np
import pandas as pd
from typing import Union

from app.config import (
    NUMERIC_FEATURES, BOOLEAN_FEATURES, DERIVED_FEATURES,
    CATEGORICAL_FEATURES,
)

# Mapping: day_of_week String → Nummer
DAY_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2,
    "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6,
    "Mo": 0, "Di": 1, "Mi": 2, "Do": 3, "Fr": 4, "Sa": 5, "So": 6,
}


def _safe_float(val, default=0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_bool(val) -> int:
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        return 1 if val.lower() in ("true", "1", "yes") else 0
    return 0


def build_features_from_dict(data: dict) -> dict:
    """Baue Feature-Dict aus einem einzelnen Signal (API Request)."""
    features = {}

    # Numerische Features
    for f in NUMERIC_FEATURES:
        features[f] = _safe_float(data.get(f))

    # Boolean Features → 0/1
    for f in BOOLEAN_FEATURES:
        features[f] = _safe_bool(data.get(f))

    # Abgeleitete Features
    features["score_spread"] = features["buy_score"] - features["sell_score"]
    features["stoch_diff"] = features["stoch_k"] - features["stoch_d"]

    price = features["price_at_signal"]
    ema20 = features["ema20"]
    ema50 = features["ema50"]
    if price and price > 0:
        features["ema_spread"] = ((ema20 - ema50) / price) * 100
    else:
        features["ema_spread"] = 0.0

    features["bb_squeeze"] = _safe_bool(data.get("bb_squeeze"))
    features["hour_of_day"] = int(data.get("hour_of_day", 12))
    features["day_of_week_num"] = _parse_day_of_week(data.get("day_of_week", 2))

    # Kategorische Features → One-Hot
    for cat_name, possible_values in CATEGORICAL_FEATURES.items():
        actual_value = str(data.get(cat_name, "")).lower()
        for val in possible_values[1:]:  # Erstes wird als Referenz weggelassen
            col_name = f"{cat_name}_{val}"
            features[col_name] = 1 if actual_value == val.lower() else 0

    return features


def build_features_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Baue Feature-DataFrame aus Trainingsdaten."""
    features = pd.DataFrame()

    # Numerische Features
    for f in NUMERIC_FEATURES:
        if f in df.columns:
            features[f] = pd.to_numeric(df[f], errors="coerce").fillna(0)
        else:
            features[f] = 0.0

    # Boolean Features → 0/1
    for f in BOOLEAN_FEATURES:
        if f in df.columns:
            features[f] = df[f].apply(_safe_bool)
        else:
            features[f] = 0

    # Abgeleitete Features
    features["score_spread"] = features["buy_score"] - features["sell_score"]
    features["stoch_diff"] = features["stoch_k"] - features["stoch_d"]

    price = features["price_at_signal"].replace(0, np.nan)
    features["ema_spread"] = (
        (features["ema20"] - features["ema50"]) / price * 100
    ).fillna(0)

    if "bb_squeeze" in df.columns:
        features["bb_squeeze"] = df["bb_squeeze"].apply(_safe_bool)
    else:
        features["bb_squeeze"] = 0

    # hour_of_day aus timestamp oder time_of_day
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        features["hour_of_day"] = ts.dt.hour.fillna(12).astype(int)
    elif "time_of_day" in df.columns:
        features["hour_of_day"] = df["time_of_day"].apply(_parse_hour).fillna(12).astype(int)
    else:
        features["hour_of_day"] = 12

    # day_of_week_num
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        features["day_of_week_num"] = ts.dt.dayofweek.fillna(2).astype(int)
    elif "day_of_week" in df.columns:
        features["day_of_week_num"] = df["day_of_week"].apply(_parse_day_of_week).fillna(2).astype(int)
    else:
        features["day_of_week_num"] = 2

    # Kategorische Features → One-Hot
    for cat_name, possible_values in CATEGORICAL_FEATURES.items():
        if cat_name in df.columns:
            actual = df[cat_name].fillna("").str.lower()
        else:
            actual = pd.Series([""] * len(df))

        for val in possible_values[1:]:  # Erstes als Referenz
            col_name = f"{cat_name}_{val}"
            features[col_name] = (actual == val.lower()).astype(int)

    return features


def get_feature_names() -> list:
    """Gibt die vollstaendige Feature-Liste zurueck."""
    names = list(NUMERIC_FEATURES) + list(BOOLEAN_FEATURES) + list(DERIVED_FEATURES)
    for cat_name, possible_values in CATEGORICAL_FEATURES.items():
        for val in possible_values[1:]:
            names.append(f"{cat_name}_{val}")
    return names


def _parse_day_of_week(val) -> int:
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        return DAY_MAP.get(val.strip(), 2)
    return 2


def _parse_hour(val) -> int:
    if val is None:
        return 12
    try:
        if isinstance(val, str) and ":" in val:
            return int(val.split(":")[0])
        return int(val)
    except (ValueError, TypeError):
        return 12
