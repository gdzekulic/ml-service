"""Feature Engineering Pipeline."""

import numpy as np
import pandas as pd

from app.config import (
    NUMERIC_FEATURES, BOOLEAN_FEATURES, DERIVED_FEATURES,
    CATEGORICAL_FEATURES,
)

# Mapping: day_of_week String -> Nummer (lowercase keys + German abbreviations)
DAY_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2,
    "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
    "mo": 0, "di": 1, "mi": 2, "do": 3, "fr": 4, "sa": 5, "so": 6,
}

# Per-feature defaults for numeric features (0.0 unless specified here)
NUMERIC_DEFAULTS = {
    "fear_greed_index": 50.0,
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
        features[f] = _safe_float(data.get(f), default=NUMERIC_DEFAULTS.get(f, 0.0))

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

    # Phase 2: Erweiterte abgeleitete Features
    bs = features["buy_score"]
    ss = features["sell_score"]
    score_sum = bs + ss
    features["score_ratio"] = bs / score_sum if score_sum > 0 else 0.5
    features["score_total"] = score_sum
    features["rsi_extreme"] = abs(features["rsi"] - 50)
    features["stoch_extreme"] = abs(features["stoch_k"] - 50)
    features["bb_position_extreme"] = abs(features["bb_percent_b"] - 0.5)
    features["macd_abs"] = abs(features["macd"])
    features["momentum_abs"] = abs(features["momentum"])
    features["atr_momentum_interaction"] = features["atr_percent"] * abs(features["momentum"])
    rsi = features["rsi"]
    stoch = features["stoch_k"]
    features["rsi_stoch_agreement"] = 1.0 if (rsi > 70 and stoch > 80) or (rsi < 30 and stoch < 20) else 0.0
    features["trend_strength"] = abs(features["ema_spread"])

    # Phase 3: Preis-Trend und Signal-Interaktionen
    features["ema_price_ratio_20"] = price / ema20 if ema20 > 0 else 1.0
    features["ema_price_ratio_50"] = price / ema50 if ema50 > 0 else 1.0
    features["ema_alignment"] = 1.0 if (
        (price > ema20 and ema20 > ema50) or (price < ema20 and ema20 < ema50)
    ) else 0.0
    features["score_dominance"] = max(bs, ss) / score_sum if score_sum > 0 else 0.5
    features["signal_confidence_product"] = (
        features["confidence"] * features["signal_strength"] / 100
        if features["signal_strength"] > 0 else 0.0
    )
    features["bb_atr_interaction"] = features["bb_percent_b"] * features["atr_percent"]
    momentum = features["momentum"]
    features["rsi_momentum_agreement"] = 1.0 if (
        (rsi > 50 and momentum > 0) or (rsi < 50 and momentum < 0)
    ) else 0.0

    # Phase 4: Sentiment-Features
    fg = features["fear_greed_index"]
    features["fear_greed_normalized"] = (fg - 50) / 50
    news_sent = features["news_sentiment_score"]
    features["sentiment_momentum_agreement"] = 1.0 if (
        (news_sent > 0.2 and momentum > 0) or
        (news_sent < -0.2 and momentum < 0)
    ) else 0.0
    features["sentiment_regime_interaction"] = news_sent * abs(features["ema_spread"])

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
        default_val = NUMERIC_DEFAULTS.get(f, 0.0)
        if f in df.columns:
            features[f] = pd.to_numeric(df[f], errors="coerce").fillna(default_val)
        else:
            features[f] = default_val

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

    # Phase 2: Erweiterte abgeleitete Features
    bs = features["buy_score"]
    ss = features["sell_score"]
    score_sum = bs + ss
    features["score_ratio"] = np.where(score_sum > 0, bs / score_sum, 0.5)
    features["score_total"] = score_sum
    features["rsi_extreme"] = (features["rsi"] - 50).abs()
    features["stoch_extreme"] = (features["stoch_k"] - 50).abs()
    features["bb_position_extreme"] = (features["bb_percent_b"] - 0.5).abs()
    features["macd_abs"] = features["macd"].abs()
    features["momentum_abs"] = features["momentum"].abs()
    features["atr_momentum_interaction"] = features["atr_percent"] * features["momentum"].abs()
    features["rsi_stoch_agreement"] = (
        ((features["rsi"] > 70) & (features["stoch_k"] > 80)) |
        ((features["rsi"] < 30) & (features["stoch_k"] < 20))
    ).astype(float)
    features["trend_strength"] = features["ema_spread"].abs()

    # Phase 3: Preis-Trend und Signal-Interaktionen
    ema20_safe = features["ema20"].replace(0, np.nan)
    ema50_safe = features["ema50"].replace(0, np.nan)
    features["ema_price_ratio_20"] = (features["price_at_signal"] / ema20_safe).fillna(1.0)
    features["ema_price_ratio_50"] = (features["price_at_signal"] / ema50_safe).fillna(1.0)
    features["ema_alignment"] = (
        ((features["price_at_signal"] > features["ema20"]) & (features["ema20"] > features["ema50"])) |
        ((features["price_at_signal"] < features["ema20"]) & (features["ema20"] < features["ema50"]))
    ).astype(float)
    features["score_dominance"] = np.where(
        score_sum > 0, np.maximum(bs, ss) / score_sum, 0.5
    )
    features["signal_confidence_product"] = np.where(
        features["signal_strength"] > 0,
        features["confidence"] * features["signal_strength"] / 100,
        0.0,
    )
    features["bb_atr_interaction"] = features["bb_percent_b"] * features["atr_percent"]
    features["rsi_momentum_agreement"] = (
        ((features["rsi"] > 50) & (features["momentum"] > 0)) |
        ((features["rsi"] < 50) & (features["momentum"] < 0))
    ).astype(float)

    # Phase 4: Sentiment-Features
    fg = features["fear_greed_index"]
    features["fear_greed_normalized"] = (fg - 50) / 50
    ns = features["news_sentiment_score"]
    mom = features["momentum"]
    features["sentiment_momentum_agreement"] = (
        ((ns > 0.2) & (mom > 0)) | ((ns < -0.2) & (mom < 0))
    ).astype(float)
    features["sentiment_regime_interaction"] = ns * features["ema_spread"].abs()

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
        return DAY_MAP.get(val.strip().lower(), 2)
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
