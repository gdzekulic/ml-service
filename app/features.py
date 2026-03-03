"""Feature Engineering Pipeline."""

import numpy as np
import pandas as pd

from app.config import (
    NUMERIC_FEATURES, BOOLEAN_FEATURES, DERIVED_FEATURES,
    TEMPORAL_FEATURES, CATEGORICAL_FEATURES,
)
from app.database import load_ticker_history, count_signals_today

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

# Temporal features that default to 0.5 instead of 0.0 (neutral baseline)
TEMPORAL_NEUTRAL_DEFAULTS = {"ticker_win_rate_10", "ticker_last_outcome"}


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

    # Phase 5: Temporal Features (Lag/Delta/Rolling/Outcome aus DB-History)
    ticker = data.get("ticker", "")
    history = _load_history_safe(ticker)

    if history:
        prev = history[0]  # neuestes gemessenes Signal
        features["rsi_lag_1"] = _safe_float(prev.get("rsi"))
        features["buy_score_lag_1"] = _safe_float(prev.get("buy_score"))
        features["momentum_lag_1"] = _safe_float(prev.get("momentum"))
        features["atr_percent_lag_1"] = _safe_float(prev.get("atr_percent"))

        features["rsi_delta"] = features["rsi"] - features["rsi_lag_1"]
        features["buy_score_delta"] = features["buy_score"] - features["buy_score_lag_1"]
        features["sell_score_delta"] = features["sell_score"] - _safe_float(prev.get("sell_score"))
        features["momentum_delta"] = features["momentum"] - features["momentum_lag_1"]

        # Rolling (max 5 Signale)
        hist_slice = history[:5]
        rsi_vals = [_safe_float(h.get("rsi")) for h in hist_slice]
        bs_vals = [_safe_float(h.get("buy_score")) for h in hist_slice]
        atr_vals = [_safe_float(h.get("atr_percent")) for h in hist_slice]

        features["rsi_rolling_mean_5"] = np.mean(rsi_vals)
        features["rsi_rolling_std_5"] = np.std(rsi_vals, ddof=0)
        features["buy_score_rolling_mean_5"] = np.mean(bs_vals)
        features["atr_percent_rolling_mean_5"] = np.mean(atr_vals)

        # Outcome-History
        measured = [h for h in history if h.get("was_correct") in ("true", "false")]
        if measured:
            wins = sum(1 for h in measured if h.get("was_correct") == "true")
            features["ticker_win_rate_10"] = wins / len(measured)
            features["ticker_last_outcome"] = 1.0 if measured[0].get("was_correct") == "true" else 0.0
        else:
            features["ticker_win_rate_10"] = 0.5
            features["ticker_last_outcome"] = 0.5
    else:
        for f in TEMPORAL_FEATURES:
            features[f] = 0.5 if f in TEMPORAL_NEUTRAL_DEFAULTS else 0.0

    # consecutive_signals_today (unabhaengig von History, eigener DB-Query)
    features["consecutive_signals_today"] = _count_signals_safe(ticker)

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

    # Phase 5: Temporal Features (Lag/Delta/Rolling/Outcome per Ticker)
    if "ticker" in df.columns:
        grp = df.groupby("ticker")

        # Lag-Features (shift=1 → vorheriges Signal)
        features["rsi_lag_1"] = grp["rsi"].shift(1).reindex(df.index).fillna(0.0).values
        features["buy_score_lag_1"] = grp["buy_score"].shift(1).reindex(df.index).fillna(0.0).values
        features["momentum_lag_1"] = grp["momentum"].shift(1).reindex(df.index).fillna(0.0).values
        features["atr_percent_lag_1"] = grp["atr_percent"].shift(1).reindex(df.index).fillna(0.0).values

        # Delta-Features
        features["rsi_delta"] = features["rsi"] - features["rsi_lag_1"]
        features["buy_score_delta"] = features["buy_score"] - features["buy_score_lag_1"]
        sell_score_lag = grp["sell_score"].shift(1).reindex(df.index).fillna(0.0).values
        features["sell_score_delta"] = features["sell_score"] - sell_score_lag
        features["momentum_delta"] = features["momentum"] - features["momentum_lag_1"]

        # Rolling-Features (Fenster=5, min_periods=1, shift(1) gegen Data Leakage)
        features["rsi_rolling_mean_5"] = grp["rsi"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        ).reindex(df.index).fillna(0.0).values
        features["rsi_rolling_std_5"] = grp["rsi"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).std(ddof=0)
        ).reindex(df.index).fillna(0.0).values
        features["buy_score_rolling_mean_5"] = grp["buy_score"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        ).reindex(df.index).fillna(0.0).values
        features["atr_percent_rolling_mean_5"] = grp["atr_percent"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        ).reindex(df.index).fillna(0.0).values

        # Outcome-History: Win-Rate der letzten 10 gemessenen Signale
        if "was_correct" in df.columns:
            # shift(1) um Data Leakage zu verhindern
            correct_bool = df["was_correct"].astype(str).str.lower().eq("true").astype(float)
            shifted_correct = correct_bool.groupby(df["ticker"]).shift(1)

            features["ticker_win_rate_10"] = shifted_correct.groupby(
                df["ticker"]
            ).transform(
                lambda x: x.rolling(10, min_periods=1).mean()
            ).fillna(0.5).values

            features["ticker_last_outcome"] = shifted_correct.fillna(0.5).values
        else:
            features["ticker_win_rate_10"] = 0.5
            features["ticker_last_outcome"] = 0.5

        # consecutive_signals_today
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            signal_date = ts.dt.date
            features["consecutive_signals_today"] = df.groupby(
                [df["ticker"], signal_date]
            ).cumcount().values.astype(float)
        else:
            features["consecutive_signals_today"] = 0.0
    else:
        for f in TEMPORAL_FEATURES:
            features[f] = 0.5 if f in TEMPORAL_NEUTRAL_DEFAULTS else 0.0

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
    names = (
        list(NUMERIC_FEATURES) + list(BOOLEAN_FEATURES)
        + list(DERIVED_FEATURES) + list(TEMPORAL_FEATURES)
    )
    for cat_name, possible_values in CATEGORICAL_FEATURES.items():
        for val in possible_values[1:]:
            names.append(f"{cat_name}_{val}")
    return names


def _load_history_safe(ticker: str) -> list:
    """Lade Ticker-History aus DB, bei Fehler leere Liste."""
    if not ticker:
        return []
    try:
        return load_ticker_history(ticker, n=10)
    except Exception:
        return []


def _count_signals_safe(ticker: str) -> float:
    """Zaehle heutige Signale, bei Fehler 0."""
    if not ticker:
        return 0.0
    try:
        return float(count_signals_today(ticker))
    except Exception:
        return 0.0


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
