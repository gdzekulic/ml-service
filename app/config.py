"""ML-Service Konfiguration."""

import os

# Database
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "trading_bot")
DB_USER = os.getenv("DB_USER", "trading_bot_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# Model
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")
MIN_TRAINING_SAMPLES = int(os.getenv("MIN_TRAINING_SAMPLES", "200"))
TRAIN_TEST_SPLIT = float(os.getenv("TRAIN_TEST_SPLIT", "0.8"))

# XGBoost Defaults
XGB_N_ESTIMATORS = int(os.getenv("XGB_N_ESTIMATORS", "200"))
XGB_MAX_DEPTH = int(os.getenv("XGB_MAX_DEPTH", "6"))
XGB_LEARNING_RATE = float(os.getenv("XGB_LEARNING_RATE", "0.1"))
XGB_SUBSAMPLE = float(os.getenv("XGB_SUBSAMPLE", "0.8"))
XGB_COLSAMPLE_BYTREE = float(os.getenv("XGB_COLSAMPLE_BYTREE", "0.8"))
XGB_EARLY_STOPPING = int(os.getenv("XGB_EARLY_STOPPING", "20"))

# Feature-Listen
NUMERIC_FEATURES = [
    "buy_score", "sell_score", "rsi", "stoch_k", "stoch_d",
    "macd", "bb_percent_b", "atr_percent", "momentum",
    "ema20", "ema50", "price_at_signal", "risk_reward",
    "confidence", "signal_strength",
]

BOOLEAN_FEATURES = [
    "price_vs_ema20", "price_vs_ema50",
    "rsi_divergence_bull", "rsi_divergence_bear",
]

DERIVED_FEATURES = [
    "score_spread", "stoch_diff", "ema_spread",
    "bb_squeeze", "hour_of_day", "day_of_week_num",
]

CATEGORICAL_FEATURES = {
    "signal_type": ["BUY", "SELL", "WATCH_BUY", "WATCH_SELL", "NONE"],
    "market_regime": [
        "VOLATILE", "TRENDING_STRONG", "TRENDING_WEAK",
        "RANGING", "SQUEEZE", "TRANSITIONAL", "UNKNOWN",
    ],
    "trend_direction": ["uptrend", "downtrend", "sideways"],
    "stoch_crossover": ["bullish", "bearish", "none"],
    "macd_crossover": ["bullish", "bearish", "none"],
}
