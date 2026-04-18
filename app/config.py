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

# Optuna Hyperparameter Tuning
ENABLE_OPTUNA = os.getenv("ENABLE_OPTUNA", "true").lower() in ("true", "1", "yes")
OPTUNA_N_TRIALS = int(os.getenv("OPTUNA_N_TRIALS", "15"))

# Multi-Regime Modelle
ENABLE_REGIME_MODELS = os.getenv("ENABLE_REGIME_MODELS", "true").lower() in ("true", "1", "yes")
MIN_REGIME_SAMPLES = int(os.getenv("MIN_REGIME_SAMPLES", "50"))
REGIME_WEIGHT = float(os.getenv("REGIME_WEIGHT", "0.3"))  # 30% Regime, 70% Global

# Feature-Listen
NUMERIC_FEATURES = [
    "buy_score", "sell_score", "rsi", "stoch_k", "stoch_d",
    "macd", "bb_percent_b", "atr_percent", "momentum",
    "ema20", "ema50", "price_at_signal", "risk_reward",
    "confidence", "signal_strength",
    "score_15min_buy", "score_15min_sell",
    "fear_greed_index", "news_sentiment_score", "news_count",
    # Phase 6: Volumen
    "volume_ratio",
    # Phase B2: Momentum-Beschleunigung (Spec 2026-04-18)
    "rsi_velocity_15m", "rsi_velocity_1h",
    "macd_hist_slope_15m", "macd_hist_slope_1h",
    "roc_acceleration_15m", "volume_surge_pattern",
    "mtf_momentum_lead",
]

BOOLEAN_FEATURES = [
    "price_vs_ema20", "price_vs_ema50",
    "rsi_divergence_bull", "rsi_divergence_bear",
    "mtf_alignment",
    "company_news_found",
    # Phase 6: Volumen
    "price_volume_div",
]

DERIVED_FEATURES = [
    "score_spread", "stoch_diff", "ema_spread",
    "bb_squeeze", "hour_of_day", "day_of_week_num",
    # Phase 2: Erweiterte abgeleitete Features
    "score_ratio", "score_total",
    "rsi_extreme", "stoch_extreme", "bb_position_extreme",
    "macd_abs", "momentum_abs",
    "atr_momentum_interaction", "rsi_stoch_agreement", "trend_strength",
    # Phase 3: Preis-Trend und Signal-Interaktionen
    "ema_price_ratio_20", "ema_price_ratio_50",
    "ema_alignment", "score_dominance",
    "signal_confidence_product", "bb_atr_interaction",
    "rsi_momentum_agreement",
    # Phase 4: Sentiment-Features
    "fear_greed_normalized",
    "sentiment_momentum_agreement",
    "sentiment_regime_interaction",
    # Phase 6: Volumen-Interaktionen
    "volume_momentum_agreement",
    "volume_breakout_signal",
]

TEMPORAL_FEATURES = [
    # Phase 5: Zeitreihen-Features
    # Lag-Features (vorheriges Signal desselben Tickers)
    "rsi_lag_1", "buy_score_lag_1", "momentum_lag_1", "atr_percent_lag_1",
    # Delta-Features (Veraenderung zum vorherigen Signal)
    "rsi_delta", "buy_score_delta", "sell_score_delta", "momentum_delta",
    # Rolling-Features (Durchschnitt/Std letzter 5 Signale)
    "rsi_rolling_mean_5", "rsi_rolling_std_5",
    "buy_score_rolling_mean_5", "atr_percent_rolling_mean_5",
    # Outcome-History (Ticker-Performance)
    "ticker_win_rate_10", "ticker_last_outcome",
    # Signal-Frequenz
    "consecutive_signals_today",
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
    "exchange": ["XETRA", "NASDAQ", "NYSE"],
    "sector": [
        "Technology", "Healthcare", "Financial", "Industrial",
        "Consumer", "Energy", "Communication", "Other",
    ],
}
