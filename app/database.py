"""PostgreSQL Datenbankverbindung."""

import psycopg2
import psycopg2.extras
import pandas as pd
from contextlib import contextmanager

from app.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD


def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


@contextmanager
def get_cursor(commit=False):
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        yield cur
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def load_training_data() -> pd.DataFrame:
    """Lade gemessene Signale fuer Training."""
    query = """
        SELECT
            signal_id, timestamp, ticker, signal_type,
            buy_score, sell_score, rsi, stoch_k, stoch_d,
            macd, bb_percent_b, atr_percent, momentum,
            ema20, ema50, price_at_signal, risk_reward,
            confidence, signal_strength,
            price_vs_ema20, price_vs_ema50,
            rsi_divergence_bull, rsi_divergence_bear,
            bb_squeeze,
            stoch_crossover, macd_crossover,
            market_regime, trend_direction,
            exchange, sector, mtf_alignment,
            score_15min_buy, score_15min_sell,
            day_of_week, time_of_day,
            was_correct
        FROM signals_extended
        WHERE outcome_status = 'measured'
          AND was_correct IN ('true', 'false')
        ORDER BY timestamp ASC
    """
    conn = get_connection()
    try:
        df = pd.read_sql(query, conn)
        return df
    finally:
        conn.close()


def insert_model_performance(
    model_version: str,
    training_samples: int,
    accuracy: float,
    precision_score: float,
    recall_score: float,
    f1_score: float,
    features: dict,
    parameters: dict,
    notes: str = "",
):
    """Speichere Modell-Performance in DB."""
    query = """
        INSERT INTO ml_model_performance
            (model_version, training_samples, accuracy, precision_score,
             recall_score, f1_score, features, parameters, notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    import json

    with get_cursor(commit=True) as cur:
        cur.execute(query, (
            model_version, training_samples, accuracy, precision_score,
            recall_score, f1_score,
            json.dumps(features), json.dumps(parameters), notes,
        ))


def insert_prediction(
    signal_id: str,
    model_version: str,
    prediction: str,
    confidence: float,
    features_used: dict,
    probability: float = None,
    feature_importance_top5: dict = None,
):
    """Speichere ML-Prediction in DB."""
    import json

    query = """
        INSERT INTO ml_predictions
            (signal_id, model_version, prediction, confidence,
             features_used, probability, feature_importance_top5)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    with get_cursor(commit=True) as cur:
        cur.execute(query, (
            signal_id, model_version, prediction, confidence,
            json.dumps(features_used),
            probability,
            json.dumps(feature_importance_top5) if feature_importance_top5 else None,
        ))
