"""FastAPI ML-Service fuer Signal-Vorhersage."""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.model import (
    load_latest_model, predict_single, predict_batch,
    train_model, get_model_version, get_model_info,
    get_feature_importances, get_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ML-Service startet...")
    loaded = load_latest_model()
    if loaded:
        logger.info(f"Modell geladen: {get_model_version()}")
    else:
        logger.info("Kein Modell vorhanden - Training erforderlich")
    yield
    logger.info("ML-Service beendet")


app = FastAPI(
    title="Trading Bot ML-Service",
    description="XGBoost Signal-Vorhersage Service",
    version="1.0.0",
    lifespan=lifespan,
)


# === Request/Response Models ===

class PredictRequest(BaseModel):
    ticker: str
    signal_type: str
    buy_score: float = 0
    sell_score: float = 0
    rsi: float = 50
    stoch_k: float = 50
    stoch_d: float = 50
    macd: float = 0
    bb_percent_b: float = 0.5
    atr_percent: float = 1.0
    momentum: float = 0
    ema20: float = 0
    ema50: float = 0
    price_at_signal: float = 0
    risk_reward: float = 1.0
    confidence: float = 0.5
    signal_strength: float = 2
    price_vs_ema20: bool = True
    price_vs_ema50: bool = True
    rsi_divergence_bull: bool = False
    rsi_divergence_bear: bool = False
    bb_squeeze: bool = False
    stoch_crossover: str = "none"
    macd_crossover: str = "none"
    market_regime: str = "UNKNOWN"
    trend_direction: str = "sideways"
    hour_of_day: int = 12
    day_of_week: int = 2
    signal_id: Optional[str] = None


class BatchPredictRequest(BaseModel):
    signals: list[PredictRequest]


class PredictResponse(BaseModel):
    prediction: str
    probability: float
    model_version: str
    feature_importance_top5: list = []
    error: Optional[str] = None


class TrainResponse(BaseModel):
    success: bool
    version: Optional[str] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    training_samples: Optional[int] = None
    test_samples: Optional[int] = None
    total_samples: Optional[int] = None
    positive_rate: Optional[float] = None
    feature_importance_top10: Optional[list] = None
    error: Optional[str] = None


# === Endpoints ===

@app.get("/health")
async def health():
    model = get_model()
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": get_model_version(),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    data = request.model_dump()
    result = predict_single(data)

    # Optional: Prediction in DB speichern
    if request.signal_id:
        try:
            from app.database import insert_prediction
            insert_prediction(
                signal_id=request.signal_id,
                model_version=result.get("model_version", "unknown"),
                prediction=result["prediction"],
                confidence=result["probability"],
                features_used=data,
                probability=result["probability"],
                feature_importance_top5=result.get("feature_importance_top5"),
            )
        except Exception as e:
            logger.error(f"Prediction-Logging fehlgeschlagen: {e}")

    return PredictResponse(**result)


@app.post("/predict/batch")
async def predict_batch_endpoint(request: BatchPredictRequest):
    items = [s.model_dump() for s in request.signals]
    results = predict_batch(items)

    # Predictions in DB speichern
    for i, (signal, result) in enumerate(zip(request.signals, results)):
        signal_id = signal.signal_id
        if signal_id and "error" not in result:
            try:
                from app.database import insert_prediction
                insert_prediction(
                    signal_id=signal_id,
                    model_version=result.get("model_version", "unknown"),
                    prediction=result["prediction"],
                    confidence=result["probability"],
                    features_used=items[i],
                    probability=result["probability"],
                    feature_importance_top5=None,
                )
            except Exception as e:
                logger.error(f"Batch Prediction-Logging fehlgeschlagen fuer {signal_id}: {e}")

    return {"predictions": results, "count": len(results)}


@app.post("/train", response_model=TrainResponse)
async def train():
    try:
        result = train_model()
        return TrainResponse(**result)
    except Exception as e:
        logger.error(f"Training fehlgeschlagen: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    info = get_model_info()
    importances = get_feature_importances()

    if not info:
        return {
            "model_loaded": False,
            "message": "Kein Modell geladen",
        }

    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    return {
        "model_loaded": True,
        "model_version": info.get("version"),
        "trained_at": info.get("trained_at"),
        "accuracy": info.get("accuracy"),
        "precision": info.get("precision"),
        "recall": info.get("recall"),
        "f1": info.get("f1"),
        "training_samples": info.get("training_samples"),
        "test_samples": info.get("test_samples"),
        "total_samples": info.get("total_samples"),
        "positive_rate_train": info.get("positive_rate_train"),
        "positive_rate_test": info.get("positive_rate_test"),
        "feature_importance_top10": [
            {"feature": k, "importance": round(v, 4)}
            for k, v in sorted_imp[:10]
        ],
        "feature_importance_all": {k: round(v, 4) for k, v in sorted_imp},
    }


@app.get("/model/info")
async def model_info():
    info = get_model_info()
    if not info:
        return {"model_loaded": False, "message": "Kein Modell geladen"}

    return {
        "model_loaded": True,
        "version": info.get("version"),
        "trained_at": info.get("trained_at"),
        "training_samples": info.get("training_samples"),
        "test_samples": info.get("test_samples"),
        "total_samples": info.get("total_samples"),
        "accuracy": info.get("accuracy"),
        "f1": info.get("f1"),
        "parameters": info.get("parameters"),
    }
