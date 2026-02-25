"""XGBoost Model Training und Prediction."""

import os
import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from app.config import (
    MODEL_DIR, MIN_TRAINING_SAMPLES, TRAIN_TEST_SPLIT,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    XGB_SUBSAMPLE, XGB_COLSAMPLE_BYTREE, XGB_EARLY_STOPPING,
)
from app.features import build_features_from_df, build_features_from_dict, get_feature_names
from app.database import load_training_data, insert_model_performance

logger = logging.getLogger(__name__)

# Globaler State
_model = None
_model_version = None
_model_info = {}
_feature_importances = {}


def get_model():
    global _model
    return _model


def get_model_version():
    return _model_version


def get_model_info():
    return _model_info.copy()


def get_feature_importances():
    return _feature_importances.copy()


def load_latest_model():
    """Lade das neueste Modell beim Start."""
    global _model, _model_version, _model_info, _feature_importances

    model_dir = Path(MODEL_DIR)
    if not model_dir.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Kein Modell-Verzeichnis, ueberspringe Laden")
        return False

    # Finde neuestes Modell
    model_files = sorted(model_dir.glob("model_*.joblib"), reverse=True)
    if not model_files:
        logger.info("Kein gespeichertes Modell gefunden")
        return False

    latest = model_files[0]
    info_file = latest.with_suffix(".json")

    try:
        _model = joblib.load(latest)
        _model_version = latest.stem.replace("model_", "")

        if info_file.exists():
            with open(info_file) as f:
                _model_info = json.load(f)
            _feature_importances = _model_info.get("feature_importances", {})
        else:
            _feature_importances = _extract_importances(_model)
            _model_info = {"version": _model_version}

        logger.info(f"Modell geladen: {_model_version} ({latest})")
        return True
    except Exception as e:
        logger.error(f"Fehler beim Laden: {e}")
        return False


def predict_single(data: dict) -> dict:
    """Einzelne Vorhersage fuer ein Signal."""
    if _model is None:
        return {"error": "Kein Modell geladen", "prediction": "unknown", "probability": 0.5}

    features = build_features_from_dict(data)
    feature_names = get_feature_names()

    X = np.array([[features.get(f, 0) for f in feature_names]])

    proba = _model.predict_proba(X)[0]
    pred_class = int(_model.predict(X)[0])

    probability = float(proba[1])  # P(correct)
    prediction = "correct" if pred_class == 1 else "incorrect"

    # Top-5 Feature Importance
    top5 = _get_top5_importance()

    return {
        "prediction": prediction,
        "probability": round(probability, 6),
        "model_version": _model_version or "none",
        "feature_importance_top5": top5,
    }


def predict_batch(items: list) -> list:
    """Batch-Vorhersage fuer mehrere Signale."""
    if _model is None:
        return [
            {"error": "Kein Modell geladen", "prediction": "unknown", "probability": 0.5}
            for _ in items
        ]

    feature_names = get_feature_names()
    results = []

    for data in items:
        features = build_features_from_dict(data)
        X = np.array([[features.get(f, 0) for f in feature_names]])

        proba = _model.predict_proba(X)[0]
        pred_class = int(_model.predict(X)[0])

        results.append({
            "ticker": data.get("ticker", ""),
            "signal_type": data.get("signal_type", ""),
            "prediction": "correct" if pred_class == 1 else "incorrect",
            "probability": round(float(proba[1]), 6),
            "model_version": _model_version or "none",
        })

    return results


def train_model() -> dict:
    """Trainiere neues XGBoost-Modell."""
    global _model, _model_version, _model_info, _feature_importances

    logger.info("Starte Training...")

    # 1. Daten laden
    df = load_training_data()
    total_samples = len(df)
    logger.info(f"Geladene Samples: {total_samples}")

    if total_samples < MIN_TRAINING_SAMPLES:
        return {
            "success": False,
            "error": f"Zu wenige Samples: {total_samples} (min. {MIN_TRAINING_SAMPLES})",
            "total_samples": total_samples,
        }

    # 2. Features bauen
    X = build_features_from_df(df)
    y = (df["was_correct"] == "true").astype(int)

    logger.info(f"Features: {X.shape[1]}, Positive Rate: {y.mean():.2%}")

    # 3. Chronologischer Split
    split_idx = int(len(df) * TRAIN_TEST_SPLIT)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # 4. XGBoost Training (mit Klassen-Balancierung)
    # sqrt-Daempfung: volle Ratio (0.39) war zu aggressiv → sqrt ergibt ~0.63
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    raw_ratio = neg_count / pos_count if pos_count > 0 else 1.0
    scale_pos_weight = float(np.sqrt(raw_ratio))

    logger.info(f"Klassen-Balance: pos={pos_count}, neg={neg_count}, scale_pos_weight={scale_pos_weight:.3f}")

    model = XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        early_stopping_rounds=XGB_EARLY_STOPPING,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # 5. Metriken
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    logger.info(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    # Per-Signal-Type Metriken
    signal_type_metrics = {}
    if "signal_type" in df.columns:
        test_signal_types = df.iloc[split_idx:]["signal_type"]
        for st in test_signal_types.unique():
            mask = test_signal_types == st
            if mask.sum() > 0:
                st_acc = accuracy_score(y_test[mask.values], y_pred[mask.values])
                signal_type_metrics[st] = {
                    "count": int(mask.sum()),
                    "accuracy": round(st_acc, 4),
                    "positive_rate": round(float(y_test[mask.values].mean()), 4),
                }
                logger.info(f"  {st}: Accuracy={st_acc:.4f} (n={mask.sum()}, pos_rate={y_test[mask.values].mean():.2%})")

    # 6. Feature Importance
    feature_names = get_feature_names()
    importances = dict(zip(feature_names, model.feature_importances_.tolist()))
    sorted_imp = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    # 7. Version + Speichern
    version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"model_{version}.joblib"
    joblib.dump(model, model_path)

    info = {
        "version": version,
        "trained_at": datetime.now().isoformat(),
        "training_samples": int(split_idx),
        "test_samples": int(len(X_test)),
        "total_samples": total_samples,
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "positive_rate_train": round(float(y_train.mean()), 4),
        "positive_rate_test": round(float(y_test.mean()), 4),
        "scale_pos_weight": round(scale_pos_weight, 4),
        "signal_type_metrics": signal_type_metrics,
        "feature_importances": sorted_imp,
        "parameters": {
            "n_estimators": XGB_N_ESTIMATORS,
            "max_depth": XGB_MAX_DEPTH,
            "learning_rate": XGB_LEARNING_RATE,
            "subsample": XGB_SUBSAMPLE,
            "colsample_bytree": XGB_COLSAMPLE_BYTREE,
            "scale_pos_weight": round(scale_pos_weight, 4),
        },
    }

    info_path = model_path.with_suffix(".json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    # 8. DB loggen
    try:
        insert_model_performance(
            model_version=version,
            training_samples=total_samples,
            accuracy=acc,
            precision_score=prec,
            recall_score=rec,
            f1_score=f1,
            features={"names": feature_names, "importances": sorted_imp},
            parameters=info["parameters"],
            notes=f"Train: {split_idx}, Test: {len(X_test)}, Positive Rate: {y.mean():.2%}",
        )
    except Exception as e:
        logger.error(f"DB-Logging fehlgeschlagen: {e}")

    # 9. Globalen State aktualisieren
    _model = model
    _model_version = version
    _model_info = info
    _feature_importances = sorted_imp

    # Alte Modelle aufraeumen (behalte letzte 5)
    _cleanup_old_models(model_dir, keep=5)

    return {
        "success": True,
        "version": version,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "training_samples": split_idx,
        "test_samples": len(X_test),
        "total_samples": total_samples,
        "positive_rate": round(float(y.mean()), 4),
        "scale_pos_weight": round(scale_pos_weight, 4),
        "signal_type_metrics": signal_type_metrics,
        "feature_importance_top10": [
            {"feature": k, "importance": round(v, 4)}
            for k, v in list(sorted_imp.items())[:10]
        ],
    }


def _extract_importances(model) -> dict:
    try:
        feature_names = get_feature_names()
        return dict(zip(feature_names, model.feature_importances_.tolist()))
    except Exception:
        return {}


def _get_top5_importance() -> list:
    if not _feature_importances:
        return []
    sorted_items = sorted(_feature_importances.items(), key=lambda x: x[1], reverse=True)
    return [
        {"feature": k, "importance": round(v, 4)}
        for k, v in sorted_items[:5]
    ]


def _cleanup_old_models(model_dir: Path, keep: int = 5):
    """Loesche alte Modelle, behalte die neuesten."""
    model_files = sorted(model_dir.glob("model_*.joblib"), reverse=True)
    for old_model in model_files[keep:]:
        try:
            old_model.unlink()
            info_file = old_model.with_suffix(".json")
            if info_file.exists():
                info_file.unlink()
            logger.info(f"Altes Modell geloescht: {old_model.name}")
        except Exception as e:
            logger.warning(f"Konnte {old_model.name} nicht loeschen: {e}")
