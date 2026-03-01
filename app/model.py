"""XGBoost Model Training und Prediction."""

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import optuna
import shap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

from app.config import (
    MODEL_DIR, MIN_TRAINING_SAMPLES, TRAIN_TEST_SPLIT,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    XGB_SUBSAMPLE, XGB_COLSAMPLE_BYTREE, XGB_EARLY_STOPPING,
    ENABLE_OPTUNA, OPTUNA_N_TRIALS,
    ENABLE_REGIME_MODELS, MIN_REGIME_SAMPLES, REGIME_WEIGHT,
)
from app.features import build_features_from_df, build_features_from_dict, get_feature_names
from app.database import load_training_data, insert_model_performance

logger = logging.getLogger(__name__)

# Globaler State
_model = None
_model_version = None
_model_info = {}
_feature_importances = {}
_regime_models = {}  # {"TRENDING_STRONG": model, "RANGING": model, ...}


def get_model():
    return _model


def get_model_version():
    return _model_version


def get_model_info():
    return _model_info.copy()


def get_feature_importances():
    return _feature_importances.copy()


def load_latest_model():
    """Lade das neueste Modell beim Start."""
    global _model, _model_version, _model_info, _feature_importances, _regime_models

    model_dir = Path(MODEL_DIR)
    if not model_dir.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Kein Modell-Verzeichnis, ueberspringe Laden")
        return False

    # Finde neuestes Modell
    model_files = sorted(model_dir.glob("model_v*.joblib"), reverse=True)
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

        # Regime-Modelle laden
        _regime_models = {}
        regime_pattern = f"model_{_model_version}_regime_*.joblib"
        for regime_file in model_dir.glob(regime_pattern):
            regime_name = regime_file.stem.split("_regime_")[-1]
            try:
                _regime_models[regime_name] = joblib.load(regime_file)
            except Exception as e:
                logger.warning(f"Regime-Modell {regime_name} konnte nicht geladen werden: {e}")

        logger.info(f"Modell geladen: {_model_version} ({latest}), Regime-Modelle: {list(_regime_models.keys()) or 'keine'}")
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

    global_proba = float(_model.predict_proba(X)[0][1])
    probability, used_regime = _blend_with_regime(global_proba, X, data.get("market_regime", ""))

    return {
        "prediction": "correct" if probability >= 0.5 else "incorrect",
        "probability": round(probability, 6),
        "model_version": _model_version or "none",
        "regime_model_used": used_regime,
        "feature_importance_top5": _get_top5_importance(),
    }


def predict_batch(items: list) -> list:
    """Batch-Vorhersage fuer mehrere Signale (vektorisiert)."""
    if _model is None:
        return [
            {"error": "Kein Modell geladen", "prediction": "unknown", "probability": 0.5}
            for _ in items
        ]

    feature_names = get_feature_names()
    top5 = _get_top5_importance()

    # Features einmal fuer alle Items bauen
    all_features = [build_features_from_dict(data) for data in items]
    X_all = np.array([[f.get(name, 0) for name in feature_names] for f in all_features])

    # Globale Prediction in einem Aufruf
    global_probas = _model.predict_proba(X_all)[:, 1]

    results = []
    for i, data in enumerate(items):
        probability, used_regime = _blend_with_regime(
            float(global_probas[i]), X_all[i:i+1], data.get("market_regime", "")
        )

        results.append({
            "ticker": data.get("ticker", ""),
            "signal_type": data.get("signal_type", ""),
            "prediction": "correct" if probability >= 0.5 else "incorrect",
            "probability": round(probability, 6),
            "model_version": _model_version or "none",
            "regime_model_used": used_regime,
            "feature_importance_top5": top5,
        })

    return results


def _blend_with_regime(global_proba: float, X_row: np.ndarray, market_regime: str) -> tuple:
    """Blende globale Prediction mit Regime-Modell.

    Returns (probability, regime_name_or_None).
    """
    regime = str(market_regime).upper()
    if regime in _regime_models:
        try:
            regime_proba = float(_regime_models[regime].predict_proba(X_row)[0][1])
            blended = (1 - REGIME_WEIGHT) * global_proba + REGIME_WEIGHT * regime_proba
            return blended, regime
        except Exception:
            pass
    return global_proba, None


def train_model() -> dict:
    """Trainiere neues XGBoost-Modell."""
    global _model, _model_version, _model_info, _feature_importances, _regime_models

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

    # 3. Chronologischer Split: Train / Validation / Test
    #    Optuna tuned auf Val, finale Metriken nur auf Test (kein Data Leakage)
    n = len(df)
    use_3way = ENABLE_OPTUNA and n >= 100
    if use_3way:
        split_idx = int(n * 0.6)
        test_start_idx = int(n * 0.8)
        X_train, X_val, X_test = X.iloc[:split_idx], X.iloc[split_idx:test_start_idx], X.iloc[test_start_idx:]
        y_train, y_val, y_test = y.iloc[:split_idx], y.iloc[split_idx:test_start_idx], y.iloc[test_start_idx:]
        logger.info(f"3-Way Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    else:
        split_idx = int(n * TRAIN_TEST_SPLIT)
        test_start_idx = split_idx
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        X_val, y_val = X_test, y_test
        logger.info(f"2-Way Split: Train={len(X_train)}, Test={len(X_test)}")

    # 4. XGBoost Training (mit Klassen-Balancierung)
    # sqrt-Daempfung: volle Ratio (0.39) war zu aggressiv → sqrt ergibt ~0.63
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    raw_ratio = neg_count / pos_count if pos_count > 0 else 1.0
    scale_pos_weight = float(np.sqrt(raw_ratio))

    logger.info(f"Klassen-Balance: pos={pos_count}, neg={neg_count}, scale_pos_weight={scale_pos_weight:.3f}")

    # Optuna Hyperparameter Tuning oder Default-Parameter
    used_optuna = False
    best_params = {
        "n_estimators": XGB_N_ESTIMATORS,
        "max_depth": XGB_MAX_DEPTH,
        "learning_rate": XGB_LEARNING_RATE,
        "subsample": XGB_SUBSAMPLE,
        "colsample_bytree": XGB_COLSAMPLE_BYTREE,
    }

    if use_3way:
        try:
            logger.info(f"Starte Optuna Tuning ({OPTUNA_N_TRIALS} Trials)...")
            best_params = _optuna_tune(X_train, y_train, X_val, y_val, scale_pos_weight)
            used_optuna = True
        except Exception as e:
            logger.warning(f"Optuna fehlgeschlagen, nutze Defaults: {e}")

    model = XGBClassifier(
        **best_params,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        early_stopping_rounds=XGB_EARLY_STOPPING,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # 5. Metriken
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    logger.info(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    # Per-Signal-Type Metriken
    signal_type_metrics = {}
    if "signal_type" in df.columns:
        test_signal_types = df.iloc[test_start_idx:]["signal_type"]
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

    # 6. Feature Importance (XGBoost native)
    feature_names = get_feature_names()
    importances = dict(zip(feature_names, model.feature_importances_.tolist()))
    sorted_imp = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    # 6b. SHAP-basierte Feature Importance (auf Sample begrenzt fuer RAM-Schutz)
    shap_imp = {}
    try:
        explainer = shap.TreeExplainer(model)
        shap_sample = X_test.sample(min(200, len(X_test)), random_state=42)
        shap_values = explainer.shap_values(shap_sample)
        shap_mean_abs = np.abs(shap_values).mean(axis=0)
        shap_imp = dict(zip(feature_names, shap_mean_abs.tolist()))
        shap_imp = dict(sorted(shap_imp.items(), key=lambda x: x[1], reverse=True))
        logger.info("SHAP Top-10: " + ", ".join(
            f"{k}={v:.4f}" for k, v in list(shap_imp.items())[:10]
        ))
    except Exception as e:
        logger.warning(f"SHAP-Berechnung fehlgeschlagen: {e}")

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
        "shap_importances": shap_imp,
        "parameters": {
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in best_params.items()},
            "scale_pos_weight": round(scale_pos_weight, 4),
            "optuna_tuned": used_optuna,
        },
    }

    info_path = model_path.with_suffix(".json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    # 8. DB loggen
    try:
        insert_model_performance(
            model_version=version,
            training_samples=split_idx,
            accuracy=acc,
            precision_score=prec,
            recall_score=rec,
            f1_score=f1,
            features={
                "names": feature_names,
                "importances": sorted_imp,
                "shap_importances": shap_imp,
            },
            parameters=info["parameters"],
            notes=f"Train: {split_idx}, Test: {len(X_test)}, Positive Rate: {y.mean():.2%}, Optuna: {used_optuna}",
        )
    except Exception as e:
        logger.error(f"DB-Logging fehlgeschlagen: {e}")

    # 9. Regime-spezifische Modelle trainieren
    regime_models_trained = {}
    _trained_regime_models = {}
    if ENABLE_REGIME_MODELS and "signal_type" in df.columns and "market_regime" in df.columns:
        regime_col = df["market_regime"].fillna("UNKNOWN").str.upper()
        for regime_name in regime_col.unique():
            if regime_name in ("UNKNOWN", ""):
                continue
            regime_mask = regime_col == regime_name
            regime_train_mask = regime_mask.iloc[:split_idx]
            regime_test_mask = regime_mask.iloc[test_start_idx:]

            n_train = int(regime_train_mask.sum())
            n_test = int(regime_test_mask.sum())

            if n_train < MIN_REGIME_SAMPLES or n_test < 10:
                logger.info(f"  Regime {regime_name}: uebersprungen (train={n_train}, test={n_test})")
                continue

            X_r_train = X_train[regime_train_mask.values]
            y_r_train = y_train[regime_train_mask.values]
            X_r_test = X_test[regime_test_mask.values]
            y_r_test = y_test[regime_test_mask.values]

            r_neg = int((y_r_train == 0).sum())
            r_pos = int((y_r_train == 1).sum())
            r_spw = float(np.sqrt(r_neg / r_pos)) if r_pos > 0 else 1.0

            r_model = XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=r_spw,
                eval_metric="logloss", use_label_encoder=False,
                random_state=42,
            )
            r_model.fit(X_r_train, y_r_train, verbose=False)

            r_pred = r_model.predict(X_r_test)
            r_acc = accuracy_score(y_r_test, r_pred)
            r_f1 = f1_score(y_r_test, r_pred, zero_division=0)

            regime_path = model_dir / f"model_{version}_regime_{regime_name}.joblib"
            joblib.dump(r_model, regime_path)
            _trained_regime_models[regime_name] = r_model

            regime_models_trained[regime_name] = {
                "train_samples": n_train, "test_samples": n_test,
                "accuracy": round(r_acc, 4), "f1": round(r_f1, 4),
            }
            logger.info(f"  Regime {regime_name}: Acc={r_acc:.4f}, F1={r_f1:.4f} (train={n_train}, test={n_test})")

    info["regime_models"] = regime_models_trained

    # Info-JSON nochmal speichern (mit Regime-Daten)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    # 10. Globalen State aktualisieren
    _model = model
    _model_version = version
    _model_info = info
    _feature_importances = sorted_imp
    _regime_models = _trained_regime_models

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
        "optuna_tuned": used_optuna,
        "signal_type_metrics": signal_type_metrics,
        "feature_importance_top10": [
            {"feature": k, "importance": round(v, 4)}
            for k, v in list(sorted_imp.items())[:10]
        ],
        "shap_importance_top10": [
            {"feature": k, "shap_value": round(v, 4)}
            for k, v in list(shap_imp.items())[:10]
        ],
        "regime_models": regime_models_trained,
    }


def _optuna_tune(X_train, y_train, X_val, y_val, scale_pos_weight: float) -> dict:
    """Finde optimale Hyperparameter mit Optuna auf dem Validation-Set."""

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        model = XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            early_stopping_rounds=20,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict(X_val)
        return f1_score(y_val, y_pred, zero_division=0)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="xgb_tuning")
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=False)

    logger.info(f"Optuna best F1: {study.best_value:.4f} nach {len(study.trials)} Trials")
    logger.info(f"Optuna best params: {study.best_params}")

    return study.best_params


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
    # Nur Haupt-Modelle (nicht regime_*) fuer Sortierung
    model_files = sorted(model_dir.glob("model_v*.joblib"), reverse=True)
    # Regime-Modelle rausfiltern
    main_models = [f for f in model_files if "_regime_" not in f.name]

    for old_model in main_models[keep:]:
        version_prefix = old_model.stem  # z.B. "model_v20260228_120000"
        try:
            old_model.unlink()
            info_file = old_model.with_suffix(".json")
            if info_file.exists():
                info_file.unlink()
            # Zugehoerige Regime-Modelle loeschen
            for regime_file in model_dir.glob(f"{version_prefix}_regime_*.joblib"):
                regime_file.unlink()
                logger.info(f"Regime-Modell geloescht: {regime_file.name}")
            logger.info(f"Altes Modell geloescht: {old_model.name}")
        except Exception as e:
            logger.warning(f"Konnte {old_model.name} nicht loeschen: {e}")
