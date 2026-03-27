# ML-Service - Trading Signal Prediction

FastAPI-basierter Machine-Learning-Service zur Vorhersage, ob Trading-Signale korrekt oder inkorrekt sind. Nutzt einen XGBoost-Classifier mit 75+ Features, Optuna-Hyperparameter-Tuning und regime-spezifischen Modellen.

## Inhaltsverzeichnis

- [Ueberblick](#ueberblick)
- [Architektur](#architektur)
- [Schnellstart](#schnellstart)
- [API-Referenz](#api-referenz)
- [ML-Modell](#ml-modell)
- [Feature-Engineering](#feature-engineering)
- [Konfiguration](#konfiguration)
- [Deployment](#deployment)
- [Datenbank](#datenbank)
- [Monitoring & Logs](#monitoring--logs)

## Ueberblick

Der ML-Service ist Teil eines automatisierten Trading-Systems und bewertet Kauf-/Verkaufssignale mit einer Wahrscheinlichkeit fuer Korrektheit. Er ist als Microservice konzipiert und kommuniziert ueber REST-API mit dem n8n-Orchestrator und dem Streamlit-Dashboard.

**Kernfunktionen:**
- Binary Classification: `correct` / `incorrect` fuer Trading-Signale
- 75+ Features aus technischer Analyse, Sentiment und Marktregimen
- Optuna-basiertes Hyperparameter-Tuning (15 Trials)
- Regime-spezifische Modelle (Blending: 30% Regime + 70% Global)
- SHAP-basierte Modell-Erklaerbarkeit
- Automatische Modell-Versionierung und Cleanup

## Architektur

```
                    +-------------------+
                    |   n8n Workflows   |
                    |  (Signal Scanner) |
                    +--------+----------+
                             |
                    POST /predict
                             |
                    +--------v----------+
                    |    ML-Service     |
                    |   (FastAPI)       |
                    |   Port 8000       |
                    +--------+----------+
                             |
              +--------------+--------------+
              |                             |
    +---------v---------+       +-----------v-----------+
    |    PostgreSQL      |       |    /app/models/       |
    | signals_extended   |       |  model_v*.joblib      |
    | ml_predictions     |       |  model_v*.json        |
    | ml_model_perf.     |       |  *_regime_*.joblib    |
    +-------------------+       +-----------------------+
```

## Schnellstart

### Voraussetzungen

- Python 3.11+
- PostgreSQL mit `signals_extended` Tabelle
- Mind. 200 gemessene Signale in der DB fuer initiales Training

### Lokal starten

```bash
# Abhaengigkeiten installieren
pip install -r requirements.txt

# Umgebungsvariablen setzen
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=trading_bot
export DB_USER=trading_bot_user
export DB_PASSWORD=your_password

# Service starten
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Mit Docker

```bash
# Image bauen
docker build -t ml-service:latest .

# Container starten
docker run -d \
  --name ml-service \
  -p 8000:8000 \
  -e DB_HOST=postgres \
  -e DB_USER=trading_bot_user \
  -e DB_PASSWORD=your_password \
  -v ml-models:/app/models \
  ml-service:latest
```

### Erstes Training ausloesen

```bash
curl -X POST http://localhost:8000/train
```

## API-Referenz

### `GET /health` - Health Check

Prueft ob der Service und das Modell bereit sind.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "v20260228_120000",
  "timestamp": "2026-02-28T12:00:00.000000"
}
```

---

### `POST /predict` - Einzelne Vorhersage

Bewertet ein einzelnes Trading-Signal.

**Request Body:**
```json
{
  "ticker": "AAPL",
  "signal_type": "BUY",
  "buy_score": 75.5,
  "sell_score": 25.3,
  "rsi": 65,
  "stoch_k": 78,
  "stoch_d": 72,
  "macd": 0.5,
  "bb_percent_b": 0.7,
  "atr_percent": 2.1,
  "momentum": 5.2,
  "ema20": 150.5,
  "ema50": 148.2,
  "price_at_signal": 151.3,
  "risk_reward": 1.5,
  "confidence": 0.75,
  "signal_strength": 3,
  "price_vs_ema20": true,
  "price_vs_ema50": true,
  "rsi_divergence_bull": false,
  "rsi_divergence_bear": false,
  "bb_squeeze": false,
  "stoch_crossover": "bullish",
  "macd_crossover": "bullish",
  "market_regime": "TRENDING_STRONG",
  "trend_direction": "uptrend",
  "hour_of_day": 14,
  "day_of_week": 2,
  "signal_id": "sig_12345"
}
```

**Response:**
```json
{
  "prediction": "correct",
  "probability": 0.847,
  "model_version": "v20260228_120000",
  "regime_model_used": "TRENDING_STRONG",
  "feature_importance_top5": [
    {"feature": "buy_score", "importance": 0.152},
    {"feature": "rsi", "importance": 0.120},
    {"feature": "signal_strength", "importance": 0.099}
  ],
  "error": null
}
```

| Feld | Typ | Beschreibung |
|------|-----|-------------|
| `prediction` | string | `"correct"` oder `"incorrect"` |
| `probability` | float | Wahrscheinlichkeit (0.0 - 1.0), >= 0.5 = correct |
| `regime_model_used` | string | Welches Regime-Modell beigemischt wurde |
| `feature_importance_top5` | array | Top-5 Features fuer diese Vorhersage |
| `signal_id` | string | Optional - wenn gesetzt, wird Vorhersage in DB gespeichert |

---

### `POST /predict/batch` - Batch-Vorhersage

Bewertet mehrere Signale auf einmal (vektorisiert, performanter als Einzelaufrufe).

**Request Body:**
```json
{
  "signals": [
    { "ticker": "AAPL", "signal_type": "BUY", "buy_score": 75.5, "..." : "..." },
    { "ticker": "SAP", "signal_type": "SELL", "buy_score": 20.0, "..." : "..." }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    { "prediction": "correct", "probability": 0.847, "..." : "..." },
    { "prediction": "incorrect", "probability": 0.312, "..." : "..." }
  ],
  "count": 2
}
```

---

### `POST /train` - Modell trainieren

Laedt Trainingsdaten aus der DB, trainiert ein neues Modell und speichert es.

**Request Body:** leer

**Response:**
```json
{
  "success": true,
  "version": "v20260228_120000",
  "accuracy": 0.782,
  "precision": 0.814,
  "recall": 0.745,
  "f1": 0.778,
  "training_samples": 800,
  "test_samples": 200,
  "total_samples": 1000,
  "positive_rate": 0.35,
  "scale_pos_weight": 1.645,
  "optuna_tuned": true,
  "signal_type_metrics": {
    "BUY": { "count": 600, "accuracy": 0.812, "positive_rate": 0.38 },
    "SELL": { "count": 400, "accuracy": 0.723, "positive_rate": 0.31 }
  },
  "feature_importance_top10": [ "..." ],
  "shap_importance_top10": [ "..." ],
  "regime_models": {
    "TRENDING_STRONG": { "train_samples": 120, "accuracy": 0.866, "f1": 0.852 },
    "RANGING": { "train_samples": 95, "accuracy": 0.708, "f1": 0.689 }
  },
  "error": null
}
```

> **Hinweis:** Training benoetigt mindestens 200 Signale mit `outcome_status = 'measured'` und `was_correct IN ('true', 'false')`.

---

### `GET /metrics` - Modell-Metriken

Gibt detaillierte Metriken des aktuell geladenen Modells zurueck.

**Response:**
```json
{
  "model_loaded": true,
  "model_version": "v20260228_120000",
  "accuracy": 0.782,
  "precision": 0.814,
  "recall": 0.745,
  "f1": 0.778,
  "training_samples": 800,
  "test_samples": 200,
  "feature_importance_top10": [ "..." ],
  "feature_importance_all": { "buy_score": 0.152, "rsi": 0.120, "..." : "..." }
}
```

---

### `GET /model/info` - Modell-Info

Kompakte Uebersicht ueber das aktuelle Modell und dessen Hyperparameter.

**Response:**
```json
{
  "model_loaded": true,
  "version": "v20260228_120000",
  "trained_at": "2026-02-28T12:00:00",
  "training_samples": 800,
  "accuracy": 0.782,
  "f1": 0.778,
  "parameters": {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 1.645,
    "optuna_tuned": true
  }
}
```

## ML-Modell

### Algorithmus

**XGBoost Classifier** (Gradient Boosted Decision Trees) fuer binaere Klassifikation.

- **Ziel:** Vorhersage ob ein Trading-Signal korrekt oder inkorrekt ist
- **Output:** Wahrscheinlichkeit (0.0 - 1.0), Schwelle bei 0.5
- **Framework:** xgboost 2.1.3

### Trainingsprozess

```
1. Daten laden
   └── PostgreSQL: signals_extended (nur outcome_status='measured')

2. Feature Engineering
   └── 75+ Features in 5 Phasen (siehe Feature-Engineering)

3. Chronologischer Split (kein Random-Split!)
   ├── Mit Optuna:  60% Train | 20% Validation | 20% Test
   └── Ohne Optuna: 80% Train | 20% Test

4. Class Balancing
   └── scale_pos_weight = sqrt(neg_count / pos_count)

5. Hyperparameter-Tuning (Optuna)
   ├── 15 Trials, Ziel: F1-Score maximieren
   └── Tuned auf Validation-Set, Metriken auf Test-Set

6. Regime-Modelle (optional)
   ├── Separates Modell pro Marktregime
   ├── Min. 50 Samples pro Regime
   └── Blending: 30% Regime + 70% Global

7. Evaluation
   ├── Accuracy, Precision, Recall, F1
   ├── SHAP Feature Importance
   └── Per-Signal-Type Metriken (BUY vs SELL)

8. Speichern
   ├── model_v{timestamp}.joblib
   ├── model_v{timestamp}.json (Metadaten)
   └── Auto-Cleanup (behaelt 5 neueste Versionen)
```

### Hyperparameter

| Parameter | Standard | Optuna-Bereich | Beschreibung |
|-----------|----------|---------------|-------------|
| `n_estimators` | 200 | 100-300 | Anzahl Baeume |
| `max_depth` | 6 | 3-10 | Maximale Baumtiefe |
| `learning_rate` | 0.1 | 0.01-0.3 | Lernrate (log-skaliert) |
| `subsample` | 0.8 | 0.6-1.0 | Anteil Samples pro Baum |
| `colsample_bytree` | 0.8 | 0.6-1.0 | Anteil Features pro Baum |
| `min_child_weight` | - | 1-10 | Min. Gewicht in Blaettern |
| `gamma` | - | 0.0-5.0 | Min. Loss-Reduktion fuer Split |
| `reg_alpha` | - | 1e-8 bis 10 | L1-Regularisierung |
| `reg_lambda` | - | 1e-8 bis 10 | L2-Regularisierung |

### Regime-Modelle

Der Service trainiert optionale Modelle pro Marktregime:

| Regime | Beschreibung |
|--------|-------------|
| `VOLATILE` | Hohe Volatilitaet, starke Schwankungen |
| `TRENDING_STRONG` | Starker Aufwaerts-/Abwaertstrend |
| `TRENDING_WEAK` | Schwacher Trend |
| `RANGING` | Seitwaertsmarkt |
| `SQUEEZE` | Bollinger-Band-Squeeze, Ausbruch erwartet |
| `TRANSITIONAL` | Uebergangsphase zwischen Regimen |

Bei Vorhersagen wird das Regime-Modell mit dem globalen Modell geblendet (Standard: 30/70).

## Feature-Engineering

Die Feature-Pipeline transformiert Rohdaten in 75+ Modell-Features ueber 5 Phasen:

### Phase 1: Roh-Features (19)

**Numerisch (15):** `buy_score`, `sell_score`, `rsi`, `stoch_k`, `stoch_d`, `macd`, `bb_percent_b`, `atr_percent`, `momentum`, `ema20`, `ema50`, `price_at_signal`, `risk_reward`, `confidence`, `signal_strength`

**Boolean (4):** `price_vs_ema20`, `price_vs_ema50`, `rsi_divergence_bull`, `rsi_divergence_bear`

### Phase 2: Abgeleitete Features (14)

| Feature | Berechnung |
|---------|-----------|
| `score_spread` | buy_score - sell_score |
| `stoch_diff` | stoch_k - stoch_d |
| `ema_spread` | (ema20 - ema50) / price * 100 |
| `score_ratio` | buy_score / (buy_score + sell_score) |
| `rsi_extreme` | \|rsi - 50\| |
| `atr_momentum_interaction` | atr_percent * \|momentum\| |
| `rsi_stoch_agreement` | 1 wenn beide ueberkauft/ueberverkauft |
| `trend_strength` | \|ema_spread\| |

### Phase 3: Preis-Trend & Interaktionen (7)

| Feature | Berechnung |
|---------|-----------|
| `ema_price_ratio_20` | price / ema20 |
| `ema_alignment` | 1 wenn EMAs ausgerichtet |
| `score_dominance` | max(buy, sell) / (buy + sell) |
| `signal_confidence_product` | confidence * signal_strength / 100 |
| `bb_atr_interaction` | bb_percent_b * atr_percent |

### Phase 4: Sentiment-Features (3)

| Feature | Berechnung |
|---------|-----------|
| `fear_greed_normalized` | (fear_greed_index - 50) / 50 |
| `sentiment_momentum_agreement` | 1 wenn Sentiment & Momentum uebereinstimmen |
| `sentiment_regime_interaction` | news_sentiment * \|ema_spread\| |

### Phase 5: Temporale Features (19)

- **Lag-Features (4):** Vorheriger Wert von rsi, buy_score, momentum, atr_percent
- **Delta-Features (4):** Aenderung gegenueber vorherigem Signal
- **Rolling-Features (4):** Mittelwert/Standardabweichung ueber letzte 5 Signale
- **Outcome-History (2):** Win-Rate der letzten 10 Signale pro Ticker, letztes Ergebnis
- **Signal-Frequenz (1):** Aufeinanderfolgende Signale am selben Tag

### Kategorische Features (One-Hot Encoded)

| Feature | Kategorien |
|---------|-----------|
| `signal_type` | BUY, SELL, WATCH_BUY, WATCH_SELL, NONE |
| `market_regime` | VOLATILE, TRENDING_STRONG, TRENDING_WEAK, RANGING, SQUEEZE, TRANSITIONAL, UNKNOWN |
| `trend_direction` | uptrend, downtrend, sideways |
| `stoch_crossover` | bullish, bearish, none |
| `macd_crossover` | bullish, bearish, none |
| `exchange` | XETRA, NASDAQ, NYSE |
| `sector` | Technology, Healthcare, Financial, Industrial, Consumer, Energy, Communication, Other |

## Konfiguration

Alle Einstellungen erfolgen ueber Umgebungsvariablen mit sinnvollen Standardwerten.

### Datenbank

| Variable | Standard | Beschreibung |
|----------|----------|-------------|
| `DB_HOST` | `localhost` | PostgreSQL Host |
| `DB_PORT` | `5432` | PostgreSQL Port |
| `DB_NAME` | `trading_bot` | Datenbankname |
| `DB_USER` | `trading_bot_user` | Datenbankbenutzer |
| `DB_PASSWORD` | (leer) | Datenbankpasswort |

### Modell

| Variable | Standard | Beschreibung |
|----------|----------|-------------|
| `MODEL_DIR` | `/app/models` | Verzeichnis fuer Modell-Dateien |
| `MIN_TRAINING_SAMPLES` | `200` | Min. Samples fuer Training |
| `TRAIN_TEST_SPLIT` | `0.8` | Train/Test-Verhaeltnis |

### XGBoost

| Variable | Standard | Beschreibung |
|----------|----------|-------------|
| `XGB_N_ESTIMATORS` | `200` | Anzahl Baeume |
| `XGB_MAX_DEPTH` | `6` | Max. Baumtiefe |
| `XGB_LEARNING_RATE` | `0.1` | Lernrate |
| `XGB_SUBSAMPLE` | `0.8` | Sample-Anteil pro Baum |
| `XGB_COLSAMPLE_BYTREE` | `0.8` | Feature-Anteil pro Baum |
| `XGB_EARLY_STOPPING` | `20` | Early-Stopping-Runden |

### Optuna

| Variable | Standard | Beschreibung |
|----------|----------|-------------|
| `ENABLE_OPTUNA` | `true` | Optuna-Tuning aktivieren |
| `OPTUNA_N_TRIALS` | `15` | Anzahl Tuning-Trials |

### Regime-Modelle

| Variable | Standard | Beschreibung |
|----------|----------|-------------|
| `ENABLE_REGIME_MODELS` | `true` | Regime-Modelle aktivieren |
| `MIN_REGIME_SAMPLES` | `50` | Min. Samples pro Regime |
| `REGIME_WEIGHT` | `0.3` | Gewichtung Regime-Modell (0.0-1.0) |

## Deployment

### Docker (Produktion)

```bash
# Image bauen (Multi-Stage Build, ca. 800MB)
docker build -t ml-service:latest .

# Starten mit Docker Compose
docker compose up -d ml-service
```

### Docker Compose Beispiel

```yaml
services:
  ml-service:
    build: ./ml-service
    ports:
      - "8000:8000"
    environment:
      - DB_HOST=postgres
      - DB_PORT=5432
      - DB_NAME=trading_bot
      - DB_USER=trading_bot_user
      - DB_PASSWORD=${DB_PASSWORD}
      - ENABLE_OPTUNA=true
      - OPTUNA_N_TRIALS=15
    volumes:
      - ml-models:/app/models
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

volumes:
  ml-models:
```

### Health Check

```bash
curl http://localhost:8000/health
```

Erwartete Antwort bei erfolgreichem Start:
```json
{"status": "healthy", "model_loaded": true, "model_version": "v20260228_120000"}
```

Falls `model_loaded: false`: Initiales Training mit `POST /train` ausloesen.

### Modell-Persistenz

Modelle werden im `MODEL_DIR` gespeichert (Standard: `/app/models`):

```
models/
  model_v20260228_120000.joblib       # Haupt-Modell
  model_v20260228_120000.json         # Metadaten & Metriken
  model_v20260228_120000_regime_VOLATILE.joblib
  model_v20260228_120000_regime_TRENDING_STRONG.joblib
  model_v20260228_120000_regime_RANGING.joblib
```

> Der Service behaelt automatisch die 5 neuesten Modell-Versionen und loescht aeltere.

## Datenbank

### Benoetigte Tabellen

**`signals_extended`** - Haupttabelle fuer Trainingsdaten (gelesen):
- Alle Signal-Features (buy_score, rsi, macd, etc.)
- `was_correct`: Ziel-Variable (`'true'` / `'false'`)
- `outcome_status`: Nur `'measured'` wird fuer Training verwendet

**`ml_model_performance`** - Training-Metriken (geschrieben):
- `model_version`, `accuracy`, `precision_score`, `recall_score`, `f1_score`
- `training_samples`, `features` (JSON), `parameters` (JSON)

**`ml_predictions`** - Vorhersage-Log (geschrieben):
- `signal_id`, `model_version`, `prediction`, `probability`
- `features_used` (JSON), `feature_importance_top5` (JSON)

## Monitoring & Logs

### Log-Format

```
2026-02-28 12:00:00 [INFO] app.model: Modell geladen: v20260228_120000 (Accuracy: 0.782)
2026-02-28 12:01:00 [INFO] app.model: Training gestartet mit 1000 Samples
2026-02-28 12:03:00 [INFO] app.model: Optuna Best Trial: F1=0.778, Params={...}
2026-02-28 12:04:00 [INFO] app.model: Training abgeschlossen: v20260228_120400
```

### Wichtige Metriken

| Metrik | Endpoint | Beschreibung |
|--------|----------|-------------|
| Modell geladen? | `GET /health` | `model_loaded: true/false` |
| Aktuelle Accuracy | `GET /metrics` | Accuracy auf Test-Set |
| Feature Importances | `GET /metrics` | Top-10 + alle Features |
| Modell-Version | `GET /model/info` | Aktuelle Version + Parameter |

### Fehlerbehandlung

| Situation | Verhalten |
|-----------|----------|
| Kein Modell geladen | Gibt `prediction: "unknown"`, `probability: 0.5` zurueck |
| Fehlende Features | Werden mit Standardwerten gefuellt (0.0 / 0.5) |
| DB nicht erreichbar | Vorhersage laeuft mit gecachten Daten weiter |
| < 200 Training-Samples | Training schlaegt fehl mit Fehlermeldung |
| Regime-Modell fehlt | Faellt automatisch auf globales Modell zurueck |

## Tech-Stack

| Komponente | Version | Zweck |
|-----------|---------|-------|
| FastAPI | 0.115.6 | Web-Framework |
| Uvicorn | 0.34.0 | ASGI-Server |
| XGBoost | 2.1.3 | ML-Modell |
| scikit-learn | 1.6.1 | Metriken & Utilities |
| Pandas | 2.2.3 | Datenverarbeitung |
| NumPy | 2.2.1 | Numerische Berechnungen |
| psycopg2 | 2.9.10 | PostgreSQL-Treiber |
| SHAP | 0.46.0 | Modell-Erklaerbarkeit |
| Optuna | 4.1.0 | Hyperparameter-Tuning |
| Pydantic | 2.10.4 | Request-Validierung |
| Python | 3.11 | Runtime |

## Lizenz

Privates Repository - Alle Rechte vorbehalten.
