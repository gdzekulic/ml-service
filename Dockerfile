FROM python:3.11-slim AS builder

WORKDIR /build

# Pip-Packages in separatem Stage bauen (begrenzt RAM-Spitze)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.11-slim

WORKDIR /app

# System-Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Vorgebaute Python-Packages kopieren (kein pip install zur Laufzeit)
COPY --from=builder /install /usr/local

# App Code
COPY app/ ./app/

# Models-Verzeichnis (wird als Volume gemounted)
RUN mkdir -p /app/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
