# FinTwin-X public demo image: Next.js static export + FastAPI + generated synthetic demo data
FROM node:22-alpine AS web-builder
WORKDIR /web
COPY apps/web/package.json ./
RUN npm install --no-audit --no-fund
COPY apps/web/ ./
ENV NEXT_TELEMETRY_DISABLED=1
ENV NEXT_PUBLIC_API_URL=
RUN npm run typecheck && npm run build

FROM python:3.12-slim AS python-builder
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONPATH=/app
COPY requirements-demo.txt .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install -r requirements-demo.txt
COPY . .
# Build a compact, deterministic synthetic portfolio into the image.
RUN /opt/venv/bin/python -m data_generator.generate --users 500 --months 18 --seed 1 \
 && /opt/venv/bin/python -m pipelines.run_pipeline \
 && /opt/venv/bin/python -c "from ml.train_all import train_all; train_all(risk_algorithms=('xgboost',))" \
 && /opt/venv/bin/python -m rag.ingestion.ingest_docs
COPY --from=web-builder /web/out /app/apps/web/out

FROM python:3.12-slim AS runtime
WORKDIR /app
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    FINTWIN_ENV=production \
    FINTWIN_ENABLE_DEMO_USER=false \
    FINTWIN_EMBEDDING_BACKEND=tfidf \
    FINTWIN_ENABLE_SHAP=false
RUN groupadd -r fintwin && useradd -r -g fintwin -m fintwin
COPY --from=python-builder /opt/venv /opt/venv
COPY --from=python-builder --chown=fintwin:fintwin /app /app
RUN mkdir -p /app/logs && chown -R fintwin:fintwin /app/logs
USER fintwin
EXPOSE 10000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import os,urllib.request; p=os.getenv('PORT','10000'); urllib.request.urlopen('http://127.0.0.1:'+p+'/health', timeout=4)"
CMD ["sh", "-c", "python -m uvicorn apps.api.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
