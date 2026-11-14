FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgomp1 curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Trained models + test samples: public release asset, pinned by SHA-256 (joblib files are pickles).
ARG ASSETS_URL=https://api.github.com/repos/adityavdn/auscultation-models/releases/assets/747508727
ARG ASSETS_SHA256=7fa7e299cb9d236dc507484dd85b81b6088cb03042f0e4e2ed7ce094bfa622f0
RUN curl -fsSL -H "Accept: application/octet-stream" -o /tmp/assets.tgz "$ASSETS_URL" \
 && echo "$ASSETS_SHA256  /tmp/assets.tgz" | sha256sum -c - \
 && tar xzf /tmp/assets.tgz -C /app && rm /tmp/assets.tgz
ENV MODELS_DIR=/app/models SAMPLES_DIR=/app/samples DB_PATH=/tmp/cases.sqlite NUMBA_CACHE_DIR=/tmp MPLCONFIGDIR=/tmp
COPY ml ml
COPY app app
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app_from_env --factory --host 0.0.0.0 --port ${PORT:-7860}"]
