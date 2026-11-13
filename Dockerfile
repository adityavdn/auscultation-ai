FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ENV MODELS_DIR=/app/models SAMPLES_DIR=/app/samples DB_PATH=/tmp/cases.sqlite NUMBA_CACHE_DIR=/tmp MPLCONFIGDIR=/tmp
COPY ml ml
COPY app app
# Trained models and test-split samples are added to the deploy bundle (not tracked in git).
COPY models models
COPY samples samples
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app_from_env --factory --host 0.0.0.0 --port ${PORT:-7860}"]
