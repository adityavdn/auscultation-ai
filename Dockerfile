FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgomp1 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH MODELS_DIR=/home/user/app/models SAMPLES_DIR=/home/user/app/samples
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user ml ml
COPY --chown=user app app
# Trained models + test-split samples are pulled from the HF model repo at build time.
ARG MODEL_REPO=adityavdn/auscultation-models
RUN --mount=type=secret,id=HF_TOKEN,mode=0444,required=false \
    python -c "import os; from huggingface_hub import snapshot_download as s; t=open('/run/secrets/HF_TOKEN').read().strip() if os.path.exists('/run/secrets/HF_TOKEN') else None; s('$MODEL_REPO', local_dir='.', allow_patterns=['models/*','samples/*'], token=t)"
EXPOSE 7860
CMD ["uvicorn", "app.main:app_from_env", "--factory", "--host", "0.0.0.0", "--port", "7860"]
