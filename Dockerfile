# CPU-only image: faster-whisper (CTranslate2) + FastAPI. No torch, no GPU, no network at runtime.
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MODEL_DIR=/models/int8
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[deploy]" && useradd -m app
USER app
EXPOSE 8000
# mount the exported model:  -v $PWD/results/deploy/ct2/int8:/models/int8:ro
CMD ["uvicorn", "fieldasr.deploy.api:app", "--host", "0.0.0.0", "--port", "8000"]
