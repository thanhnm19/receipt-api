FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY main.py ./
COPY extractor.py ./

# Bake model files into the image so runtime does not need to download.
RUN mkdir -p models \
    && MODEL_AUTO_DOWNLOAD=1 MODEL_DOWNLOAD_QUIET=0 python -c "import main; main.download_models(); print('Models baked into image')"

ENV MODEL_AUTO_DOWNLOAD=0 \
    MODEL_DOWNLOAD_QUIET=1 \
    MODELS_DIR=models \
    PORT=8080

EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
