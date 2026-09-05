FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY services ./services
COPY training ./training
COPY config ./config
RUN pip install -U pip && pip install -e '.[asr]'
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--app-dir", "services/api", "--host", "0.0.0.0", "--port", "8080"]
