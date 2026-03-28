FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# curl is not in slim images; required for HEALTHCHECK only.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home --no-log-init --uid 10001 app \
    && chown -R app:app /app
USER app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fSs http://localhost:7860/health >/dev/null || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
