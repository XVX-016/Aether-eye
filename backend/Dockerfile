FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

# System deps (add GDAL/PROJ/etc as needed for geospatial workloads)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend and ml_core so that ML core is fully isolated but available to the API.
COPY ./backend /app/backend
COPY ./ml_core /app/ml_core

WORKDIR /app

# Install ML core and backend dependencies.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir ./ml_core && \
    pip install --no-cache-dir -r backend/requirements.txt

WORKDIR /app/backend

EXPOSE 8000

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:8000", "--workers", "2"]

