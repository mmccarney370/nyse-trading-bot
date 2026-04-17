# =============================================
# NYSE Trading Bot — Docker Image
# Supports both GPU (NVIDIA CUDA) and CPU-only modes.
#
# GPU:  docker compose up -d
# CPU:  docker compose --profile cpu up -d
# =============================================

# --- Stage 1: Dependencies ---
FROM python:3.11-slim AS deps

WORKDIR /app

# System deps for building native extensions (arcticdb, hmmlearn, lightgbm, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Stage 2: Runtime ---
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy bot source code
COPY . .

# Create directories for persistent state (mounted as volumes)
RUN mkdir -p logs ppo_checkpoints/portfolio ppo_checkpoints/tft_cache \
    data_cache arcticdb_tickdb checkpoints

# Default environment variables (override via .env or docker-compose)
ENV PYTHONUNBUFFERED=1
ENV TQDM_DISABLE=1
ENV REDIS_HOST=redis
ENV OLLAMA_HOST=http://ollama:11434

# Health check — verify Python can import the bot
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "import bot; print('ok')" || exit 1

ENTRYPOINT ["python", "__main__.py"]
