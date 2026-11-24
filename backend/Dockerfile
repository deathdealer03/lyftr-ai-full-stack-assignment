# ==========================================
# Stage 1: Build Dependencies
# ==========================================
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ==========================================
# Stage 2: Minimal Runtime
# ==========================================
FROM python:3.12-slim AS runtime

WORKDIR /app

# Ensure LiteLLM uses offline model cost map and suppresses logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/root/.local/bin:$PATH \
    LITELLM_LOCAL_MODEL_COST_MAP=True

# Copy installed python dependencies from builder stage
COPY --from=builder /root/.local /root/.local

# Create persistent storage volume directory
RUN mkdir -p /data

# Copy application source
COPY app/ /app/app/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
