# ── Stage 1: Builder ─────────────────────────────────────────────────────────
# Install all Python dependencies into /install prefix.
# This stage is discarded — only the installed packages are copied to final.
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system build tools needed to compile native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# --prefix=/install puts packages outside the system Python,
# making it easy to COPY only the packages to the final stage.
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Final Runtime Image ─────────────────────────────────────────────
FROM python:3.11-slim AS final

# --- Security: run as non-root user -----------------------------------------
RUN useradd --create-home --shell /bin/bash --uid 1001 appuser

# --- Java for PySpark (ETL jobs only — exclude from API-only builds) ---------
# Java is required by PySpark. If you're running the API only (not ETL),
# you can remove these lines and set PYSPARK_PYTHON in your ETL container.
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# --- Environment ---------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/install/bin:$PATH"

# --- Copy Python packages from builder stage ----------------------------------
COPY --from=builder /install /install

# --- Copy application source --------------------------------------------------
WORKDIR /app
COPY --chown=appuser:appuser . .

# --- Switch to non-root user --------------------------------------------------
USER appuser

# --- Healthcheck --------------------------------------------------------------
# Docker checks liveness every 30s; 3 consecutive failures → container restart.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" \
    || exit 1

EXPOSE 8000

# --- Default command ----------------------------------------------------------
# Use 2 uvicorn workers for the container.
# In k8s: scale via HPA + replicas, not workers (one worker per pod is cleaner).
CMD ["uvicorn", "deployment.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info", \
     "--access-log"]
