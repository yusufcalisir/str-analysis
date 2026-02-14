# ──────────────────────────────────────────────────────────────────────
# VANTAGE-Node — Sovereign Forensic Database Container
# ──────────────────────────────────────────────────────────────────────
# Alpine-based Python image optimized for gRPC, Milvus SDK, and DSPy.
# Health check validates both local Milvus and global Orchestrator.
# ──────────────────────────────────────────────────────────────────────

FROM python:3.11-alpine AS builder

# Build dependencies for gRPC native extensions
RUN apk add --no-cache \
    gcc \
    g++ \
    musl-dev \
    linux-headers \
    libffi-dev \
    openssl-dev \
    make \
    cmake \
    git

WORKDIR /build

# Install Python dependencies in a virtual environment
COPY requirements-node.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements-node.txt

# ──────────────────────────────────────────────────────────────────────
# Runtime stage (minimal)
# ──────────────────────────────────────────────────────────────────────
FROM python:3.11-alpine

LABEL maintainer="VANTAGE-STR Engineering <ops@vantage-str.io>"
LABEL description="VANTAGE Sovereign Node — privacy-preserving forensic database"
LABEL version="2.2.0"

# Runtime-only system dependencies
RUN apk add --no-cache \
    libstdc++ \
    libffi \
    openssl \
    ca-certificates \
    curl \
    tini && \
    addgroup -S vantage && \
    adduser -S -G vantage -h /app vantage

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VANTAGE_NODE_ENV=production

# Application code
WORKDIR /app
COPY --chown=vantage:vantage . .

# Certificate mount point
RUN mkdir -p /etc/vantage/certs && \
    chown -R vantage:vantage /etc/vantage
VOLUME ["/etc/vantage/certs"]

# gRPC port
EXPOSE 50051

# Health check: verify Milvus + Orchestrator connectivity
HEALTHCHECK --interval=15s --timeout=5s --retries=3 --start-period=30s \
    CMD python -c "\
import sys; \
from node.infrastructure.security_provider import SecurityProvider; \
sp = SecurityProvider('health-check', '/etc/vantage/certs'); \
result = sp.boot_check_lenient(); \
sys.exit(0 if result.success else 1)" || exit 1

USER vantage

ENTRYPOINT ["tini", "--"]
CMD ["python", "-m", "node.main", "--host", "0.0.0.0", "--port", "50051"]
