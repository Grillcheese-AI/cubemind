FROM python:3.12-slim AS base

# System deps for Vulkan + Redis
RUN apt-get update && apt-get install -y --no-install-recommends \
    libvulkan1 vulkan-tools mesa-vulkan-drivers \
    redis-tools git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install grilly first (cached layer)
COPY pyproject.toml .
RUN pip install --no-cache-dir grilly numpy

# Install cubemind
COPY . .
RUN pip install --no-cache-dir -e ".[dev]"

# Default: run tests
CMD ["python", "-m", "pytest", "tests/", "-v"]

# ── Development stage ────────────────────────────────────────────────
FROM base AS dev

RUN pip install --no-cache-dir ruff matplotlib ipython

CMD ["ipython"]

# ── Benchmark stage ──────────────────────────────────────────────────
FROM base AS bench

COPY benchmarks/ benchmarks/
CMD ["python", "-m", "pytest", "benchmarks/", "-v", "--benchmark-only"]

# ── API stage (future) ──────────────────────────────────────────────
FROM base AS api

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "cubemind.cloud.api:app", "--host", "0.0.0.0", "--port", "8000"]
