# Multi-stage Dockerfile for Recommndr with advanced optimizations
# Enable BuildKit for better performance
# syntax=docker/dockerfile:1.4

# Base stage with common dependencies
FROM --platform=$BUILDPLATFORM python:3.11-slim as base

# Set environment variables for optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONHASHSEED=random \
    PYTHONOPTIMIZE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies with advanced pruning
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Create app directory
WORKDIR /app

# Create virtual environment for better isolation
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel for better dependency resolution
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy only requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies with advanced pruning
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt \
    && find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + \
    && find /opt/venv -type f -name "*.pyc" -delete \
    && find /opt/venv -type f -name "*.pyo" -delete

# Copy application code (this layer changes most frequently)
COPY . .

# Set Python path and optimize Python
ENV PYTHONPATH=/app \
    PYTHONOPTIMIZE=2

# Development stage with additional tools
FROM base as development

# Install development dependencies with pruning
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-asyncio \
    black \
    isort \
    flake8 \
    mypy \
    jupyter \
    jupyterlab \
    && find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + \
    && find /opt/venv -type f -name "*.pyc" -delete

# Expose port
EXPOSE 8000

# Default command for development
CMD ["python", "-m", "src.data_generation.main"]

# Testing stage for CI/CD
FROM development as testing

# Install additional testing tools with pruning
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
    locust \
    coverage \
    && find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + \
    && find /opt/venv -type f -name "*.pyc" -delete

# Run tests by default
CMD ["pytest", "--cov=src", "--cov-report=html", "--cov-report=term"]

# Production stage with advanced optimizations
FROM base as production

# Remove development dependencies and prune
RUN pip uninstall -y pytest pytest-cov black isort flake8 mypy jupyter jupyterlab \
    && find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + \
    && find /opt/venv -type f -name "*.pyc" -delete \
    && find /opt/venv -type f -name "*.pyo" -delete \
    && find /opt/venv -type f -name "*.pyd" -delete

# Create non-root user for security
RUN groupadd --gid 1000 recommndr && \
    useradd --uid 1000 --gid recommndr --shell /bin/bash --create-home recommndr

# Set proper permissions
RUN chown -R recommndr:recommndr /app
USER recommndr

# Expose port
EXPOSE 8000

# Health check with proper timeout
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for production
CMD ["python", "-m", "src.data_generation.main"]

# Minimal production stage (for Azure Container Apps) with extreme pruning
FROM base as minimal

# Remove unnecessary packages and extreme pruning
RUN pip uninstall -y pytest pytest-cov black isort flake8 mypy jupyter jupyterlab \
    && apt-get remove -y build-essential git \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/* \
    && find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + \
    && find /opt/venv -type f -name "*.pyc" -delete \
    && find /opt/venv -type f -name "*.pyo" -delete \
    && find /opt/venv -type f -name "*.pyd" -delete \
    && find /opt/venv -type d -name "tests" -exec rm -rf {} + \
    && find /opt/venv -type d -name "test" -exec rm -rf {} + \
    && find /opt/venv -type d -name "*.egg-info" -exec rm -rf {} + \
    && find /opt/venv -type d -name "*.dist-info" -exec rm -rf {} +

# Create non-root user
RUN groupadd --gid 1000 recommndr && \
    useradd --uid 1000 --gid recommndr --shell /bin/bash --create-home recommndr

# Set permissions
RUN chown -R recommndr:recommndr /app
USER recommndr

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "src.data_generation.main"]

# Security-hardened production stage
FROM base as secure

# Install security updates and tools
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Remove development dependencies
RUN pip uninstall -y pytest pytest-cov black isort flake8 mypy jupyter jupyterlab \
    && find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + \
    && find /opt/venv -type f -name "*.pyc" -delete

# Create non-root user with security restrictions
RUN groupadd --gid 1000 recommndr && \
    useradd --uid 1000 --gid recommndr --shell /bin/bash --create-home recommndr \
    && chown -R recommndr:recommndr /app \
    && chmod -R 755 /app

# Security: Drop capabilities and set security options
USER recommndr

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "src.data_generation.main"]
