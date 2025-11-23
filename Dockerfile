# Dockerfile for Federated Learning Server/Client Services
FROM python:3.10-slim

# Allow CI/CD to choose caching behavior
ARG PIP_NO_CACHE=0

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# - Local: pip install -r requirements.txt
# - CI/CD: pip install --no-cache-dir -r requirements.txt
RUN if [ "$PIP_NO_CACHE" = "1" ]; then \
        pip install --no-cache-dir -r requirements.txt ; \
    else \
        pip install -r requirements.txt ; \
    fi

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models drift_reports

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose ports for FL server
EXPOSE 8080

# Default command (can be overridden)
CMD ["python", "fl_server.py"]
