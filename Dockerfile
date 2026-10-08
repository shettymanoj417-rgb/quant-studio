FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install backend dependencies
COPY backend/requirements.txt /app/backend/requirements.txt

RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy the entire project
COPY . /app

# Runtime directories
RUN mkdir -p \
    /app/backend/uploads \
    /app/backend/quantized_models \
    /app/backend/logs

# Make backend imports work
ENV PYTHONPATH=/app/backend

# Render provides PORT at runtime
EXPOSE 10000

# Start Flask application
CMD ["sh", "-c", "python -m gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --timeout 120 app:app"]