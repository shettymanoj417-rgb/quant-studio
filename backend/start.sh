#!/bin/bash
set -e

cd /app/backend

exec python -m gunicorn app:app --bind 0.0.0.0:${PORT:-10000}