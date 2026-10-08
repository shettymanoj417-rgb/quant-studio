#!/bin/bash

# Activate virtual environment if exists
if [ -d "venv" ]; then
	source venv/bin/activate
fi

# Set environment variables
export FLASK_APP=backend/app.py
export FLASK_ENV=development
export FLASK_DEBUG=1

# Create necessary directories
mkdir -p uploads quantized_models logs

# Run Flask development server
flask run --host=0.0.0.0 --port=5000 --reload
