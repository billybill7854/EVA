#!/bin/bash

# Startup script for HuggingFace Spaces
# Initializes database and starts the application

echo "Starting EVA Telegram Bot..."

# Initialize database if it doesn't exist
if [ ! -f "/app/data/eva.db" ]; then
    echo "Initializing database..."
    python init_db.py
fi

# Run any migrations
echo "Running database migrations..."
python migrate_db.py || echo "Migration completed or not needed"

# Start the application
echo "Starting FastAPI application..."
python -m uvicorn main:app --host 0.0.0.0 --port 7860
