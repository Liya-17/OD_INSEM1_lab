#!/bin/bash
# Auto-generate demo model if not present
if [ ! -f "model/garbage_model.h5" ]; then
    echo "Model not found. Generating demo model..."
    python model/create_demo_model.py
fi
exec gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
