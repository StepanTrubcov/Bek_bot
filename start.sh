#!/bin/bash
gunicorn app:app \
  --workers 2 \
  --worker-class sync \
  --timeout 120 \
  --bind 0.0.0.0:10000 \
  --preload  