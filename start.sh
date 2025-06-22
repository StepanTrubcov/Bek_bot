#!/bin/bash

# Получаем порт из переменной окружения (Render сам его задаст)
PORT=${PORT:-10000}

# Запускаем Gunicorn
gunicorn app:app \
  --workers 2 \
  --worker-class gevent \
  --timeout 300 \
  --bind 0.0.0.0:$PORT \
  --preload \
  --access-logfile - \
  --error-logfile -