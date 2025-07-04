#!/bin/bash

# Получаем порт из переменной окружения (Render сам его задаст)
PORT=${PORT:-10000}

# Запускаем Gunicorn
gunicorn app:app \
  --workers 2 \
  --worker-class sync \
  --timeout 400 \
  --bind 0.0.0.0:$PORT \
  --access-logfile - \
  --error-logfile -