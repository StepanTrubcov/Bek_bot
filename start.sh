#!/bin/bash

PORT=${PORT:-10000}

# Запускаем Gunicorn, указывая правильное имя приложения
gunicorn app:app \
  --workers 2 \
  --worker-class sync \
  --timeout 400 \
  --bind 0.0.0.0:$PORT \
  --access-logfile - \
  --error-logfile -