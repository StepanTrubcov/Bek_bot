#!/bin/bash

# Получаем порт из переменной окружения
PORT=${PORT:-10000}

# Запускаем Gunicorn с базовыми параметрами
exec gunicorn app:app \
  --workers 1 \
  --worker-class gevent \
  --timeout 300 \
  --bind 0.0.0.0:$PORT \
  --access-logfile - \
  --error-logfile -