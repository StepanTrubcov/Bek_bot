#!/bin/bash

# Установка переменных окружения
export PYTHONUNBUFFERED=TRUE
export PYTHONPATH="${PYTHONPATH}:/app"

# Запуск Gunicorn с оптимизированными параметрами
gunicorn app:app \
  --workers 4 \                     # Увеличено количество воркеров
  --worker-class gevent \           # Используем gevent для асинхронности
  --timeout 300 \                   # Увеличен таймаут для обработки аудио
  --bind 0.0.0.0:10000 \
  --preload \
  --access-logfile - \              # Логирование запросов
  --error-logfile - \               # Логирование ошибок
  --capture-output \                # Перехват stdout/stderr
  --log-level info \                # Уровень логирования
  --max-requests 1000 \             # Автоматическая перезагрузка воркеров
  --max-requests-jitter 50