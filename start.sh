#!/bin/bash

# Установка переменных окружения
export PYTHONUNBUFFERED=TRUE
export PYTHONPATH="${PYTHONPATH}:/app"

# Запуск Gunicorn с оптимизированными параметрами (все в одной строке)
gunicorn app:app --workers 2 --worker-class gevent --timeout 300 --bind 0.0.0.0:10000 --preload --access-logfile - --error-logfile - --capture-output --log-level info