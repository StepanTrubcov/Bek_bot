# Получаем порт из переменной окружения (Render сам его задаст)
PORT=${PORT:-10000}

# Рассчитываем количество воркеров на основе доступной памяти
# На Render.com бесплатный инстанс имеет 0.5GB RAM
if [ -f /.dockerenv ] || [ "$RENDER" ]; then
  WORKERS=1  # Для Render лучше использовать 1 воркер из-за ограничений памяти
else
  WORKERS=2
fi

# Запускаем Gunicorn с оптимизированными параметрами
exec gunicorn app:app \
  --workers $WORKERS \
  --worker-class gevent \
  --timeout 300 \
  --bind 0.0.0.0:$PORT \
  --max-requests $MAX_REQUESTS \
  --max-requests-jitter $MAX_REQUESTS_JITTER \
  --access-logfile - \
  --error-logfile - \
  --preload