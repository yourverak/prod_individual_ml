FROM python:3.10-slim

WORKDIR /app

# Сначала копируем только requirements, чтобы кэшировать установку библиотек
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Теперь копируем весь остальной код
COPY . .

# Запускаем сервер на 80 порту
CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "80"]