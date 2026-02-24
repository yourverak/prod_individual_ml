FROM python:3.10-slim

WORKDIR /app

# 1. Системные зависимости (меняются реже всего)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 2. Копируем requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip

# 3. Ставим самое тяжелое (CatBoost, Pandas, Numpy) 
# Это закешируется и не будет перекачиваться при ошибках в других библиотеках
RUN pip install --no-cache-dir catboost pandas numpy

# 4. Ставим остальное из списка (FastAPI, Evidently, и т.д.)
# Если тут будет ошибка, шаги 1-3 НЕ будут перевыполняться!
RUN pip install --no-cache-dir -r requirements.txt

# 5. Копируем код (меняется чаще всего)
COPY . .

CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "80"]