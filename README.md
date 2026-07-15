![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)
![CatBoost](https://img.shields.io/badge/ML-CatBoost-yellow.svg)
![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)

# Look-a-Like Service: Banking Offer Targeting System
## A production-ready service designed to identify high-potential audiences for bank marketing offers. The system manages the full MLOps lifecycle, including real-time batch data ingestion, quality validation, data drift detection, and automated model retraining.

## Project Overview

The service facilitates targeted marketing by matching offers to potential customers while filtering out existing brand clients.

## Project Structure and Module Descriptions

```text
├── data/
│   ├── v1-2/        # Local testing data (version 1)
│   └── v2-2/        # Local testing data (version 2)
├── ml/
│   └── models/      # Pre-trained models and training artifacts
└── service/         # Source code and application logic
    ├── api.py
    ├── drift.py
    ├── features.py
    ├── metrics.py
    ├── pipeline.py
    ├── s3_utils.py
    ├── schemas.py
    ├── state.py
    ├── storage.py
    ├── train_nlp.py
    ├── train.py
    ├── upload_data.py
    └── validation.py
```
### Machine Learning (ml/)
* `models/offer_classifier`: Pre-trained model for offer description classification (trained via `train_nlp.py`). This directory stores the latest model version.
### Service Layer (service/)
1. `api.py`: FastAPI interface implementing contract endpoints (`/lookalike`, `/data/batch`, `/monitoring/drift`, etc.).
2. `drift.py`: Statistical drift detection module using Evidently AI.
3. `features.py`: Feature engineering and data preprocessing logic.
4. `metrics.py`: Implementation of the MAP@100 metric for ranking quality assessment.
5. `pipeline.py`: Orchestration of the training and validation processes.
6. `s3_utils.py`: Boto3 wrappers for MinIO (S3) interaction.
7. `schemas.py`: Pydantic schemas for strict JSON request validation.
8. `state.py`: Application state management (model versions, statuses, etc.).
9. `storage.py`: Data storage logic.
10. `train_nlp.py`: Training script for the TF-IDF offer category classifier.
11. `train.py`: Model training script and SHAP value calculation for interpretability.
12. `upload_data.py`: Utility script for local testing and data injection simulation.
13. `validation.py`: Data quality control using Great Expectations.

## Technical Stack

* **Language:** Python 3.10.
* **API Layer:** FastAPI.
* **ML Core:** CatBoost, TF-IDF.
* **Infrastructure:** Docker, Docker Compose, MinIO.
* **Tooling:** DVC, MLflow, Evidently, Great Expectations.

## EDA

The `main.ipynb` notebook documents extensive research and hypothesis testing. Key findings:

* **Balanced Dataset:** Dataset v1 is well-balanced with a baseline conversion rate of approximately 33%.
* **Correlations:** Features `has_past_rewards` and `rewards_count` show the strongest correlation with the target. Customers with previous cashback experience convert nearly twice as often (45% vs 28%).
* **Age Factor:** A direct correlation exists: older clients show a higher response rate to bank offers.
* **The "Active Client" Paradox:** Less active clients tend to respond to offers more readily than highly active banking users.

### Feature Engineering
To enhance model performance, the following features were engineered:
* `hunter_index`: `rewards_count / (tx_total_count + 1)`. Identifies deal-seeking behavior.
* `online_share`: `tx_online_count / tx_total_count`. Measures digital transaction propensity.
* `tx_sum_mean` & `tx_sum_total`: Numerical mapping of categorical transaction buckets (e.g., "10k+", "1k-5k") to mean values.
* `rec_unique_categories`: Unique category count to distinguish between specialized vs. general shoppers.

### Preprocessing & Model Selection
* **Outlier Handling:** Outliers in `tx_total_count`, `tx_sum_total`, `tx_sum_mean`, `items_count_sum`, and `items_cost_sum` were handled using the 3 IQR rule.
* **Model Selection:** CatBoost was trained to evaluate feature importance via SHAP values, guiding the final feature selection.






# Look-a-Like Service: Система подбора аудитории для банковских офферов
## Сервис предназначен для автоматизированного подбора целевой аудитории для маркетинговых офферов банка. Система реализует полный цикл MLOps: от обработки потоковых данных и валидации качества до детектирования дрейфа и автоматического переобучения моделей.

### Структура решения

```text
├── data/
│   ├── v1-2/        # Данные для локального тестирования (версия 1)
│   └── v2-2/        # Данные для локального тестирования (версия 2)
├── ml/
│   └── models/      # Папка с весами и артефактами моделей
└── service/         # Исходный код и логика сервиса
    ├── api.py
    ├── drift.py
    ├── features.py
    ├── metrics.py
    ├── pipeline.py
    ├── s3_utils.py
    ├── schemas.py
    ├── state.py
    ├── storage.py
    ├── train_nlp.py
    ├── train.py
    ├── upload_data.py
    └── validation.py
```
***

### Описание файлов:
Машинное обучение (ml/)

1. models/offer_classifier — преобученная модель для классификации описаний офферов (файл обучения train_nlp.py).
2. В эту же папку сохраняется самая новая модель.

Сервисный слой (service/)

1. `api.py` — FastApi интерфейс, реализует эндпоинты по контракту: `/lookalike`, `/data/batch`, `/monitoring/drift` и др.
2. `drift.py` - Детекция статистического дрейфа с помощью **Evidently**. 
3. `features.py` - Обработка данных и создание новых фич.
4. `metrics.py` - Реализация целевой метрики **MAP@100** ддля оценки качества ранжирования.
5. `pipeline.py` - Соединение всех процессов в единый цикл.
6. `s3_utils.py` - Обертки над `boto3` для работы с **MinlO**.
7. `schemas.py` - Pyndatic-схемы длястрогой валидации входящих JSON-запросов.
8. `state.py` - Хранилище глобального состояния сервиса (версии моделей, статусы и тд.)
9. `storage.py` - Логика хранения.
10. `train_nlp.py` - Скрипт для обучения TF-IDF классификатора категорий офферов.GR
11. `train.py` - Отвечает за обучение и расчет значений **SHAP** для интерпритации ответов.
12. `upload_data.py` - Файл для имитации работы чекера и локального тестирования.
13. `validation.py` - Контроль качества данных с помощью **great-expectations**.
***

## Cтек 

* **Язык:** Python 3.10.
* **Интерфейс API:** FastAPI.
* **ML:** CatBoost, TF-IDF.
* **Инфраструктура:** Docker, Docker Compose, MinIO.
* **MLOps:** DVC, MLflow, Evidently, Great Expectations.

## EDA

В блокноте `main.ipynb` зафиксированы результаты подробного исследования данных и проверки гипотез. Ключевые выводы:

* **Сбалансированность данных:** Датасет версии v1 хорошо сбалансирован, базовая конверсия составляет около 33%.
* **Корреляция:** Признаки `has_past_rewards` и `rewards_count` демонстрируют наиболее сильную связь с целевой переменной. Клиенты, ранее получавшие кэшбэк, совершают целевое действие почти в два раза чаще (45% против 28%).
* **Возрастной фактор:** Существует прямая зависимость: чем старше клиент, тем выше вероятность его отклика на предложения банка.
* **Парадокс активности:** Менее активные в повседневных транзакциях клиенты охотнее реагируют на маркетинговые предложения, чем гиперактивные пользователи.

### Feature Engineering
Для улучшения предсказательной способности модели были созданы следующие признаки:
* `hunter_index`: `rewards_count / (tx_total_count + 1)` — доля транзакций с кэшбэком. Позволяет выделить категорию клиентов, целенаправленно ищущих скидки («охотников за скидками»).
* `online_share`: `tx_online_count / tx_total_count` — доля покупок в интернете. Измеряет склонность пользователя к онлайн-шопингу.
* `tx_sum_mean` & `tx_sum_total`: восстановленные непрерывные значения сумм транзакций из категориальных бакетов (например, интервалы вида «10k+» или «1k-5k» были переведены в средние числовые значения).
* `rec_unique_categories`: количество уникальных категорий покупок, помогающее разделить специализированных покупателей от тех, кто совершает покупки во всех категориях подряд.

### Предобработка данных и выбор модели
* **Обработка выбросов:** Выбросы в числовых признаках (таких как `tx_total_count`, `tx_sum_total`, `tx_sum_mean`, `items_count_sum` и `items_cost_sum`) были обработаны по строгому правилу 3 IQR.
* **Выбор модели:** Модель CatBoost была обучена для оценки важности признаков с использованием значений SHAP (SHAP values), на основе которых был сформирован финальный набор переменных для обучения.

