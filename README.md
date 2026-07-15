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

Обьяснение проведенного преданализа в файле main.ipynb:

Перед началом обучения необходимо провести хорошее пользовательское исследование. В блокноте были проверено много гипотез, из них получилось вытянуть следущие выводы:

1. Датасет v1 хорошо сбалансирован, конверсия откликов на отклики составляет примерно 33%.
2. Такие фичи как has_past_rewards и rewards_count имеют самую высокую корреляцию с таргетом. Те, кто ранее уже получал кэшбек, конвертируют почти в 2 раза чаще (45% против 28%).
3. Чем старше клиент, тем чаще он откликается на офферы.
4. Также есть небольшой парадокс активных клиентов. Офферы банка лучше заходят менее активным клиентам чем более активным.

Новые добавленные фичи:

1. `hunter_index: rewards_count / (tx_total_count + 1)`. Какая доля покупок приносит кэшбэк? Если она высокая, человек целенаправленно ищет скидки.
2. `online_share: tx_online_count / tx_total_count`. Доля онлайна.
3. Фичи `tx_sum_mean` и `tx_sum_total` для каждого юзера, для этого суммы из трансакции были переведены из вида "10k+" или "1k-5k" в средние значения.
4. rec_unique_categories: смотрим количество уникальных категорий, чтобы разделять тех, кто покупает только продукты и тех, кто покупает все подряд.

Также в таких столбцах как: `tx_total_count`, `tx_sum_total`, `tx_sum_mean`, `items_count_sum`, `items_cost_sum` были замечены выбросы, которые были обработаны по правилу 3 IQR.

Далее я решила посмотреть более нетривиальные комбинации, для этого обучила catboost, построила shap, так и решила какие фичи оставить.

