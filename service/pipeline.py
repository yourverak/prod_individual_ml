import mlflow
import datetime
from service.storage import load_table
from service.features import prepare_features, TARGET_COLUMN
from service.validation import validate_data
from service.drift import check_drift, save_reference
from service.train import LookalikeModel
from service.metrics import calculate_map_at_100
from service.state import STATE, state_lock

# Устанавливаем MLflow трекинг
mlflow.set_tracking_uri("http://mlflow:5000")

def run_pipeline(version: str):
    with state_lock:
        STATE["pipeline_status"] = "running"

    try:
        tables = ["people", "segments", "transaction", "offer", "merchant", "financial_account", "offer_seens", "offer_activation", "offer_reward"]
        data_dict = {t: load_table(version, t) for t in tables}

        # 1. Валидация
        val_report = validate_data(data_dict)
        with state_lock:
            STATE["data_quality"] = val_report
            STATE["data_version"] = version

        if not val_report["valid"]:
            with state_lock:
                STATE["last_action"] = "skipped"
                STATE["pipeline_status"] = "idle"
            return

        # 2. Feature Engineering
        df = prepare_features(data_dict)
        if df.empty:
            raise ValueError("Empty features")

        # 3. Проверка дрейфа
        drift_detected, drift_score = check_drift(df)
        
        with state_lock:
            STATE["drift_detected"] = drift_detected
            STATE["drift_score"] = drift_score

        # Если дрейфа нет и модель уже обучена - пропускаем обучение (Fail-soft + Performance)
        if not drift_detected and STATE["model_version"] != "v0.0":
            with state_lock:
                STATE["last_action"] = "none"
                STATE["pipeline_status"] = "idle"
                STATE["ready"] = True
            return

        # 4. Обучение и MLflow
        mlflow.set_experiment("Lookalike_Training")
        with mlflow.start_run() as run:
            model = LookalikeModel()
            model.fit(df)
            
            df["score"] = model.predict(df)
            map100 = calculate_map_at_100(df[TARGET_COLUMN].tolist(), df["score"].tolist())
            
            mlflow.log_param("data_version", version)
            mlflow.log_metric("map_at_100", map100)
            
            # Сохраняем референс для следующего расчета дрейфа
            save_reference(df)
            
            new_model_version = f"v{int(STATE['model_version'].replace('v','').split('.')[0]) + 1}.0"
            
            exp_record = {
                "run_id": run.info.run_id,
                "data_version": version,
                "model_version": new_model_version,
                "metrics": {"map_at_100": map100},
                "timestamp": datetime.datetime.utcnow().isoformat()
            }

            with state_lock:
                STATE["model_version"] = new_model_version
                STATE["trained_on"] = version
                STATE["last_action"] = "retrained"
                STATE["experiments"].append(exp_record)
                STATE["pipeline_status"] = "idle"
                STATE["ready"] = True

    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        with state_lock:
            STATE["pipeline_status"] = "failed"
            # Не меняем ready, старая модель продолжит работать