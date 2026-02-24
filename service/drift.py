import pandas as pd
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

REFERENCE_DATA_PATH = "data/reference_features.csv"
DRIFT_THRESHOLD = 0.3

def check_drift(current_df: pd.DataFrame) -> tuple[bool, float]:
    if not os.path.exists(REFERENCE_DATA_PATH):
        return False, 0.0 # Первое обучение, не с чем сравнивать

    reference_df = pd.read_csv(REFERENCE_DATA_PATH)
    
    # Используем только численные/категориальные фичи (без ID)
    features = current_df.select_dtypes(include=['number', 'object']).columns.tolist()
    features = [f for f in features if f not in ["user_id", "target", "offer_id"]]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df[features], current_data=current_df[features])
    
    result = report.as_dict()
    drift_score = result["metrics"][0]["result"]["share_of_drifted_columns"]
    
    drift_detected = drift_score > DRIFT_THRESHOLD
    return drift_detected, drift_score

def save_reference(df: pd.DataFrame):
    os.makedirs("data", exist_ok=True)
    df.to_csv(REFERENCE_DATA_PATH, index=False)