from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List
from service.schemas import BatchRequest, CommitRequest, LookalikeRequest, LookalikeBatchRequest
from service.storage import save_batch, load_table
from service.pipeline import run_pipeline
from service.train import LookalikeModel
from service.state import STATE, state_lock
import pandas as pd

app = FastAPI()
model = LookalikeModel()
# Подгружаем веса при старте, если есть
model.load()

@app.on_event("startup")
def startup_event():
    with state_lock:
        if model.explainer is not None:
            STATE["ready"] = True

@app.get("/ready")
def ready():
    if STATE["ready"]:
        return {"status": "ok"}
    raise HTTPException(503, "Not ready")

@app.post("/data/batch")
def data_batch(req: BatchRequest):
    if not req.version or not req.table or req.batch_id is None:
        raise HTTPException(400, "Missing required fields")
        
    valid_tables = ["people", "segments", "transaction", "offer", "merchant", "financial_account", "offer_seens", "offer_activation", "offer_reward", "receipts"]
    if req.table not in valid_tables:
        raise HTTPException(400, "Unknown table")

    if req.records:
        save_batch(req.version, req.table, req.batch_id, req.records)

    return {"status": "accepted", "table": req.table, "batch_id": req.batch_id}

@app.post("/data/commit")
def data_commit(req: CommitRequest, background_tasks: BackgroundTasks):
    with state_lock:
        if STATE["pipeline_status"] == "running":
            return {"status": "accepted"}
            
    background_tasks.add_task(run_pipeline, req.version)
    return {"status": "accepted", "tables_received": ["people", "segments", "transaction", "offer", "merchant", "financial_account", "offer_seens", "offer_activation", "offer_reward", "receipts"]}

@app.get("/status")
def status():
    return {
        "ready": STATE["ready"],
        "pipeline_status": STATE["pipeline_status"],
        "data_version": STATE["data_version"],
        "model_version": STATE["model_version"]
    }

def _process_lookalike(merchant_id: int, offer_id: int, top_n: int):
    # Грузим актуальных людей
    people = load_table(STATE["data_version"], "people")
    if people is None or people.empty:
        raise HTTPException(404, "Data not found")
        
    # Исключаем текущих клиентов
    tx = load_table(STATE["data_version"], "transaction")
    merch = load_table(STATE["data_version"], "merchant")
    
    if tx is not None and merch is not None:
        # Находим brand_dk для этого merchant_id
        target_brand = merch[merch["merchant_id_offer"] == merchant_id]["brand_dk"].values
        if len(target_brand) > 0:
            past_buyers = tx[tx["brand_dk"].isin(target_brand)]["user_id"].unique()
            people = people[~people["user_id"].isin(past_buyers)].copy()

    if people.empty:
        return {"merchant_id": merchant_id, "offer_id": offer_id, "audience": [], "audience_size": 0, "model_version": STATE["model_version"], "reasons": []}

    # Загружаем фичи для инференса
    from service.features import prepare_features
    df_features = prepare_features({"people": people})
    
    model.load() # Обновляем модель в памяти если была переобучена
    scores = model.predict(df_features)
    people["score"] = scores
    
    top = people.sort_values("score", ascending=False).head(top_n)
    
    # SHAP reasons для топа
    top_features = df_features.loc[top.index]
    reasons = model.get_reasons(top_features)

    audience = []
    for i, row in enumerate(top.itertuples()):
        audience.append({"user_id": int(row.user_id), "score": float(row.score)})
        
    # Для простоты контракта API reasons возвращается один для запроса (берем усредненный или топ-1)
    # По контракту reasons: list of dicts. Вернем глобальные объяснения для этого топа.
    avg_reasons = reasons[0] if reasons else []

    return {
        "merchant_id": merchant_id,
        "offer_id": offer_id,
        "audience": audience,
        "audience_size": len(audience),
        "model_version": STATE["model_version"],
        "reasons": avg_reasons
    }

@app.post("/lookalike")
def lookalike(req: LookalikeRequest):
    if req.top_n < 1 or req.top_n > 1000:
        raise HTTPException(400, "top_n out of bounds")
    return _process_lookalike(req.merchant_id, req.offer_id, req.top_n)

@app.post("/lookalike/batch")
def lookalike_batch(req: LookalikeBatchRequest):
    results = []
    for r in req.requests:
        try:
            res = _process_lookalike(r.merchant_id, r.offer_id, r.top_n)
            results.append(res)
        except:
            pass
    return {"results": results}

@app.get("/model/info")
def model_info():
    return {
        "model_name": "lookalike-catboost",
        "model_version": STATE["model_version"],
        "trained_on": STATE["trained_on"],
        "features_count": 9,
        "train_metrics": {
            "map_at_100": STATE.get("map@100", 0.0)
        }
    }

@app.get("/monitoring/drift")
def monitoring_drift():
    return {
        "drift_detected": STATE["drift_detected"],
        "drift_score": STATE["drift_score"],
        "action_taken": STATE["last_action"]
    }

@app.get("/monitoring/data-quality")
def data_quality():
    res = STATE["data_quality"].copy()
    res["version"] = STATE["data_version"]
    return res

@app.get("/experiments")
def get_experiments():
    return {"experiments": STATE["experiments"]}