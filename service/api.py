# api.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from service.schemas import BatchRequest, CommitRequest, LookalikeRequest
from service.storage import save_batch, load_table
from service.pipeline import run_pipeline
from service.train import LookalikeModel
from service.state import STATE, state_lock

app = FastAPI()
model = LookalikeModel()

@app.get("/ready")
def ready():
    return {"status": "ok"}


@app.post("/data/batch")
def data_batch(req: BatchRequest):
    with state_lock:
        if req.version not in STATE["received_tables"]:
            STATE["received_tables"][req.version] = set()
        STATE["received_tables"][req.version].add(req.table)
    
    if req.records:
        save_batch(req.version, req.table, req.batch_id, req.records)
    return {"status": "accepted", "table": req.table, "batch_id": req.batch_id}

@app.post("/data/commit")
def data_commit(req: CommitRequest, background_tasks: BackgroundTasks):
    with state_lock:
        tables = list(STATE.get("received_tables", {}).get(req.version, []))
        if STATE["pipeline_status"] == "running":
            return {"status": "accepted"}
            
    background_tasks.add_task(run_pipeline, req.version)
    return {"status": "accepted", "tables_received": tables}

@app.get("/status")
def status():
    return {
        "ready": True,
        "pipeline_status": STATE["pipeline_status"],
        "data_version": STATE["data_version"],
        "model_version": STATE["model_version"]
    }

@app.post("/lookalike")
def lookalike(req: LookalikeRequest):
    people = load_table(STATE["data_version"], "people")
    if people is None or people.empty:
        raise HTTPException(404, "Data not found")
        
    tx = load_table(STATE["data_version"], "transaction")
    merch = load_table(STATE["data_version"], "merchant")
    
    if tx is not None and merch is not None:
        target_brand = merch[merch["merchant_id_offer"] == merchant_id]["brand_dk"].values
        if len(target_brand) > 0:
            past_buyers = tx[tx["brand_dk"].isin(target_brand)]["user_id"].unique()
            people = people[~people["user_id"].isin(past_buyers)].copy()

    if people.empty:
        return {"merchant_id": merchant_id, "offer_id": offer_id, "audience": [], "audience_size": 0, "model_version": STATE["model_version"], "reasons": []}

    inference_base = people[["user_id"]].copy()
    inference_base["offer_id"] = offer_id

    data_dict = {
        "people": people,
        "segments": load_table(STATE["data_version"], "segments"),
        "transaction": tx,
        "offer": load_table(STATE["data_version"], "offer"),
        "financial_account": load_table(STATE["data_version"], "financial_account"),
        "offer_seens": load_table(STATE["data_version"], "offer_seens"),
        "offer_reward": load_table(STATE["data_version"], "offer_reward"),
        "receipts": load_table(STATE["data_version"], "receipts")
    }

    from service.features import prepare_features
    df_features = prepare_features(data_dict, inference_base=inference_base)
    
    model.load() 
    scores = model.predict(df_features)
    df_features["score"] = scores
    
    top = df_features.sort_values("score", ascending=False).head(top_n)
    
    reasons = model.get_reasons(top)

    audience = []
    for i, row in enumerate(top.itertuples()):
        audience.append({"user_id": int(row.user_id), "score": float(row.score)})
        
    avg_reasons = reasons[0] if reasons else []

    return {
        "merchant_id": merchant_id,
        "offer_id": offer_id,
        "audience": audience,
        "audience_size": len(audience),
        "model_version": STATE["model_version"],
        "reasons": avg_reasons
    }