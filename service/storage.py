# storage.py
import json
import io
import pandas as pd
from service.s3_utils import s3, RAW_BUCKET, ensure_bucket

ensure_bucket(RAW_BUCKET)

def batch_key(version: str, table: str, batch_id: int):
    return f"{version}/{table}/batch_{batch_id}.json"

def save_batch(version: str, table: str, batch_id: int, records: list):

    key = batch_key(version, table, batch_id)

    # идемпотентность
    try:
        s3.head_object(Bucket=RAW_BUCKET, Key=key)
        return
    except:
        pass

    body = json.dumps(records).encode("utf-8")
    s3.put_object(Bucket=RAW_BUCKET, Key=key, Body=body)

def load_table(version: str, table: str) -> pd.DataFrame | None:

    prefix = f"{version}/{table}/"

    resp = s3.list_objects_v2(Bucket=RAW_BUCKET, Prefix=prefix)
    if "Contents" not in resp:
        return None

    dfs = []

    for obj in resp["Contents"]:
        body = s3.get_object(
            Bucket=RAW_BUCKET,
            Key=obj["Key"]
        )["Body"].read()

        if not body:
            continue

        df = pd.read_json(io.BytesIO(body))
        dfs.append(df)

    if not dfs:
        return None

    return pd.concat(dfs, ignore_index=True)