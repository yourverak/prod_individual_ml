from pydantic import BaseModel
from typing import List

class LookalikeRequest(BaseModel):
    merchant_id: int
    offer_id: int
    top_n: int

class LookalikeBatchItem(BaseModel):
    merchant_id: int
    offer_id: int
    top_n: int

class LookalikeBatchRequest(BaseModel):
    requests: List[LookalikeBatchItem]

class BatchRequest(BaseModel):
    version: str
    table: str
    batch_id: int
    total_batches: int
    records: List[dict]

class CommitRequest(BaseModel):
    version: str