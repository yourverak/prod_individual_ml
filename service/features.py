import pandas as pd
import numpy as np

FEATURE_COLUMNS = [
    "recency_days", "hunter_index", "rewards_count", "rec_unique_categories",
    "items_cost_mean", "age_bucket", "region", "active_accounts_count",
    "online_share", "tx_sum_total", "segment", "region_size",
    "items_count_mean", "vip_status", "entrepreneur", "gender_cd",
    "tx_total_count", "items_cost_sum", "items_count_sum",
    "merchant_id_offer", "offer_id" 
]

TARGET_COLUMN = "target"

CAT_FEATURES = [
    "age_bucket", "region", "segment", "region_size", 
    "vip_status", "entrepreneur", "gender_cd",
    "merchant_id_offer", "offer_id"
]

def map_buckets(val):
    if pd.isna(val): return 0
    v = str(val).lower()
    if '10k+' in v or '10000+' in v: return 15000
    if '5k-10k' in v or '5000-10000' in v: return 7500
    if '1k-5k' in v or '1000-5000' in v: return 3000
    if '500-1000' in v: return 750
    if '100-500' in v: return 300
    if '0-100' in v or '<100' in v: return 50
    if '<1k' in v: return 500
    return 0

def prepare_features(data_dict: dict, inference_base: pd.DataFrame = None) -> pd.DataFrame:
    if inference_base is not None:
        df = inference_base.copy()
    else:
        seens = data_dict.get("offer_seens", pd.DataFrame())
        if seens.empty:
            return pd.DataFrame()
        df = seens[["user_id", "offer_id"]].drop_duplicates().copy()

    people = data_dict.get("people", pd.DataFrame())
    if not people.empty:
        user_features = people.copy()

        if "segments" in data_dict and not data_dict["segments"].empty:
            user_features = user_features.merge(data_dict["segments"], on="user_id", how="left")

        if "financial_account" in data_dict and not data_dict["financial_account"].empty:
            accs = data_dict["financial_account"]
            active_counts = accs[accs["account_status_cd"] == "ACT"].groupby("user_id").size().reset_index(name="active_accounts_count")
            user_features = user_features.merge(active_counts, on="user_id", how="left")

        if "offer_reward" in data_dict and not data_dict["offer_reward"].empty:
            rew_counts = data_dict["offer_reward"].groupby("user_id").size().reset_index(name="rewards_count")
            user_features = user_features.merge(rew_counts, on="user_id", how="left")

        if "transaction" in data_dict and not data_dict["transaction"].empty:
            tx = data_dict["transaction"].copy()
            tx['amount_num'] = tx['amount_bucket'].apply(map_buckets)
            tx['is_online'] = (tx['online_transaction_flg'] == 'Y').astype(int)
            tx_feats = tx.groupby('user_id').agg(
                tx_total_count=('transaction_id', 'count'),
                tx_online_count=('is_online', 'sum'),
                tx_sum_total=('amount_num', 'sum'),
                tx_sum_mean=('amount_num', 'mean')
            ).reset_index()
            user_features = user_features.merge(tx_feats, on="user_id", how="left")

        if "receipts" in data_dict and not data_dict["receipts"].empty:
            rec = data_dict["receipts"].copy()
            rec['items_cost_num'] = rec['items_cost'].apply(map_buckets)
            rec['items_count'] = pd.to_numeric(rec['items_count'], errors='coerce').fillna(0)
            rec_feats = rec.groupby('user_id').agg(
                items_count_sum=('items_count', 'sum'),
                items_count_mean=('items_count', 'mean'),
                items_cost_sum=('items_cost_num', 'sum'),
                items_cost_mean=('items_cost_num', 'mean'),
                rec_unique_categories=('category_name', 'nunique')
            ).reset_index()
            user_features = user_features.merge(rec_feats, on="user_id", how="left")

        if "offer_seens" in data_dict and not data_dict["offer_seens"].empty:
            seens_dates = data_dict["offer_seens"].groupby("user_id")["start_date"].max().reset_index()
            user_features = user_features.merge(seens_dates, on="user_id", how="left")
            user_features['start_date'] = pd.to_datetime(user_features['start_date'], errors='coerce')
            user_features['last_activity_day'] = pd.to_datetime(user_features['last_activity_day'], errors='coerce')
            user_features['recency_days'] = (user_features['last_activity_day'] - user_features['start_date']).dt.days.abs()
            user_features['recency_days'] = user_features['recency_days'].fillna(999)

        user_features['tx_total_count'] = user_features.get('tx_total_count', 0).fillna(0)
        user_features['rewards_count'] = user_features.get('rewards_count', 0).fillna(0)
        user_features['tx_online_count'] = user_features.get('tx_online_count', 0).fillna(0)
        user_features['hunter_index'] = user_features['rewards_count'] / (user_features['tx_total_count'] + 1)
        user_features['online_share'] = user_features['tx_online_count'] / (user_features['tx_total_count'] + 1e-9)

        iqr_cols = ['tx_total_count', 'tx_sum_total', 'tx_sum_mean', 'items_count_sum', 'items_cost_sum']
        for col in iqr_cols:
            if col in user_features.columns:
                q1 = user_features[col].quantile(0.25)
                q3 = user_features[col].quantile(0.75)
                user_features[col] = user_features[col].clip(upper=q3 + 3 * (q3 - q1))

        df = df.merge(user_features, on="user_id", how="left")

    if "offer" in data_dict and not data_dict["offer"].empty:
        df = df.merge(data_dict["offer"][["offer_id", "merchant_id_offer"]], on="offer_id", how="left")

    if inference_base is None:
        activations = data_dict.get("offer_activation", pd.DataFrame())
        if not activations.empty:
            acts = activations[["user_id", "offer_id"]].drop_duplicates()
            acts[TARGET_COLUMN] = 1
            df = df.merge(acts, on=["user_id", "offer_id"], how="left")
            df[TARGET_COLUMN] = df[TARGET_COLUMN].fillna(0).astype(int)
        else:
            df[TARGET_COLUMN] = 0

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = "Unknown" if col in CAT_FEATURES else 0

    df[CAT_FEATURES] = df[CAT_FEATURES].fillna("Unknown").astype(str)
    num_cols = [c for c in FEATURE_COLUMNS if c not in CAT_FEATURES]
    df[num_cols] = df[num_cols].fillna(0).astype(float)

    return df