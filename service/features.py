import pandas as pd
import numpy as np

# Обновленный список самых сильных фичей из SHAP
FEATURE_COLUMNS = [
    "recency_days", "hunter_index", "rewards_count", "rec_unique_categories",
    "items_cost_mean", "age_bucket", "region", "active_accounts_count",
    "online_share", "tx_sum_total", "segment", "region_size",
    "items_count_mean", "vip_status", "entrepreneur", "gender_cd",
    "tx_total_count", "items_cost_sum", "items_count_sum"
]

TARGET_COLUMN = "target"

# Категориальные фичи для CatBoost
CAT_FEATURES = [
    "age_bucket", "region", "segment", "region_size", 
    "vip_status", "entrepreneur", "gender_cd"
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

def prepare_features(data_dict: dict) -> pd.DataFrame:
    people = data_dict.get("people", pd.DataFrame())
    if people.empty:
        return pd.DataFrame()

    df = people.copy()

    # 1. Сегменты
    if "segments" in data_dict and not data_dict["segments"].empty:
        df = df.merge(data_dict["segments"], on="user_id", how="left")

    # 2. Финансовые счета (Активные)
    if "financial_account" in data_dict and not data_dict["financial_account"].empty:
        accs = data_dict["financial_account"]
        active = accs[accs["account_status_cd"] == "ACT"]
        active_counts = active.groupby("user_id").size().reset_index(name="active_accounts_count")
        df = df.merge(active_counts, on="user_id", how="left")

    # 3. Награды в прошлом
    if "offer_reward" in data_dict and not data_dict["offer_reward"].empty:
        rew = data_dict["offer_reward"]
        rew_counts = rew.groupby("user_id").size().reset_index(name="rewards_count")
        df = df.merge(rew_counts, on="user_id", how="left")

    # 4. Транзакции (Маппинг бакетов, онлайн доля)
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
        df = df.merge(tx_feats, on="user_id", how="left")

    # 5. Чеки (receipts)
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
        df = df.merge(rec_feats, on="user_id", how="left")

    # 6. Recency Days (через offer_seens)
    if "offer_seens" in data_dict and not data_dict["offer_seens"].empty:
        seens = data_dict["offer_seens"]
        # Берем последнюю дату показа для пользователя
        seens_dates = seens.groupby("user_id")["start_date"].max().reset_index()
        df = df.merge(seens_dates, on="user_id", how="left")
        
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        df['last_activity_day'] = pd.to_datetime(df['last_activity_day'], errors='coerce')
        
        df['recency_days'] = (df['last_activity_day'] - df['start_date']).dt.days.abs()
        df['recency_days'] = df['recency_days'].fillna(999)

    # 7. Сложные индексы
    df['tx_total_count'] = df.get('tx_total_count', 0).fillna(0)
    df['rewards_count'] = df.get('rewards_count', 0).fillna(0)
    df['tx_online_count'] = df.get('tx_online_count', 0).fillna(0)
    
    df['hunter_index'] = df['rewards_count'] / (df['tx_total_count'] + 1)
    df['online_share'] = df['tx_online_count'] / (df['tx_total_count'] + 1e-9)

    # 8. Очистка от выбросов (Правило 3 IQR)
    iqr_cols = [
        'tx_total_count', 'tx_sum_total', 'tx_sum_mean', 
        'items_count_sum', 'items_cost_sum'
    ]
    for col in iqr_cols:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            upper_bound = q3 + 3 * iqr
            df[col] = df[col].clip(upper=upper_bound)

    # 9. Формирование таргета
    if "offer_activation" in data_dict and not data_dict["offer_activation"].empty:
        activations = data_dict["offer_activation"]["user_id"].unique()
        df[TARGET_COLUMN] = df["user_id"].isin(activations).astype(int)
    else:
        df[TARGET_COLUMN] = 0

    # 10. Заполнение пропусков и типов для CatBoost
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = "Unknown" if col in CAT_FEATURES else 0
            
    # Категориальные в строку
    df[CAT_FEATURES] = df[CAT_FEATURES].fillna("Unknown").astype(str)
    
    # Численные в float
    num_cols = [c for c in FEATURE_COLUMNS if c not in CAT_FEATURES]
    df[num_cols] = df[num_cols].fillna(0).astype(float)

    return df