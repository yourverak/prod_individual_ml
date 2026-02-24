import pandas as pd
import requests
import os
import numpy as np
import sys

API_URL = "http://localhost/data/batch"
COMMIT_URL = "http://localhost/data/commit"

TABLE_TO_FILE = {
    "people": "prod_clients.csv",
    "segments": "prizm_segments.csv",
    "transaction": "prod_financial_transaction.csv",
    "offer": "t_offer.csv",
    "merchant": "t_merchant.csv",
    "financial_account": "financial_account.csv",
    "offer_seens": "offer_seens.csv",
    "offer_activation": "offer_activation.csv",
    "offer_reward": "offer_reward.csv",
    "receipts": "receipts.csv"
}

def upload_version(version_folder):
    api_version = "v1" if "v1" in version_folder else "v2"
        
    base_path = f"./data/{version_folder}"
    
    if not os.path.exists(base_path):
        return

    for table_name, file_name in TABLE_TO_FILE.items():
        file_path = os.path.join(base_path, file_name)
        
        if not os.path.exists(file_path):
            continue
            
    
        df = pd.read_csv(file_path, nrows=50000) 
        df = df.replace({np.nan: None})
        
        total_rows = len(df)
        batch_size = 10000
        total_batches = (total_rows // batch_size) + (1 if total_rows % batch_size > 0 else 0)
        
        
        for i in range(0, total_rows, batch_size):
            batch = df.iloc[i : i + batch_size]
            current_batch_id = (i // batch_size) + 1
            payload = {
                "version": api_version,
                "table": table_name,
                "batch_id": current_batch_id,
                "total_batches": total_batches,
                "records": batch.to_dict(orient="records")
            }
            try:
                res = requests.post(API_URL, json=payload)
                if res.status_code != 200:
                    print(f"Ошибка в батче {current_batch_id}: {res.text}")
            except Exception as e:
                print(f"Ошибка соединения: {e}")
                return
                
    res = requests.post(COMMIT_URL, json={"version": api_version})
    if res.status_code == 200:
        print(f"Пайплайн запущен. Проверь логи Docker контейнера 'api'.")
    else:
        print(f"шибка COMMIT: {res.text}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        upload_version(sys.argv[1])
    else:
        print("Использование: python upload_data.py [имя_папки_в_data]")