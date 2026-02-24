import pandas as pd
import requests
import os
import numpy as np

API_URL = "http://localhost/data/batch"
COMMIT_URL = "http://localhost/data/commit"

files_to_upload = [
    {"path": "/Users/macbookvera/Desktop/ИИ/Prod_final_indv/ml/v1-2/prod_clients.csv", "table": "people"},
    {"path": "/Users/macbookvera/Desktop/ИИ/Prod_final_indv/ml/v1-2/prod_financial_transaction.csv", "table": "transaction"},
    {"path": "/Users/macbookvera/Desktop/ИИ/Prod_final_indv/ml/v1-2/offer_activation.csv", "table": "offer_activation"},
    {"path": "/Users/macbookvera/Desktop/ИИ/Prod_final_indv/ml/v2/receipts.csv", "table": "receipts"}
]

def upload_data():
    for item in files_to_upload:
        file_path = item["path"]
        table_name = item["table"]
        
        if not os.path.exists(file_path):
            print(f"⚠️ Файл не найден: {file_path}")
            continue
            
        print(f"--- Чтение {table_name} ---")
        df = pd.read_csv(file_path, nrows=100000)
        df = df.replace({np.nan: None})
        total_rows = len(df)
        batch_size = 10000
        total_batches = (total_rows // batch_size) + (1 if total_rows % batch_size > 0 else 0)
        print(f"Загрузка {table_name}: {total_rows} строк...")
        
        for i in range(0, total_rows, batch_size):
            batch = df.iloc[i : i + batch_size]
            current_batch_id = (i // batch_size) + 1
            payload = {
                "version": "v1",
                "table": table_name,
                "batch_id": current_batch_id,
                "total_batches": total_batches,
                "records": batch.to_dict(orient="records")
            }
            try:
                requests.post(API_URL, json=payload)
            except Exception as e:
                print(f"❌ Ошибка соединения: {e}")
                return
                
    print("--- Отправка команды COMMIT ---")
    res = requests.post(COMMIT_URL, json={"version": "v1"})
    if res.status_code == 200:
        print("🚀 Успех! Обучение пошло. Смотри логи Docker.")
    else:
        print(f"❌ Ошибка COMMIT: {res.text}")

if __name__ == "__main__":
    upload_data()