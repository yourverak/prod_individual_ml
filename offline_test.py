import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from service.features import prepare_features, FEATURE_COLUMNS, TARGET_COLUMN
from service.metrics import calculate_map_at_100

def main():

    
    try:
        
        people = pd.read_csv("/Users/macbookvera/Desktop/ИИ/Prod_final_indv/data/v1-2/prod_clients.csv", nrows=50000)
        offer_activation = pd.read_csv("/Users/macbookvera/Desktop/ИИ/Prod_final_indv/data/v1-2/offer_activation.csv")
        receipts = pd.read_csv("/Users/macbookvera/Desktop/ИИ/Prod_final_indv/data/v2/receipts.csv", nrows=100000)
    except FileNotFoundError as e:
        print(f"❌ Ошибка файла: {e}")
        return

    data_dict = {
        "people": people,
        "offer_activation": offer_activation,
        "receipts": receipts
    }

    print("⚙️ Генерация фичей (prepare_features)...")
    df = prepare_features(data_dict)
    
    if df.empty:
        print("❌ Ошибка: DataFrame после prepare_features пустой!")
        return

    print(f"📊 Размер итогового датасета: {df.shape}")
    
    # Проверяем, сгенерировались ли фичи из FEATURE_COLUMNS
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"⚠️ Внимание! В df нет этих колонок: {missing_cols}")
        
    use_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not use_cols:
        print("❌ Нет фичей для обучения. Проверь, как мэтчатся receipts и people.")
        return

    X = df[use_cols].fillna(0)
    y = df[TARGET_COLUMN].values

    print("🧠 Разбивка данных и обучение (LogisticRegression)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    if len(np.unique(y_train)) < 2:
        print("❌ Ошибка: В выборке только один класс (0). Никто не активировал оффер.")
        return

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("🎯 Считаем твою метрику...")
    y_proba = model.predict_proba(X_test)[:, 1]

    # Вызываем твою функцию подсчета MAP@100
    map100 = calculate_map_at_100(y_test, y_proba)
    
    print("="*40)
    print(f"🔥 ОФФЛАЙН MAP@100 = {map100:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()