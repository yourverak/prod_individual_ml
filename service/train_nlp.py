import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

offers = pd.read_csv('/Users/macbookvera/Desktop/ИИ/Prod_final_indv/data/v1-2/t_offer.csv')

def create_ground_truth_label(text):
    if pd.isna(text): return 'unknown'
    text = str(text).lower()
    
    # Рестораны и еда
    if any(word in text for word in ['ресторан', 'кафе', 'ням', 'бургер', 'пицца', 'кофе', 'бар', 'столовая', 'кофейня']):
        return 'food_dining'
    # Красота и уход
    elif any(word in text for word in ['beauty', 'красот', 'уход', 'студия', 'салон', 'стрижка', 'маникюр', 'барбершоп']):
        return 'beauty'
    # Аптеки и здоровье
    elif any(word in text for word in ['аптека', 'медицин', 'здоров', 'клиника', 'дент', 'стоматология']):
        return 'health'
    # Авто
    elif any(word in text for word in ['авто', 'запчасти', 'детейлинг', 'шиномонтаж', 'мойка', 'сто']):
        return 'auto'
    # Магазины продуктов (еда домой)
    elif any(word in text for word in ['сухофрукт', 'зелен', 'кондитерская', 'продукт', 'маркет', 'мед', 'мясо', 'рыба', 'пекарня']):
        return 'grocery'
    # Остальные магазины и услуги (ритейл)
    elif any(word in text for word in ['магазин', 'электрика', 'фото', 'цветов', 'одежда', 'обувь', 'мебель', 'ремонт']):
        return 'retail_services'
    # Пустые или неинформативные
    elif 'сайт не указан' in text or len(text.strip()) < 5:
        return 'no_info'
    else:
        return 'other'

offers['label'] = offers['offer_text'].apply(create_ground_truth_label)

df_train = offers.dropna(subset=['offer_text'])

X = df_train['offer_text']
y = df_train['label']

print("Размер выборки:", len(X))
print("\nРаспределение категорий:\n", y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n🤖 Обучаем TF-IDF + LogisticRegression...")
nlp_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=2000, ngram_range=(1, 2))), 
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

nlp_pipeline.fit(X_train, y_train)

y_pred = nlp_pipeline.predict(X_test)
print("\n📊 Отчет о качестве (Test):")
print(classification_report(y_test, y_pred))

os.makedirs("models", exist_ok=True)
model_path = "models/offer_classifier.pkl"
joblib.dump(nlp_pipeline, model_path)

