import numpy as np

def calculate_map_at_100(y_true, y_probas):
    """
    Вычисляет Average Precision для топ-100 предсказаний.
    y_true: массив реальных таргетов (0 или 1)
    y_probas: массив вероятностей от модели
    """
    # 1. Объединяем таргет и вероятности и сортируем по убыванию вероятности
    data = sorted(zip(y_probas, y_true), key=lambda x: x[0], reverse=True)
    
    # 2. Берем только топ-100
    top_100 = data[:100]
    
    # 3. Считаем Average Precision
    relevant_count = 0  # Сколько реальных единичек мы встретили
    precision_sum = 0   # Сумма точностей в точках, где встретили единичку
    
    for i, (prob, label) in enumerate(top_100):
        if label == 1:
            relevant_count += 1
            # Точность на текущем шаге i: (кол-во найденных / текущая позиция)
            precision_sum += relevant_count / (i + 1)
            
    if relevant_count == 0:
        return 0.0
        
    # Делим на общее количество найденных релевантных объектов в топ-100
    return precision_sum / relevant_count