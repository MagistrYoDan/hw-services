import numpy as np

def predict(model, input_data, params=None):
    """
    Функция прогона модели.
    - Для изображения: input_data — байты изображения
    - Для JSON: input_data — dict
    """
    if isinstance(input_data, bytes):
        # Пример: преобразуем байты в вектор (упрощённо)
        feature_vector = np.frombuffer(input_data, dtype=np.uint8)[:100]  # обрезаем до 100 байт
        feature_vector = feature_vector.reshape(1, -1)
    else:
        # Для JSON: преобразуем в вектор
        feature_vector = np.array([input_data.get(f"feature_{i}", 0) for i in range(10)]).reshape(1, -1)

    # Предсказание
    prediction = model.predict(feature_vector)[0]
    return {"prediction": float(prediction)}
