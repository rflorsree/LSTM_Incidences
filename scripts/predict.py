import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from datetime import datetime

# Cargar modelo y escaladores
model = tf.keras.models.load_model("models/lstm_model.keras")
scaler_X = joblib.load("models/scaler_X.pkl")
scaler_y = joblib.load("models/scaler_y.pkl")

# Parámetros estimados de entrada
fecha_str = "2025-04-02"
hora_inicio = 10
clientes_aprox = 500
tm_muerto_aprox = 150
tm_resolucion_aprox = 200
minutos_respuesta = 90

# Procesar fecha
fecha = pd.to_datetime(fecha_str)
mes = fecha.month
dia_semana = fecha.weekday()
semana_del_anio = fecha.isocalendar().week

# Construir input
X_input = np.array([[
    hora_inicio,
    mes,
    dia_semana,
    semana_del_anio,
    minutos_respuesta,
    clientes_aprox,
    tm_muerto_aprox,
    tm_resolucion_aprox
]])

# Escalar y dar forma para LSTM
X_scaled = scaler_X.transform(X_input).reshape(1, 1, -1)

# Predecir
y_pred_scaled = model.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]

# Mostrar resultados
print(f"Predicción LSTM para {fecha_str}")
print(f"- Número de incidencias:         {y_pred[0]:.0f}")
print(f"- Clientes afectados:           {y_pred[1]:.2f}")
print(f"- Tiempo promedio muerto (min): {y_pred[2]:.2f}")
print(f"- Tiempo promedio resolución:   {y_pred[3]:.2f}")
