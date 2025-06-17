import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from datetime import datetime
from dotenv import load_dotenv
import os

# Cargar variables desde archivo .env
load_dotenv("eda_outputs/.env")

# Variables desde entorno
clientes_aprox = float(os.getenv("CLIENTES_APROX"))
tm_muerto_aprox = float(os.getenv("TM_MUERTO_APROX"))
tm_resolucion_aprox = float(os.getenv("TM_RESOLUCION_APROX"))
minutos_respuesta = float(os.getenv("MINUTOS_RESPUESTA"))
hora_inicio = float(os.getenv("hora_inicio"))

# Fecha objetivo
fecha_str = "2025-04-02"


# Procesar fecha
fecha = pd.to_datetime(fecha_str)
mes = fecha.month
dia_semana = fecha.weekday()
semana_del_anio = fecha.isocalendar().week

# Construir input para el modelo
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

# Cargar modelo y escaladores
model = tf.keras.models.load_model("models/lstm_model.keras")
scaler_X = joblib.load("models/scaler_X.pkl")
scaler_y = joblib.load("models/scaler_y.pkl")

# Escalar y predecir
X_scaled = scaler_X.transform(X_input).reshape(1, 1, -1)
y_pred_scaled = model.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]

# Mostrar resultados
print(f"Predicción LSTM para {fecha_str}")
print(f"- Número de incidencias:         {y_pred[0]:.0f}")
print(f"- Clientes afectados:            {y_pred[1]:.0f}")
print(f"- Tiempo muerto (min):           {y_pred[2]:.2f}")
print(f"- Tiempo de resolución (min):    {y_pred[3]:.2f}")
