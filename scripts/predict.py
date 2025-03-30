
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from datetime import datetime

# cargar modelos 
model = tf.keras.models.load_model("models/lstm_model.keras")
ohe = joblib.load("models/encoder_causa.pkl")
scaler_y = joblib.load("models/scaler_y.pkl")

# input
fecha_str = "2025-06-02"  
causa = "Robo de CTO"      

# procesar entrada
fecha = pd.to_datetime(fecha_str)
dia_semana = fecha.weekday()

# codificar causa
causa_encoded = ohe.transform([[causa]])
X_input = np.hstack([[dia_semana], causa_encoded[0]]).reshape(1, 1, -1)  # (samples, timesteps, features)

# predecir
y_pred_scaled = model.predict(X_input)
y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]

# resultados
print(f"Predicción LSTM para {fecha_str} - {causa}")
print(f"- Número de incidencias:         {y_pred[0]:.2f}")
print(f"- Clientes afectados:           {y_pred[1]:.2f}")
print(f"- Tiempo promedio muerto (min): {y_pred[2]:.2f}")
print(f"- Tiempo promedio resolución:   {y_pred[3]:.2f}")
