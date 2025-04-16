import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_excel("data/EXCELINCIDENCIAS.xlsx", sheet_name="Sheet1")

# Procesar fechas
df["INICIO INCIDENCIA"] = pd.to_datetime(df["INICIO INCIDENCIA"])
df["HORA DE LLEGADA"] = pd.to_datetime(df["HORA DE LLEGADA"])
df["CIERRE DE INCIDENCIA"] = pd.to_datetime(df["CIERRE DE INCIDENCIA"])

# Variables temporales
df["hora_inicio"] = df["INICIO INCIDENCIA"].dt.hour
df["mes"] = df["INICIO INCIDENCIA"].dt.month
df["dia_semana"] = df["INICIO INCIDENCIA"].dt.weekday
df["semana_del_anio"] = df["INICIO INCIDENCIA"].dt.isocalendar().week.astype(int)
df["minutos_respuesta"] = (df["HORA DE LLEGADA"] - df["INICIO INCIDENCIA"]).dt.total_seconds() / 60

# Agregar incidencias por día
df["FECHA"] = df["INICIO INCIDENCIA"].dt.date
incidencias_por_dia = df.groupby("FECHA").size().reset_index(name="incidencias")
df = df.merge(incidencias_por_dia, on="FECHA", how="left")

# Entradas (X)
X = df[[
    "hora_inicio", "mes", "dia_semana", "semana_del_anio", "minutos_respuesta",
    "CLIENTES", "TIEMPO MUERTO (MIN)", "TIEMPO RESOLUCION (MIN)"
]].values

# Salidas (y)
y = df[["incidencias", "CLIENTES", "TIEMPO MUERTO (MIN)", "TIEMPO RESOLUCION (MIN)"]].values

# Escalado
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Redimensionar para LSTM
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Modelo LSTM
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_scaled.shape[1], X_scaled.shape[2])),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4)  # 4 salidas
])

model.compile(optimizer='adam', loss='mse')

# Entrenamiento
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Guardar modelo y escaladores
os.makedirs("models", exist_ok=True)
model.save("models/lstm_model.keras")
joblib.dump(scaler_X, "models/scaler_X.pkl")
joblib.dump(scaler_y, "models/scaler_y.pkl")

# Crear carpeta de salidas
os.makedirs("outputs", exist_ok=True)

# Gráfica de pérdida
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('LSTM - Pérdida durante el entrenamiento')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid()
plt.savefig("outputs/lstm_loss_curve.png")
plt.close()

# Predicción
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_orig = scaler_y.inverse_transform(y_test)

labels = ["Incidencias", "Clientes", "TM Muerto", "TM Resolución"]
maes = []
r2s = []

# Visualización y métricas
for i in range(4):
    plt.figure(figsize=(8, 4))
    plt.scatter(y_test_orig[:, i], y_pred[:, i], alpha=0.5)
    plt.xlabel("Real")
    plt.ylabel("Predicho")
    plt.title(f"LSTM - Real vs Predicho - {labels[i]}")
    plt.grid()
    plt.savefig(f"outputs/lstm_real_vs_pred_{i}_{labels[i].replace(' ', '_').lower()}.png")
    plt.close()

    maes.append(mean_absolute_error(y_test_orig[:, i], y_pred[:, i]))
    r2s.append(r2_score(y_test_orig[:, i], y_pred[:, i]))

# MAE
plt.figure(figsize=(10, 5))
plt.bar(labels, maes, color='orange')
plt.title('LSTM - MAE por variable de salida')
plt.ylabel('Mean Absolute Error')
plt.grid(axis='y')
plt.savefig("outputs/lstm_mae_comparativo.png")
plt.close()

# R²
plt.figure(figsize=(10, 5))
plt.bar(labels, r2s, color='teal')
plt.title('LSTM - R² Score por variable de salida')
plt.ylabel('R²')
plt.ylim(0, 1)
plt.grid(axis='y')
plt.savefig("outputs/lstm_r2_comparativo.png")
plt.close()

print("Modelo LSTM con incidencias entrenado, guardado y gráficas exportadas a 'outputs'.")
