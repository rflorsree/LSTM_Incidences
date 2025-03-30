
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_excel("data/EXCELINCIDENCIAS.xlsx", sheet_name="Sheet1")

# procesar fechas
df["FECHA"] = df["INICIO INCIDENCIA"].dt.date

data_grouped = df.groupby(["FECHA", "CAUSA"]).agg({
    "INICIO INCIDENCIA": "count",
    "CLIENTES": "sum",
    "TIEMPO MUERTO (MIN)": "mean",
    "TIEMPO RESOLUCION (MIN)": "mean"
}).reset_index()

data_grouped.columns = ["fecha", "causa", "incidencias", "clientes", "tm_muerto", "tm_resolucion"]
data_grouped["fecha"] = pd.to_datetime(data_grouped["fecha"])
data_grouped["dia_semana"] = data_grouped["fecha"].dt.weekday

# codificacion de causas
ohe = OneHotEncoder(sparse_output=False)
causa_encoded = ohe.fit_transform(data_grouped[["causa"]])

# targets
X = np.hstack([
    data_grouped[["dia_semana"]].values,
    causa_encoded
])

y = data_grouped[["incidencias", "clientes", "tm_muerto", "tm_resolucion"]].values

# escalar salidas
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# redimensionar X para LSTM
X = X.reshape((X.shape[0], 1, X.shape[1]))

# dividir test y train
X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42)

# MODELO LSTM
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1], X.shape[2])),
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4)  # 4 salidas
])

model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# guardar modelos y transformadores
os.makedirs("models", exist_ok=True)
model.save("models/lstm_model.keras")
joblib.dump(ohe, "models/encoder_causa.pkl")
joblib.dump(scaler_y, "models/scaler_y.pkl")

# si no existe crear carpeta 
os.makedirs("outputs", exist_ok=True)

# entrenamiento vs validacion
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Pérdida durante el entrenamiento (LSTM)')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid()
plt.savefig("outputs/lstm_loss_curve.png")
plt.close()

# real vs predicho
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_orig = scaler_y.inverse_transform(y_test)

labels = ["Incidencias", "Clientes", "TM Muerto", "TM Resolución"]
maes = []
r2s = []
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

# graficar mae y r2
plt.figure(figsize=(10, 5))
plt.bar(labels, maes, color='orange')
plt.title('LSTM - MAE por variable de salida')
plt.ylabel('Mean Absolute Error')
plt.grid(axis='y')
plt.savefig("outputs/lstm_mae_comparativo.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.bar(labels, r2s, color='teal')
plt.title('LSTM - R² Score por variable de salida')
plt.ylabel('R²')
plt.ylim(0, 1)
plt.grid(axis='y')
plt.savefig("outputs/lstm_r2_comparativo.png")
plt.close()

print("Modelo LSTM entrenado, guardado y gráficas exportadas a 'outputs'.")
