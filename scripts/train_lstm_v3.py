import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# === CARGA Y PREPROCESAMIENTO ===

df = pd.read_excel("data/EXCELINCIDENCIAS.xlsx", sheet_name="Sheet1")
df["INICIO INCIDENCIA"] = pd.to_datetime(df["INICIO INCIDENCIA"])
df["HORA DE LLEGADA"] = pd.to_datetime(df["HORA DE LLEGADA"])
df["CIERRE DE INCIDENCIA"] = pd.to_datetime(df["CIERRE DE INCIDENCIA"])
df["hora_inicio"] = df["INICIO INCIDENCIA"].dt.hour
df["mes"] = df["INICIO INCIDENCIA"].dt.month
df["dia_semana"] = df["INICIO INCIDENCIA"].dt.weekday
df["semana_del_anio"] = df["INICIO INCIDENCIA"].dt.isocalendar().week.astype(int)
df["minutos_respuesta"] = (df["HORA DE LLEGADA"] - df["INICIO INCIDENCIA"]).dt.total_seconds() / 60

df["FECHA"] = df["INICIO INCIDENCIA"].dt.date
incidencias_por_dia = df.groupby("FECHA").size().reset_index(name="incidencias")
df = df.merge(incidencias_por_dia, on="FECHA", how="left")

X = df[["hora_inicio", "mes", "dia_semana", "semana_del_anio", "minutos_respuesta"]].values
y = df[["incidencias", "CLIENTES"]].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# === CREAR CARPETAS DE SALIDA ===
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# === GRAFICAS DE ESCALADO ===
plt.figure(figsize=(10, 5))
plt.plot(X_scaled.reshape(X_scaled.shape[0], -1))
plt.title("X escalado")
plt.grid()
plt.savefig("outputs/x_scaled_plot.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(y_scaled)
plt.title("y escalado (incidencias y clientes)")
plt.grid()
plt.savefig("outputs/y_scaled_plot.png")
plt.close()

# === ENTRENAMIENTO PRINCIPAL ===
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_scaled.shape[1], X_scaled.shape[2])),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2)
])
model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

model.save("models/lstm_model_2targets.keras")
joblib.dump(scaler_X, "models/scaler_X_2targets.pkl")
joblib.dump(scaler_y, "models/scaler_y_2targets.pkl")

# === CURVA DE PERDIDA ===
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Curva de pérdida (100 épocas, batch 32)')
plt.xlabel('Época')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid()
plt.savefig("outputs/lstm_loss_curve_2targets.png")
plt.close()

# === PREDICCION Y GRAFICAS ===
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_orig = scaler_y.inverse_transform(y_test)

labels = ["Incidencias", "Clientes"]
maes, mses, r2s = [], [], []

for i in range(2):
    plt.figure(figsize=(8, 4))
    plt.scatter(y_test_orig[:, i], y_pred[:, i], alpha=0.5)
    plt.xlabel("Real")
    plt.ylabel("Predicho")
    plt.title(f"Real vs Predicho - {labels[i]}")
    plt.grid()
    plt.savefig(f"outputs/lstm_real_vs_pred_{labels[i].lower()}.png")
    plt.close()

    maes.append(mean_absolute_error(y_test_orig[:, i], y_pred[:, i]))
    mses.append(mean_squared_error(y_test_orig[:, i], y_pred[:, i]))
    r2s.append(r2_score(y_test_orig[:, i], y_pred[:, i]))

# === GRAFICAS DE METRICAS ===
plt.figure(figsize=(10, 5))
plt.bar(labels, maes, color='orange')
plt.title('MAE por variable')
plt.grid(axis='y')
plt.savefig("outputs/lstm_mae_2targets.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.bar(labels, mses, color='red')
plt.title('MSE por variable')
plt.grid(axis='y')
plt.savefig("outputs/lstm_mse_2targets.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.bar(labels, r2s, color='teal')
plt.title('R² por variable')
plt.ylim(0, 1)
plt.grid(axis='y')
plt.savefig("outputs/lstm_r2_2targets.png")
plt.close()

# === COMPARACION DE CONFIGURACIONES ===
configs = [
    {"epochs": 50, "batch_size": 16},
    {"epochs": 100, "batch_size": 32},
    {"epochs": 150, "batch_size": 64}
]

results = []
for config in configs:
    temp_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_scaled.shape[1], X_scaled.shape[2])),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2)
    ])
    temp_model.compile(optimizer='adam', loss='mse')
    temp_model.fit(X_train, y_train, epochs=config["epochs"], batch_size=config["batch_size"],
                   validation_data=(X_test, y_test), verbose=0)

    y_pred_temp = scaler_y.inverse_transform(temp_model.predict(X_test))
    y_test_temp = scaler_y.inverse_transform(y_test)

    results.append({
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "mae": mean_absolute_error(y_test_temp, y_pred_temp),
        "mse": mean_squared_error(y_test_temp, y_pred_temp),
        "r2": r2_score(y_test_temp, y_pred_temp)
    })

results_df = pd.DataFrame(results)
results_df["label"] = results_df["epochs"].astype(str) + "-" + results_df["batch_size"].astype(str)
results_df.to_csv("outputs/comparacion_configuraciones.csv", index=False)

# === GRAFICA COMPARATIVA DOBLE EJE ===
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.set_xlabel("Configuración (Épocas-Batch)")
ax1.set_ylabel("MAE / MSE", color='black')
ax1.plot(results_df["label"], results_df["mae"], marker='o', label="MAE", color='tab:blue')
ax1.plot(results_df["label"], results_df["mse"], marker='o', label="MSE", color='tab:orange')
ax1.tick_params(axis='y')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.set_ylabel("R²", color='tab:green')
ax2.plot(results_df["label"], results_df["r2"], marker='o', label="R²", color='tab:green')
ax2.tick_params(axis='y', labelcolor='tab:green')   
ax2.set_ylim(0, 1)

fig.suptitle("Comparación de Configuraciones (Épocas-Batch)")
fig.tight_layout()
plt.grid(True)
plt.savefig("outputs/comparacion_configuraciones_doble_eje.png")
plt.close()
