import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, median_absolute_error
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

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

# Agrupar por fecha para obtener `incidencias`
df["FECHA"] = df["INICIO INCIDENCIA"].dt.date
incidencias_por_dia = df.groupby("FECHA").size().reset_index(name="incidencias")
df = df.merge(incidencias_por_dia, on="FECHA", how="left")

# Entradas (X)
X = df[[
    "hora_inicio", "mes", "dia_semana", "semana_del_anio", "minutos_respuesta",
    "CLIENTES", "TIEMPO MUERTO (MIN)", "TIEMPO RESOLUCION (MIN)"]].values

# Salidas (y)
y = df[["incidencias", "CLIENTES", "TIEMPO MUERTO (MIN)", "TIEMPO RESOLUCION (MIN)"]].values

# Escalado
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Redimensionar para LSTM
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# División entrenamiento/prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Crear carpeta outputs
os.makedirs("outputs", exist_ok=True)

# Comparar combinaciones de epochs y batch_size
epochs_list = [30, 60, 90]
batch_sizes = [16, 32, 64]
results = []

for epochs in epochs_list:
    for batch in batch_sizes:
        model_temp = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_scaled.shape[1], X_scaled.shape[2])),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(4)
        ])
        model_temp.compile(optimizer='adam', loss='mse')
        history_temp = model_temp.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch,
            validation_data=(X_test, y_test),
            verbose=0
        )
        val_loss = history_temp.history['val_loss'][-1]
        results.append({
            'epochs': epochs,
            'batch_size': batch,
            'val_loss': val_loss
        })

# Mejor configuración
df_results = pd.DataFrame(results)
pivot_table = df_results.pivot(index='epochs', columns='batch_size', values='val_loss')
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="YlGnBu")
plt.title("LSTM - Comparación de Val Loss por Configuración")
plt.xlabel("Batch Size")
plt.ylabel("Epochs")
plt.savefig("outputs/lstm_config_comparativa.png")
plt.close()

mejor = df_results.loc[df_results['val_loss'].idxmin()]
mejores_epochs = int(mejor['epochs'])
mejor_batch_size = int(mejor['batch_size'])

# Entrenamiento final
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_scaled.shape[1], X_scaled.shape[2])),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4)
])
model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, y_train, epochs=mejores_epochs, batch_size=mejor_batch_size, validation_data=(X_test, y_test))

# Guardar modelo y escaladores
os.makedirs("models", exist_ok=True)
model.save("models/lstm_model.keras")
joblib.dump(scaler_X, "models/scaler_X.pkl")
joblib.dump(scaler_y, "models/scaler_y.pkl")

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

# Predicciones
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_orig = scaler_y.inverse_transform(y_test)

# Métricas
titles = ["Incidencias", "Clientes", "TM Muerto", "TM Resolución"]
mses, maes, medaes, rmses, r2s_percent = [], [], [], [], []

for i in range(4):
    plt.figure(figsize=(8, 4))
    plt.scatter(y_test_orig[:, i], y_pred[:, i], alpha=0.5)
    plt.xlabel("Real")
    plt.ylabel("Predicho")
    plt.title(f"LSTM - Real vs Predicho - {titles[i]}")
    plt.grid()
    plt.savefig(f"outputs/lstm_real_vs_pred_{i}_{titles[i].replace(' ', '_').lower()}.png")
    plt.close()

    y_true = y_test_orig[:, i]
    y_pred_i = y_pred[:, i]
    mses.append(mean_squared_error(y_true, y_pred_i))
    maes.append(mean_absolute_error(y_true, y_pred_i))
    medaes.append(median_absolute_error(y_true, y_pred_i))
    rmses.append(np.sqrt(mean_squared_error(y_true, y_pred_i)))
    r2s_percent.append(r2_score(y_true, y_pred_i) * 100)

# DataFrame de métricas
df_metrics = pd.DataFrame({
    "Variable": titles,
    "MSE": mses,
    "MAE": maes,
    "MedAE": medaes,
    "RMSE": rmses,
    "R² (%)": r2s_percent
})

# Guardar CSV y PNG de la tabla
formatted_values = df_metrics.copy()
formatted_values[df_metrics.columns[1:]] = df_metrics[df_metrics.columns[1:]].applymap(lambda x: f"{x:.2f}")

# Crear y guardar la tabla como imagen
fig, ax = plt.subplots(figsize=(10, 2))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=formatted_values.values, colLabels=formatted_values.columns, cellLoc='center', loc='center')
table.scale(1, 2)
table.auto_set_font_size(False)
table.set_fontsize(10)
plt.tight_layout()
plt.savefig("outputs/gru_metrics_table.png", dpi=300)
plt.close()

# MAE comparativo
plt.figure(figsize=(10, 5))
plt.bar(titles, maes, color='purple')
plt.title('LSTM - MAE por variable de salida')
plt.ylabel('Mean Absolute Error')
plt.grid(axis='y')
plt.savefig("outputs/lstm_mae_comparativo.png")
plt.close()

# R² comparativo
plt.figure(figsize=(10, 5))
plt.bar(titles, r2s_percent, color='coral')
plt.title('LSTM - R² Score por variable de salida')
plt.ylabel('R² (%)')
plt.ylim(0, 100)
plt.grid(axis='y')
plt.savefig("outputs/lstm_r2_comparativo.png")
plt.close()

print("\nModelo LSTM entrenado, métricas calculadas y gráficas exportadas en 'outputs/'.")
