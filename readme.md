# Proyecto: Predicción de Incidencias con LSTM


## Estructura del Proyecto
```
LSTM_Prediccion_Incidencias/
|
├── data/              # Datos originales (Excel)
│   └── EXCELINCIDENCIAS.xlsx
|
├── scripts/           # Códigos de entrenamiento y predicción
│   ├── train_ltsm.py
│   └── predict.py
|
├── models/            # Modelo entrenado y transformadores
│   ├── ltsm_model/     # Modelo de TensorFlow
│   ├── encoder_causa.pkl
│   └── scaler_y.pkl
|
└── outputs/           # Resultados, gráficas o logs (opcional)

```

---

## Entrenamiento del Modelo

El archivo `scripts/train_lstm.py` entrena una red MLP con los siguientes datos:
- **Input**: día de la semana (de la fecha), y causa (codificada)
- **Output**:
  - Número de incidencias
  - Número de clientes afectados
  - Tiempo promedio muerto (min)
  - Tiempo promedio de resolución (min)

### Ejecución:
```bash
cd scripts
python train_lstm.py
```
El modelo y los preprocesadores se guardarán en la carpeta `../models/`.

---

## Predicción

El archivo `scripts/predict.py` permite hacer una predicción ingresando:
- Fecha (YYYY-MM-DD)
- Causa de incidencia (texto)

### Ejecución:
```bash
cd scripts
python predict.py
```
Los resultados se mostrarán en consola.

---

## Requisitos

Instala las dependencias necesarias:

```bash
pip install tensorflow pandas numpy scikit-learn openpyxl joblib
```
