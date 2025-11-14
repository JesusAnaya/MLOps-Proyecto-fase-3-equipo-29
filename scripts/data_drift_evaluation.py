import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score

# --- Configuración de Rutas ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ruta de entrada: El archivo de deriva que CONTIENE X y Y
DRIFTED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'drift_south_test_data.csv')

# Rutas del modelo y preprocesador ya entrenados
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.joblib')
PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'models', 'preprocessor.joblib')

# Ruta de salida para las métricas de deriva
OUTPUT_PATH = os.path.join(BASE_DIR, 'models', 'drift_results.json')

# Nombre de la variable objetivo (Asegúrate de que este nombre sea CORRECTO)
TARGET_COLUMN = 'kredit'

def load_data_and_assets():
   """Carga los datos de deriva, el modelo y el preprocesador, y separa X e Y."""
   print("Cargando modelo y datos...")
  
   # 1. Cargar Assets (Modelo y Preprocesador)
   try:
       model = joblib.load(MODEL_PATH)
       preprocessor = joblib.load(PREPROCESSOR_PATH)
       print(f"Modelo cargado desde: {MODEL_PATH}")
   except FileNotFoundError:
       print("ERROR: Asegúrate de haber ejecutado 'v run mlops-train' primero. Faltan best_model.joblib o preprocessor.joblib.")
       return None, None, None, None


   # 2. Cargar y Separar el Conjunto de Datos de Deriva
   try:
       # Cargar el archivo único de deriva (que incluye la columna TARGET_COLUMN)
       df_drift = pd.read_csv(DRIFTED_DATA_PATH)
       print(f"Datos de deriva cargados desde: {DRIFTED_DATA_PATH}. Shape: {df_drift.shape}")
      
       # Separar características (X) y etiqueta (y)
       X_drift = df_drift.drop(columns=[TARGET_COLUMN])
       y_drift = df_drift[TARGET_COLUMN]
      
       print(f"Datos separados. X (Características) shape: {X_drift.shape}, y (Etiquetas) shape: {y_drift.shape}")
      
   except FileNotFoundError:
       print(f"ERROR: No se encontró el archivo de deriva: {DRIFTED_DATA_PATH}.")
       return None, None, None, None
   except KeyError:
       print(f"ERROR: No se encontró la columna objetivo '{TARGET_COLUMN}' en el archivo de deriva. Verifica el nombre de la columna.")
       return None, None, None, None


   return model, preprocessor, X_drift, y_drift


def evaluate_drift(model, preprocessor, X_drift, y_drift):
   """Preprocesa los datos y evalúa el modelo."""
  
   # 1. Aplicar Preprocesamiento a los nuevos datos
   print("Aplicando preprocesador a los datos de deriva...")
   X_drift_processed = preprocessor.transform(X_drift)
  
   # 2. Hacer Predicciones
   print("Realizando predicciones...")
   y_pred = model.predict(X_drift) # (X_drift_processed)
   y_proba = model.predict_proba(X_drift)[:, 1] if hasattr(model, 'predict_proba') else None


   # 3. Calcular Métricas
   print("Calculando métricas de desempeño...")
   metrics = {
       "accuracy": accuracy_score(y_drift, y_pred),
       "f1_score": f1_score(y_drift, y_pred),
       "recall": recall_score(y_drift, y_pred),
       "precision": precision_score(y_drift, y_pred),
   }
  
   if y_proba is not None:
       metrics["roc_auc_score"] = roc_auc_score(y_drift, y_proba)


   return metrics


def save_metrics(metrics):
   """Guarda las métricas en un archivo JSON."""
   os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True) # Asegura que la carpeta models exista
   with open(OUTPUT_PATH, 'w') as f:
       json.dump(metrics, f, indent=4)
   print(f"\nMétricas de desempeño con Deriva guardadas en: {OUTPUT_PATH}")


if __name__ == '__main__':
   model, preprocessor, X_drift, y_drift = load_data_and_assets()
  
   if model is not None and X_drift is not None:


       print('Evaluación del data drift')
       drift_metrics = evaluate_drift(model, preprocessor, X_drift, y_drift)


       print('Almacenando las métricas')
       save_metrics(drift_metrics)


       print("\n--- Resultados de la Evaluación con Deriva (DRIFT) ---")
       for metric, value in drift_metrics.items():
           print(f"{metric.ljust(15)}: {value:.4f}")
       print("-------------------------------------------------------")
