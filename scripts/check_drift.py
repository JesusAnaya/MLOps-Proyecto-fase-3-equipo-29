import pandas as pd
import os
from evidently import Report
from evidently.metrics import *
from evidently.presets import *
import joblib 

# --- RUTA ---
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
# Datos de Referencia
INPUT_REF_DATA = os.path.join(BASE_DIR, 'data', 'processed', 'data_clean.csv') 
# Datos de Producción (Simulados)
INPUT_DRIFT_DATA = os.path.join(BASE_DIR, 'data', 'processed', 'drift_south_test_data.csv') 
REPORT_PATH = os.path.join(BASE_DIR, 'reports', 'evidently_drift_report.html')

# RUTA DEL MEJOR MODELO
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.joblib')

# Variable objetivo
TARGET_COLUMN = 'kredit' 
# Nombre de la columna de predicciones del modelo
PREDICTION_COLUMN = 'prediction' 

# --- VARIABLES DE INTERÉS PARA EL DRIFT ---
CRITICAL_DRIFT_FEATURES = ['laufkont', 'hoehe', 'verw', 'laufzeit']

def run_evidently_report():
    """Carga los datos y genera un reporte HTML interactivo con Evidently AI."""
    print("--- Inicializando Reporte Interactivo con Evidently AI ---")

    # 1. Cargar Datos
    try:
        ref_data = pd.read_csv(INPUT_REF_DATA)
        drift_data = pd.read_csv(INPUT_DRIFT_DATA)
        
        # Eliminar NaN en la columna objetivo si existen (necesario para la métrica de rendimiento)
        ref_data.dropna(subset=[TARGET_COLUMN], inplace=True)
        drift_data.dropna(subset=[TARGET_COLUMN], inplace=True)

        print(f"Datos de Referencia cargados: {len(ref_data)} filas")
        print(f"Datos de Producción cargados: {len(drift_data)} filas")
        
    except FileNotFoundError as e:
        print(f"Error al cargar datos: {e}. Asegúrate de que las rutas son correctas.")
        return

    # 2. Cargar y Simular la columna de Predicciones en Producción
    
    model = None
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Modelo cargado exitosamente desde: {MODEL_PATH}")
        
        # Usar solo las features necesarias para la predicción
        # Ajusta esta lista de features si tu modelo usa un subset diferente.
        # Asumimos que las columnas son todas las que no son el TARGET_COLUMN
        #INPUT_FEATURES = ref_data.drop(columns=[TARGET_COLUMN]).columns.tolist() 

        # Generar predicciones para los datos de producción
        #drift_data[PREDICTION_COLUMN] = model.predict(drift_data[INPUT_FEATURES])
        
    except FileNotFoundError:
        print(f"ERROR CRÍTICO: No se encontró el modelo en la ruta esperada: {MODEL_PATH}")
        print("El reporte se generará, pero las métricas de Rendimiento (ClassificationPreset) serán inexactas.")
        # Generar predicciones simuladas (inutilizables para rendimiento real, pero evita que Evidently falle)
        drift_data[PREDICTION_COLUMN] = drift_data[TARGET_COLUMN].apply(lambda x: 1 if x == 0 else 0)
    except Exception as e:
        print(f"ERROR: No se pudo cargar o usar el modelo. {e}")
        print("El reporte se generará, pero las métricas de Rendimiento (ClassificationPreset) serán inexactas.")
        # Fallback de predicciones
        drift_data[PREDICTION_COLUMN] = drift_data[TARGET_COLUMN].apply(lambda x: 1 if x == 0 else 0)

    # 3. Definir el Reporte
    data_monitoring_report = Report(metrics=[
        # 1. Monitoreo de Data Drift (AHORA SOLO PARA LAS VARIABLES CRÍTICAS ESPECIFICADAS)
        DataDriftPreset(
            features=CRITICAL_DRIFT_FEATURES 
        ),
        # 2. Monitoreo de Target Drift (Cambio en la distribución de la variable objetivo real, 'kredit')
        TargetDriftPreset(
            target_name=TARGET_COLUMN, # Es 'kredit'
            prediction_name=PREDICTION_COLUMN
        ),
        # 3. Monitoreo de Clasificación (Rendimiento del modelo: F1, ROC AUC, etc.)
        ClassificationPreset(
            target_name=TARGET_COLUMN, 
            prediction_name=PREDICTION_COLUMN,
            probas=None 
        )
    ])

    # 4. Generar el Reporte
    print("\nGenerando reporte, esto puede tardar unos segundos...")
    data_monitoring_report.run(
        reference_data=ref_data, 
        current_data=drift_data,
        column_mapping=None
    )

    # 5. Guardar el Reporte como HTML
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    data_monitoring_report.save_html(REPORT_PATH)
    
    print("\n================== REPORTE GENERADO ==================")
    print(f"Reporte HTML guardado exitosamente en: {REPORT_PATH}")
    print("¡Abre este archivo en tu navegador para ver el dashboard interactivo!")
    print("======================================================")

if __name__ == "__main__":
    run_evidently_report()