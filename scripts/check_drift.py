import pandas as pd
import os
from eventually.detector import ModelMonitor

# --- CONFIGURACIÓN DE RUTAS (Ajusta según tu proyecto) ---
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..') 
BASELINE_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'data_clean.csv') 
TARGET_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'drift_south_test_data.csv')

# --- DEFINICIÓN DE CONFIGURACIÓN ---
# Puedes definir qué columnas monitorear y qué métricas usar.
MONITOR_CONFIG = {
    'hoehe': ['PSI', 'KolmogorovSmirnov'], # Métricas para la columna numérica
    'laufkont': ['KolmogorovSmirnov'],    # Métrica para la columna categórica/ordinal
    # Agrega más columnas y métricas aquí
}

def run_drift_check():
    """
    Carga los datos, ejecuta el monitoreo de drift y genera el reporte.
    """
    try:
        df_baseline = pd.read_csv(BASELINE_PATH)
        df_target = pd.read_csv(TARGET_PATH)
    except FileNotFoundError:
        print("ERROR: No se encontraron los archivos de baseline o target.")
        return
    
    # 1. Inicializar el monitor
    monitor = ModelMonitor(
        baseline=df_baseline, 
        target=df_target,
        metric_configs=MONITOR_CONFIG
    )
    
    print("Iniciando monitoreo de Data Drift...")
    
    # 2. Ejecutar la detección
    monitor.fit()
    
    # 3. Generar el reporte
    drift_report = monitor.create_report()
    
    # 4. Imprimir resultados y guardar el reporte
    print("\n--- RESUMEN DEL DRIFT ---")
    print(drift_report.summary)
    
    # Guarda el reporte detallado en un archivo HTML o JSON para revisión
    report_file_path = os.path.join(BASE_DIR, 'reports', 'drift_report.html')
    # Asegúrate de que la carpeta 'reports' exista
    os.makedirs(os.path.dirname(report_file_path), exist_ok=True)
    
    with open(report_file_path, 'w') as f:
        f.write(drift_report.to_html())
        
    print(f"\nReporte HTML de Drift guardado en: {report_file_path}")

if __name__ == "__main__":
    run_drift_check()