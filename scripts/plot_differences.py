import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURACIÓN DE RUTAS ---
# Directorio base para encontrar los datos (asume que este script está en una subcarpeta)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(CURRENT_DIR, '..') 
ORIGINAL_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'data_clean.csv')
DRIFTED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'drift_south_test_data.csv')

# --- PARÁMETROS DE PLOTEO ---
VARIABLE_TO_PLOT = 'hoehe'
# Definición de la ruta donde se guardará el archivo
OUTPUT_PLOT_PATH = os.path.join(CURRENT_DIR, f'comparison_distribution_{VARIABLE_TO_PLOT}.png')


def plot_distribution_comparison(original_path: str, drifted_path: str, column_name: str, output_path: str):
    """
    Carga dos DataFrames y plotea la distribución (histograma + KDE) de una columna específica para compararlas,
    guardando el resultado en un archivo.
    """
    try:
        print(f"Cargando datos originales desde: {original_path}")
        df_original = pd.read_csv(original_path)
        
        print(f"Cargando datos con drift desde: {drifted_path}")
        df_drifted = pd.read_csv(drifted_path)

    except FileNotFoundError as e:
        print(f"ERROR: No se encontró el archivo: {e.filename}. Asegúrate de que las rutas y nombres de archivo sean correctos.")
        return
    except Exception as e:
        print(f"Ocurrió un error al cargar los datos: {e}")
        return

    if column_name not in df_original.columns or column_name not in df_drifted.columns:
        print(f"ERROR: La columna '{column_name}' no se encuentra en uno o ambos DataFrames.")
        return

    # --- Configuración del Plot ---
    plt.figure(figsize=(10, 6))
    
    # Plotear la distribución del dataset Original
    sns.histplot(
        df_original[column_name], 
        kde=True, 
        stat="density", 
        linewidth=0,
        color="blue",
        label="Datos Originales (data_clean.csv)", 
        alpha=0.4
    )
    
    # Plotear la distribución del dataset con Drift
    sns.histplot(
        df_drifted[column_name], 
        kde=True, 
        stat="density", 
        linewidth=0,
        color="red",
        label="Datos con Drift (drift_south_test_data.csv)", 
        alpha=0.4
    )

    # Añadir título y etiquetas
    plt.title(f'Comparación de Distribución de la Variable: {column_name.upper()} (Monto del Crédito)', fontsize=14)
    plt.xlabel(column_name.capitalize(), fontsize=12)
    plt.ylabel('Densidad', fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout() # Ajusta el layout para que no se corten los elementos
    
    # --- CAMBIO CLAVE: Guardar el archivo en lugar de mostrarlo ---
    plt.savefig(output_path)
    plt.close() # Cierra la figura para liberar memoria
    
    print(f"\nPlot guardado exitosamente en: {output_path}")
    
    # Imprimir estadísticas clave para referencia
    print("\n--- Estadísticas Descriptivas ---")
    print("Datos Originales:")
    print(df_original[column_name].describe())
    print("\nDatos con Drift:")
    print(df_drifted[column_name].describe())


if __name__ == "__main__":
    plot_distribution_comparison(
        original_path=ORIGINAL_DATA_PATH,
        drifted_path=DRIFTED_DATA_PATH,
        column_name=VARIABLE_TO_PLOT,
        output_path=OUTPUT_PLOT_PATH # Pasamos la ruta de salida
    )