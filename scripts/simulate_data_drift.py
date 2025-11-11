import pandas as pd
import numpy as np
import os
from collections import Counter

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
INPUT_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'data_clean.csv')
OUTPUT_DRIFT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'drift_south_test_data.csv')

# --- PARÁMETROS DE SIMULACIÓN ---
DRIFT_SAMPLE_SIZE = 0.4 # Aplicamos la deriva al 40% de los datos

# --- DRIFT 1: Categórica (laufkont) - Desplazamiento al Riesgo ---
FEATURE_1_CAT = 'laufkont'

NEW_DISTRIBUTION_WEIGHTS = {
    1: 0.50, 
    2: 0.25, 
    3: 0.15, 
    4: 0.10  
}

# --- DRIFT 2: Numérica (hoehe) - Desplazamiento a Montos Menores ---
FEATURE_2_NUM = 'hoehe'
# Simulamos que el banco es más conservador, reduciendo el monto promedio de los préstamos.
# Aplicaremos un factor de reducción aleatorio entre 10% y 30%.
MIN_REDUCTION_FACTOR = 0.70
MAX_REDUCTION_FACTOR = 0.90

def apply_numeric_drift(series: pd.Series) -> pd.Series:
    """Aplica una reducción aleatoria al monto del crédito."""
    # Genera un array de factores de escala (entre 0.7 y 0.9)
    reduction_factors = np.random.uniform(MIN_REDUCTION_FACTOR, MAX_REDUCTION_FACTOR, size=len(series))
    # Aplica la reducción y redondea al entero más cercano
    return (series * reduction_factors).round().astype(int)

def generate_multi_drift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un conjunto de datos de prueba con Data Drift combinada en 'laufkont' y 'hoehe'.
    """
    print(f"--- Simulación de Data Drift Múltiple ---")

    # Asegurarse de que 'laufkont' sea int para el muestreo categórico
    df[FEATURE_1_CAT] = df[FEATURE_1_CAT].astype(int)
    
    # 1. Mostrar distribuciones originales
    print("\nDistribución Original de laufkont:")
    print(df[FEATURE_1_CAT].value_counts(normalize=True).sort_index())
    print(f"\nEstadísticas Originales de {FEATURE_2_NUM} (Monto del Crédito):")
    print(df[FEATURE_2_NUM].describe())


    # 2. Separar los datos para aplicar la deriva
    drift_sample = df.sample(frac=DRIFT_SAMPLE_SIZE, random_state=42)
    remaining_data = df.drop(drift_sample.index)

    # 3. Aplicar DRIFT 1 (Categórica: laufkont)
    category_list = list(NEW_DISTRIBUTION_WEIGHTS.keys())
    weight_list = list(NEW_DISTRIBUTION_WEIGHTS.values())

    new_status_values = np.random.choice(
        category_list,
        size=len(drift_sample),
        p=weight_list
    ).astype(int)

    drift_sample[FEATURE_1_CAT] = new_status_values

    # 4. Aplicar DRIFT 2 (Numérica: hoehe)
    drift_sample[FEATURE_2_NUM] = apply_numeric_drift(drift_sample[FEATURE_2_NUM])

    # 5. Unir los datos restantes con la muestra de deriva y mezclar
    drifted_df = pd.concat([remaining_data, drift_sample], ignore_index=True).sample(frac=1).reset_index(drop=True)

    # 6. Verificar las nuevas distribuciones globales
    print("\n\n--- Distribuciones Finales con Drift (Parcial) ---")
    print(f"\nDistribución Final de {FEATURE_1_CAT} (laufkont):")
    print(drifted_df[FEATURE_1_CAT].value_counts(normalize=True).sort_index())
    print(f"\nEstadísticas Finales de {FEATURE_2_NUM} (Monto del Crédito):")
    print(drifted_df[FEATURE_2_NUM].describe())
    
    # Mostrar la diferencia en la media como indicador de drift
    original_mean = df[FEATURE_2_NUM].mean()
    drifted_mean = drifted_df[FEATURE_2_NUM].mean()
    percentage_change = ((drifted_mean - original_mean) / original_mean) * 100
    print(f"\nCambio de media en '{FEATURE_2_NUM}': {original_mean:.2f} -> {drifted_mean:.2f} ({percentage_change:.2f}%)")


    return drifted_df

if __name__ == "__main__":
    try:
        data = pd.read_csv(INPUT_DATA_PATH, index_col=0)

        # Generar el dataset con deriva
        drifted_data = generate_multi_drift(data)

        # Guardar el nuevo conjunto de datos de prueba
        drifted_data.to_csv(OUTPUT_DRIFT_PATH, index=False)
        print(f"\nConjunto de datos con Data Drift Múltiple guardado en: {OUTPUT_DRIFT_PATH}")

    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo de datos en {INPUT_DATA_PATH}. Por favor, verifica el nombre del archivo y la ruta.")
    except Exception as e:
        print(f"Ocurrió un error durante la simulación: {e}")