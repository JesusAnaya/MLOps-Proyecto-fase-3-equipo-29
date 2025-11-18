import pandas as pd
import numpy as np
import os
from collections import Counter

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
INPUT_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'data_clean.csv')
OUTPUT_DRIFT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'drift_south_test_data.csv')

# --- PARÁMETROS DE SIMULACIÓN ---
DRIFT_SAMPLE_SIZE = 0.5
TARGET_COLUMN = 'kredit'  # Columna objetivo

# --- DRIFT 1: Categórica (laufkont) ---
FEATURE_1_CAT = 'laufkont'
NEW_DISTRIBUTION_WEIGHTS = {1: 0.50, 2: 0.30, 3: 0.15, 4: 0.05}

# --- DRIFT 2: Numérica (hoehe) ---
FEATURE_2_NUM = 'hoehe'
MIN_REDUCTION_FACTOR = 0.40
MAX_REDUCTION_FACTOR = 0.90

# --- DRIFT 3: Conceptual (laufzeit vs. verw) ---
FEATURE_3_CONCEPT_A = 'verw'
FEATURE_3_CONCEPT_B = 'laufzeit'
SHORT_TERM_THRESHOLD = 36
LONG_TERM_FACTOR_MIN = 0.5
LONG_TERM_FACTOR_MAX = 2.5

# --- DRIFT 4: Label Shift (kredit) ---
NEW_TARGET_PROBABILITY_OF_ONE = 0.50

# --- DRIFT 5: Categórica (verw) ---
FEATURE_5_CAT = 'verw'
NEW_DISTRIBUTION_WEIGHTS_VERW = {0: 0.10, 1: 0.20, 2: 0.40, 3: 0.30}

# --- FUNCIONES DE DRIFT ---
def apply_numeric_drift(series: pd.Series) -> pd.Series:
    """Aplica reducción aleatoria a hoehe."""
    reduction_factors = np.random.uniform(MIN_REDUCTION_FACTOR, MAX_REDUCTION_FACTOR, size=len(series))
    return (series * reduction_factors).round().astype(int)

def apply_mean_increase(series: pd.Series, increase_factor: float = 0.3) -> pd.Series:
    """Aumenta la media de la serie en un porcentaje dado."""
    mean_original = series.mean()
    target_mean = mean_original * (1 + increase_factor)
    shift = target_mean - mean_original
    return (series + shift).round().astype(int)


def apply_concept_drift(df_sample: pd.DataFrame) -> pd.DataFrame:
    """Simula la deriva de concepto aumentando la duración de ciertos créditos."""
    df_transformed = df_sample.copy()
    drift_mask = (df_transformed[FEATURE_3_CONCEPT_A].isin([3, 4])) & \
                 (df_transformed[FEATURE_3_CONCEPT_B] < SHORT_TERM_THRESHOLD)
    num_drifted_rows = drift_mask.sum()

    if num_drifted_rows > 0:
        increment_factors = np.random.uniform(LONG_TERM_FACTOR_MIN, LONG_TERM_FACTOR_MAX, size=num_drifted_rows)
        df_transformed.loc[drift_mask, FEATURE_3_CONCEPT_B] = \
            (df_transformed.loc[drift_mask, FEATURE_3_CONCEPT_B] * increment_factors).round().astype(int)

    print(f"✓ Drift Conceptual aplicado: {num_drifted_rows} filas afectadas.")
    return df_transformed


def apply_label_shift(df_sample: pd.DataFrame, target_col: str, new_prob_one: float) -> pd.DataFrame:
    """Aplica Label Shift para forzar una proporción de clase positiva."""
    df_transformed = df_sample.copy()
    num_rows = len(df_transformed)
    num_ones_needed = int(num_rows * new_prob_one)

    new_target_values = np.zeros(num_rows, dtype=int)
    indices_for_one = np.random.choice(num_rows, size=num_ones_needed, replace=False)
    new_target_values[indices_for_one] = 1

    df_transformed[target_col] = new_target_values
    return df_transformed

# --- GENERACIÓN DEL DRIFT ---
def generate_multi_drift(df: pd.DataFrame) -> pd.DataFrame:
    print(f"--- Simulación de Data Drift Múltiple y Target Shift ---")
    original_rate_one = (df[TARGET_COLUMN] == 1).mean()
    print(f"\nProporción Original de '{TARGET_COLUMN}=1': {original_rate_one:.2f}")

    drift_sample = df.sample(frac=DRIFT_SAMPLE_SIZE, random_state=42)
    remaining_data = df.drop(drift_sample.index)

    # DRIFT 1: Laufkont
    new_status_values = np.random.choice(list(NEW_DISTRIBUTION_WEIGHTS.keys()),
                                         size=len(drift_sample),
                                         p=list(NEW_DISTRIBUTION_WEIGHTS.values())).astype(int)
    drift_sample[FEATURE_1_CAT] = new_status_values
    print("✓ Drift categórico aplicado a 'laufkont'.")

    # DRIFT 2: Hoehe
    drift_sample[FEATURE_2_NUM] = apply_numeric_drift(drift_sample[FEATURE_2_NUM])
    print("✓ Drift numérico aplicado a 'hoehe'.")

    # DRIFT 3: Conceptual
    drift_sample = apply_concept_drift(drift_sample)

    # DRIFT 5: Verw
    new_verw_values = np.random.choice(list(NEW_DISTRIBUTION_WEIGHTS_VERW.keys()),
                                       size=len(drift_sample),
                                       p=list(NEW_DISTRIBUTION_WEIGHTS_VERW.values())).astype(int)
    drift_sample[FEATURE_5_CAT] = new_verw_values
    print("✓ Drift categórico aplicado a 'verw'.")

    # DRIFT 7: Laufzeit
    drift_sample['laufzeit'] = apply_mean_increase(drift_sample['laufzeit'], increase_factor=0.3)
    print("✓ Drift numérico aplicado a 'laufzeit' (+30% media).")

    # DRIFT 4: Label shift
    drift_sample = apply_label_shift(drift_sample, TARGET_COLUMN, NEW_TARGET_PROBABILITY_OF_ONE)
    print(f"✓ Drift de label aplicado a '{TARGET_COLUMN}'.")

    # Unir los datos y mezclar
    drifted_df = pd.concat([remaining_data, drift_sample], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    # Estadísticas finales
    print("\n--- Distribuciones finales ---")
    print(f"{FEATURE_1_CAT}:\n{drifted_df[FEATURE_1_CAT].value_counts(normalize=True)}")
    print(f"{FEATURE_5_CAT}:\n{drifted_df[FEATURE_5_CAT].value_counts(normalize=True)}")
    print(f"Media 'laufzeit': {drifted_df['laufzeit'].mean():.2f} (original: {df['laufzeit'].mean():.2f})")
    print(f"Media 'hoehe': {drifted_df['hoehe'].mean():.2f} (original: {df['hoehe'].mean():.2f})")
    print(f"Proporción 'kredit=1': {drifted_df[TARGET_COLUMN].mean():.2f}")

    return drifted_df

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    try:
        data = pd.read_csv(INPUT_DATA_PATH)
        data = data.reset_index(drop=True)

        # --- LIMPIEZA BÁSICA (ejemplo: imputar nulos con moda para categóricas y media para numéricas) ---
        for col in ['laufkont', 'verw']:
            data[col].fillna(data[col].mode()[0], inplace=True)
        for col in ['laufzeit', 'hoehe']:
            data[col].fillna(data[col].mean(), inplace=True)

        # Generar dataset con drift
        drifted_data = generate_multi_drift(data)

        # Guardar
        drifted_data.to_csv(OUTPUT_DRIFT_PATH, index=False)
        print(f"\nConjunto de datos con drift guardado en: {OUTPUT_DRIFT_PATH}")

    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo de datos en {INPUT_DATA_PATH}")
    except Exception as e:
        print(f"Ocurrió un error: {e}")