import pandas as pd
import numpy as np
import os
from collections import Counter


# --- CONFIGURACIÓN DE RUTAS ---
# BASE_DIR apunta a la carpeta raíz del proyecto (un nivel arriba de scripts)
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
INPUT_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'data_clean.csv')
OUTPUT_DRIFT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'drift_south_test_data.csv')


# --- PARÁMETROS DE SIMULACIÓN ---
# Aplicamos la deriva al 50% de los datos
DRIFT_SAMPLE_SIZE = 0.5
TARGET_COLUMN = 'kredit' # Columna objetivo


# --- DRIFT 1: Categórica (laufkont) - Desplazamiento al Riesgo ---
FEATURE_1_CAT = 'laufkont'
# La nueva distribución favorece los estados 1 (crítico) y 2 (otros bancos)
NEW_DISTRIBUTION_WEIGHTS = {
   1: 0.50, # Alta probabilidad de riesgo
   2: 0.30,
   3: 0.15,
   4: 0.05 
}


# --- DRIFT 2: Numérica (hoehe) - Desplazamiento a Montos Menores ---
FEATURE_2_NUM = 'hoehe'
# Reducción aleatoria del monto entre 10% y 30%
MIN_REDUCTION_FACTOR = 0.40
MAX_REDUCTION_FACTOR = 0.90


# --- DRIFT 3: Relacional/Concepto (laufzeit vs. verwendungszweck) ---
FEATURE_3_CONCEPT_A = 'verw'
FEATURE_3_CONCEPT_B = 'laufzeit'
SHORT_TERM_THRESHOLD = 36
LONG_TERM_FACTOR_MIN = 0.5
LONG_TERM_FACTOR_MAX = 2.5


# --- DRIFT 4: Target Objective (kredit) - Label Shift ---
# Forzamos que la clase positiva (1: mal crédito) represente el 50% en la muestra de deriva
NEW_TARGET_PROBABILITY_OF_ONE = 0.50




# --- CLASE DE MANEJO DE DATOS INVÁLIDOS (SIN CAMBIOS) ---
class InvalidDataHandler:


   def __init__(self, target_column_name: str, valid_values_map: dict, mixed_type_col: str = None):
       self.target_col = target_column_name
       self.rules = valid_values_map
       self.valid_rules = valid_values_map
       self.mixed_col = mixed_type_col
       self.cols_to_impute = list(valid_values_map.keys())


   def clean_and_transform(self, df_input : pd.DataFrame) -> pd.DataFrame:
       df_transformed = df_input.copy()
       df_transformed = self._apply_numeric_conversion_logic(df_transformed)
       df_transformed = self._apply_drop_logic(df_transformed)
       df_transformed = self._apply_impute_logic(df_transformed)
       return df_transformed
  
   def _apply_numeric_conversion_logic(self, df: pd.DataFrame) -> pd.DataFrame:
       data_temp = df.copy()
       non_numeric_summary = {}


       for col in data_temp.columns:
           temp_numeric = pd.to_numeric(data_temp[col], errors='coerce')
           non_numeric_elements = temp_numeric.isna()
           if non_numeric_elements.any():
               non_numeric_values = data_temp[col][non_numeric_elements]
               non_numeric_summary[col] = non_numeric_values.value_counts().to_dict()


       non_numeric_series = pd.Series(non_numeric_summary)
       all_unique_non_numeric_elements = []
       for col, count_dict in non_numeric_series.items():
           if col != self.mixed_col:
                all_unique_non_numeric_elements.extend(count_dict.keys())


       final_elements = list(set(all_unique_non_numeric_elements))
       data_clean = data_temp.replace(final_elements, np.nan)
      
       columnas_numericas = data_clean.columns.tolist()
       if self.mixed_col and self.mixed_col in columnas_numericas:
           columnas_numericas.remove(self.mixed_col)


       data_clean[columnas_numericas] = data_clean[columnas_numericas].apply(
           lambda x: pd.to_numeric(x, errors='coerce')
       )
      
       return data_clean




   def _apply_drop_logic(self, df: pd.DataFrame) -> pd.DataFrame:
       target_col = self.target_col
       if target_col not in df.columns:
           print(f"Error (Drop logic): La columna objetivo {target_col} no existe en el DataFrame.")
           return df.copy()
       # Elimina NaNs en el target y filas con valores fuera de {0, 1}
       df_clean_target = df[(~df[target_col].isnull()) & (df[target_col].isin([0, 1]))].copy()
       return df_clean_target


   def _apply_impute_logic(self, df: pd.DataFrame) -> pd.DataFrame:
       df_imputed = df.copy()
       for col, valid_values in self.valid_rules.items():
           if col in df_imputed.columns:
               invalid_mask = (~df_imputed[col].isin(valid_values)) | df_imputed[col].isnull()
               count_invalid = invalid_mask.sum()


               if count_invalid > 0:
                   imputation_mode_value = df_imputed.loc[df_imputed[col].isin(valid_values), col].mode()
                   if not imputation_mode_value.empty:
                       mode_value = imputation_mode_value.iloc[0]
                       df_imputed.loc[invalid_mask, col] = mode_value
       return df_imputed




# --- FUNCIONES DE DRIFT (DRIFT 1, 2, 3 sin cambios) ---


def apply_numeric_drift(series: pd.Series) -> pd.Series:
   """Aplica una reducción aleatoria al monto del crédito."""
   reduction_factors = np.random.uniform(MIN_REDUCTION_FACTOR, MAX_REDUCTION_FACTOR, size=len(series))
   return (series * reduction_factors).round().astype(int)


def apply_concept_drift(df_sample: pd.DataFrame) -> pd.DataFrame:
   """Simula la deriva de concepto aumentando la duración de los créditos de Educación y Negocios."""
   df_transformed = df_sample.copy()


   drift_mask = (df_transformed[FEATURE_3_CONCEPT_A].isin([3, 4])) & \
                (df_transformed[FEATURE_3_CONCEPT_B] < SHORT_TERM_THRESHOLD)
               
   num_drifted_rows = drift_mask.sum()
  
   if num_drifted_rows > 0:
       increment_factors = np.random.uniform(LONG_TERM_FACTOR_MIN, LONG_TERM_FACTOR_MAX, size=num_drifted_rows)
      
       df_transformed.loc[drift_mask, FEATURE_3_CONCEPT_B] = \
           (df_transformed.loc[drift_mask, FEATURE_3_CONCEPT_B] * increment_factors).round().astype(int)


   print(f"✓ Drift 3 (Concept Drift) aplicado: {num_drifted_rows} créditos ({FEATURE_3_CONCEPT_A}=3/4) fueron extendidos.")
  
   return df_transformed




# --- NUEVA FUNCIÓN DE DRIFT (DRIFT 4) ---


def apply_label_shift(df_sample: pd.DataFrame, target_col: str, new_prob_one: float) -> pd.DataFrame:
   """
   Aplica Label Shift (Target Drift) forzando una nueva proporción de la clase positiva (1).
   """
   df_transformed = df_sample.copy()
   num_rows = len(df_transformed)
   num_ones_needed = int(num_rows * new_prob_one)
  
   # 1. Crear un array de target values (0s)
   new_target_values = np.zeros(num_rows, dtype=int)
  
   # 2. Seleccionar índices aleatorios para convertirlos a 1
   indices_for_one = np.random.choice(
       num_rows,
       size=num_ones_needed,
       replace=False # Asegura que los índices sean únicos
   )
   new_target_values[indices_for_one] = 1
  
   # 3. Aplicar los nuevos valores al dataframe
   df_transformed[target_col] = new_target_values
  
   return df_transformed




def generate_multi_drift(df: pd.DataFrame) -> pd.DataFrame:
   """
   Genera un conjunto de datos de prueba con Data Drift y Target Shift combinados.
   """
   print(f"--- Simulación de Data Drift Múltiple y Target Shift ---")


   # Mostrar estadísticos originales
   original_rate_one = (df[TARGET_COLUMN] == 1).mean()
   print(f"\nProporción Original de 'kredit=1': {original_rate_one:.2f}")


   # 1. Separar los datos para aplicar la deriva
   drift_sample = df.sample(frac=DRIFT_SAMPLE_SIZE, random_state=42)
   remaining_data = df.drop(drift_sample.index)


   # 2. Aplicar DRIFT 1 (Categórica: laufkont)
   category_list = list(NEW_DISTRIBUTION_WEIGHTS.keys())
   weight_list = list(NEW_DISTRIBUTION_WEIGHTS.values())


   new_status_values = np.random.choice(
       category_list,
       size=len(drift_sample),
       p=weight_list
   ).astype(int)


   drift_sample[FEATURE_1_CAT] = new_status_values
   print("\n✓ Drift 1 (Categórica) aplicada a 'laufkont'.")




   # 3. Aplicar DRIFT 2 (Numérica: hoehe)
   drift_sample[FEATURE_2_NUM] = apply_numeric_drift(drift_sample[FEATURE_2_NUM])
   print("✓ Drift 2 (Numérica) aplicada a 'hoehe'.")




   # 4. Aplicar DRIFT 3 (Relacional/Concepto: laufzeit vs. verwendungszweck)
   drift_sample = apply_concept_drift(drift_sample)


  
   # 5. Aplicar DRIFT 4 (Target/Label Shift: kredit)
   drift_sample = apply_label_shift(
       drift_sample,
       TARGET_COLUMN,
       NEW_TARGET_PROBABILITY_OF_ONE
   )
   print(f"✓ Drift 4 (Label Shift) aplicada a '{TARGET_COLUMN}'. Nueva tasa: {NEW_TARGET_PROBABILITY_OF_ONE:.2f}")




   # 6. Unir los datos restantes con la muestra de deriva y mezclar
   drifted_df = pd.concat([remaining_data, drift_sample], ignore_index=True).sample(frac=1).reset_index(drop=True)


   # 7. Verificar las nuevas distribuciones globales
   print("\n\n--- Distribuciones Finales Globales (con Drift Parcial) ---")
  
   final_rate_one = (drifted_df[TARGET_COLUMN] == 1).mean()
   print(f"Proporción Final de 'kredit=1': {final_rate_one:.2f} (Original: {original_rate_one:.2f})")
  
   # Verificación de Deriva 1 (laufkont)
   print(f"\nDistribución Final de {FEATURE_1_CAT} (laufkont):")
   print(drifted_df[FEATURE_1_CAT].value_counts(normalize=True).sort_index())
  
   # Verificación de Deriva 2 (Hoehe)
   original_mean_hoehe = df[FEATURE_2_NUM].mean()
   drifted_mean_hoehe = drifted_df[FEATURE_2_NUM].mean()
   print(f"\nMedia Final de {FEATURE_2_NUM}: {drifted_mean_hoehe:.2f} (Original: {original_mean_hoehe:.2f})")


   return drifted_df


if __name__ == "__main__":
   try:
       print('Lectura de la base y configuración del entorno.')
      
       if 'PYTHONPATH' not in os.environ:
            os.environ['PYTHONPATH'] = '.'
       elif '.' not in os.environ['PYTHONPATH']:
           os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + os.pathsep + '.'


       data = pd.read_csv(INPUT_DATA_PATH, index_col=0)
       data = data.reset_index(drop=True)
      
       # --- LÓGICA DE LIMPIEZA DE DATOS ---
       print('\n--- Limpieza de datos (InvalidDataHandler) ---')
      
       VALID_VALUES_MAP = {
           'laufkont': [1, 2, 3, 4],
           'hoehe': data['hoehe'].unique().tolist(),
           'laufzeit': data['laufzeit'].unique().tolist(),
           'verw': data['verw'].unique().tolist(),
           # Debes expandir este mapa con todas tus features categóricas y sus valores válidos
       }
      
       cleaner = InvalidDataHandler(
           target_column_name=TARGET_COLUMN,
           valid_values_map=VALID_VALUES_MAP,
           mixed_type_col=None
       )


       data_clean = cleaner.clean_and_transform(data)
       print(f"Filas después de la limpieza: {len(data_clean)}")
      
       # -----------------------------------


       # Generar el dataset con deriva
       drifted_data = generate_multi_drift(data_clean)


       # Guardar el nuevo conjunto de datos de prueba
       drifted_data.to_csv(OUTPUT_DRIFT_PATH, index=False)
       print(f"\nConjunto de datos con 4 Derivas guardado en: {OUTPUT_DRIFT_PATH}")


   except FileNotFoundError:
       print(f"ERROR: No se encontró el archivo de datos en {INPUT_DATA_PATH}. Por favor, verifica el nombre del archivo y la ruta.")
   except Exception as e:
       print(f"Ocurrió un error durante la simulación: {e}")