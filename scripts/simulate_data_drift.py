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

# --- CLASE DE MANEJO DE DATOS INVÁLIDOS ---
class InvalidDataHandler: 

    """
    Clase para detectar valores no válidos dentro de nuestra BD y realizar el proceso de imputación 
    de los datos inconsistentes a través de la moda (para el caso categórico)
    """

    # Se ajusta el constructor para que 'target_column_name' se guarde como 'target_col'
    def __init__(self, target_column_name: str, valid_values_map: dict, mixed_type_col: str = None):

        self.target_col = target_column_name # Renombrado para consistencia con _apply_drop_logic
        self.rules = valid_values_map
        self.valid_rules = valid_values_map
        self.mixed_col = mixed_type_col
        ## Detect the columns we're about to clean
        self.cols_to_impute = list(valid_values_map.keys())


    def clean_and_transform(self, df_input : pd.DataFrame) -> pd.DataFrame: 

        """
        Método para realizar la limpieza de los datos que no son válidos dentro de nuestro dataframe

        """
        df_transformed = df_input.copy()

        ## Conversión de los datos numéricos
        df_transformed = self._apply_numeric_conversion_logic(df_transformed)

        ## Limpieza particular para la variable objetivo
        df_transformed = self._apply_drop_logic(df_transformed)
        
        ## Limpieza para las variables features (variables características)
        df_transformed = self._apply_impute_logic(df_transformed)

        return df_transformed
    
    def _apply_numeric_conversion_logic(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        Identifica y reemplaza elementos no numéricos atípicos por NaN, 
        luego convierte las columnas relevantes a tipo numérico.
        """

        data_temp = df.copy()

        non_numeric_summary = {}

        # 1. Detección de elementos no numéricos
        for col in data_temp.columns:
            # Intentar convertir a numérico, forzando errores a NaN
            temp_numeric = pd.to_numeric(data_temp[col], errors='coerce')
            non_numeric_elements = temp_numeric.isna()

            if non_numeric_elements.any():
                non_numeric_values = data_temp[col][non_numeric_elements]
                # Solo contamos los elementos que *realmente* eran texto o caracteres atípicos
                non_numeric_summary[col] = non_numeric_values.value_counts().to_dict()

        non_numeric_series = pd.Series(non_numeric_summary)

        # 2. Identificar elementos a substituir por NaN (excluyendo la columna de tipo mixto)
        all_unique_non_numeric_elements = []
        for col, count_dict in non_numeric_series.items():
            if col != self.mixed_col:
                 all_unique_non_numeric_elements.extend(count_dict.keys())

        # 3. Reemplazar los valores por NaN
        final_elements = list(set(all_unique_non_numeric_elements))
        data_clean = data_temp.replace(final_elements, np.nan)
        
        # 4. Conversión final a tipo numérico
        columnas_numericas = data_clean.columns.tolist()
        if self.mixed_col and self.mixed_col in columnas_numericas:
            columnas_numericas.remove(self.mixed_col)

        data_clean[columnas_numericas] = data_clean[columnas_numericas].apply(
            lambda x: pd.to_numeric(x, errors='coerce')
        )
        
        return data_clean


    def _apply_drop_logic(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        Lógica para eliminar las filas que no tienen valores válidos. 
        Debido a que la variable objetivo que tenemos es uno de los elementos más importantes en el proceso de desarrollo del modelo, 
        en lugar de imputarla, como al resto de las variables, eliminaremos los valores no válidos, con el objetivo de no incluir ruido en nuestro 
        modelo.
        """

        target_col = self.target_col

        if target_col not in df.columns: 
            print(f"Error (Drop logic): La columna objetivo {target_col} no existe en el DataFrame.")
            return df.copy()

        # Asumiendo que la columna objetivo es binaria (0 o 1) y se deben eliminar NaNs y valores fuera de {0, 1}
        df_clean_target = df[(~df[target_col].isnull()) & (df[target_col].isin([0, 1]))].copy()

        return df_clean_target

    def _apply_impute_logic(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        Lógica para imputar los valores no válidos. Con la moda para el caso de las variables categóricas. 
        Nota: Se realiza la imputación de datos, pues el EDA arrojo que eliminar las variables nos dejaría con muy poca información para el entrenamiento. 
        """
        df_imputed = df.copy()

        for col, valid_values in self.valid_rules.items():
            if col in df_imputed.columns: 
                # Verificar si los valores de la columna son de tipo float/numérico (probablemente NaN) o fuera de los valid_values
                invalid_mask = (~df_imputed[col].isin(valid_values)) | df_imputed[col].isnull()
                count_invalid = invalid_mask.sum()

                if count_invalid > 0: 
                    # Calcula la moda solo de los valores válidos para evitar imputar con un valor que ya es inválido
                    imputation_mode_value = df_imputed.loc[df_imputed[col].isin(valid_values), col].mode()

                    if not imputation_mode_value.empty:
                        mode_value = imputation_mode_value.iloc[0]

                        ## Reemplazar los valores inválidos por la moda
                        df_imputed.loc[invalid_mask, col] = mode_value

        return df_imputed


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
        print('Lectura de la base')
        data = pd.read_csv(INPUT_DATA_PATH, index_col=0)

        # Reseteamos el index para tener la variable 'laufkont' (y la columna objetivo)
        data = data.reset_index()
        print(f"Filas iniciales: {len(data)}")

        # --- LÓGICA DE LIMPIEZA DE DATOS ---
        print('\n--- Limpieza de datos (InvalidDataHandler) ---')
        
        # Definir los valores válidos para las columnas relevantes. 
        # NOTA: Debes adaptar estos diccionarios a tu dataset real.
        VALID_VALUES_MAP = {
            'laufkont': [1, 2, 3, 4], # Valores válidos para FEATURE_1_CAT
            'hoehe': data['hoehe'].unique().tolist(), # Asumiendo que todos los valores iniciales son válidos, aunque _apply_numeric_conversion_logic se encargará de atípicos.
            # Agrega aquí otras columnas y sus valores válidos si es necesario
        }
        
        # Asumiendo que la columna objetivo es 'kredit' (o la que corresponda)
        TARGET_COLUMN = 'kredit'
        
        # Crear y aplicar el limpiador
        cleaner = InvalidDataHandler(
            target_column_name=TARGET_COLUMN, 
            valid_values_map=VALID_VALUES_MAP, 
            # Define 'mixed_type_col' si tienes una columna que deliberadamente contiene texto y números.
            mixed_type_col=None 
        )

        data_clean = cleaner.clean_and_transform(data)
        
        print(f"Filas después de la limpieza (Drop Logic): {len(data_clean)}")
        print(data_clean[FEATURE_1_CAT].value_counts())
        
        # -----------------------------------

        # Generar el dataset con deriva sobre los datos limpios
        drifted_data = generate_multi_drift(data_clean)

        # Guardar el nuevo conjunto de datos de prueba
        drifted_data.to_csv(OUTPUT_DRIFT_PATH, index=False)
        print(f"\nConjunto de datos con Data Drift Múltiple guardado en: {OUTPUT_DRIFT_PATH}")

    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo de datos en {INPUT_DATA_PATH}. Por favor, verifica el nombre del archivo y la ruta.")
    except Exception as e:
        print(f"Ocurrió un error durante la simulación: {e}")