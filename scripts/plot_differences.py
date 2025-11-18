import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, chi2_contingency

# --- CONFIGURACIÓN DE RUTAS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(CURRENT_DIR, '..') 
ORIGINAL_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'data_clean.csv')
DRIFTED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'drift_south_test_data.csv')

# Variables categóricas y continuas
CATEGORICAL_VARS = {
    'laufkont': [1, 2, 3, 4],
    'verw': [0, 1, 2, 3]
}
CONTINUOUS_VARS = ['laufzeit', 'hoehe']

# Carpeta de salida para plots
OUTPUT_PLOTS_DIR = os.path.join(CURRENT_DIR, 'plots')
os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)


def clean_categorical(df, col, valid_values):
    """Imputa valores no válidos con la moda."""
    mode_value = df[col][df[col].isin(valid_values)].mode()[0]
    df[col] = df[col].apply(lambda x: x if x in valid_values else mode_value)
    return df


def clean_continuous(df, col):
    """Imputa nulos con la media y outliers usando IQR reemplazando por la media."""
    # Imputar NaN con la media
    mean_value = df[col].mean()
    df[col] = df[col].fillna(mean_value)
    
    # Detectar outliers con IQR y reemplazar por la media
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    df[col] = df[col].apply(lambda x: mean_value if x < lower or x > upper else x)
    return df


def plot_distribution_comparison(original_path, drifted_path, output_dir):
    # Cargar datasets
    df_orig = pd.read_csv(original_path)
    df_drift = pd.read_csv(drifted_path)

    # --- LIMPIEZA ---
    for col, valid_values in CATEGORICAL_VARS.items():
        df_orig = clean_categorical(df_orig, col, valid_values)
        df_drift = clean_categorical(df_drift, col, valid_values)

    for col in CONTINUOUS_VARS:
        df_orig = clean_continuous(df_orig, col)
        df_drift = clean_continuous(df_drift, col)

    # --- ANÁLISIS Y PLOTS ---
    for col in CATEGORICAL_VARS.keys():
        # Tabla de contingencia
        contingency = pd.crosstab(df_orig[col], df_drift[col])
        chi2, p, dof, ex = chi2_contingency(contingency)
        print(f"\n--- Chi-Cuadrado: {col} ---")
        print("Chi2 statistic:", chi2)
        print("p-value:", p)

        # Plot de barras
        plt.figure(figsize=(8, 5))
        sns.countplot(x=col, data=df_orig, color="blue", alpha=0.5, label="Original")
        sns.countplot(x=col, data=df_drift, color="red", alpha=0.5, label="Drift")
        plt.title(f'{col.upper()} (Chi2={chi2:.3f}, p={p:.3f})')
        plt.legend(["Original", "Drift"])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'categorical_{col}.png'))
        plt.close()
        print(f"Plot guardado: categorical_{col}.png")

    for col in CONTINUOUS_VARS:
        # Convertir a float y eliminar infinitos
        data_orig = pd.to_numeric(df_orig[col], errors='coerce')
        data_orig = data_orig[np.isfinite(data_orig)]

        data_drift = pd.to_numeric(df_drift[col], errors='coerce')
        data_drift = data_drift[np.isfinite(data_drift)]

        # KS test solo si hay datos válidos
        if len(data_orig) > 0 and len(data_drift) > 0:
            ks_stat, ks_pvalue = ks_2samp(data_orig, data_drift)
        else:
            ks_stat, ks_pvalue = np.nan, np.nan

        print(f"\n--- K-S Test: {col} ---")
        print("KS statistic:", ks_stat)
        print("p-value:", ks_pvalue)

        # Plot distribución
        plt.figure(figsize=(10, 6))
        sns.histplot(data_orig, kde=True, stat="density", linewidth=0,
                     color="blue", label="Datos Originales", alpha=0.4)
        sns.histplot(data_drift, kde=True, stat="density", linewidth=0,
                     color="red", label="Datos con Drift", alpha=0.4)
        plt.title(f'{col.upper()} (KS={ks_stat:.3f}, p={ks_pvalue:.3f})', fontsize=14)
        plt.xlabel(col.capitalize())
        plt.ylabel('Densidad')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'continuous_{col}.png'))
        plt.close()
        print(f"Plot guardado: continuous_{col}.png")


if __name__ == "__main__":
    plot_distribution_comparison(ORIGINAL_DATA_PATH, DRIFTED_DATA_PATH, OUTPUT_PLOTS_DIR)