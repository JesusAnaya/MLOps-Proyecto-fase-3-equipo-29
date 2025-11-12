import pandas as pd
import os
from mlops_project.dataset import load_and_prepare_data # Asumimos que esta función te da los datos limpios de train
from mlops_project.features import prepare_features     # Asumimos que tiene la lógica del preprocesador
from mlops_project.modeling.train import train_model
from mlops_project.modeling.predict import predict_and_evaluate

# --- NUEVA CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..') 
DRIFTED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'drift_south_test_data.csv')
TARGET_COLUMN = 'kredit' # Asegúrate de que este es el nombre de tu columna objetivo

# --- PRIMEROS PASOS (Entrenamiento y Preprocesador) ---

# 1. Preparar datos originales (solo necesitamos el X_train y y_train para el entrenamiento y el preprocesador)
X_train, _, y_train, _ = load_and_prepare_data(
    filepath="data/raw/german_credit_modified.csv",
    save_processed=True
)

# 2. Preparar features (Se necesita X_test vacío o un placeholder para definir el preprocessor)
# Usaremos una copia de X_train como placeholder, aunque la función prepare_features 
# idealmente solo ajusta el preprocesador con X_train. 
X_train_t, X_test_placeholder_t, preprocessor = prepare_features(
    X_train=X_train,
    X_test=X_train.head(1), # Placeholder para la función, ya que no usaremos X_test
    save_preprocessor=True
)

# 3. Entrenar modelo 
pipeline, results = train_model(
    X_train=X_train,
    y_train=y_train,
    preprocessor=preprocessor,
    model_name="logistic_regression",
    use_smote=True,
    evaluate=True,
    save_model=True
)

print("\n--- Evaluando con Base de Data Drift ---")

# 4a. Cargar la base de Data Drift
try:
    df_drifted = pd.read_csv(DRIFTED_DATA_PATH)
    
    # Separar características (X) y target (y)
    X_drift = df_drifted.drop(columns=[TARGET_COLUMN])
    y_drift = df_drifted[TARGET_COLUMN]

except FileNotFoundError:
    print(f"ERROR: No se encontró el archivo de Data Drift en {DRIFTED_DATA_PATH}.")
    exit()
except KeyError:
    print(f"ERROR: La columna objetivo '{TARGET_COLUMN}' no se encontró en el archivo de Data Drift.")
    exit()

# 4b. Evaluar con los datos de drift
y_pred_drift, y_proba_drift, metrics_drift = predict_and_evaluate(
    model=pipeline,
    X_test=X_drift,
    y_test=y_drift,
    # Se recomienda cambiar el nombre del archivo de predicciones guardado para reflejar que es el set de drift
    save_predictions_name="drift_predictions.csv", 
    save_predictions=True
)

# 5. Mostrar resultados
print(f"\nMétricas de Evaluación (Data Drift - {len(X_drift)} muestras):")
print(f"ROC-AUC: {metrics_drift['roc_auc']:.4f}")
print(f"F1-Score: {metrics_drift['f1']:.4f}")