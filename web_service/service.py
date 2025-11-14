"""
Service contiene la lógica de negocio del servicio web.
"""
import logging
from typing import Dict, List
import joblib
import pandas as pd

from mlops_project.config import (
    BEST_MODEL_FILENAME,
    MLFLOW_MODEL_VERSION,
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    get_model_path,
)

# Configurar logging
logger = logging.getLogger(__name__)

# Variable global para almacenar el modelo
_model = None


def load_model():
    model_path = get_model_path(BEST_MODEL_FILENAME)
    logger.info(f"Cargando modelo desde: {model_path}")

    if not model_path.exists():
        error_msg = (
            f"Modelo no encontrado en: {model_path}\n"
            f"Por favor, ejecuta: dvc pull {BEST_MODEL_FILENAME}.dvc"
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        model = joblib.load(model_path)
        logger.info("Modelo cargado exitosamente")
        return model
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {e}")
        raise


def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


def is_model_loaded() -> bool:
    global _model
    if _model is None:
        # Intentar cargar si no está cargado
        try:
            model_path = get_model_path(BEST_MODEL_FILENAME)
            if model_path.exists():
                _model = load_model()
                return True
            return False
        except Exception:
            return False
    return True


def _features_to_dataframe(features: Dict) -> pd.DataFrame:
    # numéricas, ordinales, nominales
    feature_order = NUMERIC_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES

    # Crear DataFrame en el orden correcto
    data = {feature: [features[feature]] for feature in feature_order}
    df = pd.DataFrame(data)

    return df


def _instances_to_dataframe(instances: List[Dict]) -> pd.DataFrame:
    # numéricas, ordinales, nominales
    feature_order = NUMERIC_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES

    # Crear lista de diccionarios en el orden correcto
    ordered_data = []
    for instance in instances:
        ordered_dict = {feature: instance[feature] for feature in feature_order}
        ordered_data.append(ordered_dict)

    df = pd.DataFrame(ordered_data)
    return df


def predict_single(features: Dict) -> Dict:
    """
    Realiza una predicción individual.
    """
    try:
        model = get_model()

        # Convertir features a DataFrame
        X = _features_to_dataframe(features)

        # Realizar predicción
        prediction = model.predict(X)[0]

        # Obtener probabilidad si el modelo lo soporta
        probability = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            # Probabilidad de clase positiva (índice 1)
            probability = float(proba[1])
        else:
            # Si no hay predict_proba, usar 1.0 si predice 1, 0.0 si predice 0
            probability = float(prediction)

        result = {
            "prediction": int(prediction),
            "probability": probability,
        }

        logger.debug(f"Predicción realizada: {result}")
        return result

    except Exception as e:
        logger.error(f"Error en predicción individual: {e}")
        raise


def predict_batch(instances: List[Dict]) -> List[int]:
    """
    Realiza predicciones por lotes.
    """
    try:
        model = get_model()

        # Convertir instancias a DataFrame
        X = _instances_to_dataframe(instances)

        # Realizar predicciones
        predictions = model.predict(X)

        # Convertir a lista de enteros
        result = [int(pred) for pred in predictions]

        logger.debug(f"Predicciones por lotes realizadas: {len(result)} instancias")
        return result

    except Exception as e:
        logger.error(f"Error en predicción por lotes: {e}")
        raise


def get_model_info() -> Dict[str, str]:
    """
    Obtiene información del modelo desplegado.
    """
    # Intentar extraer el nombre del modelo del archivo o usar default
    model_name = "logistic_regression"  # Default, puede mejorarse extrayendo del modelo

    return {
        "model_name": model_name,
        "model_version": MLFLOW_MODEL_VERSION,
    }