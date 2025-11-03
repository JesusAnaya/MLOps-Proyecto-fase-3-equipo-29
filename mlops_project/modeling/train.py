"""
Training module for MLOps project with MLflow integration.

Este módulo maneja el entrenamiento y evaluación de modelos:
- Entrenamiento con cross-validation
- Evaluación de métrricas
- Guardado de modelos
- Soporte para balanceo de clases con SMOTE
- **Integración con MLflow**: tracking de runs, logging de params/métricas,
  logging/registro de modelos en Model Registry
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple
import warnings
import subprocess
from datetime import datetime

from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline as SkPipeline


# Import diferido para que el script funcione incluso si falta mlflow
_MLFLOW_AVAILABLE = True
try:
    import mlflow
    import mlflow.sklearn as mlsk
    import mlflow.xgboost as mlxg  # MLflow specialized library for XGBoost
    from mlflow.models.signature import infer_signature
except Exception:
    _MLFLOW_AVAILABLE = False
    mlsk = None
    mlxg = None

from mlops_project.config import (
    AVAILABLE_MODELS,
    BEST_MODEL_FILENAME,
    CV_FOLDS,
    CV_REPEATS,
    MLFLOW_DEFAULT_TAGS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_VERSION,
    MLFLOW_REGISTER_MODELS,
    MLFLOW_TRACKING_URI,
    RANDOM_SEED,
    RESULTS_FILENAME,
    SMOTE_CONFIG,
    get_model_path,
)

# Suprimir warnings durante entrenamiento
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "no-git"


def get_model_instance(model_name: str) -> BaseEstimator:
    """
    Crea una instancia del modelo según el nombre.
    """
    model_mapping = {
        "logistic_regression": LogisticRegression,
        "decision_tree": DecisionTreeClassifier,
        "random_forest": RandomForestClassifier,
        "svm": SVC,
        "xgboost": XGBClassifier,
        "mlp": MLPClassifier,
    }

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Modelo '{model_name}' no disponible. Opciones: {list(AVAILABLE_MODELS.keys())}"
        )

    model_class = model_mapping[model_name]
    model_params = AVAILABLE_MODELS[model_name]["params"]

    return model_class(**model_params)


def get_smote_instance(smote_method: str = "SMOTE") -> SMOTE | BorderlineSMOTE:
    """
    Crea una instancia de SMOTE según el método especificado.
    """
    if smote_method == "BorderlineSMOTE":
        return BorderlineSMOTE(
            random_state=SMOTE_CONFIG["random_state"],
            k_neighbors=SMOTE_CONFIG["k_neighbors"],
            m_neighbors=SMOTE_CONFIG["m_neighbors"],
        )
    else:
        return SMOTE(random_state=SMOTE_CONFIG["random_state"])


def create_training_pipeline(
    preprocessor: Any,
    model: BaseEstimator,
    use_smote: bool = True,
    smote_method: str = "BorderlineSMOTE",
) -> ImbPipeline:
    """
    Crea el pipeline completo de entrenamiento compatible con imblearn:
    - Si el preprocessor es un sklearn.Pipeline, se 'aplana' sus steps.
    - Si no, se agrega como un transformador normal.
    """
    steps = []

    # 1) Aplanar si es sklearn.Pipeline; de lo contrario, agregar como transformador único
    if isinstance(preprocessor, SkPipeline):
        # ejemplo: [('coltransform', ColumnTransformer(...)), ('scaler', StandardScaler())]
        steps.extend(preprocessor.steps)
    else:
        steps.append(("preprocessor", preprocessor))

    # 2) SMOTE (si aplica)
    if use_smote:
        smote_instance = get_smote_instance(smote_method)
        steps.append(("smote", smote_instance))

    # 3) Modelo
    steps.append(("model", model))

    return ImbPipeline(steps=steps)


def evaluate_model(
    pipeline: ImbPipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = CV_FOLDS,
    cv_repeats: int = CV_REPEATS,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Evalúa el modelo usando validación cruzada.
    """
    cv = RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=cv_repeats, random_state=RANDOM_SEED)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "geometric_mean": make_scorer(geometric_mean_score),
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_results = cross_validate(
            pipeline, X, np.ravel(y), scoring=scoring, cv=cv, return_train_score=True
        )

    results = {}
    for metric_name in scoring.keys():
        test_key = f"test_{metric_name}"
        train_key = f"train_{metric_name}"

        results[metric_name] = {
            "test_mean": float(np.mean(cv_results[test_key])),
            "test_std": float(np.std(cv_results[test_key])),
            "train_mean": float(np.mean(cv_results[train_key])),
            "train_std": float(np.std(cv_results[train_key])),
        }

    if verbose:
        print("\n" + "=" * 60)
        print("RESULTADOS DE VALIDACIÓN CRUZADA")
        print("=" * 60)
        print(
            f"CV: {cv_folds} folds x {cv_repeats} repeats = {cv_folds * cv_repeats} evaluaciones"
        )
        print("-" * 60)

        for metric_name, scores in results.items():
            print(f"{metric_name.upper():20s}:")
            print(f"  Test:  {scores['test_mean']:.4f} (± {scores['test_std']:.3f})")
            print(f"  Train: {scores['train_mean']:.4f} (± {scores['train_std']:.3f})")

        print("=" * 60)

    return results


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: Any,
    model_name: str = "logistic_regression",
    use_smote: bool = True,
    smote_method: str = "BorderlineSMOTE",
    evaluate: bool = True,
    save_model: bool = True,
    model_filename: Optional[str] = None,
) -> Tuple[ImbPipeline, Optional[Dict[str, Dict[str, float]]]]:
    """
    Entrena un modelo con el pipeline completo.
    """
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO DE MODELO")
    print("=" * 60)

    model = get_model_instance(model_name)
    model_display_name = AVAILABLE_MODELS[model_name]["name"]

    print(f"\nModelo: {model_display_name}")
    print(f"SMOTE: {'Sí' if use_smote else 'No'} ({smote_method if use_smote else 'N/A'})")
    print(f"Datos: X{X_train.shape}, y{y_train.shape}")

    pipeline = create_training_pipeline(
        preprocessor=preprocessor,
        model=model,
        use_smote=use_smote,
        smote_method=smote_method,
    )

    evaluation_results = None
    if evaluate:
        print("\n[1/2] Evaluando modelo con validación cruzada...")
        evaluation_results = evaluate_model(pipeline, X_train, y_train)

    print(f"\n[{'2/2' if evaluate else '1/1'}] Entrenando modelo con todos los datos de entrenamiento...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(X_train, y_train)

    print("✓ Modelo entrenado exitosamente")

    if save_model:
        filename = model_filename or BEST_MODEL_FILENAME
        model_path = get_model_path(filename)
        joblib.dump(pipeline, model_path)
        print(f"✓ Modelo guardado en: {model_path}")

    print("\n" + "=" * 60)

    return pipeline, evaluation_results


def save_results(
    results: Dict[str, Dict[str, float]],
    model_name: str,
    filename: Optional[str] = None,
) -> str:
    """
    Guarda los resultados de evaluación en formato JSON. Devuelve la ruta escrita.
    """
    filename = filename or RESULTS_FILENAME
    results_path = get_model_path(filename)

    output = {
        "model": model_name,
        "model_display_name": AVAILABLE_MODELS[model_name]["name"],
        "metrics": results,
        "config": {
            "cv_folds": CV_FOLDS,
            "cv_repeats": CV_REPEATS,
            "random_seed": RANDOM_SEED,
            "smote_config": SMOTE_CONFIG,
        },
    }

    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"✓ Resultados guardados en: {results_path}")
    return results_path


def _convert_param_value(value: Any) -> Any:
    """
    Convierte valores de parámetros a tipos compatibles con MLflow.
    MLflow requiere que los parámetros sean strings, números o booleanos.
    """
    if isinstance(value, (str, int, float, bool)):
        return value
    elif isinstance(value, (list, tuple)):
        # MLflow no soporta listas directamente, convertir a string JSON
        return json.dumps(value)
    elif value is None:
        return "None"
    else:
        # Convertir otros tipos a string
        return str(value)


def _log_hyperparameters(model_name: str, model_params: Dict[str, Any]) -> None:
    """
    Logs hyperparameters with correct types for each model type.
    Uses model-specific parameter names and converts values appropriately.
    """
    # Log model identifier
    mlflow.log_param("model_key", model_name)
    
    # Convert and log each hyperparameter with correct type
    for param_name, param_value in model_params.items():
        converted_value = _convert_param_value(param_value)
        
        # Use model-specific parameter naming
        if model_name == "xgboost":
            # XGBoost parameters - preserve as is with proper typing
            mlflow.log_param(f"model.{param_name}", converted_value)
        else:
            # sklearn models - use standard naming
            mlflow.log_param(f"model__{param_name}", converted_value)


def _mlflow_log_run(
    *,
    X_sample: pd.DataFrame,
    pipeline: ImbPipeline,
    model_name: str,
    model_display_name: str,
    model_params: Dict[str, Any],
    evaluation_results: Optional[Dict[str, Dict[str, float]]],
    used_smote: bool,
    smote_method: Optional[str],
    results_json_path: Optional[str],
    args_namespace: argparse.Namespace,
) -> None:
    """
    Integra MLflow con logging completo:
    - Versión del modelo
    - Hyperparameters con tipos correctos
    - Métricas de evaluación
    - Resultados relevantes
    - Registro en Model Registry usando librerías especializadas
    """
    if not _MLFLOW_AVAILABLE or args_namespace.mlflow_disable:
        print("ℹ MLflow deshabilitado o no disponible; se omite logging.")
        return

    try:
        # Configuración base desde config.py (puede ser sobrescrita por args)
        tracking_uri = args_namespace.mlflow_uri or MLFLOW_TRACKING_URI
        experiment = args_namespace.mlflow_experiment or MLFLOW_EXPERIMENT_NAME
        model_version = args_namespace.model_version or MLFLOW_MODEL_VERSION
        
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

        # Tags con información del proyecto
        run_name = args_namespace.mlflow_run_name or f"train_{model_name}_{_ts()}"
        tags = {
            **MLFLOW_DEFAULT_TAGS.copy(),
            "model_version": model_version,
            "model_name": model_name,
            "model_display_name": model_display_name,
            "git_commit": _git_commit(),
            "timestamp": _ts(),
        }
        
        if args_namespace.mlflow_tags:
            # Permitir JSON de tags extra
            try:
                tags.update(json.loads(args_namespace.mlflow_tags))
            except Exception:
                pass

        with mlflow.start_run(run_name=run_name, tags=tags):
            # === 1. VERSIÓN DEL MODELO ===
            mlflow.log_param("version", model_version)
            
            # === 2. HYPERPARAMETERS ===
            _log_hyperparameters(model_name, model_params)
            
            # Parámetros de preprocesamiento y SMOTE
            mlflow.log_params({
                "cv_folds": CV_FOLDS,
                "cv_repeats": CV_REPEATS,
                "random_seed": RANDOM_SEED,
                "smote__used": used_smote,
            })
            
            if used_smote:
                mlflow.log_params({
                    "smote__method": smote_method or "N/A",
                    "smote__k_neighbors": SMOTE_CONFIG.get("k_neighbors", "N/A"),
                    "smote__m_neighbors": SMOTE_CONFIG.get("m_neighbors", "N/A"),
                })

            # === 3. MÉTRICAS DE EVALUACIÓN ===
            if evaluation_results:
                # Log métricas principales (promedios de CV)
                for metric, stats in evaluation_results.items():
                    # Métricas de test (las más importantes)
                    mlflow.log_metric(f"{metric}_test_mean", stats["test_mean"])
                    mlflow.log_metric(f"{metric}_test_std", stats["test_std"])
                    
                    # Métricas de train (para detectar overfitting)
                    mlflow.log_metric(f"{metric}_train_mean", stats["train_mean"])
                    mlflow.log_metric(f"{metric}_train_std", stats["train_std"])
                    
                    # Métrica principal para comparación (test_mean)
                    if metric in ["roc_auc", "f1", "accuracy"]:
                        mlflow.log_metric(f"{metric}", stats["test_mean"])
                
                # Log métricas agregadas para facilitar comparación
                if "roc_auc" in evaluation_results:
                    mlflow.log_metric("best_metric", evaluation_results["roc_auc"]["test_mean"])
                    mlflow.log_param("best_metric_name", "roc_auc")

            # === 4. RESULTADOS RELEVANTES ===
            if results_json_path and os.path.exists(results_json_path):
                mlflow.log_artifact(results_json_path, artifact_path="results")
                # También log el path como parámetro para referencia
                mlflow.log_param("results_json_path", str(results_json_path))
            
            # Log información del dataset
            mlflow.log_params({
                "dataset_shape": f"{X_sample.shape[0]}x{X_sample.shape[1]}",
                "n_features": X_sample.shape[1],
            })

            # === 5. REGISTRO DEL MODELO ===
            try:
                signature = infer_signature(X_sample, pipeline.predict(X_sample))
            except Exception:
                signature = None

            registered_name = (
                args_namespace.mlflow_reg_name or model_display_name.replace(" ", "_").lower()
            )

            # Usar librerías especializadas según el tipo de modelo
            # Nota: Para pipelines que incluyen preprocesamiento, usamos mlflow.sklearn
            # ya que puede manejar cualquier modelo sklearn-compatible (incluyendo XGBoost)
            # Para XGBoost, también registramos metadata específica pero usamos sklearn logging
            
            if model_name == "xgboost":
                # Para XGBoost en pipeline, loggear el pipeline completo con sklearn
                # pero agregar tags y metadata específicos de XGBoost
                mlflow.set_tag("model_framework", "xgboost")
                mlflow.set_tag("model_type", "gradient_boosting")
                
                # Loggear también el modelo XGBoost puro para referencia
                if mlxg is not None and "model" in pipeline.named_steps:
                    try:
                        xgb_model = pipeline.named_steps["model"]
                        # Loggear modelo XGBoost standalone (opcional, solo para referencia)
                        mlxg.log_model(
                            xgb_model,
                            artifact_path="xgb_model_only",
                        )
                    except Exception:
                        pass  # No crítico si falla
                
                # Loggear pipeline completo con sklearn
                if MLFLOW_REGISTER_MODELS:
                    mlsk.log_model(
                        pipeline,
                        artifact_path="model",
                        input_example=X_sample,
                        signature=signature,
                        registered_model_name=registered_name,
                    )
                else:
                    mlsk.log_model(
                        pipeline,
                        artifact_path="model",
                        input_example=X_sample,
                        signature=signature,
                    )
            else:
                # sklearn models - usar mlflow.sklearn
                mlflow.set_tag("model_framework", "sklearn")
                mlflow.set_tag("model_type", model_name)
                
                if MLFLOW_REGISTER_MODELS:
                    mlsk.log_model(
                        pipeline,
                        artifact_path="model",
                        input_example=X_sample,
                        signature=signature,
                        registered_model_name=registered_name,
                    )
                else:
                    mlsk.log_model(
                        pipeline,
                        artifact_path="model",
                        input_example=X_sample,
                        signature=signature,
                    )

            # Log información del run
            run_id = mlflow.active_run().info.run_id
            print(f"✓ MLflow run registrado:")
            print(f"  - Experiment: '{experiment}'")
            print(f"  - Run ID: {run_id}")
            print(f"  - Model: {model_display_name}")
            print(f"  - Version: {model_version}")
            if MLFLOW_REGISTER_MODELS:
                print(f"  - Registered as: '{registered_name}'")
                
    except Exception as e:
        # No interrumpir entrenamiento por fallas de tracking
        print(f"⚠ Aviso MLflow: no se pudo registrar el run → {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    Función principal para ejecutar el script desde línea de comandos.
    """
    parser = argparse.ArgumentParser(
        description="Entrena un modelo de clasificación para el pipeline de MLOps (con MLflow)"
    )
    parser.add_argument("--X-train", type=str, required=True, help="CSV con features de entrenamiento")
    parser.add_argument("--y-train", type=str, required=True, help="CSV con target de entrenamiento")
    parser.add_argument("--preprocessor", type=str, required=True, help="Ruta del preprocessor .joblib")
    parser.add_argument(
        "--model",
        type=str,
        default="logistic_regression",
        choices=list(AVAILABLE_MODELS.keys()),
        help="Modelo a entrenar",
    )
    parser.add_argument("--no-smote", action="store_true", help="No usar SMOTE para balanceo de clases")
    parser.add_argument(
        "--smote-method",
        type=str,
        default="BorderlineSMOTE",
        choices=["SMOTE", "BorderlineSMOTE"],
        help="Método de SMOTE a usar",
    )
    parser.add_argument("--no-evaluate", action="store_true", help="No evaluar con cross-validation")
    parser.add_argument("--output", type=str, help="Nombre del archivo de salida para el modelo (opcional)")

    # -------- Banderas de MLflow --------
    parser.add_argument("--mlflow-disable", action="store_true", help="Deshabilitar logging a MLflow")
    parser.add_argument("--mlflow-uri", type=str, help=f"Tracking URI (default: {MLFLOW_TRACKING_URI})")
    parser.add_argument("--mlflow-experiment", type=str, help=f"Nombre de experimento (default: {MLFLOW_EXPERIMENT_NAME})")
    parser.add_argument("--mlflow-run-name", type=str, help="Nombre del run (si no, usa plantilla)")
    parser.add_argument("--mlflow-reg-name", type=str, help="Nombre a registrar en Model Registry")
    parser.add_argument("--mlflow-tags", type=str, help='Tags extra en JSON (e.g. \'{"dataset":"south_german"}\')')
    parser.add_argument("--model-version", type=str, help=f"Versión semántica del modelo (default: {MLFLOW_MODEL_VERSION})")

    args = parser.parse_args()

    try:
        # Cargar datos
        print("Cargando datos...")
        X_train = pd.read_csv(args.X_train)
        y_train = pd.read_csv(args.y_train).iloc[:, 0]  # Primera columna

        # Cargar preprocessor
        print(f"Cargando preprocessor desde: {args.preprocessor}")
        preprocessor = joblib.load(args.preprocessor)

        # Entrenar modelo
        pipeline, results = train_model(
            X_train=X_train,
            y_train=y_train,
            preprocessor=preprocessor,
            model_name=args.model,
            use_smote=not args.no_smote,
            smote_method=args.smote_method,
            evaluate=not args.no_evaluate,
            save_model=True,
            model_filename=args.output,
        )

        # Guardar resultados si se evaluó
        results_json_path = None
        if results:
            results_json_path = save_results(results, args.model)

        # -------- MLflow logging/registro --------
        model_display_name = AVAILABLE_MODELS[args.model]["name"]
        model_params = AVAILABLE_MODELS[args.model]["params"]
        X_sample = X_train.head(min(5, len(X_train)))  # input_example para firma

        _mlflow_log_run(
            X_sample=X_sample,
            pipeline=pipeline,
            model_name=args.model,
            model_display_name=model_display_name,
            model_params=model_params,
            evaluation_results=results,
            used_smote=(not args.no_smote),
            smote_method=(args.smote_method if not args.no_smote else None),
            results_json_path=results_json_path,
            args_namespace=args,
        )

        print("\n✓ Pipeline de entrenamiento completado exitosamente")
        return 0

    except Exception as e:
        print(f"\n✗ Error en el pipeline de entrenamiento: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
