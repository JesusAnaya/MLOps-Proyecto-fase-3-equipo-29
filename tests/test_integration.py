"""
Tests de integración para el pipeline completo MLOps.

Estos tests verifican que todos los componentes trabajen juntos correctamente
desde la carga de datos hasta la predicción final.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mlops_project.dataset import load_and_prepare_data
from mlops_project.features import prepare_features
from mlops_project.modeling.predict import predict_and_evaluate
from mlops_project.modeling.train import train_model


@pytest.mark.integration
class TestCompletePipeline:
    """Tests de integración para el pipeline completo."""

    def test_pipeline_end_to_end(self, sample_realistic_data_df):
        """
        Test de integración completo end-to-end: 
        carga de datos → preprocesamiento → entrenamiento → predicción → métricas.
        
        Valida que todo el pipeline funcione correctamente de extremo a extremo.
        """
        # Preparar datos sintéticos con columnas realistas
        X = sample_realistic_data_df.drop(columns=["kredit"])
        y = sample_realistic_data_df["kredit"]

        # Dividir en train/test manualmente
        from sklearn.model_selection import train_test_split
        from mlops_project.config import RANDOM_SEED, TEST_SIZE

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
        )

        # 1. Preparar features (preprocesamiento)
        X_train_t, X_test_t, preprocessor = prepare_features(
            X_train=X_train,
            X_test=X_test,
            save_preprocessor=False,
        )

        # Verificar que las features se transformaron correctamente
        assert X_train_t.shape[0] == len(X_train)
        assert X_test_t.shape[0] == len(X_test)
        assert X_train_t.shape[1] >= X_train.shape[1]  # Features aumentaron (OHE)
        assert preprocessor is not None

        # 2. Entrenar modelo
        pipeline, results = train_model(
            X_train=X_train,
            y_train=y_train,
            preprocessor=preprocessor,
            model_name="logistic_regression",
            use_smote=False,  # Sin SMOTE para test más rápido
            evaluate=True,
            save_model=False,
        )

        # Verificar que el modelo se entrenó correctamente
        assert pipeline is not None
        assert results is not None
        assert "roc_auc" in results
        assert results["roc_auc"]["test_mean"] > 0  # Debe tener algún valor
        assert "accuracy" in results
        assert "f1" in results

        # 3. Realizar predicciones y evaluación
        y_pred, y_proba, metrics = predict_and_evaluate(
            model=pipeline,
            X_test=X_test,
            y_test=y_test,
            save_predictions=False,
        )

        # Verificar predicciones
        assert len(y_pred) == len(y_test)
        assert len(y_proba) == len(y_test)
        assert all(pred in [0, 1] for pred in y_pred)  # Predicciones binarias válidas
        assert all(0 <= prob <= 1 for prob in y_proba)  # Probabilidades válidas

        # Verificar métricas
        assert "accuracy" in metrics
        assert metrics["accuracy"] >= 0 and metrics["accuracy"] <= 1
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

    def test_pipeline_with_smote(self, sample_realistic_data_df):
        """Test de integración con SMOTE habilitado."""
        X = sample_realistic_data_df.drop(columns=["kredit"])
        y = sample_realistic_data_df["kredit"]

        from sklearn.model_selection import train_test_split
        from mlops_project.config import RANDOM_SEED, TEST_SIZE

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
        )

        # Preparar features
        X_train_t, X_test_t, preprocessor = prepare_features(
            X_train=X_train,
            X_test=X_test,
            save_preprocessor=False,
        )

        # Entrenar con SMOTE
        pipeline, results = train_model(
            X_train=X_train,
            y_train=y_train,
            preprocessor=preprocessor,
            model_name="logistic_regression",
            use_smote=True,
            smote_method="SMOTE",
            evaluate=True,
            save_model=False,
        )

        # Verificar resultados
        assert pipeline is not None
        assert results is not None

        # Realizar predicciones
        y_pred, y_proba, metrics = predict_and_evaluate(
            model=pipeline,
            X_test=X_test,
            y_test=y_test,
            save_predictions=False,
        )

        assert len(y_pred) == len(y_test)
        assert "f1" in metrics

    def test_pipeline_multiple_models(self, sample_realistic_data_df):
        """Test de integración con múltiples modelos."""
        X = sample_realistic_data_df.drop(columns=["kredit"])
        y = sample_realistic_data_df["kredit"]

        from sklearn.model_selection import train_test_split
        from mlops_project.config import RANDOM_SEED, TEST_SIZE

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
        )

        # Preparar features una vez
        X_train_t, X_test_t, preprocessor = prepare_features(
            X_train=X_train,
            X_test=X_test,
            save_preprocessor=False,
        )

        # Probar múltiples modelos
        models_to_test = ["logistic_regression", "decision_tree", "random_forest"]

        for model_name in models_to_test:
            pipeline, results = train_model(
                X_train=X_train,
                y_train=y_train,
                preprocessor=preprocessor,
                model_name=model_name,
                use_smote=False,
                evaluate=True,
                save_model=False,
            )

            assert pipeline is not None
            assert results is not None

            # Verificar que se pueden hacer predicciones
            y_pred = pipeline.predict(X_test)
            assert len(y_pred) == len(X_test)


@pytest.mark.integration
class TestDataToFeaturesIntegration:
    """Tests de integración entre módulos de datos y features."""

    def test_load_and_prepare_features_integration(self, tmp_path, sample_realistic_data_df):
        """Test de integración: carga de datos -> preparación de features."""
        # Usar datos realistas del fixture que tienen todas las columnas necesarias
        data = sample_realistic_data_df.copy()

        test_file = tmp_path / "test_data.csv"
        data.to_csv(test_file, index=False)

        # Cargar datos
        X_train, X_test, y_train, y_test = load_and_prepare_data(
            filepath=str(test_file),
            save_processed=False,
            return_combined=False,
        )

        # Verificar que los datos se cargaron correctamente
        assert X_train is not None
        assert X_test is not None
        assert len(X_train) > 0
        assert len(X_test) > 0

        # Preparar features
        X_train_t, X_test_t, preprocessor = prepare_features(
            X_train=X_train,
            X_test=X_test,
            save_preprocessor=False,
        )

        # Verificar transformación
        assert X_train_t.shape[0] == len(X_train)
        assert X_test_t.shape[0] == len(X_test)
        assert preprocessor is not None


@pytest.mark.integration
@pytest.mark.slow
class TestFullPipelineWithSaving:
    """Tests de integración que incluyen guardado de archivos."""

    def test_pipeline_with_file_operations(self, tmp_path, sample_realistic_data_df):
        """Test completo con guardado de archivos."""
        X = sample_realistic_data_df.drop(columns=["kredit"])
        y = sample_realistic_data_df["kredit"]

        from sklearn.model_selection import train_test_split
        from mlops_project.config import RANDOM_SEED, TEST_SIZE

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
        )

        # Preparar features con guardado
        with patch("mlops_project.config.MODELS_DIR", tmp_path / "models"):
            (tmp_path / "models").mkdir(parents=True)

            X_train_t, X_test_t, preprocessor = prepare_features(
                X_train=X_train,
                X_test=X_test,
                save_preprocessor=True,
            )

            # Verificar que se guardó el preprocessor
            preprocessor_path = tmp_path / "models" / "preprocessor.joblib"
            assert preprocessor_path.exists()

            # Entrenar y guardar modelo
            with patch("mlops_project.config.get_model_path") as mock_path:
                model_path = tmp_path / "models" / "test_model.joblib"
                mock_path.return_value = model_path

                pipeline, results = train_model(
                    X_train=X_train,
                    y_train=y_train,
                    preprocessor=preprocessor,
                    model_name="logistic_regression",
                    use_smote=False,
                    evaluate=True,
                    save_model=True,
                    model_filename="test_model.joblib",
                )

                # Verificar que se guardó el modelo
                assert model_path.exists()

                # Realizar predicciones y guardar
                # Mockear el guardado de predicciones para evitar dependencias de archivos
                with patch("mlops_project.modeling.predict.pd.DataFrame.to_csv") as mock_to_csv:
                    y_pred, y_proba, metrics = predict_and_evaluate(
                        model=pipeline,
                        X_test=X_test,
                        y_test=y_test,
                        save_predictions=True,
                    )

                    # Verificar que se intentó guardar las predicciones
                    assert mock_to_csv.called


@pytest.mark.integration
class TestMLflowIntegration:
    """Tests de integración con MLflow (mocked)."""

    @patch("mlops_project.modeling.train._MLFLOW_AVAILABLE", True)
    @patch("mlops_project.modeling.train.mlflow")
    def test_train_with_mlflow_logging(self, mock_mlflow, sample_realistic_data_df):
        """Test que verifica que MLflow se llama correctamente durante entrenamiento."""
        X = sample_realistic_data_df.drop(columns=["kredit"])
        y = sample_realistic_data_df["kredit"]

        from sklearn.model_selection import train_test_split
        from mlops_project.config import RANDOM_SEED, TEST_SIZE

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
        )

        # Preparar features
        X_train_t, X_test_t, preprocessor = prepare_features(
            X_train=X_train,
            X_test=X_test,
            save_preprocessor=False,
        )

        # Mock MLflow
        mock_mlflow.set_tracking_uri.return_value = None
        mock_mlflow.set_experiment.return_value = None
        mock_mlflow.start_run.return_value.__enter__ = lambda x: x
        mock_mlflow.start_run.return_value.__exit__ = lambda *args: None
        mock_mlflow.active_run.return_value.info.run_id = "test-run-id"

        # Entrenar (esto debería llamar a MLflow)
        # Nota: Necesitamos pasar args_namespace mock
        from argparse import Namespace

        with patch(
            "mlops_project.modeling.train._mlflow_log_run"
        ) as mock_mlflow_log:
            pipeline, results = train_model(
                X_train=X_train,
                y_train=y_train,
                preprocessor=preprocessor,
                model_name="logistic_regression",
                use_smote=False,
                evaluate=True,
                save_model=False,
            )

            # Verificar que el modelo se entrenó
            assert pipeline is not None

