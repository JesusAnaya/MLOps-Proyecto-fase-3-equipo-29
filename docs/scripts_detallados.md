# Documentación Detallada de Scripts

Esta guía documenta todos los scripts CLI disponibles en el proyecto.

## Tabla de Contenidos

- [mlops-prepare-data](#mlops-prepare-data)
- [mlops-prepare-features](#mlops-prepare-features)
- [mlops-train](#mlops-train)
- [mlops-predict](#mlops-predict)

## mlops-prepare-data

**Función**: Carga datos crudos, limpia valores inválidos, elimina outliers y divide en train/test.

### Uso Básico

```bash
uv run mlops-prepare-data \
    --input data/raw/german_credit_modified.csv \
    --save
```

### Opciones

- `--input`: Ruta al archivo CSV de entrada (requerido)
- `--save`: Guardar datos procesados en data/processed/
- `--combined`: Retornar datos combinados sin dividir

### Salida

- `data/processed/data_clean.csv`: Datos limpios
- `data/processed/Xtraintest.csv`: Features combinadas (train + test)
- `data/processed/ytraintest.csv`: Variable objetivo combinada

### Ejemplos

```bash
# Solo preparar datos sin guardar
uv run mlops-prepare-data --input data/raw/german_credit_modified.csv

# Preparar y guardar
uv run mlops-prepare-data \
    --input data/raw/german_credit_modified.csv \
    --save

# Con datos combinados
uv run mlops-prepare-data \
    --input data/raw/german_credit_modified.csv \
    --save \
    --combined
```

## mlops-prepare-features

**Función**: Transforma features aplicando imputación, escalado y encoding.

### Uso Básico

```bash
uv run mlops-prepare-features \
    --train data/processed/Xtraintest.csv \
    --save-preprocessor
```

### Opciones

- `--train`: Archivo CSV con datos de entrenamiento (requerido)
- `--test`: Archivo CSV con datos de prueba (opcional)
- `--output-train`: Archivo de salida para train (default: X_train_processed.csv)
- `--output-test`: Archivo de salida para test (default: X_test_processed.csv)
- `--save-preprocessor`: Guardar el preprocessor en models/

### Salida

- Features transformadas (impresas en consola o guardadas)
- `models/preprocessor.joblib`: Preprocessor guardado (si se usa --save-preprocessor)

### Ejemplos

```bash
# Solo entrenamiento
uv run mlops-prepare-features \
    --train data/processed/Xtraintest.csv \
    --save-preprocessor

# Con datos de test
uv run mlops-prepare-features \
    --train data/processed/Xtraintest.csv \
    --test data/processed/X_test.csv \
    --output-train X_train_transformed.csv \
    --output-test X_test_transformed.csv \
    --save-preprocessor
```

## mlops-train

**Función**: Entrena modelos con cross-validation y guarda el mejor modelo. Incluye integración completa con MLflow.

### Uso Básico

```bash
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model logistic_regression
```

### Opciones Principales

- `--X-train`: Features de entrenamiento (requerido)
- `--y-train`: Variable objetivo de entrenamiento (requerido)
- `--preprocessor`: Ruta al preprocessor guardado (requerido)
- `--model`: Modelo a entrenar (default: logistic_regression)
  - Opciones: `logistic_regression`, `random_forest`, `decision_tree`, `svm`, `xgboost`
- `--no-smote`: No usar SMOTE para balanceo de clases
- `--smote-method`: Método de SMOTE (default: BorderlineSMOTE)
  - Opciones: `SMOTE`, `BorderlineSMOTE`
- `--no-evaluate`: No evaluar con cross-validation (entrenamiento más rápido)
- `--output`: Nombre del archivo de salida para el modelo

### Opciones MLflow

- `--mlflow-disable`: Deshabilitar logging a MLflow
- `--mlflow-uri`: Tracking URI (default: desde config.py)
- `--mlflow-experiment`: Nombre de experimento (default: desde config.py)
- `--mlflow-run-name`: Nombre del run (default: generado automáticamente)
- `--mlflow-reg-name`: Nombre a registrar en Model Registry
- `--mlflow-tags`: Tags extra en JSON (ej: `'{"dataset":"south_german"}'`)
- `--model-version`: Versión semántica del modelo (default: 0.1.0)

### Salida

- `models/best_model.joblib`: Modelo entrenado
- `models/model_results.json`: Métricas de evaluación
- MLflow: Run registrado con hiperparámetros, métricas y modelo

### Ejemplos

```bash
# Entrenamiento básico
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model logistic_regression

# Sin SMOTE
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model random_forest \
    --no-smote

# Sin evaluación (solo entrenar)
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model logistic_regression \
    --no-evaluate

# Con MLflow personalizado
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model xgboost \
    --mlflow-run-name "xgboost_experimento_v1" \
    --model-version "1.2.0"
```

Para más detalles sobre MLflow, ver [mlflow_guia.md](mlflow_guia.md).

## mlops-predict

**Función**: Realiza predicciones con un modelo entrenado y evalúa métricas.

### Uso Básico

```bash
uv run mlops-predict \
    --model models/best_model.joblib \
    --X-test data/processed/Xtraintest.csv \
    --y-test data/processed/ytraintest.csv \
    --save
```

### Opciones

- `--model`: Ruta al modelo guardado (default: models/best_model.joblib)
- `--X-test`: Features de test (requerido)
- `--y-test`: Variable objetivo de test (opcional, para evaluación)
- `--output`: Archivo de salida para predicciones (default: predictions.csv)
- `--batch-size`: Tamaño del batch para predicciones (default: 1000)
- `--proba`: Guardar probabilidades en lugar de clases
- `--save`: Guardar predicciones en archivo

### Salida

- `predictions.csv`: Predicciones guardadas
- `evaluation_metrics.json`: Métricas de evaluación (si se provee y_test)

### Ejemplos

```bash
# Predicción con evaluación
uv run mlops-predict \
    --model models/best_model.joblib \
    --X-test data/processed/Xtraintest.csv \
    --y-test data/processed/ytraintest.csv \
    --save

# Solo predicción (sin evaluación)
uv run mlops-predict \
    --model models/best_model.joblib \
    --X-test data/processed/X_new.csv \
    --output predictions.csv \
    --save

# Predicción con probabilidades
uv run mlops-predict \
    --model models/best_model.joblib \
    --X-test data/processed/X_test.csv \
    --proba \
    --save

# Predicción en batches grandes
uv run mlops-predict \
    --model models/best_model.joblib \
    --X-test data/processed/X_large.csv \
    --batch-size 1000 \
    --save
```

## Ayuda

Para ver ayuda de cualquier script:

```bash
uv run mlops-prepare-data --help
uv run mlops-prepare-features --help
uv run mlops-train --help
uv run mlops-predict --help
```

