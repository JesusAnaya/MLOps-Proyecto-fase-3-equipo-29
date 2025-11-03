# Ejemplos del Pipeline MLOps

Esta guía muestra ejemplos detallados de cómo ejecutar el pipeline completo de diferentes maneras.

## Tabla de Contenidos

- [Pipeline Completo Automatizado](#pipeline-completo-automatizado)
- [Pipeline Manual Paso a Paso](#pipeline-manual-paso-a-paso)
- [Entrenar Múltiples Modelos](#entrenar-múltiples-modelos)
- [Pipeline Programático](#pipeline-programático)
- [Ejemplos con MLflow](#ejemplos-con-mlflow)

## Pipeline Completo Automatizado

### Usando Make

```bash
# Ejecutar todo el pipeline de una vez
make pipeline
```

Este comando ejecuta secuencialmente:
1. `make prepare-data` - Preparación de datos
2. `make prepare-features` - Transformación de features
3. `make train` - Entrenamiento del modelo

## Pipeline Manual Paso a Paso

### 1. Preparar Datos

```bash
uv run mlops-prepare-data \
    --input data/raw/german_credit_modified.csv \
    --save
```

**Salida esperada:**
- `data/processed/data_clean.csv`
- `data/processed/Xtraintest.csv`
- `data/processed/ytraintest.csv`

### 2. Preparar Features

```bash
uv run mlops-prepare-features \
    --train data/processed/Xtraintest.csv \
    --save-preprocessor
```

**Salida esperada:**
- Features transformadas
- `models/preprocessor.joblib`

### 3. Entrenar Modelo

```bash
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model logistic_regression
```

**Salida esperada:**
- `models/best_model.joblib`
- `models/model_results.json`
- Run en MLflow

### 4. Realizar Predicciones

```bash
uv run mlops-predict \
    --model models/best_model.joblib \
    --X-test data/processed/Xtraintest.csv \
    --y-test data/processed/ytraintest.csv \
    --save
```

**Salida esperada:**
- `predictions.csv`
- `evaluation_metrics.json`

## Entrenar Múltiples Modelos

### Script Bash

```bash
# Iterar sobre diferentes modelos
for model in logistic_regression random_forest decision_tree xgboost; do
    echo "Entrenando $model..."
    uv run mlops-train \
        --X-train data/processed/Xtraintest.csv \
        --y-train data/processed/ytraintest.csv \
        --preprocessor models/preprocessor.joblib \
        --model $model \
        --output ${model}_model.joblib \
        --mlflow-run-name "comparacion_${model}"
done
```

### Comparar Modelos en MLflow

Después de ejecutar el script anterior, puedes comparar los modelos en la UI de MLflow usando los nombres de los runs (`comparacion_logistic_regression`, etc.).

## Pipeline Programático

### Ejemplo Completo en Python

```python
from mlops_project.dataset import load_and_prepare_data
from mlops_project.features import prepare_features
from mlops_project.modeling.train import train_model
from mlops_project.modeling.predict import predict_and_evaluate

# 1. Preparar datos
X_train, X_test, y_train, y_test = load_and_prepare_data(
    filepath="data/raw/german_credit_modified.csv",
    save_processed=True
)

# 2. Preparar features
X_train_t, X_test_t, preprocessor = prepare_features(
    X_train=X_train,
    X_test=X_test,
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

# 4. Evaluar
y_pred, y_proba, metrics = predict_and_evaluate(
    model=pipeline,
    X_test=X_test,
    y_test=y_test,
    save_predictions=True
)

print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
print(f"F1-Score: {metrics['f1']:.4f}")
```

## Ejemplos con MLflow

### Pipeline con Tracking Completo

```bash
# 1. Preparar datos
uv run mlops-prepare-data \
    --input data/raw/german_credit_modified.csv \
    --save

# 2. Preparar features
uv run mlops-prepare-features \
    --train data/processed/Xtraintest.csv \
    --save-preprocessor

# 3. Entrenar con MLflow
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model logistic_regression \
    --mlflow-run-name "pipeline_completo_lr" \
    --model-version "1.0.0"
```

### Comparar Diferentes Configuraciones

```bash
# Configuración 1: Con SMOTE
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model logistic_regression \
    --mlflow-run-name "lr_con_smote" \
    --mlflow-tags '{"smote":"true"}'

# Configuración 2: Sin SMOTE
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model logistic_regression \
    --no-smote \
    --mlflow-run-name "lr_sin_smote" \
    --mlflow-tags '{"smote":"false"}'
```

### Experimentar con Hiperparámetros

Los hiperparámetros están en `mlops_project/config.py`. Puedes modificar `AVAILABLE_MODELS` y ejecutar:

```bash
# Después de modificar hiperparámetros en config.py
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model random_forest \
    --mlflow-run-name "rf_hyperparams_modificados"
```

Compara los resultados en MLflow para ver el impacto de los cambios.

## Pipeline con Validación

### Validar cada paso

```bash
# Paso 1: Verificar datos
uv run mlops-prepare-data \
    --input data/raw/german_credit_modified.csv \
    --save
ls -lh data/processed/*.csv

# Paso 2: Verificar features
uv run mlops-prepare-features \
    --train data/processed/Xtraintest.csv \
    --save-preprocessor
ls -lh models/preprocessor.joblib

# Paso 3: Verificar entrenamiento
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model logistic_regression
cat models/model_results.json

# Paso 4: Verificar predicción
uv run mlops-predict \
    --model models/best_model.joblib \
    --X-test data/processed/Xtraintest.csv \
    --y-test data/processed/ytraintest.csv \
    --save
head predictions.csv
```

## Troubleshooting

### Error: "Archivo no encontrado"

Verifica que los archivos anteriores se generaron correctamente:
```bash
ls -la data/processed/
ls -la models/
```

### Error: "Preprocessor no encontrado"

Asegúrate de ejecutar `mlops-prepare-features` antes de `mlops-train`.

### Error: "Modelo no encontrado"

Asegúrate de ejecutar `mlops-train` antes de `mlops-predict`.

