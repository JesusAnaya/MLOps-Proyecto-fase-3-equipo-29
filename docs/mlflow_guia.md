# Guía Completa de MLflow

Esta guía explica cómo usar MLflow en el proyecto MLOps para tracking de experimentos, versionado de modelos y registro en Model Registry.

## Tabla de Contenidos

- [Introducción](#introducción)
- [Configuración](#configuración)
- [Uso Básico](#uso-básico)
- [Opciones Avanzadas](#opciones-avanzadas)
- [Ejemplos Detallados](#ejemplos-detallados)
- [Visualización de Resultados](#visualización-de-resultados)
- [Model Registry](#model-registry)

## Introducción

MLflow está integrado en el script de entrenamiento (`mlops-train`) y registra automáticamente:

- **Versión del modelo**: Versión semántica configurable
- **Hiperparámetros**: Todos los hiperparámetros del modelo con tipos correctos
- **Métricas de evaluación**: Métricas de cross-validation (accuracy, precision, recall, f1, roc_auc, etc.)
- **Resultados relevantes**: Configuración, dataset info, resultados JSON
- **Modelos**: Modelos completos registrados en Model Registry

## Configuración

### Configuración Base

La configuración se encuentra en `mlops_project/config.py`:

```python
MLFLOW_TRACKING_URI = "https://mlflow-equipo-29.robomous.ai"
MLFLOW_EXPERIMENT_NAME = "equipo-29"
MLFLOW_MODEL_VERSION = "0.1.0"
MLFLOW_REGISTER_MODELS = True
```

### Sobrescribir con Variables de Entorno

```bash
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_EXPERIMENT="mi-experimento"
export MODEL_VERSION="2.0.0"
```

### Sobrescribir con Argumentos CLI

Todos los parámetros de MLflow pueden sobrescribirse desde la línea de comandos.

## Uso Básico

### Entrenamiento con MLflow (Automático)

El uso más simple - MLflow está habilitado por defecto:

```bash
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model logistic_regression
```

**Salida esperada:**
```
✓ MLflow run registrado:
  - Experiment: 'equipo-29'
  - Run ID: abc123def456...
  - Model: Logistic Regression
  - Version: 0.1.0
  - Registered as: 'logistic_regression'
🏃 View run at: https://mlflow-equipo-29.robomous.ai/#/experiments/1/runs/abc123...
```

### Entrenamiento sin MLflow

Si necesitas deshabilitar MLflow:

```bash
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model logistic_regression \
    --mlflow-disable
```

## Opciones Avanzadas

### Personalizar Nombre del Run

```bash
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model random_forest \
    --mlflow-run-name "experimento_rf_sin_smote"
```

### Personalizar Versión del Modelo

```bash
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model logistic_regression \
    --model-version "1.2.3"
```

### Cambiar Experimento

```bash
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model decision_tree \
    --mlflow-experiment "experimentos-decision-trees"
```

### Cambiar Tracking URI

```bash
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model logistic_regression \
    --mlflow-uri "http://localhost:5000"
```

### Personalizar Nombre en Model Registry

```bash
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model xgboost \
    --mlflow-reg-name "xgboost-credit-classifier"
```

### Agregar Tags Personalizados

```bash
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model random_forest \
    --mlflow-tags '{"dataset":"south_german_credit","experiment_type":"baseline"}'
```

## Ejemplos Detallados

### Ejemplo 1: Entrenamiento Básico con Evaluación

```bash
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model logistic_regression
```

**Qué se registra:**
- Versión: 0.1.0
- Hiperparámetros del modelo (C, penalty, solver, max_iter, etc.)
- Métricas de cross-validation (15 evaluaciones: 5 folds × 3 repeats)
- Resultados JSON como artefacto
- Modelo completo en Model Registry

### Ejemplo 2: Entrenamiento sin Evaluación (Más Rápido)

```bash
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model random_forest \
    --no-evaluate \
    --mlflow-run-name "rf_fast_training"
```

**Qué se registra:**
- Versión e hiperparámetros
- Configuración del dataset
- Modelo completo
- **NO** métricas de cross-validation (se omite evaluación)

### Ejemplo 3: Entrenamiento sin SMOTE

```bash
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model decision_tree \
    --no-smote \
    --mlflow-run-name "dt_sin_balanceo"
```

**Qué se registra:**
- Hiperparámetros indicando `smote__used: False`
- Métricas comparables sin balanceo de clases

### Ejemplo 4: XGBoost con Configuración Personalizada

```bash
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model xgboost \
    --model-version "2.0.0" \
    --mlflow-reg-name "xgboost-produccion" \
    --mlflow-tags '{"stage":"production","priority":"high"}'
```

**Qué se registra:**
- Modelo XGBoost usando librería especializada `mlflow.xgboost`
- Pipeline completo con sklearn
- Tags personalizados para filtrado
- Versión 2.0.0 en Model Registry

### Ejemplo 5: Comparar Múltiples Modelos

```bash
# Entrenar múltiples modelos con nombres descriptivos
for model in logistic_regression random_forest decision_tree xgboost; do
    uv run mlops-train \
        --X-train data/processed/Xtraintest.csv \
        --y-train data/processed/ytraintest.csv \
        --preprocessor models/preprocessor.joblib \
        --model $model \
        --mlflow-run-name "comparacion_${model}" \
        --model-version "1.0.0"
done
```

Luego puedes comparar los modelos en la UI de MLflow usando los nombres de los runs.

### Ejemplo 6: Entrenamiento con Versión Específica

```bash
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model logistic_regression \
    --model-version "1.3.0-beta" \
    --mlflow-run-name "lr_v1.3.0_beta"
```

## Visualización de Resultados

### Acceso a la UI de MLflow

Después de entrenar, el script proporciona URLs:

```
🏃 View run train_logistic_regression_20251102_163346 at: 
   https://mlflow-equipo-29.robomous.ai/#/experiments/1/runs/2e89e5c881d44273944bfca8238a80fe

🧪 View experiment at: 
   https://mlflow-equipo-29.robomous.ai/#/experiments/1
```

### Qué Ver en la UI

1. **Parámetros**:
   - `version`: Versión del modelo
   - `model_key`: Tipo de modelo
   - `model__*`: Hiperparámetros específicos
   - `cv_folds`, `cv_repeats`: Configuración de validación
   - `smote__*`: Configuración de SMOTE

2. **Métricas**:
   - `accuracy_test_mean`, `accuracy_test_std`
   - `roc_auc_test_mean`, `roc_auc_test_std`
   - `f1_test_mean`, `f1_test_std`
   - Y todas las demás métricas con sus desviaciones estándar

3. **Artefactos**:
   - `model/`: Modelo completo serializado
   - `results/model_results.json`: Resultados detallados en JSON
   - Para XGBoost: `xgb_model_only/`: Modelo XGBoost standalone

4. **Tags**:
   - `project`, `team`, `script`
   - `model_version`, `model_name`, `model_display_name`
   - `git_commit`, `timestamp`
   - Tags personalizados si se proporcionaron

## Model Registry

### Registro Automático

Por defecto, todos los modelos se registran en Model Registry con nombre basado en el tipo de modelo.

**Nombres por defecto:**
- `logistic_regression` → Logistic Regression
- `random_forest` → Random Forest
- `decision_tree` → Decision Tree
- `xgboost` → XGBoost
- `svm` → Support Vector Machine

### Versiones

Cada entrenamiento crea una nueva versión del modelo. Las versiones se incrementan automáticamente:
- Primera ejecución: Versión 1
- Segunda ejecución: Versión 2
- Y así sucesivamente...

### Promoción de Modelos

Desde la UI de MLflow puedes:
1. Marcar versiones como "Staging"
2. Marcar versiones como "Production"
3. Archivar versiones antiguas

### Cargar Modelo desde Registry

```python
import mlflow

# Cargar última versión
model = mlflow.sklearn.load_model("models:/logistic_regression/latest")

# Cargar versión específica
model = mlflow.sklearn.load_model("models:/logistic_regression/1")

# Cargar versión en producción
model = mlflow.sklearn.load_model("models:/logistic_regression/Production")
```

## Librerías Especializadas

### sklearn Models

Los modelos de sklearn (LogisticRegression, DecisionTree, RandomForest, SVM) usan `mlflow.sklearn`:

- Pipeline completo incluido (preprocessing + modelo)
- Compatibilidad total con scikit-learn
- Modelo listo para producción

### XGBoost

XGBoost usa **ambas** librerías:

1. **`mlflow.sklearn`**: Para el pipeline completo (preprocessing + XGBoost)
2. **`mlflow.xgboost`**: Para el modelo XGBoost standalone (referencia)

Esto permite:
- Usar el pipeline completo para predicciones
- Acceder al modelo XGBoost puro si es necesario
- Mejor tracking de hiperparámetros específicos de XGBoost

## Troubleshooting

### MLflow no se conecta

Verifica:
1. Credenciales de autenticación si el servidor las requiere
2. URL del tracking URI: `--mlflow-uri "http://correcto:puerto"`
3. Conectividad de red

### Error al registrar modelo

- Verifica que el modelo se entrenó correctamente
- Revisa permisos en el Model Registry
- Usa `--mlflow-disable` para entrenar sin MLflow si es necesario

### Métricas no aparecen

- Asegúrate de **no** usar `--no-evaluate`
- Verifica que la evaluación se completó sin errores
- Revisa los logs del script

## Mejores Prácticas

1. **Nombres descriptivos**: Usa `--mlflow-run-name` para identificar fácilmente los runs
2. **Versionado semántico**: Sigue `MAJOR.MINOR.PATCH` para `--model-version`
3. **Tags útiles**: Usa tags para filtrar y organizar experimentos
4. **Comparación de modelos**: Usa nombres consistentes para comparar fácilmente
5. **Documentación**: Anota cambios importantes en los tags

## Referencias

- [Documentación oficial de MLflow](https://mlflow.org/docs/latest/index.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)

