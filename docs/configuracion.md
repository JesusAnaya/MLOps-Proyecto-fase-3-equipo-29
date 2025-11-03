# Configuración del Proyecto (config.py)

Este documento explica el propósito y la configuración del archivo `mlops_project/config.py`, que centraliza todas las configuraciones del proyecto MLOps.

## Tabla de Contenidos

- [Propósito](#propósito)
- [Estructura General](#estructura-general)
- [Secciones de Configuración](#secciones-de-configuración)
- [Cómo Configurar](#cómo-configurar)
- [Ejemplos de Modificación](#ejemplos-de-modificación)
- [Mejores Prácticas](#mejores-prácticas)

## Propósito

El archivo `config.py` sirve como **único punto de configuración** del proyecto, centralizando:

- Rutas de datos y modelos
- Hiperparámetros de modelos
- Parámetros de preprocesamiento
- Configuración de validación cruzada
- Configuración de MLflow
- Definición de features
- Semillas aleatorias para reproducibilidad

**Ventajas:**
- Fácil mantenimiento: cambios en un solo lugar
- Reproducibilidad: semillas y parámetros centralizados
- Consistencia: todos los módulos usan las mismas configuraciones
- Documentación: valores claramente definidos

## Estructura General

El archivo está organizado en secciones lógicas:

1. **Rutas del proyecto**: Directorios y paths
2. **Configuración de datos**: Columnas objetivo, archivos
3. **Features**: Definición de tipos de variables
4. **Preprocesamiento**: Estrategias de imputación, escalado
5. **Validación cruzada**: Folds, repeats, método
6. **Modelos**: Hiperparámetros de cada modelo
7. **MLflow**: Tracking URI, experimentos, versionado
8. **Funciones utilitarias**: Helpers para paths

## Secciones de Configuración

### 1. Rutas del Proyecto

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
```

**Propósito**: Define las rutas base del proyecto de forma relativa.

**No modificar normalmente**, a menos que cambies la estructura de directorios.

### 2. Configuración de Datos

```python
TARGET_COLUMN = "kredit"
MIXED_TYPE_COLUMN = "mixed_type_col"
RAW_DATA_ORIGINAL = "german_credit_original.csv"
RAW_DATA_MODIFIED = "german_credit_modified.csv"
```

**Propósito**: Define la columna objetivo y nombres de archivos.

**Modificar si**: Cambias de dataset o nombre de columna objetivo.

### 3. Semilla Aleatoria

```python
RANDOM_SEED = 42
```

**Propósito**: Garantiza reproducibilidad en todos los procesos aleatorios.

**Modificar si**: Quieres experimentar con diferentes semillas (pero documenta el cambio).

**⚠️ Importante**: Cambiar la semilla afectará todos los resultados (entrenamiento, validación, etc.).

### 4. División de Datos

```python
TEST_SIZE = 0.3
STRATIFY = True
```

**Propósito**: Configura cómo se dividen train/test.

- `TEST_SIZE`: Proporción para test (0.3 = 30%)
- `STRATIFY`: Mantener proporción de clases

**Modificar si**: Necesitas diferente proporción de división.

### 5. Definición de Features

```python
NUMERIC_FEATURES: List[str] = ["laufzeit", "hoehe", "alter"]
ORDINAL_FEATURES: List[str] = ["beszeit", "rate", ...]
NOMINAL_FEATURES: List[str] = ["laufkont", "moral", ...]
```

**Propósito**: Clasifica las variables por tipo para aplicar transformaciones correctas.

**Modificar si**: 
- Agregas nuevas features al dataset
- Cambias el tipo de alguna feature existente

**Ejemplo de modificación:**
```python
# Agregar nueva feature numérica
NUMERIC_FEATURES: List[str] = [
    "laufzeit",
    "hoehe",
    "alter",
    "nueva_feature_numerica"  # Nueva
]
```

### 6. Validación de Variables Categóricas

```python
CATEGORICAL_VALIDATION_RULES: Dict[str, List[int]] = {
    "laufkont": [1, 2, 3, 4],
    "moral": [0, 1, 2, 3, 4],
    ...
}
```

**Propósito**: Define valores válidos para cada variable categórica (usado en limpieza de datos).

**Modificar si**: Agregas nuevas categorías o cambias los valores válidos.

### 7. Detección de Outliers

```python
OUTLIER_METHOD = "IQR"  # Opciones: 'IQR' o 'Percentiles'
OUTLIER_PERCENTILES: Tuple[float, float] = (0.05, 0.95)
OUTLIER_VARIABLES: List[str] = ["laufzeit", "wohnzeit", "alter", ...]
```

**Propósito**: Configura cómo se detectan y manejan outliers.

**Modificar si**:
- Cambias el método de detección
- Agregas/quitas variables para análisis de outliers

**Ejemplo:**
```python
# Usar método de percentiles
OUTLIER_METHOD = "Percentiles"
OUTLIER_PERCENTILES = (0.01, 0.99)  # Más estricto

# Agregar variable
OUTLIER_VARIABLES = ["laufzeit", "wohnzeit", "alter", "nueva_variable"]
```

### 8. Preprocesamiento

```python
NUMERIC_IMPUTE_STRATEGY = "median"
CATEGORICAL_IMPUTE_STRATEGY = "most_frequent"
NUMERIC_SCALER_RANGE: Tuple[int, int] = (1, 2)
```

**Propósito**: Define estrategias para imputación y escalado.

**Opciones disponibles:**
- `NUMERIC_IMPUTE_STRATEGY`: `"mean"`, `"median"`, `"mode"`, o constante numérica
- `CATEGORICAL_IMPUTE_STRATEGY`: `"most_frequent"` o constante categórica
- `NUMERIC_SCALER_RANGE`: Rango para MinMaxScaler (min, max)

**Modificar si**: Quieres probar diferentes estrategias de preprocesamiento.

### 9. Validación Cruzada

```python
CV_FOLDS = 5
CV_REPEATS = 3
CV_METHOD = "RepeatedStratifiedKFold"
```

**Propósito**: Configura la validación cruzada.

- `CV_FOLDS`: Número de folds (5 = 5-fold CV)
- `CV_REPEATS`: Número de repeticiones (3 = 15 evaluaciones totales)
- `CV_METHOD`: Método de validación (actualmente solo `RepeatedStratifiedKFold`)

**Modificar si**: Necesitas más/menos folds o repeticiones.

**⚠️ Consideraciones**:
- Más folds = más tiempo de entrenamiento, más confiabilidad
- Más repeats = más robustez estadística

### 10. Hiperparámetros de Modelos

#### Logistic Regression (Modelo Base)

```python
BEST_MODEL_PARAMS: Dict[str, any] = {
    "penalty": "l2",
    "solver": "newton-cg",
    "max_iter": 1000,
    "C": 1,
    "random_state": RANDOM_SEED,
}
```

#### Todos los Modelos

```python
AVAILABLE_MODELS: Dict[str, Dict] = {
    "logistic_regression": {...},
    "random_forest": {...},
    "decision_tree": {...},
    "svm": {...},
    "xgboost": {...},
}
```

**Propósito**: Define los hiperparámetros por defecto para cada modelo.

**Modificar si**: Quieres ajustar hiperparámetros o agregar nuevos modelos.

**Ejemplo: Ajustar Random Forest**
```python
"random_forest": {
    "name": "Random Forest",
    "params": {
        "n_estimators": 300,  # Aumentado de 200
        "max_depth": 5,        # Aumentado de 3
        "min_samples_split": 30,  # Reducido de 50
        "random_state": RANDOM_SEED,
    },
}
```

### 11. Configuración de SMOTE

```python
SMOTE_CONFIG: Dict[str, any] = {
    "method": "BorderlineSMOTE",
    "random_state": RANDOM_SEED,
    "k_neighbors": 5,
    "m_neighbors": 10,
}
```

**Propósito**: Configura el balanceo de clases.

**Opciones de método**:
- `"SMOTE"`: SMOTE estándar
- `"BorderlineSMOTE"`: SMOTE con enfoque en bordes (recomendado)
- `"ADASYN"`: Adaptive Synthetic Sampling

**Modificar si**: Quieres probar diferentes métodos de balanceo.

### 12. Métricas de Evaluación

```python
EVALUATION_METRICS: List[str] = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
]
```

**Propósito**: Define qué métricas se calculan durante la evaluación.

**Modificar si**: Quieres agregar/quitar métricas.

**Métricas disponibles** (sklearn):
- `"accuracy"`, `"precision"`, `"recall"`, `"f1"`
- `"roc_auc"`, `"average_precision"`
- Cualquier scorer de sklearn

### 13. Configuración de MLflow

```python
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow-equipo-29.robomous.ai")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "equipo-29")
MLFLOW_MODEL_VERSION = os.getenv("MODEL_VERSION", "0.1.0")
MLFLOW_REGISTER_MODELS = True
MLFLOW_ENABLE_AUTOLOG = False
```

**Propósito**: Configura la integración con MLflow.

**Modificar si**: 
- Cambias el servidor de MLflow
- Quieres un experimento diferente
- Cambias la versión por defecto

**Recomendación**: Usar variables de entorno para sobrescribir:

```bash
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_EXPERIMENT="mi-experimento"
export MODEL_VERSION="2.0.0"
```

## Cómo Configurar

### Verificar Configuración Actual

```bash
# Ejecutar config.py directamente
uv run python mlops_project/config.py
```

Esto mostrará un resumen de la configuración actual.

### Modificar Configuración

1. **Editar el archivo**: `mlops_project/config.py`
2. **Seguir la estructura existente**: Mantener el formato y tipos
3. **Documentar cambios**: Agregar comentarios si es necesario
4. **Probar**: Ejecutar tests o pipeline para verificar

### Usar Variables de Entorno

Para configuración dinámica (especialmente MLflow):

```bash
# Antes de ejecutar scripts
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_EXPERIMENT="experimento-local"
export MODEL_VERSION="1.0.0"

# Ejecutar entrenamiento
uv run mlops-train ...
```

## Ejemplos de Modificación

### Ejemplo 1: Agregar Nueva Feature Numérica

```python
# En la sección NUMERIC_FEATURES
NUMERIC_FEATURES: List[str] = [
    "laufzeit",
    "hoehe",
    "alter",
    "nueva_feature",  # Nueva feature
]

# Si es necesario, agregar a detección de outliers
OUTLIER_VARIABLES: List[str] = [
    "laufzeit",
    "wohnzeit",
    "alter",
    "bishkred",
    "hoehe",
    "nueva_feature",  # Si aplica
]
```

### Ejemplo 2: Cambiar Estrategia de Preprocesamiento

```python
# Cambiar imputación numérica a media en lugar de mediana
NUMERIC_IMPUTE_STRATEGY = "mean"

# Cambiar rango de escalado
NUMERIC_SCALER_RANGE: Tuple[int, int] = (0, 1)  # En lugar de (1, 2)
```

### Ejemplo 3: Ajustar Validación Cruzada

```python
# Aumentar robustez estadística
CV_FOLDS = 10        # De 5 a 10
CV_REPEATS = 5       # De 3 a 5
# Total: 10 × 5 = 50 evaluaciones (más tiempo, más confiabilidad)
```

### Ejemplo 4: Modificar Hiperparámetros de XGBoost

```python
"xgboost": {
    "name": "XGBoost",
    "params": {
        "booster": "gbtree",
        "n_estimators": 200,        # Aumentado de 100
        "max_depth": 5,             # Aumentado de 3
        "learning_rate": 0.05,      # Aumentado de 0.01
        "subsample": 0.8,           # Aumentado de 0.7
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    },
}
```

### Ejemplo 5: Cambiar Método SMOTE

```python
SMOTE_CONFIG: Dict[str, any] = {
    "method": "SMOTE",  # Cambiar de BorderlineSMOTE a SMOTE estándar
    "random_state": RANDOM_SEED,
    "k_neighbors": 5,
}
# Nota: m_neighbors solo se usa en BorderlineSMOTE
```

## Mejores Prácticas

### 1. Documentar Cambios

Cuando modifiques valores, agrega comentarios explicando el motivo:

```python
# Cambiado de 5 a 10 para mayor robustez estadística
CV_FOLDS = 10
```

### 2. Mantener Reproducibilidad

- **No cambiar `RANDOM_SEED`** a menos que sea intencional
- **Versionar cambios** importantes en Git
- **Documentar** cambios en commit messages

### 3. Probar Cambios

Después de modificar configuración:

```bash
# Ejecutar tests
make test

# Verificar configuración
uv run python mlops_project/config.py

# Ejecutar pipeline completo
make pipeline
```

### 4. Usar Variables de Entorno para MLflow

En lugar de editar el archivo para diferentes entornos:

```bash
# Desarrollo local
export MLFLOW_TRACKING_URI="http://localhost:5000"

# Producción
export MLFLOW_TRACKING_URI="https://mlflow-equipo-29.robomous.ai"
```

### 5. Validar Tipos

Asegúrate de mantener los tipos correctos:

- `List[str]` para listas de strings
- `Dict[str, any]` para diccionarios de configuración
- `Tuple[int, int]` para rangos

### 6. No Modificar Funciones Utilitarias

Las funciones `get_data_path()`, `get_model_path()`, y `ensure_directories()` están optimizadas. No modificarlas a menos que sea necesario.

## Estructura del Archivo

```
config.py
│
├── Rutas (líneas 14-24)
├── Datos (líneas 26-36)
├── Semilla y división (líneas 38-43)
├── Features (líneas 45-73)
├── Validación categóricas (líneas 75-99)
├── Outliers (líneas 101-104)
├── Preprocesamiento (líneas 106-109)
├── Validación cruzada (líneas 111-114)
├── Modelos (líneas 116-182)
├── Archivos salida (líneas 184-187)
├── MLflow (líneas 189-205)
└── Funciones utilitarias (líneas 208-266)
```

## Referencias

- [Documentación de scikit-learn](https://scikit-learn.org/stable/)
- [Documentación de XGBoost](https://xgboost.readthedocs.io/)
- [Guía de MLflow](mlflow_guia.md)
- [Scripts Detallados](scripts_detallados.md)

## Troubleshooting

### Error: "Feature no encontrada"

**Problema**: Feature definida en `NUMERIC_FEATURES` pero no existe en el dataset.

**Solución**: Verificar que el nombre de la feature coincide exactamente con la columna del CSV.

### Error: "Invalid configuration"

**Problema**: Tipo incorrecto en alguna configuración.

**Solución**: Verificar tipos (List, Dict, Tuple) y sintaxis Python.

### Cambios no se reflejan

**Problema**: Modificaste config.py pero los cambios no aparecen.

**Solución**: 
1. Verificar que guardaste el archivo
2. Reiniciar Python/interpreter
3. Verificar que no hay caché: `uv run python -c "import mlops_project.config; print(mlops_project.config.RANDOM_SEED)"`

