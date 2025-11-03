# Modelos Disponibles

Descripción de los modelos de clasificación soportados en el proyecto.

## Modelos Soportados

### 1. Logistic Regression (Recomendado)

**Nombre CLI**: `logistic_regression`

**Características:**
- Mejor performance en validación cruzada
- ROC-AUC: ~0.77
- Interpretable y rápido
- Ideal para problemas de clasificación binaria

**Hiperparámetros por defecto:**
```python
{
    "penalty": "l2",
    "solver": "newton-cg",
    "max_iter": 1000,
    "C": 1,
    "random_state": 42
}
```

**Uso:**
```bash
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model logistic_regression
```

### 2. Random Forest

**Nombre CLI**: `random_forest`

**Características:**
- Ensemble method robusto
- Buena capacidad de generalización
- Maneja bien features no lineales
- Menos propenso a overfitting que árboles individuales

**Hiperparámetros por defecto:**
```python
{
    "n_estimators": 200,
    "max_depth": 3,
    "min_samples_split": 50,
    "random_state": 42
}
```

**Uso:**
```bash
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model random_forest
```

### 3. Decision Tree

**Nombre CLI**: `decision_tree`

**Características:**
- Modelo simple e interpretable
- Útil para análisis exploratorio
- Visualizaciones claras del árbol
- Base para Random Forest

**Hiperparámetros por defecto:**
```python
{
    "max_depth": 3,
    "min_samples_split": 20,
    "random_state": 42
}
```

**Uso:**
```bash
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model decision_tree
```

### 4. Support Vector Machine (SVM)

**Nombre CLI**: `svm`

**Características:**
- Kernel RBF
- Bueno para clasificación no lineal
- Efectivo con datos de alta dimensionalidad
- Puede ser lento en datasets grandes

**Hiperparámetros por defecto:**
```python
{
    "kernel": "rbf",
    "C": 10,
    "gamma": "auto",
    "random_state": 42
}
```

**Uso:**
```bash
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model svm
```

### 5. XGBoost

**Nombre CLI**: `xgboost`

**Características:**
- Gradient boosting avanzado
- Alta performance en muchos casos
- Maneja bien datasets grandes
- Requiere más tiempo de entrenamiento

**Hiperparámetros por defecto:**
```python
{
    "booster": "gbtree",
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.01,
    "subsample": 0.7,
    "random_state": 42,
    "n_jobs": -1
}
```

**Uso:**
```bash
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model xgboost
```

**Nota**: XGBoost usa librerías especializadas de MLflow para mejor tracking.

## Comparación de Modelos

### Performance Esperada

| Modelo | ROC-AUC Aprox. | Velocidad | Interpretabilidad |
|--------|----------------|-----------|-------------------|
| Logistic Regression | 0.77 | ⚡⚡⚡ | ⭐⭐⭐ |
| Random Forest | 0.72 | ⚡⚡ | ⭐⭐ |
| Decision Tree | 0.69 | ⚡⚡⚡ | ⭐⭐⭐ |
| SVM | 0.71 | ⚡ | ⭐ |
| XGBoost | 0.75 | ⚡⚡ | ⭐ |

### Cuándo Usar Cada Modelo

- **Logistic Regression**: Producción, interpretabilidad importante
- **Random Forest**: Datos complejos, robustez
- **Decision Tree**: Exploración, interpretabilidad máxima
- **SVM**: Datos de alta dimensionalidad
- **XGBoost**: Máxima performance, competencias

## Configuración de Modelos

Los hiperparámetros están configurados en `mlops_project/config.py`:

```python
AVAILABLE_MODELS: Dict[str, Dict] = {
    "logistic_regression": {...},
    "random_forest": {...},
    ...
}
```

Para modificar hiperparámetros, edita `config.py` y re-entrena.

