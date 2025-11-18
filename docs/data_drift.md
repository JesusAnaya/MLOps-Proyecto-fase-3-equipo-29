# Detección de Data Drift con Evidently AI

Esta guía explica cómo simular y detectar data drift en el proyecto usando Evidently AI.

## ¿Qué es Data Drift?

El **data drift** (deriva de datos) ocurre cuando la distribución de los datos de entrada cambia con el tiempo, lo que puede degradar el rendimiento del modelo. Este proyecto implementa:

1. **Simulación de drift**: Genera datos con múltiples tipos de deriva
2. **Detección con Evidently AI**: Identifica automáticamente cambios en las distribuciones
3. **Evaluación de impacto**: Mide cómo afecta el drift al rendimiento del modelo

## Tipos de Drift Simulados

El script `scripts/simulate_data_drift.py` simula los siguientes tipos de drift:

### 1. Drift Categórico (`laufkont`)
Cambia la distribución de valores categóricos:
- Distribución original modificada a: `{1: 0.50, 2: 0.30, 3: 0.15, 4: 0.05}`

### 2. Drift Numérico (`hoehe`)
Reduce valores numéricos aleatoriamente entre 40% y 90% del valor original.

### 3. Drift Conceptual (`laufzeit` vs `verw`)
Modifica la relación entre variables: aumenta `laufzeit` cuando `verw` es 3 o 4 y `laufzeit` es menor a 36 meses.

### 4. Label Shift (`kredit`)
Cambia la distribución de la variable objetivo a 50% de clase positiva (originalmente diferente).

### 5. Drift Categórico (`verw`)
Modifica la distribución de `verw` a: `{0: 0.10, 1: 0.20, 2: 0.40, 3: 0.30}`

### 6. Drift Numérico (`laufzeit`)
Aumenta la media de `laufzeit` en un 30%.

## Instalación de Dependencias

Evidently AI está disponible como dependencia opcional. Instálalo con:

```bash
uv sync --extra evidently
```

Esto instalará Evidently AI y todas sus dependencias en el entorno del proyecto.

## Reproducción Local

### Paso 1: Preparar Datos Base

Asegúrate de tener los datos procesados:

```bash
# Ejecutar pipeline hasta obtener data_clean.csv
dvc repro prepare_data
# O manualmente:
uv run mlops-prepare-data --input data/raw/german_credit_modified.csv --save
```

### Paso 2: Simular Data Drift

Genera un dataset con drift simulado:

```bash
uv run python scripts/simulate_data_drift.py
```

**Salida esperada:**
- Archivo: `data/processed/drift_south_test_data.csv`
- Mensajes de confirmación para cada tipo de drift aplicado
- Estadísticas de distribuciones finales

**Parámetros configurables** (en el script):
- `DRIFT_SAMPLE_SIZE`: Proporción de datos a modificar (default: 0.5)
- `NEW_DISTRIBUTION_WEIGHTS`: Distribuciones para variables categóricas
- `MIN_REDUCTION_FACTOR` / `MAX_REDUCTION_FACTOR`: Rango de reducción numérica

### Paso 3: Detectar Drift con Evidently AI

Ejecuta la detección automática:

```bash
uv run python scripts/check_drift.py
```

**Requisitos previos:**
- Modelo entrenado en `models/best_model.joblib`
- Datos de referencia: `data/processed/data_clean.csv`
- Datos con drift: `data/processed/drift_south_test_data.csv`

**Salida:**
- `reports/evidently_drift_results.json`: Resultados detallados en JSON
- `reports/evidently_drift_report.html`: Reporte visual en HTML

**Métricas incluidas:**
- Número de columnas con drift detectado
- P-values para cada variable (Kolmogorov-Smirnov para numéricas, Chi-cuadrado para categóricas)
- Threshold de detección: 0.05 (configurable)

### Paso 4: Evaluar Impacto en el Modelo

Mide cómo el drift afecta el rendimiento:

```bash
uv run python scripts/data_drift_evaluation.py
```

**Salida:**
- `models/drift_results.json`: Métricas de desempeño con datos con drift
- Métricas incluidas: `accuracy`, `f1_score`, `recall`, `precision`, `roc_auc_score`

**Comparación:**
Compara `models/drift_results.json` con `models/model_results.json` (resultados con datos originales).

### Paso 5: Visualizar Diferencias (Opcional)

Genera gráficos comparativos:

```bash
uv run python scripts/plot_differences.py
```

**Salida:**
- `scripts/plots/categorical_laufkont.png`
- `scripts/plots/categorical_verw.png`
- `scripts/plots/continuous_hoehe.png`
- `scripts/plots/continuous_laufzeit.png`

Cada gráfico muestra:
- Distribuciones originales vs. con drift
- Estadísticas de prueba (Chi-cuadrado o Kolmogorov-Smirnov)
- P-values

## Flujo Completo

```bash
# 0. Instalar dependencias de Evidently (solo la primera vez)
uv sync --extra evidently

# 1. Preparar datos base
dvc repro prepare_data

# 2. Simular drift
uv run python scripts/simulate_data_drift.py

# 3. Detectar drift
uv run python scripts/check_drift.py

# 4. Evaluar impacto
uv run python scripts/data_drift_evaluation.py

# 5. Visualizar (opcional)
uv run python scripts/plot_differences.py
```

## Interpretación de Resultados

### Reporte Evidently (`evidently_drift_results.json`)

- **`DriftedColumnsCount`**: Número y proporción de columnas con drift
- **`ValueDrift`**: Para cada columna:
  - `p-value < 0.05`: Drift detectado (distribución cambió significativamente)
  - `p-value >= 0.05`: Sin drift detectado

### Métricas de Desempeño (`drift_results.json`)

Compara con `model_results.json`:
- **Disminución en accuracy/F1**: Indica degradación por drift
- **Cambio en recall/precision**: Puede indicar shift en distribución de clases

## Archivos Generados

```
data/processed/
  └── drift_south_test_data.csv          # Datos con drift simulado

reports/
  ├── evidently_drift_results.json       # Resultados de detección (JSON)
  └── evidently_drift_report.html        # Reporte visual (HTML)

models/
  └── drift_results.json                 # Métricas de desempeño con drift

scripts/plots/
  ├── categorical_laufkont.png
  ├── categorical_verw.png
  ├── continuous_hoehe.png
  └── continuous_laufzeit.png
```

## Referencias

- [Evidently AI Documentation](https://docs.evidentlyai.com/)

