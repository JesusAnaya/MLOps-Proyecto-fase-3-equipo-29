# Pipeline DVC - Automatización del Flujo MLOps

Esta guía explica cómo usar el pipeline DVC para automatizar el flujo de machine learning del proyecto.

## ¿Qué es el Pipeline DVC?

El pipeline DVC (`dvc.yaml`) automatiza la ejecución lineal del **pipeline de machine learning**:

1. **Procesamiento de Datos**: Preparación y limpieza
2. **Ingeniería de Features**: Transformación y preprocesamiento
3. **Entrenamiento**: Entrenamiento del modelo de machine learning
4. **Despliegue**: Containerización con Docker (pendiente)

**Nota importante**: Los **tests son manejados por Make**, no por DVC. El pipeline DVC se enfoca exclusivamente en el flujo de ML.

El pipeline es **lineal y secuencial** para facilitar su comprensión y seguimiento.

### Ventajas

- **Reproducibilidad**: Cada ejecución es reproducible y versionada
- **Caché Inteligente**: Solo ejecuta etapas que han cambiado
- **Dependencias Automáticas**: Gestiona automáticamente las dependencias entre etapas

## Flujo del Pipeline

El pipeline DVC sigue un flujo **lineal y secuencial** enfocado en el pipeline de ML:

```
┌──────────────────┐
│  prepare_data    │  Preparación de datos
└────────┬─────────┘
         │
┌────────▼──────────────┐
│ prepare_features     │  Transformación de features
└────────┬─────────────┘
         │
┌────────▼──────────────┐
│     train            │  Entrenamiento del modelo
└───────────────────────┘
```

Cada etapa depende de la anterior, garantizando un flujo claro y fácil de seguir.

## Uso de los Scripts

Todos los scripts están definidos en `pyproject.toml` y se ejecutan usando `uv run`:

### Scripts Disponibles

| Script | Comando | Descripción |
|--------|---------|-------------|
| `mlops-prepare-data` | `uv run mlops-prepare-data` | Prepara y limpia los datos |
| `mlops-prepare-features` | `uv run mlops-prepare-features` | Transforma y preprocesa features |
| `mlops-train` | `uv run mlops-train` | Entrena el modelo |
| `mlops-predict` | `uv run mlops-predict` | Realiza predicciones |

### Ejecución del Pipeline

**Opción 1: Pipeline ML con DVC (Recomendado)**
```bash
# Usando DVC directamente (más simple)
dvc repro

# O usando Make
make pipeline-dvc
```

Este comando ejecuta el pipeline ML completo: preparación de datos, features y entrenamiento.

**Ejecutar etapas específicas del pipeline ML:**
```bash
# Solo procesamiento de datos
dvc repro prepare_data

# Solo preparación de features
dvc repro prepare_features

# Solo entrenamiento
dvc repro train

# Pipeline completo desde una etapa específica
dvc repro prepare_data prepare_features train
```

**Opción adicional: Tests + Pipeline ML completo**

Si deseas ejecutar los tests antes del pipeline ML, puedes usar:

```bash
make run-full-pipeline
```

Este comando ejecuta:
1. `make test` → Tests (unitarios + integración) vía Make
2. `make pipeline-dvc` → Pipeline ML vía DVC

**Ejecutar solo tests (sin pipeline ML):**
```bash
make test
```

## Ejecución Local

### Requisitos Previos

1. **Dependencias instaladas**:
   ```bash
   make init
   # O: uv sync
   ```

2. **Datos disponibles**:
   ```bash
   dvc pull data/raw/german_credit_modified.csv.dvc
   ```

3. **Credenciales AWS** (si necesitas descargar datos):
   ```bash
   aws configure
   ```

### Ejecutar el Pipeline

```bash
# Opción recomendada: Pipeline ML con DVC
dvc repro

# O usando Make
make pipeline-dvc

# Opción adicional: Tests + Pipeline ML completo
make run-full-pipeline

# Ver estado
dvc status

# Ver grafo del pipeline
dvc dag
```

## Etapas del Pipeline

**Nota**: Los tests son manejados por Make (ver `make test`), no por DVC. El pipeline DVC se enfoca exclusivamente en el flujo de machine learning.

### 1. Preparación de Datos (`prepare_data`)

**Script usado**: `uv run mlops-prepare-data --input data/raw/german_credit_modified.csv --save`

**Dependencias**: Datos raw, módulo dataset, configuración

**Salidas**: `data/processed/data_clean.csv`, `Xtraintest.csv`, `ytraintest.csv`

**Propósito**: Prepara y limpia los datos raw, dividiéndolos en conjuntos de entrenamiento y prueba.

### 2. Preparación de Features (`prepare_features`)

**Script usado**: `uv run mlops-prepare-features --train data/processed/Xtraintest.csv --save-preprocessor`

**Dependencias**: Datos procesados, módulo features, configuración

**Salidas**: `models/preprocessor.joblib`

### 3. Entrenamiento (`train`)

**Script usado**: `uv run mlops-train --X-train data/processed/Xtraintest.csv --y-train data/processed/ytraintest.csv --preprocessor models/preprocessor.joblib --model logistic_regression`

**Dependencias**: Datos de entrenamiento, preprocesador, módulo train, configuración

**Salidas**: `models/best_model.joblib`, `models/model_results.json` (métricas)

### 4. Despliegue Docker (`deploy_docker`) - PENDIENTE

Esta etapa está comentada en `dvc.yaml` hasta que el despliegue esté completamente implementado.

## Verificar Resultados

```bash
# Ver métricas del modelo
dvc metrics show
cat models/model_results.json

# Verificar archivos generados
ls -lh data/processed/*.csv
ls -lh models/*.joblib
```

## Troubleshooting

### Error: "Dependency not found"

```bash
# Descargar datos desde DVC
dvc pull data/raw/german_credit_modified.csv.dvc
```

### Error: "Stage failed"

```bash
# Ver logs detallados
dvc repro -v

# Ejecutar etapa manualmente
dvc repro --force train
```

### Pipeline no detecta cambios

```bash
# Forzar re-ejecución
dvc repro --force
```

## Referencias

- [Documentación de DVC](https://dvc.org/doc)
- [Guía de DVC en el Proyecto](dvc_configuracion.md)
- [Docker README](../docker/README.md)

---

**Equipo 29** - TC5044.10  
**Fecha**: Noviembre 2025
