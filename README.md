# Proyecto MLOps - Equipo 29

Proyecto de MLOps para la materia TC5044.10

------------------------
Implementación de pipeline automatizado para clasificación de crédito bancario usando el dataset South German Credit.

## Información del Equipo

**Equipo:** 29  
**Materia:** Operaciones de aprendizaje automático

### Integrantes del Equipo:
- Jesús Armando Anaya Orozco
- Oliver Josué De León Milian
- Isaura Yutsil Flores Escamilla
- Ovidio Alejandro Hernández Ruano
- Owen Jáuregui Borbón

## Requisitos

- **Python**: 3.12.0
- **UV**: Gestor de paquetes Python (se instala automáticamente)
- **DVC**: Data Version Control con soporte para S3
- **AWS CLI**: Para acceso a bucket S3
- **Git**: Para control de versiones de código

## Instalación

### 1. Clonar el Repositorio

```bash
git clone git@github.com:JesusAnaya/MLOps-Proyecto-fase-3-equipo-29.git
cd MLOps-Proyecto-fase-3-equipo-29
```

### 2. Instalar UV

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Inicializar el Proyecto

```bash
make init
```

Este comando crea el entorno virtual e instala todas las dependencias base del proyecto.

**Nota**: Las dependencias de FastAPI (para el servicio web) no se instalan por defecto. Si necesitas usar el servicio web, ver la sección [Instalación del Servicio Web](#instalación-del-servicio-web).

### 4. Configurar DVC

**IMPORTANTE: NO ejecutar `dvc init`** - El proyecto ya está configurado.

**Configurar credenciales AWS:**
```bash
aws configure
```

Se solicitarán:
- AWS Access Key ID
- AWS Secret Access Key
- Default region name: `us-east-1`
- Default output format: `json`

**Sincronizar datos:**
```bash
dvc pull  # Descarga datos desde S3
dvc status  # Verifica estado
```

Para más detalles sobre DVC, ver [docs/dvc_configuracion.md](docs/dvc_configuracion.md).

## Estructura del Proyecto

```
proyecto_etapa_2/
├── mlops_project/          # Código fuente principal
│   ├── config.py           # Configuración centralizada
│   ├── dataset.py          # Carga y preparación de datos
│   ├── features.py         # Ingeniería de features
│   └── modeling/           # Módulo de modelado
│       ├── train.py        # Entrenamiento / MLflow tracking
│       └── predict.py      # Predicción e inferencia
├── tests/                  # Tests automatizados
├── data/                   # Datos (versionados con DVC)
│   ├── raw/               # Datos originales
│   └── processed/         # Datos procesados
├── models/                # Modelos entrenados (DVC)
├── notebooks/             # Notebooks de exploración
├── docs/                  # Documentación adicional
└── pyproject.toml         # Configuración del proyecto
```

## Uso Rápido

### Comandos Principales

```bash
# Ver todos los comandos disponibles
make help

# Ejecutar pipeline completo
make pipeline

# Ejecutar todos los tests (unitarios + integración)
make test

# Ejecutar tests con reporte de coverage
make test-cov
```

### Testing

**Ejecutar todos los tests con un solo comando:**

```bash
# Opción recomendada: usando Make
make test

# O directamente con UV
uv run pytest
```

Este comando ejecuta:
- Todos los tests unitarios (128 tests)
- Todos los tests de integración (10 tests)
- Total: 138 tests
- Genera reporte de coverage automáticamente

**Ejecutar tests por separado:**

```bash
# Solo tests unitarios - usando Make
make test-unit

# Solo tests de integración - usando Make
make test-integration

# O directamente con pytest usando marcadores
uv run pytest -m unit          # Solo tests unitarios
uv run pytest -m integration  # Solo tests de integración
```

**Opciones adicionales:**
```bash
# Excluir tests lentos
uv run pytest -m "not slow"

# Ver reporte de coverage en HTML
make test-cov && open htmlcov/index.html

# Ejecutar test específico
uv run pytest tests/test_config.py -v
```

**Nota:** Los tests están marcados con `@pytest.mark.unit` o `@pytest.mark.integration` para permitir ejecutarlos por separado usando los marcadores de pytest (`-m unit` o `-m integration`).

Para documentación completa de testing, ver [docs/testing.md](docs/testing.md).

### Scripts Disponibles

El proyecto incluye 4 scripts CLI principales:

1. **Preparación de datos:**
```bash
uv run mlops-prepare-data --input data/raw/german_credit_modified.csv --save
```

2. **Preparación de features:**
```bash
uv run mlops-prepare-features --train data/processed/Xtraintest.csv --save-preprocessor
```

3. **Entrenamiento:**
```bash
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model logistic_regression
```

4. **Predicción:**
```bash
uv run mlops-predict \
    --model models/best_model.joblib \
    --X-test data/processed/Xtraintest.csv \
    --y-test data/processed/ytraintest.csv \
    --save
```

Para documentación detallada de scripts, ver [docs/scripts_detallados.md](docs/scripts_detallados.md).

### Servicio Web

Si has instalado las dependencias web (`uv sync --extra web`), puedes iniciar el servicio FastAPI:

```bash
# Opción 1: Usando el script proporcionado
uv run mlops-web-service

# Opción 2: Usando uvicorn directamente
uv run uvicorn web_service.app:app --reload --host 0.0.0.0 --port 8000
```

El servicio estará disponible en:
- **API**: http://localhost:8000
- **Documentación interactiva**: http://localhost:8000/docs
- **Documentación alternativa**: http://localhost:8000/redoc

Para más detalles sobre el servicio web, ver [web_service/README.md](web_service/README.md).

## MLflow

El proyecto está integrado con MLflow para tracking de experimentos, versionado de modelos y registro en Model Registry.

### Configuración

La configuración de MLflow está en `mlops_project/config.py`:

```python
MLFLOW_TRACKING_URI = "https://mlflow-equipo-29.robomous.ai"
MLFLOW_EXPERIMENT_NAME = "equipo-29"
MLFLOW_MODEL_VERSION = "0.1.0"
```

### Uso Básico

**Entrenar con MLflow (automático):**
```bash
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model logistic_regression
```

El script automáticamente:
- Registra el run en MLflow
- Loggea hiperparámetros y métricas
- Registra el modelo en Model Registry
- Genera URLs para ver los resultados

**Personalizar el run:**
```bash
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model random_forest \
    --mlflow-run-name "experimento_rf_v1" \
    --model-version "1.2.0"
```

**Deshabilitar MLflow:**
```bash
uv run mlops-train ... --mlflow-disable
```

Para guía completa de MLflow con ejemplos detallados, ver [docs/mlflow_guia.md](docs/mlflow_guia.md).

## Actualizar Paquetes

```bash
# Actualizar dependencias
uv sync

# Agregar nueva dependencia
uv add nombre_paquete

# Actualizar dependencia específica
uv add "nombre_paquete>=version"
```

## Documentación Adicional

- **[Configuración del Proyecto](docs/configuracion.md)**: Guía completa de config.py y cómo configurarlo
- **[Guía de UV](docs/uv_guia.md)**: Ventajas y uso detallado de UV
- **[Scripts Detallados](docs/scripts_detallados.md)**: Documentación completa de todos los scripts
- **[Guía de MLflow](docs/mlflow_guia.md)**: Guía completa con ejemplos de MLflow
- **[Pipeline y Ejemplos](docs/pipeline_ejemplos.md)**: Ejemplos detallados del pipeline
- **[Configuración DVC](docs/dvc_configuracion.md)**: Configuración detallada de DVC
- **[Testing](docs/testing.md)**: Información sobre tests y coverage
- **[Modelos Disponibles](docs/modelos_disponibles.md)**: Descripción de modelos soportados
- **[Comandos Make](docs/comandos_make.md)**: Referencia de comandos Make
- **[Roadmap](docs/roadmap.md)**: Próximos pasos y mejoras futuras
- **[Web Service](web_service/README.md)**: Documentación del servicio web FastAPI
- **[Docker](docker/README.md)**: Guía de containerización del servicio

Reglas de estilo
-------------------------
- Todo el código escrito en Inglés.
- Comentarios dentro del código escritos en Español.
- Documentación en archivos Markdown escrita en Español.

-------------------------
Maestría en Inteligencia Artificial Aplicada - MNA

**Fecha**: Noviembre 2025  
**Versión**: 0.0.1  
**Python**: 3.12.0
