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

**Nota**: Las dependencias de FastAPI (para el servicio web) no se instalan por defecto. Si necesitas usar el servicio web, instala las dependencias con `uv sync --extra web` y consulta [web_service/README.md](web_service/README.md).

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
proyecto_etapa_3/
├── mlops_project/          # Código fuente principal
│   ├── config.py           # Configuración centralizada
│   ├── dataset.py          # Carga y preparación de datos
│   ├── features.py         # Ingeniería de features
│   ├── plots.py            # Visualizaciones
│   └── modeling/            # Módulo de modelado
│       ├── train.py        # Entrenamiento / MLflow tracking
│       └── predict.py      # Predicción e inferencia
├── web_service/            # Servicio FastAPI
│   ├── app.py              # Aplicación principal
│   ├── service.py          # Lógica del servicio
│   └── models.py           # Modelos Pydantic
├── tests/                  # Tests automatizados
├── data/                   # Datos (versionados con DVC)
│   ├── raw/               # Datos originales
│   ├── processed/         # Datos procesados
│   ├── interim/           # Datos intermedios
│   └── external/          # Datos externos
├── models/                # Modelos entrenados (DVC)
├── notebooks/             # Notebooks de exploración
│   ├── exploring/         # EDA y análisis
│   ├── preprocessing/     # Preprocesamiento
│   └── modeling/          # Modelado
├── docs/                  # Documentación
├── docker/                 # Configuración Docker
│   ├── Dockerfile
│   └── docker-compose.yml
├── references/            # Referencias y recursos
├── reports/               # Reportes y figuras
├── scripts/               # Scripts de automatización
│   └── run_pipeline.sh    # Script para ejecutar pipeline completo
├── .github/               # Configuración GitHub Actions
│   └── workflows/         # Workflows de CI/CD
├── dvc.yaml               # Pipeline DVC
├── Makefile               # Comandos Make
├── pyproject.toml         # Configuración del proyecto
└── uv.lock                # Lock file de UV
```

## Uso Rápido

### Comandos Esenciales

```bash
# Ver todos los comandos disponibles
make help

# Ejecutar pipeline ML completo (recomendado)
dvc repro
# O usando Make
make pipeline-dvc

# Ejecutar todos los tests
make test

# Ejecutar tests + pipeline completo
make run-full-pipeline
```

### Servicio Web

Iniciar el servicio FastAPI (requiere `uv sync --extra web`):
```bash
uv run mlops-web-service
```

El servicio estará disponible en http://localhost:8000/docs

Para más detalles, ver [web_service/README.md](web_service/README.md).

## Documentación

Para información detallada sobre cada componente del proyecto:

- **[Comandos Make](docs/comandos_make.md)**: Lista completa de comandos Make disponibles
- **[Pipeline DVC](docs/dvc_pipeline.md)**: Guía del pipeline automatizado con DVC
- **[Scripts CLI](docs/scripts_detallados.md)**: Documentación completa de todos los scripts
- **[Testing](docs/testing.md)**: Información sobre tests y coverage
- **[MLflow](docs/mlflow_guia.md)**: Guía completa de MLflow con ejemplos
- **[Configuración](docs/configuracion.md)**: Guía completa de config.py
- **[DVC](docs/dvc_configuracion.md)**: Configuración detallada de DVC
- **[UV](docs/uv_guia.md)**: Ventajas y uso detallado de UV
- **[Modelos](docs/modelos_disponibles.md)**: Descripción de modelos soportados
- **[Pipeline Ejemplos](docs/pipeline_ejemplos.md)**: Ejemplos detallados del pipeline
- **[CI/CD Pipeline](docs/cicd_pipeline.md)**: Automatización con GitHub Actions
- **[Docker](docker/README.md)**: Guía de containerización del servicio
- **[Roadmap](docs/roadmap.md)**: Próximos pasos y mejoras futuras

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
