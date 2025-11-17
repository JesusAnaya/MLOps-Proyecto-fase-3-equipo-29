# Comandos Make

Referencia completa de todos los comandos Make disponibles en el proyecto.

## Gestión del Proyecto

```bash
make help              # Mostrar todos los comandos disponibles
make init              # Inicializar proyecto (crear venv e instalar dependencias)
make requirements      # Instalar/actualizar dependencias con uv sync
make clean             # Limpiar archivos temporales y cachés
```

## Gestión de Datos con DVC

```bash
make dvc-pull          # Descargar datos desde S3 con DVC
make dvc-push          # Subir datos a S3 con DVC
make dvc-status        # Ver estado de archivos DVC
make dvc-add           # Mostrar ayuda para agregar archivos a DVC
```

**Nota**: Para comandos DVC más detallados, ver [dvc_configuracion.md](dvc_configuracion.md).

## Calidad de Código

```bash
make lint              # Verificar código con ruff (usa UV internamente)
make format            # Formatear código con ruff (usa UV internamente)
make check             # Ejecutar lint + tests (todo con UV)
```

## Pipeline

```bash
make prepare-data      # Preparar datos (mlops-prepare-data)
make prepare-features # Preparar features (mlops-prepare-features)
make train             # Entrenar modelo (mlops-train)
make predict           # Realizar predicciones (mlops-predict)
make pipeline          # Ejecutar pipeline completo (prepare-data + prepare-features + train)
make pipeline-dvc      # Ejecutar pipeline ML con DVC (recomendado)
make run-full-pipeline # Ejecutar tests + pipeline ML completo (opción adicional)
```

**Explicación:**
- `make pipeline`: Ejecuta el pipeline completo usando Make (prepare-data + prepare-features + train)
- `make pipeline-dvc`: Ejecuta el pipeline ML usando DVC (`dvc repro`). Esta es la opción recomendada para el pipeline ML. No incluye tests.
- `make run-full-pipeline`: Ejecuta primero los tests (`make test`) y luego el pipeline ML (`make pipeline-dvc`). Opción adicional si deseas ejecutar tests antes del pipeline.

## Testing

```bash
make test              # Ejecutar todos los tests (unitarios + integración)
make test-unit         # Ejecutar solo tests unitarios
make test-integration  # Ejecutar solo tests de integración
make test-cov          # Ejecutar tests con reporte de coverage
```

**Detalles:**
- `make test`: Ejecuta todos los tests (138 tests: 128 unitarios + 10 integración)
- `make test-unit`: Solo tests unitarios (128 tests)
- `make test-integration`: Solo tests de integración (10 tests)
- `make test-cov`: Tests con reporte de coverage (HTML y terminal)

**Nota**: Para más detalles sobre testing, ver [testing.md](testing.md).

## Utilidades

```bash
make tree              # Mostrar estructura del proyecto
make scripts           # Listar scripts disponibles
make python-version    # Mostrar versión de Python
```

## Ejemplos de Uso

### Inicializar Proyecto Nuevo

```bash
# Clonar repositorio
git clone ...

# Inicializar
make init

# Configurar DVC
aws configure
make dvc-pull
```

### Desarrollo Diario

```bash
# Actualizar dependencias
make requirements

# Verificar código
make check

# Ejecutar pipeline ML (recomendado)
make pipeline-dvc

# Opción adicional: Tests + Pipeline ML completo
make run-full-pipeline

# Verificar estado de datos
make dvc-status
```
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>
read_file

### Antes de Commit

```bash
# Formatear código
make format

# Verificar calidad
make check

# Ejecutar tests
make test
```

