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
```

## Testing

```bash
make test              # Ejecutar todos los tests
make test-cov          # Ejecutar tests con coverage
```

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

# Ejecutar pipeline
make pipeline

# Verificar estado de datos
make dvc-status
```

### Antes de Commit

```bash
# Formatear código
make format

# Verificar calidad
make check

# Ejecutar tests
make test
```

