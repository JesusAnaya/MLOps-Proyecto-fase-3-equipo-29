# Guía de UV

Este proyecto usa **UV** como gestor de paquetes en lugar de pip/virtualenv tradicional.

## Ventajas de UV

### 1. Sin Activación de Entorno Virtual

UV maneja el entorno automáticamente, no necesitas activarlo manualmente:

```bash
# Tradicional (NO necesario con UV)
source .venv/bin/activate
python script.py

# Con UV (recomendado)
uv run python script.py
```

### 2. Comandos Más Rápidos

UV es significativamente más rápido que pip:

```bash
uv sync           # Instalar dependencias (más rápido que pip install)
uv add pandas     # Agregar dependencia (más rápido que pip install)
```

### 3. Gestión Automática del Entorno

UV crea y gestiona el venv automáticamente:

```bash
uv venv --python 3.12  # Crea venv (si no existe)
uv sync                # Crea venv automáticamente si es necesario
```

### 4. Compatibilidad Total

UV es compatible con pyproject.toml y pip:

```bash
# Todos los comandos tradicionales funcionan con uv run
uv run python -m pytest
uv run python -m black .
```

## Equivalencias Comunes

| Tradicional | Con UV |
|-------------|--------|
| `python script.py` | `uv run python script.py` |
| `pytest tests/` | `uv run pytest tests/` |
| `pip install package` | `uv add package` |
| `pip install -e .` | `uv sync` |
| `pip list` | `uv pip list` |
| `source .venv/bin/activate` | No necesario (usar `uv run`) |

## Uso en el Proyecto

### Ejecutar Scripts

```bash
# Scripts del proyecto
uv run mlops-prepare-data --help
uv run mlops-train --help
uv run mlops-predict --help

# Tests
uv run pytest tests/

# Python directamente
uv run python script.py
```

### Instalar Dependencias

```bash
# Instalar todas las dependencias
uv sync

# Agregar nueva dependencia
uv add pandas

# Agregar dependencia con versión específica
uv add "pandas>=2.0.0"

# Agregar dependencia de desarrollo
uv add --dev pytest
```

### Verificar Entorno

```bash
# Ver paquetes instalados
uv pip list

# Ver información del entorno
uv python list
```

## Activación Manual (Opcional)

Si prefieres trabajar en el shell activado:

```bash
# Unix/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

**Nota**: Con UV, no es necesario activar el entorno virtual. Puedes usar `uv run` para ejecutar cualquier comando Python.

## Referencias

- [Documentación oficial de UV](https://docs.astral.sh/uv/getting-started/installation/)
