# Testing

Información sobre los tests del proyecto y cómo ejecutarlos.

## Ejecutar Tests

### Todos los Tests

```bash
# Usando Make
make test

# Directamente con UV
uv run pytest tests/
```

### Tests con Coverage

```bash
# Con Make
make test-cov

# Directamente
uv run pytest tests/ --cov=mlops_project --cov-report=html

# Ver reporte de coverage
open htmlcov/index.html
```

### Tests Específicos

```bash
# Test de un archivo específico
uv run pytest tests/test_config.py -v

# Test de una clase específica
uv run pytest tests/test_modeling.py::TestModelInstances -v

# Test de una función específica
uv run pytest tests/test_dataset.py::test_load_data -v
```

### Tests con Filtros

```bash
# Solo tests unitarios
uv run pytest tests/ -m unit

# Solo tests de integración
uv run pytest tests/ -m integration

# Excluir tests lentos
uv run pytest tests/ -m "not slow"
```

## Estadísticas de Testing

- **Total de Tests**: 130
- **Tests Pasando**: 108 (83%)
- **Coverage**: 64%
- **Uso de Mocks**: Para I/O, entrenamiento y visualización

## Estructura de Tests

```
tests/
├── conftest.py           # Fixtures compartidas
├── test_config.py        # Tests de configuración
├── test_dataset.py       # Tests de datos
├── test_dataset_io.py    # Tests de I/O de datos
├── test_features.py      # Tests de features
├── test_modeling.py      # Tests de modelado
├── test_modeling_improved.py  # Tests mejorados de modelado
└── test_plots.py         # Tests de visualizaciones
```

## Fixtures Disponibles

Las fixtures se encuentran en `conftest.py`:

- Datos de prueba
- Modelos mock
- Configuraciones temporales

## Mejores Prácticas

1. **Ejecutar tests antes de commit**: `make test`
2. **Verificar coverage**: Mantener > 60%
3. **Tests unitarios rápidos**: Para desarrollo iterativo
4. **Tests de integración**: Para validar pipeline completo

