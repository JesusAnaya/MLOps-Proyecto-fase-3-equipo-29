# Testing

Información completa sobre los tests del proyecto y cómo ejecutarlos.

## Filosofía de Testing

El proyecto se enfoca en:
- **Tests unitarios**: Validar todas las funciones y módulos críticos del proyecto
- **Tests de integración**: Validar el flujo end-to-end completo (carga de datos → preprocesamiento → predicción → métricas)
- **Mocks completos**: Usar mocks completos de módulos externos cuando sea necesario (no se requiere mocking granular)
- **Optimización**: Testear módulos específicos cuando hay problemas, solo ejecutar todos los tests después de arreglar
- **Cobertura práctica**: Enfocarse en cubrir funciones y módulos críticos, no buscar 100% de cobertura

## Tabla de Contenidos

- [Ejecutar Todos los Tests](#ejecutar-todos-los-tests)
- [Ejecutar Tests por Separado](#ejecutar-tests-por-separado)
- [Tipos de Tests](#tipos-de-tests)
- [Ejecutar Tests Específicos](#ejecutar-tests-específicos)
- [Tests con Coverage](#tests-con-coverage)
- [Estructura de Tests](#estructura-de-tests)
- [Estrategias de Mocking](#estrategias-de-mocking)
- [Cobertura de Tests](#cobertura-de-tests)
- [Mejores Prácticas](#mejores-prácticas)
- [Comandos Útiles](#comandos-útiles)

## Ejecutar Todos los Tests

### Comando Único (Recomendado)

**Para ejecutar todos los tests del proyecto con un solo comando:**

```bash
# Opción 1: Usando Make (recomendado)
make test

# Opción 2: Directamente con UV
uv run pytest

# Opción 3: Especificando directorio explícitamente
uv run pytest tests/
```

**Este comando ejecuta:**
- Todos los tests unitarios (128 tests)
- Todos los tests de integración (10 tests)
- Total: 138 tests
- Genera reporte de coverage automáticamente

**Parámetros de pytest utilizados:**
- Sin marcadores: Ejecuta todos los tests sin filtrar
- El comando `make test` encapsula `uv run pytest` sin parámetros adicionales

## Ejecutar Tests por Separado

El proyecto permite ejecutar tests unitarios e integración por separado usando marcadores de pytest o comandos make convenientes.

### Usando Comandos Make (Recomendado)

Los comandos make encapsulan los parámetros de pytest para facilitar la ejecución:

```bash
# Ejecutar todos los tests (unitarios + integración)
make test

# Ejecutar solo tests unitarios
make test-unit

# Ejecutar solo tests de integración
make test-integration
```

**Ventajas de usar Make:**
- Comandos más cortos y fáciles de recordar
- Encapsula parámetros de pytest (`-m unit`, `-m integration`)
- Consistente con otros comandos del proyecto
- No necesitas recordar los marcadores de pytest

### Usando Parámetros de pytest Directamente

Si prefieres usar pytest directamente, puedes usar los marcadores:

```bash
# Ejecutar todos los tests
uv run pytest

# Ejecutar solo tests unitarios
uv run pytest -m unit

# Ejecutar solo tests de integración
uv run pytest -m integration
```

**Explicación de los parámetros:**
- `-m unit`: Filtra tests marcados con `@pytest.mark.unit` (128 tests)
- `-m integration`: Filtra tests marcados con `@pytest.mark.integration` (10 tests)
- Sin `-m`: Ejecuta todos los tests sin filtrar (138 tests)

**Cuándo usar cada opción:**
- **Make**: Para uso diario y desarrollo rápido
- **pytest directo**: Para mayor control o cuando necesites parámetros adicionales

### Tests con Coverage

```bash
# Con Make
make test-cov

# Directamente
uv run pytest --cov=mlops_project --cov-report=html --cov-report=term

# Ver reporte HTML
open htmlcov/index.html
```

## Tipos de Tests

El proyecto usa marcadores de pytest para categorizar los tests:

### Tests Unitarios

Tests que verifican componentes individuales de forma aislada.

**Ejecutar tests unitarios:**

```bash
# Opción recomendada: usando Make
make test-unit

# O directamente con pytest usando marcador
uv run pytest -m unit
```

**Cobertura:**
- Configuración (`test_config.py`)
- Carga de datos (`test_dataset.py`, `test_dataset_io.py`)
- Transformación de features (`test_features.py`)
- Modelos individuales (`test_modeling.py`, `test_modeling_improved.py`)
- Visualizaciones (`test_plots.py`)

**Parámetros de pytest utilizados:**
- `-m unit`: Filtra tests marcados con `@pytest.mark.unit`
- Ejecuta aproximadamente 128 tests unitarios

### Tests de Integración

Tests que verifican que múltiples componentes trabajen juntos.

**Ejecutar tests de integración:**

```bash
# Opción recomendada: usando Make
make test-integration

# O directamente con pytest usando marcador
uv run pytest -m integration
```

**Parámetros de pytest utilizados:**
- `-m integration`: Filtra tests marcados con `@pytest.mark.integration`
- Ejecuta aproximadamente 10 tests de integración

**Cobertura:**
- Pipeline completo (datos → features → entrenamiento → predicción)
- Integración con MLflow
- Operaciones de archivos completas

### Tests Lentos

Tests que toman más tiempo (entrenamiento real, operaciones de archivos).

```bash
# Excluir tests lentos (más rápido)
uv run pytest -m "not slow"

# Ejecutar solo tests lentos
uv run pytest -m slow
```

## Ejecutar Tests Específicos

### Por Archivo

```bash
# Test de un archivo específico
uv run pytest tests/test_config.py -v

# Test de un módulo completo
uv run pytest tests/test_modeling.py -v
```

### Por Clase

```bash
# Test de una clase específica
uv run pytest tests/test_modeling.py::TestModelInstances -v

# Test de múltiples clases
uv run pytest tests/test_modeling.py::TestModelInstances tests/test_modeling.py::TestTrainingPipeline -v
```

### Por Función

```bash
# Test de una función específica
uv run pytest tests/test_dataset.py::TestDataLoader::test_load_data_success -v

# Test de múltiples funciones
uv run pytest tests/test_config.py::TestConfig::test_random_seed_is_int tests/test_config.py::TestConfig::test_target_column_defined -v
```

### Por Patrón

```bash
# Tests que contengan "model" en el nombre
uv run pytest -k "model" -v

# Tests que contengan "integration" o "pipeline"
uv run pytest -k "integration or pipeline" -v
```

## Tests con Coverage

### Ver Coverage en Terminal

```bash
uv run pytest --cov=mlops_project --cov-report=term-missing
```

Muestra:
- Cobertura por módulo
- Líneas faltantes (missing lines)

### Ver Coverage en HTML

```bash
# Generar reporte HTML
uv run pytest --cov=mlops_project --cov-report=html

# Abrir en navegador
open htmlcov/index.html
```

El reporte HTML muestra:
- Cobertura por archivo
- Líneas cubiertas/no cubiertas

**Nota:** El proyecto se enfoca en cubrir las funciones y módulos críticos del pipeline. No se busca 100% de cobertura, sino asegurar que todos los componentes importantes estén validados.

## Estrategias de Mocking

El proyecto utiliza estrategias de mocking siguiendo las mejores prácticas para asegurar que los tests sean rápidos, confiables y no dependan de recursos externos.

### Principios de Mocking

1. **Mockear recursos externos**: Todos los recursos fuera del alcance del proyecto (MLflow, AWS S3, archivos del sistema) están mockeados.
2. **Mockear dependencias costosas**: Operaciones de I/O, llamadas a APIs externas, y operaciones de matplotlib están mockeadas.
3. **Usar fixtures realistas**: Los datos sintéticos usan fixtures que replican la estructura del dataset real.
4. **Verificar comportamiento, no implementación**: Los tests verifican que las funciones se llamen correctamente, no detalles de implementación.

### Ejemplos de Mocking

#### Mocking de MLflow

```python
@patch("mlops_project.modeling.train.mlflow")
def test_train_with_mlflow_logging(mock_mlflow, sample_realistic_data_df):
    """Test que verifica que MLflow se llama correctamente."""
    # Configurar mocks
    mock_mlflow.active_run.return_value.info.run_id = "test-run-id"
    
    # Ejecutar función
    train_model(...)
    
    # Verificar llamadas
    assert mock_mlflow.set_tracking_uri.called
    assert mock_mlflow.start_run.called
```

#### Mocking de Matplotlib

```python
@patch("mlops_project.plots.plt")
@patch.object(pd.Series, "hist")
def test_plot_distribution_basic(mock_hist, mock_plt, sample_data_df):
    """Test que verifica creación de gráficos sin mostrar ventanas."""
    # Configurar mocks
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_ax.figure = mock_fig
    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    mock_hist.return_value = None
    
    # Ejecutar función
    plots.plot_distribution(sample_data_df, save_path=None)
    
    # Verificar llamadas
    mock_plt.subplots.assert_called_once()
    mock_plt.show.assert_called_once()
```

#### Mocking de I/O

```python
@patch("mlops_project.dataset.pd.read_csv")
def test_load_data_calls_read_csv(mock_read_csv, sample_data_df):
    """Test que verifica carga de datos sin leer archivos reales."""
    mock_read_csv.return_value = sample_data_df
    
    loader = DataLoader("fake_path.csv")
    data = loader.load_data()
    
    # Verificar que se llamó con el path correcto
    expected_path = Path("fake_path.csv")
    mock_read_csv.assert_called_once_with(expected_path, sep=",")
```

#### Mocking de sklearn Pipelines

```python
@patch("mlops_project.modeling.train.cross_validate")
def test_evaluate_model_with_mock(mock_cross_validate, sample_realistic_data_df):
    """Test que verifica evaluación sin ejecutar CV real."""
    # Configurar mock con todas las métricas necesarias
    mock_cv_results = {
        "test_accuracy": np.array([0.8, 0.85, 0.9]),
        "train_accuracy": np.array([0.82, 0.87, 0.92]),
        "test_precision": np.array([0.75, 0.80, 0.85]),
        "train_precision": np.array([0.77, 0.82, 0.87]),
        "test_recall": np.array([0.70, 0.75, 0.80]),
        "train_recall": np.array([0.72, 0.77, 0.82]),
        "test_f1": np.array([0.72, 0.77, 0.82]),
        "train_f1": np.array([0.74, 0.79, 0.84]),
        "test_roc_auc": np.array([0.78, 0.83, 0.88]),
        "train_roc_auc": np.array([0.80, 0.85, 0.90]),
        "test_average_precision": np.array([0.76, 0.81, 0.86]),
        "train_average_precision": np.array([0.78, 0.83, 0.88]),
        "test_geometric_mean": np.array([0.71, 0.76, 0.81]),
        "train_geometric_mean": np.array([0.73, 0.78, 0.83]),
    }
    mock_cross_validate.return_value = mock_cv_results
    
    # Ejecutar evaluación
    results = evaluate_model(pipeline, X, y)
    
    # Verificar resultados
    assert results["accuracy"]["test_mean"] == pytest.approx(0.85)
```

### Fixtures de Datos Realistas

El proyecto incluye fixtures que proporcionan datos sintéticos con la estructura correcta:

```python
@pytest.fixture
def sample_realistic_data_df():
    """Fixture con columnas que coinciden con el dataset real."""
    return pd.DataFrame({
        # Numéricas
        "laufzeit": np.random.randint(1, 100, n_samples),
        "hoehe": np.random.randint(100, 10000, n_samples),
        "alter": np.random.randint(18, 80, n_samples),
        # Ordinales
        "beszeit": np.random.choice([1, 2, 3, 4, 5], n_samples),
        "rate": np.random.choice([1, 2, 3, 4], n_samples),
        "wohnzeit": np.random.choice([1, 2, 3, 4], n_samples),
        # ... todas las columnas necesarias
        "kredit": np.random.choice([0, 1], n_samples),  # Target
    })
```

**Uso en tests:**
```python
def test_pipeline_end_to_end(sample_realistic_data_df):
    """Test de integración con datos realistas."""
    X = sample_realistic_data_df.drop(columns=["kredit"])
    y = sample_realistic_data_df["kredit"]
    # ... resto del test
```

## Estructura de Tests

```
tests/
├── conftest.py                    # Fixtures compartidas
├── test_config.py                 # Tests de configuración (unit)
├── test_dataset.py                # Tests de carga de datos (unit)
├── test_dataset_io.py             # Tests de I/O de datos (unit + integration)
├── test_features.py                # Tests de features (unit)
├── test_modeling.py               # Tests de modelado (unit)
├── test_modeling_improved.py      # Tests mejorados de modelado (unit)
├── test_plots.py                  # Tests de visualizaciones (unit)
└── test_integration.py            # Tests de integración completa (integration)
```

### Fixtures Disponibles

Las fixtures en `conftest.py` están disponibles para todos los tests:

- `sample_data_df`: DataFrame de muestra para tests
- `sample_X_y`: Features y target separados
- `sample_feature_names`: Nombres de features por tipo
- `mock_trained_model`: Modelo mock pre-entrenado
- `mock_preprocessor`: Preprocessor mock
- `sample_cv_results`: Resultados mock de cross-validation
- `sample_predictions`: Predicciones y probabilidades de muestra

**Uso:**
```python
def test_example(sample_data_df, mock_trained_model):
    # Usar fixtures automáticamente
    X = sample_data_df.drop(columns=["target"])
    predictions = mock_trained_model.predict(X)
    assert len(predictions) == len(X)
```

## Cobertura de Tests

### Módulos Cubiertos

#### ✅ Configuración (`config.py`)
- Rutas y directorios
- Parámetros de configuración
- Funciones utilitarias
- Validación de valores

#### ✅ Datos (`dataset.py`)
- `DataLoader`: Carga de CSV
- `DataCleaner`: Limpieza de datos
- `DataSplitter`: División train/test
- `load_and_prepare_data`: Función principal

#### ✅ Features (`features.py`)
- `InvalidDataHandler`: Manejo de valores inválidos
- `OutlierHandler`: Detección y manejo de outliers
- `FeaturePreprocessor`: Preprocesamiento completo
- `create_feature_pipeline`: Creación de pipeline
- `prepare_features`: Función principal

#### ✅ Modelado - Entrenamiento (`modeling/train.py`)
- `get_model_instance`: Creación de modelos
- `get_smote_instance`: Creación de SMOTE
- `create_training_pipeline`: Pipeline de entrenamiento
- `evaluate_model`: Evaluación con CV
- `train_model`: Función principal
- `save_results`: Guardado de resultados
- `_mlflow_log_run`: Integración MLflow (mocked)

#### ✅ Modelado - Predicción (`modeling/predict.py`)
- `load_model`: Carga de modelos
- `predict`: Predicciones
- `evaluate_predictions`: Evaluación de predicciones
- `predict_and_evaluate`: Función principal
- `batch_predict`: Predicción en batches

#### ✅ Visualizaciones (`plots.py`)
- Todas las funciones de plotting (con mocks de matplotlib y pandas)
- Tests verifican llamadas a funciones sin crear gráficos reales

### Módulos Parcialmente Cubiertos

#### ⚠️ MLflow (`modeling/train.py`)
- Funciones internas de MLflow están mockeadas
- Tests de integración verifican llamadas a MLflow

### Estadísticas Actuales

- **Total de Tests**: 138
- **Tests Unitarios**: 128 (cubren todas las funciones y módulos críticos)
- **Tests de Integración**: 10 (validan flujo end-to-end completo)
- **Tests Pasando**: 138/138 (100%)

## Mejores Prácticas

### 1. Ejecutar Tests Antes de Commit

```bash
# Verificación completa
make check  # Lint + tests

# Solo tests
make test
```

### 2. Desarrollo Iterativo

```bash
# Tests rápidos (sin integración ni lentos)
uv run pytest -m "unit and not slow" -v

# Test específico mientras desarrollas
uv run pytest tests/test_features.py::TestFeaturePreprocessor::test_fit_transform -v
```

### 3. Verificar Coverage Regularmente

```bash
# Ver coverage actual
uv run pytest --cov=mlops_project --cov-report=term-missing | grep TOTAL
```

### 4. Tests de Integración en CI/CD

Los tests de integración deben ejecutarse en CI/CD:

```bash
# En CI/CD: todos los tests
uv run pytest

# En desarrollo local: excluir lentos
uv run pytest -m "not slow"
```

### 5. Agregar Nuevos Tests

Al agregar nueva funcionalidad:

1. **Crear test unitario** para funciones y módulos críticos
2. **Agregar test de integración** si afecta el flujo end-to-end
3. **Marcar apropiadamente** (`@pytest.mark.unit` o `@pytest.mark.integration`)
4. **Usar mocks completos** de módulos externos cuando sea necesario
5. **Enfocarse en validar comportamiento**, no implementación interna

**Ejemplo:**
```python
import pytest

@pytest.mark.unit
def test_nueva_funcionalidad():
    """Test para nueva funcionalidad."""
    # Test code here
    assert resultado == esperado
```

## Comandos Útiles

### Resumen Rápido

```bash
# Todos los tests
make test

# Tests con coverage
make test-cov

# Solo unitarios
uv run pytest -m unit

# Solo integración
uv run pytest -m integration

# Excluir lentos
uv run pytest -m "not slow"

# Test específico
uv run pytest tests/test_config.py::TestConfig::test_random_seed_is_int -v

# Ver qué tests se ejecutarían (sin ejecutar)
uv run pytest --collect-only
```

### Debugging Tests

```bash
# Ejecutar con output detallado
uv run pytest -v -s

# Ejecutar con pdb (debugger)
uv run pytest --pdb

# Ejecutar hasta primer fallo
uv run pytest -x

# Ejecutar con timeout
uv run pytest --timeout=300
```

## Troubleshooting

### Tests Fallan Inesperadamente

1. **Verificar que los datos estén disponibles:**
   ```bash
   dvc pull  # Asegurar que datos estén descargados
   ```

2. **Limpiar caché de pytest:**
   ```bash
   rm -rf .pytest_cache
   uv run pytest
   ```

3. **Verificar entorno:**
   ```bash
   uv run python -c "import mlops_project; print('OK')"
   ```

### Coverage Bajo

1. **Identificar módulos sin coverage:**
   ```bash
   uv run pytest --cov=mlops_project --cov-report=term-missing | grep -E "mlops_project|TOTAL"
   ```

2. **Agregar tests para módulos faltantes**

3. **Verificar que no haya código muerto**

### Tests Lentos

1. **Usar marcador `@pytest.mark.slow`** para tests que toman tiempo
2. **Ejecutar solo tests rápidos** durante desarrollo: `uv run pytest -m "not slow"`
3. **Ejecutar todos** antes de commit: `make test`

## Referencias

- [Documentación de pytest](https://docs.pytest.org/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [pytest-mock](https://pytest-mock.readthedocs.io/)
