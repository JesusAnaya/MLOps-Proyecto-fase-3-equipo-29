# Roadmap - Próximos Pasos

Plan de mejoras y funcionalidades futuras para el proyecto.

## Integración con MLflow (Completado)

- Tracking de experimentos
- Registro de modelos en Model Registry
- Logging de hiperparámetros y métricas
- Soporte para modelos sklearn y XGBoost

## Mejoras Pendientes

### Orquestación con Airflow

Para automatización completa del pipeline:

```python
# Ejemplo de DAG de Airflow (futuro)
from airflow import DAG
from airflow.operators.bash import BashOperator

dag = DAG('mlops_pipeline', schedule_interval='@daily')

prepare_data = BashOperator(
    task_id='prepare_data',
    bash_command='uv run mlops-prepare-data --input data/raw/data.csv --save',
    dag=dag
)

prepare_features = BashOperator(
    task_id='prepare_features',
    bash_command='uv run mlops-prepare-features --train data/processed/X.csv --save-preprocessor',
    dag=dag
)

train_model = BashOperator(
    task_id='train_model',
    bash_command='uv run mlops-train ...',
    dag=dag
)

prepare_data >> prepare_features >> train_model
```

### Deployment

- **FastAPI**: Crear API REST para serving de modelos
- **Docker**: Containerización del pipeline completo
- **CI/CD**: GitHub Actions para automatización
- **Monitoring**: Integrar con Prometheus/Grafana

### Mejoras de Modelado

- Hiperparameter tuning automático con Optuna
- AutoML con AutoSklearn
- Ensemble methods avanzados
- Feature selection automático

### Mejoras de Infraestructura

- Caché de resultados intermedios
- Pipeline paralelizado
- Optimización de memoria para datasets grandes
- Soporte para streaming de datos

### Documentación

- Tutorial interactivo
- Video demos
- Casos de uso reales
- Best practices documentadas

## Contribuciones

Para contribuir al proyecto:

1. Fork el repositorio
2. Crea una rama para tu feature
3. Implementa cambios siguiendo las convenciones
4. Ejecuta `make check` antes de commit
5. Crea un Pull Request

## Feedback

Para reportar bugs o sugerir mejoras, abre un issue en GitHub.

