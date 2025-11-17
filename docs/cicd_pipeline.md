# CI/CD Pipeline

Pipeline automatizado que ejecuta el ML pipeline, registra resultados en MLflow y sube artefactos a DVC/S3.

## Configuración

### Secrets en GitHub

**Settings** → **Secrets and variables** → **Actions** → **New repository secret**

- `AWS_ACCESS_KEY`: AWS Access Key ID
- `AWS_SECRET_KEY`: AWS Secret Access Key

## Triggers

- **Pull Requests a `main`**: Valida cambios
- **Push a `main`**: Ejecuta después de merge
- **Manual**: Workflow dispatch desde GitHub Actions

## Ejecución Local

**Requisito**: AWS CLI debe estar configurado localmente (`aws configure`)

```bash
# Ejecutar pipeline
bash scripts/run_pipeline.sh

# Con push a DVC
PUSH_TO_DVC=true bash scripts/run_pipeline.sh
```

El script verifica que AWS CLI esté instalado y configurado correctamente.

## Artefactos

- `models/best_model.joblib`
- `models/preprocessor.joblib`
- `models/model_results.json`

Todos se versionan en DVC y se suben a S3.

## Ver Resultados

**GitHub Actions**: Pestaña **Actions** → **ML Pipeline CI/CD**

**MLflow**: `https://mlflow-equipo-29.robomous.ai` (experiment: `equipo-29`)

## Troubleshooting

**AWS credentials not found**
- Local: Configura AWS CLI con `aws configure` antes de ejecutar el script
- CI: Verificar que los secrets estén configurados en GitHub

**DVC pull failed**
```bash
aws s3 ls s3://dvc-mna-mlops-equipo-29-datos-projecto/
```

**Pipeline failed**
```bash
dvc repro -v
dvc repro --force
```

## Flujo

**Pull Request**: Ejecuta pipeline → Valida cambios → No commit `.dvc`

**Push a main**: Ejecuta pipeline → Sube a S3 → Commit `.dvc` si hay cambios
