#!/bin/bash
set -e

# Verificar directorio
[ -f "pyproject.toml" ] || { echo "Error: ejecutar desde raíz del proyecto"; exit 1; }

# Verificar AWS CLI
command -v aws >/dev/null || { echo "Error: AWS CLI no instalado"; exit 1; }
aws sts get-caller-identity >/dev/null || { echo "Error: AWS credentials no configuradas"; exit 1; }

# Instalar UV si falta
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.cargo/bin:$PATH"; }

# Instalar dependencias
uv sync

# Descargar datos si faltan
[ -f "data/raw/german_credit_modified.csv" ] || dvc pull data/raw/german_credit_modified.csv.dvc

# Ejecutar pipeline
dvc repro

# Verificar artefactos
[ -f "models/best_model.joblib" ] && [ -f "models/preprocessor.joblib" ] && [ -f "models/model_results.json" ] || { echo "Error: artefactos no generados"; exit 1; }

# Push a DVC si está habilitado
if [ "$PUSH_TO_DVC" = "true" ] || [ -n "$CI" ]; then
    dvc push
fi

echo "✓ Pipeline completado"
