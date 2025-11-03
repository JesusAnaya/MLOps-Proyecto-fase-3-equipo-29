# Configuración Detallada de DVC

Esta guía explica en detalle cómo trabajar con DVC (Data Version Control) en el proyecto.

## Tabla de Contenidos

- [Introducción](#introducción)
- [Configuración Inicial](#configuración-inicial)
- [Flujo de Trabajo](#flujo-de-trabajo)
- [Comandos Útiles](#comandos-útiles)
- [Archivos Versionados](#archivos-versionados)
- [Troubleshooting](#troubleshooting)

## Introducción

El proyecto usa una estrategia **Git + DVC**:
- **Git**: Para código, configuración y metadatos (archivos `.dvc`)
- **DVC**: Para datos grandes y artefactos de modelos (almacenados en S3)

### Arquitectura

```
Git Repository (GitHub):
  ✓ Código fuente (.py, .toml, .md)
  ✓ Metadatos DVC (.dvc files)
  ✓ Archivos de configuración
  ✗ NO datos CSV
  ✗ NO modelos .joblib

DVC Repository (S3):
  ✓ Todos los datasets CSV
  ✓ Todos los modelos entrenados (.joblib)
  ✓ Archivos grandes y binarios
```

## Configuración Inicial

### 1. Instalar DVC

```bash
# Con UV (recomendado)
uv add "dvc[s3]"

# O globalmente
pip install "dvc[s3]"
```

### 2. Configurar AWS Credentials

**IMPORTANTE**: DVC requiere credenciales AWS para acceder al bucket S3.

```bash
# Instalar AWS CLI
# macOS:
brew install awscli

# Windows:
choco install awscli

# Configurar credenciales
aws configure
```

**Información requerida:**
- **AWS Access Key ID**: (proporcionada por el equipo)
- **AWS Secret Access Key**: (proporcionada por el equipo)
- **Default region name**: `us-east-1`
- **Default output format**: `json`

### 3. Verificar Configuración DVC

El proyecto **ya está configurado** con DVC. NO ejecutar `dvc init`.

Verificar configuración:
```bash
cat .dvc/config
```

Deberías ver:
```
[core]
    remote = storage
['remote "storage"']
    url = s3://dvc-mna-mlops-equipo-29-datos-projecto
```

## Flujo de Trabajo

### Sincronizar Datos (Primera Vez)

```bash
# 1. Clonar repositorio Git
git clone git@github.com:JesusAnaya/MLOps-Proyecto-fase-2-equipo-29.git
cd MLOps-Proyecto-fase-2-equipo-29

# 2. Configurar AWS credentials
aws configure

# 3. Descargar datos desde S3
dvc pull

# 4. Verificar
dvc status
```

### Agregar Nuevos Datos

```bash
# 1. Agregar archivo a DVC
dvc add data/raw/nuevo_dataset.csv

# 2. IMPORTANTE: Commit metadatos en Git PRIMERO
git add data/raw/nuevo_dataset.csv.dvc data/raw/.gitignore
git commit -m "Agregar nuevo_dataset a DVC"

# 3. Subir datos a S3
dvc push

# 4. Subir cambios de código a GitHub
git push
```

**REGLA CRÍTICA**: Siempre hacer commit de archivos `.dvc` en Git **ANTES** de `dvc push`.

### Agregar Modelos Entrenados

```bash
# 1. Agregar modelo a DVC
dvc add models/mi_modelo.joblib

# 2. Commit metadatos
git add models/mi_modelo.joblib.dvc models/.gitignore
git commit -m "Agregar modelo entrenado"

# 3. Push a S3
dvc push
```

### Sincronizar con Otros Miembros

```bash
# 1. Actualizar código desde GitHub
git pull

# 2. Descargar datos actualizados desde S3
dvc pull

# 3. Verificar estado
dvc status
```

## Comandos Útiles

### Estado y Verificación

```bash
# Ver estado de archivos DVC
dvc status

# Ver diferencias con remoto
dvc diff

# Verificar que archivos están sincronizados
dvc status
```

### Listar Archivos

```bash
# Listar archivos en data/raw
dvc list . data/raw

# Listar archivos en data/processed
dvc list . data/processed

# Listar modelos
dvc list . models
```

### Información Detallada

```bash
# Ver información de un archivo específico
dvc get . data/raw/german_credit_modified.csv.dvc

# Ver log de cambios en JSON
dvc diff --show-json
```

### Limpieza (Cuidado)

```bash
# Eliminar archivos locales pero mantener en S3
dvc gc --workspace

# Eliminar caché local
dvc cache dir
```

## Archivos Versionados

### Datos Raw

- `data/raw/german_credit_modified.csv` (96 KB)
- `data/raw/german_credit_original.csv` (46 KB)

### Datos Procesados

- `data/processed/Xtraintest.csv` (83 KB)
- `data/processed/ytraintest.csv` (4 KB)
- `data/processed/data_clean.csv` (88 KB)

### Modelos

- `models/best_model.joblib` (440 KB)
- `models/preprocessor.joblib` (10 KB)

**Nota**: Estos archivos NO están en Git. Solo los archivos `.dvc` (metadatos) están versionados en Git.

## Estrategia Git + DVC

### Qué está en Git

```bash
git ls-files | grep -E "\.(dvc|gitignore)$"
```

- `*.dvc`: Metadatos de DVC (hashes, tamaños, rutas S3)
- `*.gitignore`: Archivos generados por DVC para ignorar datos

### Qué NO está en Git

```bash
git check-ignore data/**/*.csv models/**/*.joblib
```

- Archivos CSV en `data/`
- Archivos `.joblib` en `models/`

### Protección

El proyecto tiene `.gitignore` configurado para prevenir agregar accidentalmente:

- `*.csv`: Todos los CSV (excepto `.dvc` files)
- `models/*.joblib`: Todos los modelos

## Troubleshooting

### Error: "No se puede conectar a S3"

**Solución:**
1. Verifica credenciales AWS: `aws sts get-caller-identity`
2. Verifica configuración: `cat .dvc/config`
3. Prueba acceso directo: `aws s3 ls s3://dvc-mna-mlops-equipo-29-datos-projecto`

### Error: "Archivo no encontrado en DVC"

**Solución:**
```bash
# Re-descargar todos los datos
dvc pull --force

# Verificar qué archivos están en DVC
dvc list . data/
```

### Datos Locales Desincronizados

**Solución:**
```bash
# Forzar actualización
dvc pull --force

# O eliminar y re-descargar
rm data/raw/*.csv
dvc pull
```

### Git muestra archivos CSV como nuevos

**Problema**: Archivos CSV no deberían estar en Git.

**Solución:**
```bash
# Verificar que están ignorados
git check-ignore data/processed/data_clean.csv

# Si no están ignorados, actualizar .gitignore
# Luego remover del tracking de Git
git rm --cached data/processed/data_clean.csv
```

## Mejores Prácticas

1. **Siempre commit .dvc antes de push**: Garantiza metadatos versionados
2. **Verificar antes de push**: `dvc status` antes de `dvc push`
3. **Pull después de git pull**: Siempre `dvc pull` después de `git pull`
4. **No modificar .dvc manualmente**: Dejar que DVC los gestione
5. **Backup importante**: Los datos están en S3, pero mantén backups locales si es crítico

## Referencias

- [Documentación oficial de DVC](https://dvc.org/doc)
- [DVC con S3](https://dvc.org/doc/user-guide/data-management/remote-storage/amazon-s3)

