# Módulo Docker

Este módulo contiene la documentación para containerizar el servicio web de predicción de riesgo crediticio utilizando Docker.

## Objetivos

1. **Serving y Portabilidad del Modelo con FastAPI**: Exponer el modelo vía API y asegurar su portabilidad entre entornos.
2. **Integración del Modelo en un Contenedor (Docker)**: Empaquetar el servicio y sus dependencias en una imagen reproducible y eficiente.

## Información del Modelo

**Ruta del modelo**: `models/best_model.joblib`  
**Versión del modelo**: `0.1.0` (definida en `mlops_project/config.py` como `MLFLOW_MODEL_VERSION`)

## Dockerfile

El Dockerfile debe instalar el proyecto usando UV y `pyproject.toml`, incluyendo las dependencias de FastAPI. Ejemplo mínimo:

```dockerfile
FROM python:3.12-slim

# Instalar UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copiar configuración y código
COPY pyproject.toml uv.lock ./
COPY mlops_project/ ./mlops_project/
COPY web_service/ ./web_service/
COPY models/ ./models/

# Instalar dependencias del proyecto + dependencias web (FastAPI)
RUN uv sync --system --extra web

# Exponer puerto y ejecutar servidor FastAPI
EXPOSE 8000
CMD ["uv", "run", "mlops-web-service"]
```

**Puntos clave**:
- `--extra web` instala FastAPI, uvicorn y pydantic (definidos en `pyproject.toml` como dependencias opcionales)
- `EXPOSE 8000` expone el puerto para el servicio
- `CMD` ejecuta el script `mlops-web-service` que inicia el servidor FastAPI en `0.0.0.0:8000`

## Docker Compose

Archivo `docker-compose.yml` para facilitar el desarrollo:

```yaml
version: '3.8'

services:
  ml-service:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    image: public.ecr.aws/f8w5v4x1/mna/mlops-eqipo-29:latest
    ports:
      - "8000:8000"
    volumes:
      - ../models:/app/models:ro
    environment:
      - PYTHONUNBUFFERED=1
```

**Uso**:
```bash
cd docker
docker-compose up --build
```

## Repositorio

**Repositorio público en Amazon ECR Public**:
```
public.ecr.aws/f8w5v4x1/mna/mlops-eqipo-29
```

**Nota**: La construcción de la imagen se realiza localmente, pero se usa el nombre del repositorio público para facilitar el push posterior a ECR.

## Construir la Imagen (Local)

```bash
docker build -f docker/Dockerfile -t public.ecr.aws/f8w5v4x1/mna/mlops-eqipo-29:latest .
```

La imagen se construye localmente con el nombre del repositorio público para poder publicarla después.

## Ejecutar el Contenedor (Local)

```bash
docker run -p 8000:8000 public.ecr.aws/f8w5v4x1/mna/mlops-eqipo-29:latest
```

El servicio estará disponible en http://localhost:8000 (documentación en `/docs`).

## Publicar en Amazon ECR Public

### 1. Autenticación

```bash
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/f8w5v4x1
```

**Nota**: Si aparece algún error, asegúrate de tener instalada la última versión del AWS CLI y de Docker.

### 2. Etiquetar para Versión Específica (Opcional)

```bash
docker tag public.ecr.aws/f8w5v4x1/mna/mlops-eqipo-29:latest public.ecr.aws/f8w5v4x1/mna/mlops-eqipo-29:0.1.0
```

### 3. Publicar

**Publicar versión latest**:
```bash
docker push public.ecr.aws/f8w5v4x1/mna/mlops-eqipo-29:latest
```

**Publicar versión específica (opcional)**:
```bash
docker push public.ecr.aws/f8w5v4x1/mna/mlops-eqipo-29:0.1.0
```

**Tags recomendados**: `latest` (última versión estable), `0.1.0` (versión del modelo)

## Notas Adicionales

- En producción, los modelos deberían descargarse desde DVC/S3 en lugar de copiarlos en la imagen
- El Dockerfile actual copia los modelos localmente; para producción, agregar descarga remota
- Considerar agregar health checks y ejecutar como usuario no root en despliegues de producción

## Referencias

- [README del Web Service](../web_service/README.md) - Documentación del servicio FastAPI
- [Documentación de Docker](https://docs.docker.com/)

---

**Equipo 29** - TC5044.10  
**Fecha**: Noviembre 2025

