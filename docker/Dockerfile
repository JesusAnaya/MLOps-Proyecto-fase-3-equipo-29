FROM python:3.12-slim

# Copiar binario de UV desde la imagen oficial
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar archivos de configuración del proyecto
COPY pyproject.toml uv.lock ./
COPY README.md LICENSE ./

# Copiar código fuente y servicio web
COPY mlops_project/ ./mlops_project/
COPY web_service/ ./web_service/

# Copiar modelos (incluye best_model.joblib y demás artefactos)
COPY models/ ./models/

# Instalar dependencias del proyecto + extras para web (FastAPI, uvicorn, etc.)
RUN uv sync --extra web

# Exponer puerto del servicio
EXPOSE 8000

# Comando por defecto: levantar el servicio FastAPI
CMD ["uv", "run", "mlops-web-service"]
