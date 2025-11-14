# Módulo Web Service

Este módulo contiene la implementación del servicio web para el proyecto MLOps, proporcionando una API REST para el servicio de modelos de machine learning.

## Propósito

El módulo `web_service` expone los modelos entrenados del proyecto a través de una API REST utilizando FastAPI. Este servicio permite:

- Realizar predicciones individuales y por lotes de riesgo crediticio
- Consultar información sobre los modelos desplegados
- Verificar el estado de salud del servicio
- Proporcionar documentación interactiva mediante Swagger/OpenAPI

## Estructura del Módulo

```
web_service/
├── README.md                    # Este archivo
├── __init__.py                  # Inicialización del módulo
├── app.py                       # Aplicación FastAPI con endpoints
├── service.py                   # Lógica de negocio y carga de modelos
├── models.py                    # Modelos Pydantic para validación de datos
└── postman/                     # Colecciones de Postman para pruebas
    └── postman_schema_validation.json  # Colección completa con validación de estructura
```

## Estado Actual

El módulo está **implementado y funcional**. Contiene:

- **Aplicación FastAPI completa**: Con todos los endpoints implementados
- **Validación de datos**: Usando Pydantic para validar entradas y salidas
- **Carga de modelos**: Integración con modelos entrenados desde `models/best_model.joblib`
- **Manejo de errores**: Implementación robusta de manejo de excepciones
- **Logging**: Sistema de logging integrado
- **Colecciones de Postman**: Tests funcionales y de validación de estructura

### Colecciones de Postman

#### `postman/postman_schema_validation.json`
Colección completa que incluye tests funcionales y de validación de estructura para todos los endpoints:

**Endpoints incluidos:**
- `GET /health`: Verificación de salud del servicio
- `GET /model-info`: Información del modelo desplegado
- `POST /predict`: Predicción individual (solicitudes válidas e inválidas)
- `POST /predict-batch`: Predicción por lotes
- Validación de headers HTTP

**Validaciones implementadas:**
- Códigos de estado HTTP correctos
- Validación de JSON válido
- Verificación de tipos de datos
- Validación de campos requeridos
- Validación de formatos (ej: versiones semánticas)
- Verificación de headers HTTP
- Tests de tiempo de respuesta
- Validación de estructura de errores

## Instalación

### Instalar Dependencias del Servicio Web

Las dependencias de FastAPI no se instalan por defecto. Para usar el servicio web, primero instala las dependencias opcionales:

```bash
# Instalar dependencias web (FastAPI, uvicorn, pydantic)
uv sync --extra web
```

**Nota**: Si ya tienes el proyecto instalado, este comando agregará las dependencias web sin afectar las dependencias existentes.

### Verificar Instalación

Para verificar que las dependencias están instaladas:

```bash
# Verificar que el script está disponible
uv run mlops-web-service --help

# O verificar importación de FastAPI
uv run python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')"
```

## Cómo Ejecutar el Servicio

### Prerrequisitos

1. **Dependencias web instaladas**: Asegúrate de haber instalado las dependencias web
   ```bash
   uv sync --extra web
   ```

2. **Modelo entrenado disponible**: El servicio requiere que el modelo esté disponible en `models/best_model.joblib`
   ```bash
   # Si el modelo no está disponible, descargarlo con DVC
   dvc pull models/best_model.joblib.dvc
   ```

### Ejecutar el Servicio

#### Opción 1: Usando el script proporcionado (Recomendado)

```bash
# Desde la raíz del proyecto
uv run mlops-web-service
```

Este script está definido en `pyproject.toml` y ejecuta el servicio con configuración por defecto (puerto 8000, recarga automática habilitada).

#### Opción 2: Usando uvicorn directamente

```bash
# Desde la raíz del proyecto
uv run uvicorn web_service.app:app --reload --host 0.0.0.0 --port 8000
```

#### Opción 3: Usando uvicorn con configuración personalizada

```bash
# Con recarga automática (desarrollo)
uv run uvicorn web_service.app:app --reload

# Sin recarga (producción)
uv run uvicorn web_service.app:app --host 0.0.0.0 --port 8000

# Con workers múltiples (producción)
uv run uvicorn web_service.app:app --host 0.0.0.0 --port 8000 --workers 4
```

#### Opción 4: Como módulo Python

```bash
# Ejecutar como módulo
uv run python -m web_service
```

#### Opción 5: Agregar comando al Makefile

Puedes agregar un comando al `Makefile` para facilitar la ejecución:

```makefile
## Iniciar servicio web
.PHONY: web-service
web-service:
	uv run mlops-web-service
	@echo ">>> Servicio web iniciado en http://localhost:8000"
```

Luego ejecutar con:
```bash
make web-service
```

### Acceder a la Documentación

Una vez que el servicio esté ejecutándose, puedes acceder a:

- **Documentación interactiva (Swagger UI)**: http://localhost:8000/docs
- **Documentación alternativa (ReDoc)**: http://localhost:8000/redoc
- **Esquema OpenAPI (JSON)**: http://localhost:8000/openapi.json

### Verificar que el Servicio Está Funcionando

```bash
# Verificar health check
curl http://localhost:8000/health

# Verificar información del modelo
curl http://localhost:8000/model-info
```

## Endpoints Disponibles

### 1. Health Check
```
GET /health
```
Verifica que el servicio esté operativo.

**Respuesta:**
```json
{
  "status": "ok"
}
```

**Códigos de estado:**
- `200`: Servicio operativo

### 2. Información del Modelo
```
GET /model-info
```
Retorna información sobre el modelo actualmente desplegado.

**Respuesta:**
```json
{
  "model_name": "logistic_regression",
  "model_version": "0.1.0"
}
```

**Códigos de estado:**
- `200`: Información obtenida exitosamente
- `503`: Error al obtener información del modelo

### 3. Predicción Individual
```
POST /predict
```
Realiza una predicción de riesgo crediticio para una instancia única.

**Request Body:**
```json
{
  "features": {
    "laufzeit": 24,
    "hoehe": 5000,
    "alter": 35,
    "beszeit": 2,
    "rate": 1,
    "wohnzeit": 2,
    "verm": 2,
    "bishkred": 1,
    "beruf": 2,
    "laufkont": 1,
    "moral": 2,
    "verw": 3,
    "sparkont": 2,
    "famges": 1,
    "buerge": 1,
    "weitkred": 1,
    "wohn": 1,
    "pers": 1,
    "telef": 1,
    "gastarb": 1
  }
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.85
}
```

**Descripción de campos:**
- `prediction`: Predicción binaria (0: riesgo bajo, 1: riesgo alto)
- `probability`: Probabilidad de clase positiva (riesgo alto)

**Códigos de estado:**
- `200`: Predicción realizada exitosamente
- `422`: Error de validación en los datos de entrada
- `503`: Modelo no disponible
- `500`: Error interno del servidor

### 4. Predicción por Lotes
```
POST /predict-batch
```
Realiza predicciones para múltiples instancias de forma eficiente.

**Request Body:**
```json
{
  "instances": [
    {
      "laufzeit": 24,
      "hoehe": 5000,
      "alter": 35,
      "beszeit": 2,
      "rate": 1,
      "wohnzeit": 2,
      "verm": 2,
      "bishkred": 1,
      "beruf": 2,
      "laufkont": 1,
      "moral": 2,
      "verw": 3,
      "sparkont": 2,
      "famges": 1,
      "buerge": 1,
      "weitkred": 1,
      "wohn": 1,
      "pers": 1,
      "telef": 1,
      "gastarb": 1
    },
    {
      "laufzeit": 36,
      "hoehe": 10000,
      "alter": 45,
      "beszeit": 3,
      "rate": 2,
      "wohnzeit": 3,
      "verm": 3,
      "bishkred": 2,
      "beruf": 3,
      "laufkont": 2,
      "moral": 3,
      "verw": 4,
      "sparkont": 3,
      "famges": 2,
      "buerge": 2,
      "weitkred": 2,
      "wohn": 2,
      "pers": 2,
      "telef": 2,
      "gastarb": 2
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [1, 0]
}
```

**Descripción de campos:**
- `predictions`: Lista de predicciones binarias (0 o 1) en el mismo orden que las instancias de entrada

**Códigos de estado:**
- `200`: Predicciones realizadas exitosamente
- `422`: Error de validación en los datos de entrada
- `503`: Modelo no disponible
- `500`: Error interno del servidor

**Límites:**
- Máximo 100 instancias por lote

## Descripción del Servicio

### ¿Qué hace el servicio?

El servicio web es una API REST que expone el modelo de machine learning entrenado para realizar predicciones de riesgo crediticio. El servicio:

1. **Carga el modelo entrenado**: Al iniciar, carga el modelo desde `models/best_model.joblib` (si está disponible)
2. **Valida las entradas**: Utiliza modelos Pydantic para validar que los datos de entrada cumplan con los requisitos del modelo
3. **Realiza predicciones**: Procesa las características del solicitante y retorna la predicción de riesgo crediticio
4. **Proporciona probabilidades**: Cuando el modelo lo soporta, también retorna la probabilidad de la predicción

### Características del Modelo

El modelo utiliza **20 características** del solicitante de crédito:

**Características numéricas:**
- `laufzeit`: Duración del crédito (meses)
- `hoehe`: Monto del crédito
- `alter`: Edad del solicitante

**Características ordinales:**
- `beszeit`: Duración del empleo (1-5)
- `rate`: Tasa de pago (1-4)
- `wohnzeit`: Tiempo de residencia actual (1-4)
- `verm`: Propiedad (1-4)
- `bishkred`: Número de créditos previos (1-4)
- `beruf`: Trabajo (1-4)

**Características nominales:**
- `laufkont`: Estado de cuenta (1-4)
- `moral`: Historial crediticio (0-4)
- `verw`: Propósito del crédito (0-10)
- `sparkont`: Cuenta de ahorros (1-5)
- `famges`: Estado personal/sexo (1-4)
- `buerge`: Otros deudores (1-3)
- `weitkred`: Otros planes de pago (1-3)
- `wohn`: Vivienda (1-3)

**Características binarias:**
- `pers`: Personas a cargo (1-2)
- `telef`: Teléfono (1-2)
- `gastarb`: Trabajador extranjero (1-2)

### Validación de Datos

El servicio valida automáticamente:
- Tipos de datos correctos
- Rangos válidos para cada característica
- Valores permitidos para características categóricas
- Presencia de todos los campos requeridos

Si algún dato no es válido, el servicio retorna un error `422` con detalles sobre qué campos son inválidos.

## Convenciones de Código

Siguiendo las reglas de estilo del proyecto:

- **Código**: Escrito en inglés
- **Comentarios**: Escritos en español
- **Documentación**: Escrita en español (modo formal de México)

## Testing

### Estrategia de Workspaces en Postman

El equipo está trabajando bajo una estrategia de **Postman Workspaces** para facilitar la colaboración y el versionado de las colecciones de pruebas.

**Workspace:** `MLOps equipo 29`

Todos los miembros del equipo han sido invitados a formar parte de este workspace, lo que permite:
- Colaboración en tiempo real en las colecciones
- Sincronización automática de cambios
- Historial de versiones
- Compartir variables de entorno entre el equipo
- Ejecutar tests desde cualquier miembro del equipo

### Importar la Colección en Postman

#### Opción 1: Importar desde el Workspace (Recomendado)

Si ya estás en el workspace `MLOps equipo 29`:

1. Abrir Postman y asegurarse de estar en el workspace `MLOps equipo 29`
2. Click en "Import" (esquina superior izquierda o botón "+ New")
3. Seleccionar la pestaña "File"
4. Hacer click en "Upload Files" o arrastrar el archivo
5. Seleccionar `web_service/postman/postman_schema_validation.json`
6. Click en "Import"
7. La colección se importará con todos los endpoints y tests configurados

#### Opción 2: Importar desde URL o Archivo Local

1. Abrir Postman
2. Click en "Import" (esquina superior izquierda)
3. Seleccionar la pestaña "File"
4. Hacer click en "Upload Files" o arrastrar el archivo
5. Navegar a `web_service/postman/postman_schema_validation.json` en el proyecto
6. Seleccionar el archivo y hacer click en "Import"
7. La colección aparecerá en el panel izquierdo con el nombre "Servicio ML Model FastAPI - Tests Completos"

#### Opción 3: Compartir desde el Workspace

Si otro miembro del equipo ya importó la colección al workspace:

1. Abrir Postman y asegurarse de estar en el workspace `MLOps equipo 29`
2. La colección debería aparecer automáticamente en el panel izquierdo
3. Si no aparece, hacer click en "Sync" para sincronizar con el workspace

### Configurar Variables de Entorno

La colección utiliza la variable `baseUrl` que por defecto está configurada como:
```
http://localhost:8000
```

**Para modificar la URL base:**

1. En Postman, hacer click en el ícono de engranaje (⚙️) en la esquina superior derecha
2. Seleccionar "Manage Environments" o hacer click en el dropdown de entornos
3. Crear un nuevo entorno o editar el existente (recomendado: crear entornos para desarrollo, staging y producción)
4. Agregar la variable:
   - **Variable:** `baseUrl`
   - **Initial Value:** `http://localhost:8000` (desarrollo) o la URL correspondiente
   - **Current Value:** Dejar igual o ajustar según necesidad
5. Guardar el entorno
6. Seleccionar el entorno desde el dropdown en la esquina superior derecha

**Ejemplo de entornos:**
- **Desarrollo:** `http://localhost:8000`
- **Staging:** `https://staging-api.ejemplo.com`
- **Producción:** `https://api.ejemplo.com`

### Ejecutar los Tests

Una vez importada la colección:

1. **Ejecutar un endpoint individual:**
   - Hacer click en el endpoint deseado
   - Click en "Send"
   - Revisar los resultados en la pestaña "Test Results"

2. **Ejecutar toda la colección:**
   - Click derecho en la colección "Servicio ML Model FastAPI - Tests Completos"
   - Seleccionar "Run collection"
   - Revisar el resumen de resultados
   - Todos los tests deberían pasar si el servicio está funcionando correctamente

3. **Ejecutar con el Runner:**
   - Click en "Runner" en la barra superior
   - Seleccionar la colección
   - Configurar iteraciones y delay si es necesario
   - Click en "Run [nombre de la colección]"

## Integración con el Proyecto

El servicio web está integrado con:

- **Modelos entrenados**: Carga modelos desde `models/best_model.joblib` usando DVC
- **Configuración**: Accede a `mlops_project/config.py` para parámetros del modelo y definiciones de features
- **Validación**: Utiliza las reglas de validación definidas en `CATEGORICAL_VALIDATION_RULES` del config
- **Pipeline MLOps**: Se integra con el pipeline de entrenamiento existente

### Flujo de Trabajo

1. **Entrenar el modelo**: Usar `make train` o `uv run mlops-train` para entrenar y guardar el modelo
2. **Verificar modelo disponible**: El modelo debe estar en `models/best_model.joblib` (puede requerir `dvc pull`)
3. **Iniciar el servicio**: Ejecutar `uv run uvicorn web_service.app:app --reload`
4. **Probar el servicio**: Usar Postman o acceder a `/docs` para probar los endpoints

## Próximos Pasos y Mejoras Pendientes

### Implementación Completada ✅

- ✅ Aplicación FastAPI con todos los endpoints definidos
- ✅ Integración con carga de modelos desde DVC
- ✅ Validación de entrada con Pydantic
- ✅ Manejo de errores robusto
- ✅ Logging integrado
- ✅ Documentación automática con Swagger/OpenAPI
- ✅ Colecciones de Postman para testing

### Pendiente para Próximas Etapas

#### 1. Testing Automatizado
- [ ] Tests unitarios para `service.py` (carga de modelos, predicciones)
- [ ] Tests unitarios para `app.py` (endpoints, manejo de errores)
- [ ] Tests de integración end-to-end
- [ ] Tests de carga y rendimiento
- [ ] Integración con pytest y coverage

#### 2. Preprocesador
- [ ] Integrar el preprocesador (`models/preprocessor.joblib`) para transformar features antes de la predicción
- [ ] Actualmente el modelo espera que las features ya estén preprocesadas
- [ ] Validar que el preprocesador se carga correctamente al inicio

#### 3. Mejoras de Producción
- [ ] Configuración de variables de entorno para diferentes entornos (dev, staging, prod)
- [ ] Rate limiting para prevenir abuso de la API
- [ ] Autenticación y autorización (API keys, JWT, etc.)
- [ ] CORS configurado apropiadamente
- [ ] Health checks más detallados (verificar modelo, memoria, etc.)

#### 4. Monitoreo y Observabilidad
- [ ] Métricas de Prometheus (tiempo de respuesta, número de predicciones, errores)
- [ ] Integración con sistemas de logging centralizados (ELK, CloudWatch, etc.)
- [ ] Alertas para errores y degradación del servicio
- [ ] Tracking de uso y analytics

#### 5. Despliegue
- [ ] Dockerfile para containerización
- [ ] docker-compose.yml para desarrollo local
- [ ] Configuración de CI/CD (GitHub Actions)
- [ ] Despliegue en cloud (AWS, GCP, Azure)
- [ ] Configuración de load balancer y auto-scaling

#### 6. Documentación Adicional
- [ ] Ejemplos de uso con diferentes lenguajes (Python, JavaScript, curl)
- [ ] Guía de troubleshooting
- [ ] Diagramas de arquitectura
- [ ] Guía de contribución para el módulo

#### 7. Optimizaciones
- [ ] Caché de predicciones frecuentes
- [ ] Batch processing asíncrono para lotes grandes
- [ ] Optimización de carga de modelos (lazy loading, versionado)
- [ ] Soporte para múltiples modelos simultáneos (A/B testing)

#### 8. Seguridad
- [ ] Validación de entrada más estricta (sanitización)
- [ ] Protección contra ataques comunes (SQL injection, XSS, etc.)
- [ ] Encriptación de datos sensibles
- [ ] Auditoría de accesos y predicciones

## Referencias

- [Documentación de FastAPI](https://fastapi.tiangolo.com/)
- [Guía de Postman](https://learning.postman.com/docs/)
- [Roadmap del Proyecto](../docs/roadmap.md)

---

**Equipo 29** - TC5044.10  
**Fecha**: Noviembre 2025

