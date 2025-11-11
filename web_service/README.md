# Módulo Web Service

Este módulo contiene la implementación del servicio web para el proyecto MLOps, proporcionando una API REST para el servicio de modelos de machine learning.

## Propósito

El módulo `web_service` tiene como objetivo exponer los modelos entrenados del proyecto a través de una API REST utilizando FastAPI. Este servicio permitirá:

- Realizar predicciones individuales y por lotes
- Consultar información sobre los modelos desplegados
- Verificar el estado de salud del servicio
- Proporcionar documentación interactiva mediante Swagger/OpenAPI

## Estructura del Módulo

```
web_service/
├── README.md                    # Este archivo
├── postman/                     # Colecciones de Postman para pruebas
│   └── postman_schema_validation.json  # Colección completa con validación de estructura
└── [implementación futura]     # Código de FastAPI (pendiente)
```

## Estado Actual

Actualmente, el módulo contiene:

- **Colecciones de Postman**: Archivos JSON con definiciones de endpoints y tests para validar la funcionalidad y estructura de la API una vez implementada.

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

## Implementación Futura

### Endpoints Planificados

#### 1. Health Check
```
GET /health
```
Verifica que el servicio esté operativo.

**Respuesta esperada:**
```json
{
  "status": "ok"
}
```

#### 2. Información del Modelo
```
GET /model-info
```
Retorna información sobre el modelo actualmente desplegado.

**Respuesta esperada:**
```json
{
  "model_name": "logistic_regression",
  "model_version": "0.1.0"
}
```

#### 3. Predicción Individual
```
POST /predict
```
Realiza una predicción para una instancia única.

**Request:**
```json
{
  "features": {
    "feature_1": 35,
    "feature_2": 1200.5,
    "feature_3": 2
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

#### 4. Predicción por Lotes
```
POST /predict-batch
```
Realiza predicciones para múltiples instancias.

**Request:**
```json
{
  "instances": [
    {
      "feature_1": 35,
      "feature_2": 1200.5,
      "feature_3": 2
    },
    {
      "feature_1": 52,
      "feature_2": 3000.0,
      "feature_3": 1
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

El servicio web se integrará con:

- **Modelos entrenados**: Cargará modelos desde `models/best_model.joblib`
- **Preprocesador**: Utilizará `models/preprocessor.joblib` para transformar features
- **Configuración**: Accederá a `mlops_project/config.py` para parámetros del modelo

## Próximos Pasos

1. Implementar aplicación FastAPI con los endpoints definidos
2. Integrar carga de modelos y preprocesador
3. Agregar validación de entrada con Pydantic
4. Implementar manejo de errores robusto
5. Agregar logging y monitoreo
6. Configurar documentación automática con Swagger/OpenAPI
7. Agregar tests unitarios e integración con pytest
8. Configurar despliegue con Docker (futuro)

## Referencias

- [Documentación de FastAPI](https://fastapi.tiangolo.com/)
- [Guía de Postman](https://learning.postman.com/docs/)
- [Roadmap del Proyecto](../docs/roadmap.md)

---

**Equipo 29** - TC5044.10  
**Fecha**: Noviembre 2025

