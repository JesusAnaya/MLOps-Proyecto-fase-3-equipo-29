"""
App contiene la aplicación FastAPI del servicio web.
"""
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse

from web_service.models import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from web_service.service import (
    get_model_info,
    is_model_loaded,
    predict_batch as predict_batch_service,
    predict_single,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title="Credit Prediction API - MLOps Grupo 29",
    description="API para predicción de riesgo crediticio, elaborada por el equipo 29 del curso MLOps",
    version="0.1.0",
)


@app.on_event("startup")
async def startup_event():
    logger.info("Iniciando aplicación...")
    try:
        if is_model_loaded():
            logger.info("Modelo cargado exitosamente")
        else:
            logger.warning(
                "Modelo no disponible. Ejecuta 'dvc pull models/best_model.joblib.dvc' "
            )
    except Exception as e:
        logger.warning(f"No se pudo verificar el modelo al inicio: {e}")
    logger.info("Aplicación iniciada correctamente")


@app.get("/", include_in_schema=False)
async def root():
    """
    Endpoint raíz para presentar el OpenAPI / Swagger de la API.
    """
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    return HealthResponse()


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """
    Endpoint para obtener información del modelo desplegado.
    """
    try:
        info = get_model_info()
        # Agregar estado de disponibilidad
        model_available = is_model_loaded()
        response = ModelInfoResponse(**info)
        return response
    except Exception as e:
        logger.error(f"Error al obtener información del modelo: {e}")
        raise HTTPException(
            status_code=503,
            detail="Error al obtener información del modelo.",
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    # Verificar que el modelo esté disponible
    if not is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail=(
                "Modelo no disponible. Por favor, ejecuta: dvc pull models/best_model.joblib.dvc"
            ),
        )

    try:
        # Convertir FeaturesModel a dict para el servicio
        features_dict = request.features.model_dump()

        # Realizar predicción
        result = predict_single(features_dict)

        return PredictionResponse(**result)

    except FileNotFoundError as e:
        logger.error(f"Modelo no encontrado: {e}")
        raise HTTPException(
            status_code=503,
            detail=(
                "Modelo no disponible. Por favor, ejecuta: dvc pull models/best_model.joblib.dvc"
                
            ),
        )
    except ValueError as e:
        logger.error(f"Error de validación en predicción: {e}")
        raise HTTPException(status_code=422, detail=f"Error de validación: {str(e)}")
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error al realizar predicción: {str(e)}")


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["BatchPrediction"])
async def predict_batch_endpoint(request: BatchPredictionRequest):
    # Verificar que el modelo esté disponible
    if not is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail=(
                "Modelo no disponible. Por favor, ejecuta: dvc pull models/best_model.joblib.dvc"
            ),
        )

    try:
        # Convertir List[FeaturesModel] a List[dict]
        instances_dict = [instance.model_dump() for instance in request.instances]

        # Realizar predicciones
        predictions = predict_batch_service(instances_dict)

        return BatchPredictionResponse(predictions=predictions)

    except FileNotFoundError as e:
        logger.error(f"Modelo no encontrado: {e}")
        raise HTTPException(
            status_code=503,
            detail=(
                "Modelo no disponible. Por favor, ejecuta: dvc pull models/best_model.joblib.dvc"
            ),
        )
    except ValueError as e:
        logger.error(f"Error de validación en predicción por lotes: {e}")
        raise HTTPException(status_code=422, detail=f"Error de validación: {str(e)}")
    except Exception as e:
        logger.error(f"Error en predicción por lotes: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error al realizar predicciones por lotes: {str(e)}"
        )


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request, exc):
    """Handler para errores de archivo no encontrado."""
    logger.error(f"Archivo no encontrado: {exc}")
    return JSONResponse(
        status_code=503,
        content={"detail": "Modelo no disponible. El archivo del modelo no fue encontrado."},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handler para errores """
    logger.error(f"Error no manejado: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Error interno del servidor. Por favor, intente más tarde."},
    )