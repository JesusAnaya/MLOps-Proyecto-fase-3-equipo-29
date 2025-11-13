"""
Modelos Pydantic para validación de entrada y salida de la API.

Módulo que define los esquemas de datos para las solicitudes y respuestas
de los endpoints de la API de predicción de riesgo crediticio.
"""

from typing import List

from pydantic import BaseModel, Field, field_validator

from mlops_project.config import CATEGORICAL_VALIDATION_RULES, MLFLOW_MODEL_VERSION


class HealthResponse(BaseModel):

    status: str = Field(default="ok", description="Estado del servicio")

    class Config:
        json_schema_extra = {"example": {"status": "ok"}}


class FeaturesModel(BaseModel):

    # Features numéricas
    laufzeit: int = Field(..., description="Duración del crédito", ge=1)
    hoehe: int = Field(..., description="Monto del crédito", ge=0)
    alter: int = Field(..., description="Edad del solicitante", ge=18, le=100)

    # Features ordinales
    beszeit: int = Field(..., description="Duración del empleo", ge=1, le=5)
    rate: int = Field(..., description="Tasa de pago", ge=1, le=4)
    wohnzeit: int = Field(..., description="Tiempo de residencia actual", ge=1, le=4)
    verm: int = Field(..., description="Propiedad", ge=1, le=4)
    bishkred: int = Field(..., description="Número de créditos previos", ge=1, le=4)
    beruf: int = Field(..., description="Trabajo", ge=1, le=4)

    # Features nominales
    laufkont: int = Field(..., description="Estado de cuenta")
    moral: int = Field(..., description="Historial crediticio")
    verw: int = Field(..., description="Propósito del crédito")
    sparkont: int = Field(..., description="Cuenta de ahorros")
    famges: int = Field(..., description="Estado personal/sexo")
    buerge: int = Field(..., description="Otros deudores")
    weitkred: int = Field(..., description="Otros planes de pago")
    wohn: int = Field(..., description="Vivienda")
    pers: int = Field(..., description="Personas a cargo (binaria)")
    telef: int = Field(..., description="Teléfono (binaria)")
    gastarb: int = Field(..., description="Trabajador extranjero (binaria)")

    @field_validator("laufkont", "moral", "verw", "sparkont", "famges", "buerge", "weitkred", "wohn")
    @classmethod
    def validate_nominal_features(cls, v: int, info) -> int:
        field_name = info.field_name
        if field_name in CATEGORICAL_VALIDATION_RULES:
            allowed_values = CATEGORICAL_VALIDATION_RULES[field_name]
            if v not in allowed_values:
                raise ValueError(
                    f"{field_name} debe ser uno de {allowed_values}, recibido: {v}"
                )
        return v

    @field_validator("pers", "telef", "gastarb")
    @classmethod
    def validate_binary_features(cls, v: int) -> int:
        if v not in [1, 2]:
            raise ValueError(f"Debe ser 1 o 2, recibido: {v}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
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
                "gastarb": 1,
            }
        }


class PredictionRequest(BaseModel):
    features: FeaturesModel = Field(..., description="Características del solicitante")

    class Config:
        json_schema_extra = {
            "example": {
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
                    "gastarb": 1,
                }
            }
        }


class PredictionResponse(BaseModel):

    prediction: int = Field(..., description="Predicción (0: riesgo bajo, 1: riesgo alto)", ge=0, le=1)
    probability: float = Field(..., description="Probabilidad de clase positiva", ge=0.0, le=1.0)

    class Config:
        json_schema_extra = {"example": {"prediction": 1, "probability": 0.85}}


class BatchPredictionRequest(BaseModel):

    instances: List[FeaturesModel] = Field(
        ..., description="Lista de instancias para predecir", min_length=1, max_length=100
    )

    class Config:
        json_schema_extra = {
            "example": {
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
                        "gastarb": 1,
                    },
                    {
                        "laufzeit": 36,
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
                        "gastarb": 1,
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):

    predictions: List[int] = Field(..., description="Lista de predicciones (0 o 1)")

    class Config:
        json_schema_extra = {"example": {"predictions": [1, 0]}}


class ModelInfoResponse(BaseModel):

    model_name: str = Field(..., description="Nombre del modelo")
    model_version: str = Field(default=MLFLOW_MODEL_VERSION, description="Versión del modelo")

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "logistic_regression",
                "model_version": "0.1.0",
            }
        }
