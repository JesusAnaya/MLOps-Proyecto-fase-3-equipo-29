"""
Script entry point para ejecutar el servicio FastAPI.
Permite ejecutar el servicio con: python -m web_service
o mediante el script: mlops-web-service
"""
import uvicorn


def main():
    """Función principal para ejecutar el servicio FastAPI."""
    uvicorn.run(
        "web_service.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()

