"""FastAPI application para el pipeline de análisis de plato.

Endpoint:
    POST /analyze — sube una imagen, devuelve PlateAnalysis (Pydantic).

GET /health — healthcheck para Railway.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from plate_vision_pipeline.api.deps import build_pipeline, decode_image, get_graph
from plate_vision_pipeline.config import Settings
from plate_vision_pipeline.graph import RouteAfterMeasure
from plate_vision_pipeline.models import DetectModel, SegmentModel, DescribeModel, MeasureModel
from plate_vision_pipeline.state import PipelineState

logger = logging.getLogger(__name__)


class PlateAnalysis(BaseModel):
    """Schema de salida del pipeline — validado por Pydantic."""

    detections: list[dict[str, Any]]
    description: str
    structure: dict[str, Any]
    latency_ms: dict[str, float]
    errors: list[str]


class HealthResponse(BaseModel):
    status: str
    uptime: float


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Crea la app FastAPI con lifespan que carga modelos una vez."""
    app = FastAPI(title="Plate Vision Pipeline", version="0.1.0")

    @app.on_event("startup")
    async def startup():
        """Carga modelos y compila el grafo una sola vez al arrancar."""
        settings = Settings()
        logger.info("Iniciando pipeline — VLM_MODE=%s, DEVICE=%s", settings.vlm_mode, settings.device)
        try:
            graph = build_pipeline(settings)
            app.state.graph = graph
            logger.info("Pipeline compilado exitosamente.")
        except Exception as e:
            logger.error("Error al compilar pipeline: %s", e, exc_info=True)
            raise

    @app.on_event("shutdown")
    async def shutdown():
        """Limpieza de recursos."""
        logger.info("Pipeline detenido.")

    @app.post("/analyze", response_model=PlateAnalysis)
    async def analyze(
        file: UploadFile,
        graph: PipelineState = Depends(get_graph),
    ):
        """Analiza una imagen: detecta, segmenta, describe, mide.

        Request: multipart /analyze con `image` (UploadFile).
        Response: PlateAnalysis (Pydantic) — si el pipeline falla,
        HTTP 500 con error en `errors`.
        """
        # 1. Decodificar la imagen
        image_np = decode_image(file.file.read())

        # 2. Construir el estado inicial (helper del endpoint)
        state = create_initial_pipelinestate(image_np)

        # 3. Ejecutar el grafo
        graph = app.state.graph
        try:
            result = graph.invoke(state)
        except Exception as e:
            logger.error("Error ejecutando grafo: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        # 4. Extraer resultados
        detections = result.get("detections", [])
        description = result.get("description", "")
        structure = result.get("structure", {})
        latency_ms = result.get("latency_ms", {})
        errors = result.get("errors", [])

        # 5. Si no hubo estructura válida, el pipeline falló
        if not isinstance(structure, dict) or not structure:
            logger.warning("Pipeline no produjo estructura válida: %s", structure)
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline falló al producir estructura. Errores: {errors}",
            )

        # 6. Logar latencia (solo para debug, no en el body)
        for node, ms in latency_ms.items():
            logger.info("Node %s took %.3f ms", node, ms)

        # 7. Validar que la respuesta tiene los campos esperados
        if not isinstance(detections, list):
            raise HTTPException(status_code=500, detail="Pipeline no produjo detecciones.")
        if not isinstance(description, str):
            raise HTTPException(status_code=500, detail="Pipeline no produjo descripción.")
        if not isinstance(structure, dict):
            raise HTTPException(status_code=500, detail="Pipeline no produjo estructura.")

        return PlateAnalysis(
            detections=detections,
            description=description,
            structure=structure,
            latency_ms=latency_ms,
            errors=errors,
        )

    @app.get("/health")
    async def health():
        """Healthcheck rápido."""
        return HealthResponse(
            status="ok",
            uptime=0.0,
        )

    return app


app = create_app()
