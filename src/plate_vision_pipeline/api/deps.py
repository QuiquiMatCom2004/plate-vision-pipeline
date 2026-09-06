"""Dependencias inyectables de la API.

Todas las funciones acá son *factories* de FastAPI (usadas con ``Depends``)
para que los tests puedan reemplazarlas vía ``app.dependency_overrides``
sin tocar Sami 2, instructor, ni la red.
"""

from __future__ import annotations

import io
import logging
from functools import lru_cache
from typing import Annotated

import anthropic
import numpy as np
from fastapi import Depends, HTTPException, Request
from PIL import Image

from plate_vision_pipeline.config import Settings
from plate_vision_pipeline.graph import RouteAfterMeasure, build_graph
from plate_vision_pipeline.models import (
    DescribeModel,
    DetectModel,
    MeasureModel,
    SegmentModel,
)
from plate_vision_pipeline.state import PipelineState, create_initial_pipelinestate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Convierte bytes de una imagen (PNG/JPG/etc.) a np.ndarray RGB.

    Separa el IO (formato, canales) del contrato de datos del PipelineState.
    Usa PIL para decodificar y fuerza a RGB para que YOLO/SAM2 reciban siempre
    3 canales.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(img)


# ---------------------------------------------------------------------------
# Settings (cached — se lee una vez por proceso)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# ---------------------------------------------------------------------------
# Modelos y grafo (cargados una sola vez en lifespan)
# ---------------------------------------------------------------------------


def get_graph(request: Request) -> PipelineState:
    """Devuelve el grafo compilado guardado en ``app.state`` durante el lifespan.

    Si el lifespan no corrió (p.ej. en tests sin lifespan), levanta 503.
    """
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline no inicializado. El servidor está arrancando o falló el lifespan.",
        )
    return graph


def get_route_after_measure(
    settings: Annotated[Settings, Depends(get_settings)],
) -> RouteAfterMeasure:
    """Instancia la ruta de reintentos con el max_attempts de Settings."""
    return RouteAfterMeasure(max_attempts=settings.max_measure_attempts)


# ---------------------------------------------------------------------------
# Factory: construye el grafo completo a partir de Settings.
# Se llama una sola vez en lifespan.
# ---------------------------------------------------------------------------


def build_pipeline(settings: Settings) -> PipelineState:
    """Construye el grafo LangGraph con todos los modelos configurados.

    Cada modelo se instancia con los paths y parámetros de Settings.
    Si un modelo falla al cargar (p.ej. VLM sin GPU), el __init__ de
    DescribeModel no levanta — deja model=None y el nodo decide el fallback.
    """
    detect_model = DetectModel(
        modelpath=settings.yolo_weights_path,
        device=settings.device,
        min_yolo_conf=settings.min_yolo_conf,
    )

    segment_model = SegmentModel(
        config_file="sam2_hiera_b+.yaml",
        checkpoint_path=settings.sam2_checkpoint_path,
        device=settings.device,
    )

    describe_model = DescribeModel(
        vlm_mode=settings.vlm_mode,
        vlm_name=settings.vlm_name,
        device=settings.device,
        apikey=settings.apikey,
    )

    measure_model = MeasureModel(
        raw_client=anthropic.Anthropic(api_key=settings.apikey),
        model_name=settings.measure_model_name,
        max_tokens=settings.measure_max_tokens,
    )

    route = RouteAfterMeasure(max_attempts=settings.max_measure_attempts)

    return build_graph(
        detect_node=detect_model,
        segment_node=segment_model,
        describe_node=describe_model,
        measure_node=measure_model,
        route_after_measure=route,
    )
