"""Tests de la API con TestClient y dependency overrides.

Los tests mockean el grafo LangGraph para devolver un estado predefinido,
evitando tocar Sami 2, instructor, YOLO o cualquier red. Así los tests
corren en CPU pura y validan únicamente la capa de FastAPI.
"""

from __future__ import annotations

import io
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from plate_vision_pipeline.api.main import app
from plate_vision_pipeline.api.deps import get_graph
from plate_vision_pipeline.state import PipelineState


@pytest.fixture
def client():
    return TestClient(app)


def _valid_png_bytes() -> bytes:
    """PNG real de 1x1 px — decode_image() lo tiene que poder abrir de verdad.

    Con el 422 de UnidentifiedImageError ya implementado en el endpoint,
    cualquier test que espere pasar la decodificación necesita bytes de
    imagen genuinos, no un placeholder como b"fake".
    """
    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), color=(255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_endpoint(client):
    """El endpoint /health debe devolver 200 y JSON con status=ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_analyze_endpoint_success(client, monkeypatch):
    """POST /analyze con imagen válida devuelve 200 y PlateAnalysis."""
    # 1. Armar un estado de salida del grafo que simule éxito
    fake_state: PipelineState = {
        "image": None,  # no se usa después del invoke
        "detections": [
            {"bbox": [10, 10, 50, 50], "cls": "pizza", "conf": 0.9},
            {"bbox": [60, 20, 100, 80], "cls": "ensalada", "conf": 0.8},
        ],
        "description": "Una pizza con ensalada al lado.",
        "structure": {
            "food_items": [
                {
                    "name": "Pizza",
                    "coco_class": "pizza",
                    "portion_estimate_g": 200,
                    "calories_estimate": 500,
                    "macros": {"protein_g": 15.0, "carbs_g": 40.0, "fat_g": 20.0},
                },
                {
                    "name": "Ensalada",
                    "coco_class": "ensalada",
                    "portion_estimate_g": 100,
                    "calories_estimate": 50,
                    "macros": {"protein_g": 2.0, "carbs_g": 8.0, "fat_g": 1.0},
                },
            ],
            "meal_type_guess": "almuerzo",
            "dietary_flags": ["alto en carbohidratos"],
            "macros_total": {"protein_g": 17.0, "carbs_g": 48.0, "fat_g": 21.0},
            "balanced_score": 0.75,
            "notes": "Comida balanceada.",
        },
        "latency_ms": {"detect": 120.5, "segment": 80.0, "describe": 300.0, "measure": 50.0},
        "errors": [],
    }

    # 2. Mockeamos la dependencia get_graph para que devuelva un grafo que
    #    al invocar con cualquier estado devuelva exactamente fake_state.
    class FakeGraph:
        def invoke(self, state: PipelineState) -> PipelineState:
            return fake_state

    def override_get_graph(_request):
        return FakeGraph()

    app.dependency_overrides[get_graph] = override_get_graph

    try:
        # 3. Llamamos al endpoint con un PNG real (decode_image debe poder abrirlo)
        files = {"image": ("test.png", _valid_png_bytes(), "image/png")}
        response = client.post("/analyze", files=files)

        # 4. Validaciones
        assert response.status_code == 200
        data = response.json()

        # El cuerpo debe coincidir exactamente con el PlateAnalysis que armamos
        assert data == {
            "detections": fake_state["detections"],
            "description": fake_state["description"],
            "structure": fake_state["structure"],
            "latency_ms": fake_state["latency_ms"],
            "errors": fake_state["errors"],
        }

        # Además, verificamos que el modelo Pydantic lo valida (si faltara
        # algún campo o tipo incorrecto, el test fallaría aquí).
        from plate_vision_pipeline.api.main import PlateAnalysis

        PlateAnalysis(**data)  # no debe lanzar

    finally:
        # Limpiamos el override para no afectar otros tests
        app.dependency_overrides.clear()


def test_analyze_endpoint_invalid_image(client):
    """POST /analyze con archivo no imagen debe devolver 422 (Unprocessable Entity)."""

    # No debería llegar a usar el grafo, pero lo overrideamos igual: sin esto,
    # la dependencia real get_graph corta con 503 antes de validar la imagen
    # (el client fixture no dispara el lifespan real, a propósito).
    class UnusedGraph:
        def invoke(self, state):
            raise AssertionError("no debería invocarse el grafo con una imagen inválida")

    def override_get_graph(_request):
        return UnusedGraph()

    app.dependency_overrides[get_graph] = override_get_graph

    try:
        # Enviamos un archivo que no es imagen (texto plano)
        files = {"image": ("test.txt", b"not an image", "text/plain")}
        response = client.post("/analyze", files=files)
        # El endpoint debe capturar el fallo de decodificación y devolver 422
        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_analyze_endpoint_pipeline_fails(client, monkeypatch):
    """Si el grafo lanza una excepción, el endpoint debe devolver 500."""
    class FailingGraph:
        def invoke(self, state: PipelineState):
            raise RuntimeError("fallo interno del grafo")

    def override_get_graph(_request):
        return FailingGraph()

    app.dependency_overrides[get_graph] = override_get_graph

    try:
        files = {"image": ("test.png", _valid_png_bytes(), "image/png")}
        response = client.post("/analyze", files=files)
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "fallo interno del grafo" in data["detail"]
    finally:
        app.dependency_overrides.clear()


def test_analyze_endpoint_no_structure(client, monkeypatch):
    """Si el grafo devuelve estado sin estructura válida, endpoint 500."""
    fake_state_no_struct: PipelineState = {
        "image": None,
        "detections": [],
        "description": "",
        "structure": {},  # vacío → se considera fallo
        "latency_ms": {},
        "errors": ["describe falló"],
    }

    class FakeGraph:
        def invoke(self, state: PipelineState) -> PipelineState:
            return fake_state_no_struct

    def override_get_graph(_request):
        return FakeGraph()

    app.dependency_overrides[get_graph] = override_get_graph

    try:
        files = {"image": ("test.png", _valid_png_bytes(), "image/png")}
        response = client.post("/analyze", files=files)
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Pipeline falló al producir estructura" in data["detail"]
    finally:
        app.dependency_overrides.clear()