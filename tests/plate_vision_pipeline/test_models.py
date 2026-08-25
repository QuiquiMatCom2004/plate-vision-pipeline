import numpy as np
import pytest

import plate_vision_pipeline.models as models_module
from plate_vision_pipeline.models import DescribeModel, DetectModel, MeasureModel, SegmentModel


class FakeBox:
    """Doble de un `ultralytics.engine.results.Boxes` individual."""

    def __init__(self, bbox, conf, cls_idx):
        self.xyxy = [np.array(bbox, dtype=float)]
        self.conf = [conf]
        self.cls = [cls_idx]


class FakeResult:
    """Doble del primer elemento que devuelve `YOLO.predict(...)`."""

    def __init__(self, boxes=None, names=None):
        self.boxes = boxes
        self.names = names if names is not None else {}


class FakeYOLO:
    """Doble de `ultralytics.YOLO`: registra cómo fue construido/llamado."""

    def __init__(self, modelpath):
        self.modelpath = modelpath
        self.device = None
        self.predict_calls = []
        self.result = FakeResult(boxes=[], names={})

    def to(self, device):
        self.device = device

    def predict(self, image, conf):
        self.predict_calls.append({"image": image, "conf": conf})
        return [self.result]


@pytest.fixture
def patch_yolo(monkeypatch):
    monkeypatch.setattr(models_module, "YOLO", FakeYOLO)


def test_init_loads_model_and_moves_to_device(patch_yolo):
    model = DetectModel(modelpath="yolov11n.pt", device="cpu", min_yolo_conf=0.5)

    assert model.model.modelpath == "yolov11n.pt"
    assert model.model.device == "cpu"
    assert model.min_conf == 0.5


def test_predict_calls_model_with_min_conf_threshold(patch_yolo):
    model = DetectModel(modelpath="yolov11n.pt", device="cpu", min_yolo_conf=0.35)

    model.predict(np.zeros((10, 10, 3)))

    assert model.model.predict_calls[0]["conf"] == 0.35


def test_predict_no_boxes_returns_empty_list(patch_yolo):
    model = DetectModel(modelpath="yolov11n.pt", device="cpu", min_yolo_conf=0.5)
    model.model.result = FakeResult(boxes=None, names={})

    result = model.predict(np.zeros((10, 10, 3)))

    assert result == []


def test_predict_empty_boxes_returns_empty_list(patch_yolo):
    model = DetectModel(modelpath="yolov11n.pt", device="cpu", min_yolo_conf=0.5)
    model.model.result = FakeResult(boxes=[], names={})

    result = model.predict(np.zeros((10, 10, 3)))

    assert result == []


def test_predict_maps_boxes_to_detections(patch_yolo):
    model = DetectModel(modelpath="yolov11n.pt", device="cpu", min_yolo_conf=0.5)
    boxes = [
        FakeBox(bbox=[0.0, 0.0, 10.0, 10.0], conf=0.91, cls_idx=0),
        FakeBox(bbox=[20.0, 20.0, 40.0, 40.0], conf=0.85, cls_idx=1),
    ]
    model.model.result = FakeResult(boxes=boxes, names={0: "pizza", 1: "salad"})

    result = model.predict(np.zeros((10, 10, 3)))

    assert result == [
        {"bbox": [0.0, 0.0, 10.0, 10.0], "conf": 0.91, "cls": "pizza"},
        {"bbox": [20.0, 20.0, 40.0, 40.0], "conf": 0.85, "cls": "salad"},
    ]


def test_predict_returns_native_python_types(patch_yolo):
    model = DetectModel(modelpath="yolov11n.pt", device="cpu", min_yolo_conf=0.5)
    boxes = [FakeBox(bbox=[1.0, 2.0, 3.0, 4.0], conf=0.7, cls_idx=0)]
    model.model.result = FakeResult(boxes=boxes, names={0: "pizza"})

    result = model.predict(np.zeros((10, 10, 3)))
    detection = result[0]

    assert isinstance(detection["conf"], float)
    assert isinstance(detection["bbox"], list)
    assert isinstance(detection["cls"], str)


# ---------------------------------------------------------------------------
# SegmentModel
#
# Contrato (confirmado):
#   SegmentModel(config_file, checkpoint_path, device)
#   .predict(image: np.ndarray, boxes: list[list[float]]) -> list[np.ndarray]
#
# Internamente usa la API estándar de Meta SAM2:
#   - build_sam2(config_file=..., ckpt_path=..., device=...) -> sam_model
#   - SAM2ImagePredictor(sam_model) -> predictor (queda en self.predictor)
#   - predictor.set_image(image)     # UNA vez por imagen (caro)
#   - predictor.predict(box=..., multimask_output=False) -> (masks, scores, logits)
#                                    # una vez POR bbox (barato)
#
# `config_file` NO es un setting de entorno — lo inyecta el container de
# dependencias del nodo. Open/Closed: si mañana mejora una arquitectura de
# SAM2, no tocamos el modelo, cambiamos el config inyectado.
#
# Edge case: boxes == [] -> NO se llama set_image (paso caro), devuelve [].
# ---------------------------------------------------------------------------


class FakeSam2Predictor:
    """Doble de `SAM2ImagePredictor`.

    Replica el contrato real: `predict()` devuelve una tupla
    (masks, scores, logits), donde masks tiene shape (1, H, W) bool.
    """

    def __init__(self, *args, **kwargs):
        self.set_image_calls = []
        self.predict_calls = []
        self.masks_to_return = []

    def set_image(self, image):
        self.set_image_calls.append(image)

    def predict(self, *args, box=None, multimask_output=False, **kwargs):
        idx = len(self.predict_calls)
        self.predict_calls.append({"box": box, "multimask_output": multimask_output})
        mask = self.masks_to_return[idx]
        masks = mask[np.newaxis, ...]
        scores = np.array([0.95])
        logits = np.zeros_like(masks, dtype=float)
        return masks, scores, logits


class FakeBuildSam2:
    """Doble de `build_sam2`. Registra cómo fue construido el modelo interno."""

    def __init__(self):
        self.calls = []
        self.return_value = object()

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        return self.return_value


@pytest.fixture
def patch_sam2(monkeypatch):
    fake_build = FakeBuildSam2()
    fake_predictor_cls = FakeSam2Predictor
    monkeypatch.setattr(models_module, "build_sam2", fake_build)
    monkeypatch.setattr(models_module, "SAM2ImagePredictor", fake_predictor_cls)
    return fake_build


# --- Construction: build_sam2 recibe los tres parámetros con los nombres
# correctos de la API real de Meta. El predictor se construye con el modelo
# devuelto por build_sam2. -----------------------------------------------


def test_segment_init_calls_build_sam2_with_correct_kwargs(patch_sam2):
    SegmentModel(
        config_file="sam2_hiera_b+.yaml",
        checkpoint_path="sam2.pt",
        device="cpu",
    )

    assert patch_sam2.calls == [
        {
            "config_file": "sam2_hiera_b+.yaml",
            "ckpt_path": "sam2.pt",
            "device": "cpu",
        }
    ]


def test_segment_init_builds_predictor_from_sam_model(patch_sam2, monkeypatch):
    sam_model = object()
    patch_sam2.return_value = sam_model
    predictor_instances = []

    def fake_predictor_cls(sam):
        instance = FakeSam2Predictor()
        instance.received_sam_model = sam
        predictor_instances.append(instance)
        return instance

    monkeypatch.setattr(models_module, "SAM2ImagePredictor", fake_predictor_cls)

    model = SegmentModel(
        config_file="sam2_hiera_b+.yaml",
        checkpoint_path="sam2.pt",
        device="cpu",
    )

    assert predictor_instances[0].received_sam_model is sam_model
    assert model.predictor is predictor_instances[0]


# --- predict: set_image una sola vez ---------------------------------------


def test_segment_predict_calls_set_image_exactly_once(patch_sam2):
    model = SegmentModel(
        config_file="sam2_hiera_b+.yaml", checkpoint_path="sam2.pt", device="cpu"
    )
    model.predictor.masks_to_return = [
        np.ones((5, 5), dtype=bool),
        np.zeros((5, 5), dtype=bool),
    ]
    image = np.zeros((10, 10, 3))
    boxes = [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 40.0]]

    model.predict(image, boxes)

    assert len(model.predictor.set_image_calls) == 1
    assert model.predictor.set_image_calls[0] is image


# --- predict: una llamada a predictor.predict por box ----------------------


def test_segment_predict_calls_predict_once_per_box(patch_sam2):
    model = SegmentModel(
        config_file="sam2_hiera_b+.yaml", checkpoint_path="sam2.pt", device="cpu"
    )
    model.predictor.masks_to_return = [
        np.ones((5, 5), dtype=bool),
        np.zeros((5, 5), dtype=bool),
        np.ones((5, 5), dtype=bool),
    ]
    boxes = [
        [0.0, 0.0, 10.0, 10.0],
        [20.0, 20.0, 40.0, 40.0],
        [5.0, 5.0, 15.0, 15.0],
    ]

    model.predict(np.zeros((10, 10, 3)), boxes)

    assert len(model.predictor.predict_calls) == 3


def test_segment_predict_passes_each_box_to_predictor(patch_sam2):
    """El box que llega al predictor es el mismo que recibió predict."""
    model = SegmentModel(
        config_file="sam2_hiera_b+.yaml", checkpoint_path="sam2.pt", device="cpu"
    )
    model.predictor.masks_to_return = [
        np.ones((5, 5), dtype=bool),
        np.zeros((5, 5), dtype=bool),
    ]
    boxes = [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 40.0]]

    model.predict(np.zeros((10, 10, 3)), boxes)

    assert model.predictor.predict_calls[0]["box"] == boxes[0]
    assert model.predictor.predict_calls[1]["box"] == boxes[1]


# --- predict: pasaje de multimask_output -----------------------------------


def test_segment_predict_passes_multimask_output_false_to_predictor(patch_sam2):
    """Decisión de diseño: pedimos 1 máscara por box en vez de 3 candidatos.
    Evita la decisión de argmax(scores) downstream y reduce costo de compute.
    """
    model = SegmentModel(
        config_file="sam2_hiera_b+.yaml", checkpoint_path="sam2.pt", device="cpu"
    )
    model.predictor.masks_to_return = [np.ones((5, 5), dtype=bool)]

    model.predict(np.zeros((10, 10, 3)), [[0.0, 0.0, 10.0, 10.0]])

    assert model.predictor.predict_calls[0]["multimask_output"] is False


# --- predict: contrato de retorno (máscaras 2D en orden) -------------------


def test_segment_predict_returns_masks_in_box_order(patch_sam2):
    model = SegmentModel(
        config_file="sam2_hiera_b+.yaml", checkpoint_path="sam2.pt", device="cpu"
    )
    mask_a = np.ones((5, 5), dtype=bool)
    mask_b = np.zeros((5, 5), dtype=bool)
    model.predictor.masks_to_return = [mask_a, mask_b]
    boxes = [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 40.0]]

    result = model.predict(np.zeros((10, 10, 3)), boxes)

    assert len(result) == 2
    assert np.array_equal(result[0], mask_a)
    assert np.array_equal(result[1], mask_b)


def test_segment_predict_returns_list_of_2d_ndarrays(patch_sam2):
    """Cada elemento del resultado debe ser un ndarray 2D (H, W) bool, no 3D."""
    model = SegmentModel(
        config_file="sam2_hiera_b+.yaml", checkpoint_path="sam2.pt", device="cpu"
    )
    model.predictor.masks_to_return = [
        np.ones((5, 5), dtype=bool),
        np.zeros((5, 5), dtype=bool),
    ]

    result = model.predict(np.zeros((10, 10, 3)), [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 40.0]])

    assert all(isinstance(m, np.ndarray) for m in result)
    assert all(m.ndim == 2 for m in result)
    assert all(m.dtype == bool for m in result)


# --- edge case: boxes vacíos ahorra el set_image (paso caro) ----------------


def test_segment_predict_empty_boxes_returns_empty_list_without_set_image(patch_sam2):
    """Si no hay boxes, no tiene sentido gastar el set_image (paso caro).
    Early-return [] antes de tocar el predictor. Defensivo + performante.
    """
    model = SegmentModel(
        config_file="sam2_hiera_b+.yaml", checkpoint_path="sam2.pt", device="cpu"
    )

    result = model.predict(np.zeros((10, 10, 3)), [])

    assert result == []
    assert model.predictor.set_image_calls == []
    assert model.predictor.predict_calls == []


# ---------------------------------------------------------------------------
# MeasureModel
#
# Contrato confirmado: recibe el client crudo y lo patchea con `instructor`
# dentro del __init__. Se asume `from instructor import patch`, en la misma
# línea que el `from ultralytics import YOLO` ya usado en models.py — avisar
# si en cambio se prefiere `import instructor; instructor.patch(...)`.
#
# predict(description) -> dict todavía no se testea: el estilo de llamada al
# client patcheado (OpenAI-style vs Anthropic-style) no está decidido.
# ---------------------------------------------------------------------------


class FakePatchedClient:
    """Marca que `patch()` fue aplicado, sin ser el client crudo original."""

    def __init__(self, raw_client):
        self.raw_client = raw_client


@pytest.fixture
def patch_instructor(monkeypatch):
    calls = []

    def fake_patch(client):
        calls.append(client)
        return FakePatchedClient(client)

    monkeypatch.setattr(models_module, "patch", fake_patch)
    return calls


def test_measure_model_patches_raw_client_with_instructor(patch_instructor):
    raw_client = object()

    model = MeasureModel(raw_client)

    assert patch_instructor == [raw_client]
    assert isinstance(model.client, FakePatchedClient)
    assert model.client.raw_client is raw_client


# ---------------------------------------------------------------------------
# DescribeModel
#
# Contrato confirmado: una sola clase que rama internamente según `vlm_mode`
# recibido en el __init__. Prueba los candidatos de `vlm_name[vlm_mode]` en
# orden, llamando a un método propio `self._load_candidate(name)` hasta que
# uno no lance excepción. Si todos fallan, el __init__ NO lanza.
#
# Se asume la firma `DescribeModel(vlm_mode, vlm_name, device, apikey=None)`
# (mismos nombres que en Settings) y que el resultado cargado queda en
# `self.model` (None si ningún candidato cargó). Ajustar si difiere.
# ---------------------------------------------------------------------------


def test_describe_tries_candidates_in_order_until_one_succeeds(monkeypatch):
    calls = []
    loaded_model = object()

    def fake_load_candidate(self, name):
        calls.append(name)
        if name == "model-a":
            raise RuntimeError("no disponible")
        return loaded_model

    monkeypatch.setattr(DescribeModel, "_load_candidate", fake_load_candidate)
    vlm_name = {"local": ["model-a", "model-b"], "api": ["claude-3-5-sonnet"]}

    model = DescribeModel(vlm_mode="local", vlm_name=vlm_name, device="cpu", apikey=None)

    assert calls == ["model-a", "model-b"]
    assert model.model is loaded_model


def test_describe_stops_after_first_successful_candidate(monkeypatch):
    calls = []

    def fake_load_candidate(self, name):
        calls.append(name)
        return f"loaded:{name}"

    monkeypatch.setattr(DescribeModel, "_load_candidate", fake_load_candidate)
    vlm_name = {"local": ["model-a", "model-b", "model-c"], "api": []}

    DescribeModel(vlm_mode="local", vlm_name=vlm_name, device="cpu", apikey=None)

    assert calls == ["model-a"]


def test_describe_only_tries_candidates_for_current_mode(monkeypatch):
    calls = []

    def fake_load_candidate(self, name):
        calls.append(name)
        raise RuntimeError("no disponible")

    monkeypatch.setattr(DescribeModel, "_load_candidate", fake_load_candidate)
    vlm_name = {"local": ["local-model"], "api": ["api-model-1", "api-model-2"]}

    DescribeModel(vlm_mode="api", vlm_name=vlm_name, device="cpu", apikey="sk-test")

    assert calls == ["api-model-1", "api-model-2"]


def test_describe_construction_does_not_raise_when_all_candidates_fail(monkeypatch):
    def fake_load_candidate(self, name):
        raise RuntimeError("no disponible")

    monkeypatch.setattr(DescribeModel, "_load_candidate", fake_load_candidate)
    vlm_name = {"local": ["model-a", "model-b"], "api": []}

    model = DescribeModel(vlm_mode="local", vlm_name=vlm_name, device="cpu", apikey=None)

    assert model.model is None
