import numpy as np

from plate_vision_pipeline.nodes import DescribeNode, DetectNode, MeasureNode, SegmentNode


class FakeDetector:
    """Doble de prueba para el contrato Detector: predict(image) -> list[Detection]."""

    def __init__(self, detections=None, raises=None):
        self._detections = detections if detections is not None else []
        self._raises = raises

    def predict(self, image):
        if self._raises is not None:
            raise self._raises
        return self._detections


class FakeSegmenter:
    """Doble de prueba para el contrato Segmenter: predict(image, boxes) -> list[np.ndarray]."""

    def __init__(self, masks=None, raises=None):
        self._masks = masks if masks is not None else []
        self._raises = raises
        self.calls = 0

    def predict(self, image, boxes):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return self._masks


class FakeDescriber:
    """Doble de prueba para el contrato Describer: predict(image, detections) -> str."""

    def __init__(self, description="", raises=None):
        self._description = description
        self._raises = raises

    def predict(self, image, detections):
        if self._raises is not None:
            raise self._raises
        return self._description


class FakeMeasurer:
    """Doble de prueba para el contrato Measurer: predict(description) -> dict."""

    def __init__(self, structure=None, raises=None):
        self._structure = structure
        self._raises = raises

    def predict(self, description):
        if self._raises is not None:
            raise self._raises
        return self._structure


def make_state(detections=None, description="", measure_attempts=0):
    return {
        "image": np.zeros((10, 10, 3)),
        "detections": detections if detections is not None else [],
        "description": description,
        "measure_attempts": measure_attempts,
    }


def test_detect_happy_path():
    detections = [
        {"bbox": [0.0, 0.0, 10.0, 10.0], "cls": "pizza", "conf": 0.91},
        {"bbox": [20.0, 20.0, 40.0, 40.0], "cls": "salad", "conf": 0.85},
    ]
    node = DetectNode(model=FakeDetector(detections=detections))

    result = node(make_state())

    assert result["detections"] == detections
    assert isinstance(result["latency_ms"]["detect"], float)
    assert result["latency_ms"]["detect"] >= 0
    assert "errors" not in result


def test_detect_no_detections_is_not_an_error():
    node = DetectNode(model=FakeDetector(detections=[]))

    result = node(make_state())

    assert result["detections"] == []
    assert "errors" not in result


def test_detect_model_failure_is_captured_as_error():
    node = DetectNode(model=FakeDetector(raises=RuntimeError("modelo no cargado")))

    result = node(make_state())

    assert result["detections"] == []
    assert result["errors"] == ["detect: modelo no cargado"]
    assert isinstance(result["latency_ms"]["detect"], float)


def test_segment_happy_path():
    detections = [
        {"bbox": [0.0, 0.0, 10.0, 10.0], "cls": "pizza", "conf": 0.91},
        {"bbox": [20.0, 20.0, 40.0, 40.0], "cls": "salad", "conf": 0.85},
    ]
    mask_a = np.ones((5, 5), dtype=bool)
    mask_b = np.zeros((5, 5), dtype=bool)
    node = SegmentNode(model=FakeSegmenter(masks=[mask_a, mask_b]))

    result = node(make_state(detections=detections))

    assert len(result["segmentations"]) == 2
    assert result["segmentations"][0]["bbox"] == detections[0]["bbox"]
    assert result["segmentations"][0]["cls"] == detections[0]["cls"]
    assert result["segmentations"][0]["conf"] == detections[0]["conf"]
    assert np.array_equal(result["segmentations"][0]["mask"], mask_a)
    assert np.array_equal(result["segmentations"][1]["mask"], mask_b)
    assert isinstance(result["latency_ms"]["segment"], float)
    assert "errors" not in result


def test_segment_no_detections_skips_model_call():
    fake = FakeSegmenter(masks=[])
    node = SegmentNode(model=fake)

    result = node(make_state(detections=[]))

    assert result["segmentations"] == []
    assert fake.calls == 0
    assert "errors" not in result


def test_segment_model_failure_is_captured_as_error():
    detections = [{"bbox": [0.0, 0.0, 10.0, 10.0], "cls": "pizza", "conf": 0.91}]
    node = SegmentNode(model=FakeSegmenter(raises=RuntimeError("sam2 oom")))

    result = node(make_state(detections=detections))

    assert result["segmentations"] == []
    assert result["errors"] == ["segment: sam2 oom"]
    assert isinstance(result["latency_ms"]["segment"], float)


def test_describe_happy_path():
    detections = [{"bbox": [0.0, 0.0, 10.0, 10.0], "cls": "pizza", "conf": 0.91}]
    node = DescribeNode(model=FakeDescriber(description="Hawaiian pizza con ensalada."))

    result = node(make_state(detections=detections))

    assert result["description"] == "Hawaiian pizza con ensalada."
    assert isinstance(result["latency_ms"]["describe"], float)
    assert "errors" not in result


def test_describe_model_failure_is_captured_as_error():
    node = DescribeNode(model=FakeDescriber(raises=RuntimeError("vlm timeout")))

    result = node(make_state())

    assert result["description"] == ""
    assert result["errors"] == ["describe: vlm timeout"]
    assert isinstance(result["latency_ms"]["describe"], float)


def test_measure_happy_path():
    structure = {
        "food_items": [],
        "meal_type_guess": "lunch",
        "dietary_flags": [],
        "macros_total": {"protein_g": 10.0, "carbs_g": 30.0, "fat_g": 7.0},
        "balanced_score": 0.8,
    }
    node = MeasureNode(model=FakeMeasurer(structure=structure))

    result = node(make_state(description="Hawaiian pizza con ensalada.", measure_attempts=0))

    assert result["structure"] == structure
    assert result["measure_attempts"] == 1
    assert isinstance(result["latency_ms"]["measure"], float)
    assert "errors" not in result


def test_measure_increments_attempts_from_existing_state():
    node = MeasureNode(model=FakeMeasurer(structure={"meal_type_guess": "lunch"}))

    result = node(make_state(description="texto", measure_attempts=1))

    assert result["measure_attempts"] == 2


def test_measure_failure_is_captured_as_error_and_still_increments_attempts():
    node = MeasureNode(model=FakeMeasurer(raises=RuntimeError("json invalido")))

    result = node(make_state(description="texto", measure_attempts=1))

    assert "structure" not in result
    assert result["errors"] == ["measure: json invalido"]
    assert result["measure_attempts"] == 2
    assert isinstance(result["latency_ms"]["measure"], float)
