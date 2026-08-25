import json

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsError

from plate_vision_pipeline.config import Settings


# yolo_weights_path, sam2_checkpoint_path y vlm_name no tienen default: hay
# que darles un valor dummy en cada test que no los esté probando
# específicamente, o Settings() explota antes de llegar al assert que nos
# interesa.
REQUIRED_ENV = {
    "YOLO_WEIGHTS_PATH": "/models/yolo.pt",
    "SAM2_CHECKPOINT_PATH": "/models/sam2.pt",
    "VLM_NAME": json.dumps({
        "local": ["Qwen/Qwen2-VL-2B-Instruct"],
        "api": ["claude-3-5-sonnet-20241022"],
    }),
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Evita que env vars reales del shell contaminen los tests de default."""
    for var in [
        "DEVICE", "VLM_MODE", "MAX_MEASURE_ATTEMPTS", "APIKEY",
        "TRT_ENGINE_PATH", "YOLO_WEIGHTS_PATH", "SAM2_CHECKPOINT_PATH",
        "MIN_YOLO_CONF", "VLM_NAME",
    ]:
        monkeypatch.delenv(var, raising=False)
    for var, value in REQUIRED_ENV.items():
        monkeypatch.setenv(var, value)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def test_default_device_is_cpu():
    assert Settings().device == "cpu"


def test_default_vlm_mode_is_api():
    assert Settings().vlm_mode == "api"


def test_default_max_measure_attempts_is_three():
    assert Settings().max_measure_attempts == 3


def test_default_apikey_is_none():
    assert Settings().apikey is None


def test_default_trt_engine_path_is_none():
    assert Settings().trt_engine_path is None


def test_default_min_yolo_conf_is_0_3():
    assert Settings().min_yolo_conf == 0.3


# ---------------------------------------------------------------------------
# Lectura desde variables de entorno
# ---------------------------------------------------------------------------

def test_device_reads_from_env_var(monkeypatch):
    monkeypatch.setenv("DEVICE", "cuda")
    assert Settings().device == "cuda"


def test_vlm_mode_reads_from_env_var(monkeypatch):
    monkeypatch.setenv("VLM_MODE", "local")
    assert Settings().vlm_mode == "local"


def test_max_measure_attempts_reads_from_env_var(monkeypatch):
    monkeypatch.setenv("MAX_MEASURE_ATTEMPTS", "5")
    assert Settings().max_measure_attempts == 5


def test_trt_engine_path_reads_from_env_var(monkeypatch):
    monkeypatch.setenv("TRT_ENGINE_PATH", "/models/yolo.fp16.engine")
    assert Settings().trt_engine_path == "/models/yolo.fp16.engine"


def test_yolo_weights_path_reads_from_env_var(monkeypatch):
    monkeypatch.setenv("YOLO_WEIGHTS_PATH", "/other/yolo.pt")
    assert Settings().yolo_weights_path == "/other/yolo.pt"


def test_min_yolo_conf_reads_from_env_var(monkeypatch):
    monkeypatch.setenv("MIN_YOLO_CONF", "0.5")
    assert Settings().min_yolo_conf == 0.5


# ---------------------------------------------------------------------------
# Campos obligatorios: fallan rápido si faltan
# ---------------------------------------------------------------------------

def test_missing_yolo_weights_path_is_rejected(monkeypatch):
    monkeypatch.delenv("YOLO_WEIGHTS_PATH", raising=False)
    with pytest.raises(ValidationError):
        Settings()


def test_missing_sam2_checkpoint_path_is_rejected(monkeypatch):
    monkeypatch.delenv("SAM2_CHECKPOINT_PATH", raising=False)
    with pytest.raises(ValidationError):
        Settings()


# ---------------------------------------------------------------------------
# Valores inválidos
# ---------------------------------------------------------------------------

def test_invalid_device_rejected(monkeypatch):
    monkeypatch.setenv("DEVICE", "tpu")
    with pytest.raises(ValidationError):
        Settings()


def test_invalid_vlm_mode_rejected(monkeypatch):
    monkeypatch.setenv("VLM_MODE", "cloud")
    with pytest.raises(ValidationError):
        Settings()


# ---------------------------------------------------------------------------
# vlm_name: diccionario de candidatos {"local": [...], "api": [...]} usado
# para intentar cargar modelos en orden hasta que uno funcione. Las keys son
# los mismos literales que vlm_mode.
# ---------------------------------------------------------------------------

def test_vlm_name_reads_from_env_var(monkeypatch):
    monkeypatch.setenv("VLM_NAME", json.dumps({
        "local": ["Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct"],
        "api": ["claude-3-5-sonnet-20241022"],
    }))

    settings = Settings()

    assert settings.vlm_name["local"] == [
        "Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct",
    ]
    assert settings.vlm_name["api"] == ["claude-3-5-sonnet-20241022"]


def test_vlm_name_missing_is_rejected(monkeypatch):
    monkeypatch.delenv("VLM_NAME", raising=False)
    with pytest.raises(ValidationError):
        Settings()


def test_vlm_name_rejects_unknown_key(monkeypatch):
    monkeypatch.setenv("VLM_NAME", json.dumps({
        "cpu": ["Qwen/Qwen2-VL-2B-Instruct"],
        "api": ["claude-3-5-sonnet-20241022"],
    }))
    with pytest.raises(ValidationError):
        Settings()


def test_vlm_name_rejects_non_list_value(monkeypatch):
    monkeypatch.setenv("VLM_NAME", json.dumps({
        "local": "Qwen/Qwen2-VL-2B-Instruct",
        "api": ["claude-3-5-sonnet-20241022"],
    }))
    with pytest.raises(ValidationError):
        Settings()


def test_vlm_name_rejects_invalid_json(monkeypatch):
    monkeypatch.setenv("VLM_NAME", "not-json")
    with pytest.raises(SettingsError):
        Settings()
