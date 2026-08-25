from typing import Annotated, NotRequired, TypedDict
import numpy as np
import operator

def merge_latency(old: dict, new: dict) -> dict:
    return {**old, **new}

class Detection(TypedDict):
    bbox: list[float]
    cls: str
    conf: float

class Segmentation(Detection):
    mask: np.ndarray

class PipelineState(TypedDict):
    image: np.ndarray
    detections: list[Detection]
    segmentations: NotRequired[list[Segmentation]]
    description: NotRequired[str]
    structure: NotRequired[dict]

    errors: Annotated[list[str], operator.add]
    latency_ms: Annotated[dict[str,float], merge_latency]

    measure_attempts: int


def create_initial_pipelinestate(image:np.ndarray) -> PipelineState:
    return PipelineState(image=image,detections=[],errors=[],latency_ms={},measure_attempts=0)
