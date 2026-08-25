import numpy as np
from langgraph.graph import END

from plate_vision_pipeline.graph import build_graph, route_after_detect, RouteAfterMeasure
from plate_vision_pipeline.state import create_initial_pipelinestate


# ---------------------------------------------------------------------------
# Funciones de ruteo puras — sin LangGraph, sin nodos, solo la decisión.
# ---------------------------------------------------------------------------

def test_route_after_detect_no_detections_goes_to_end():
    assert route_after_detect({"detections": []}) == [END]


def test_route_after_detect_with_detections_fans_out_to_segment_and_describe():
    state = {"detections": [{"bbox": [0.0, 0.0, 1.0, 1.0], "cls": "pizza", "conf": 0.9}]}
    assert route_after_detect(state) == ["segment", "describe"]


def test_route_after_measure_success_goes_to_end():
    state = {"structure": {"meal_type_guess": "lunch"}, "measure_attempts": 1}
    assert RouteAfterMeasure()(state) == END


def test_route_after_measure_failure_below_max_retries_to_describe():
    state = {"measure_attempts": 1}
    assert RouteAfterMeasure()(state) == "describe"


def test_route_after_measure_failure_at_max_retries_goes_to_end():
    state = {"measure_attempts": 3}
    assert RouteAfterMeasure()(state) == END


def test_route_after_measure_respects_custom_max_attempts():
    route = RouteAfterMeasure(max_attempts=1)
    assert route({"measure_attempts": 1}) == END


def test_route_after_measure_custom_max_attempts_still_retries_below_threshold():
    route = RouteAfterMeasure(max_attempts=5)
    assert route({"measure_attempts": 3}) == "describe"


# ---------------------------------------------------------------------------
# Grafo completo con nodos fake — confirma el cableado, no solo la decisión.
# ---------------------------------------------------------------------------

def make_initial_state():
    return create_initial_pipelinestate(np.zeros((10, 10, 3)))


def ok_detect(state):
    detections = [{"bbox": [0.0, 0.0, 1.0, 1.0], "cls": "pizza", "conf": 0.9}]
    return {"detections": detections, "latency_ms": {"detect": 1.0}}


def empty_detect(state):
    return {"detections": [], "latency_ms": {"detect": 1.0}}


def ok_segment(state):
    return {"segmentations": [], "latency_ms": {"segment": 1.0}}


class CountingDescribe:
    def __init__(self):
        self.calls = 0

    def __call__(self, state):
        self.calls += 1
        return {"description": f"intento {self.calls}", "latency_ms": {"describe": 1.0}}


class FlakyMeasure:
    """Falla las primeras `fail_times` veces, después siempre tiene éxito."""

    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.calls = 0

    def __call__(self, state):
        self.calls += 1
        attempts = state.get("measure_attempts", 0) + 1
        if self.calls <= self.fail_times:
            return {
                "errors": [f"measure: intento {self.calls} fallido"],
                "measure_attempts": attempts,
                "latency_ms": {"measure": 1.0},
            }
        return {
            "structure": {"meal_type_guess": "lunch"},
            "measure_attempts": attempts,
            "latency_ms": {"measure": 1.0},
        }


class CountingNode:
    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, state):
        self.calls += 1
        return self.fn(state)


def test_graph_ends_early_when_no_detections():
    segment = CountingNode(ok_segment)
    describe = CountingDescribe()
    measure = FlakyMeasure(fail_times=0)

    graph = build_graph(empty_detect, segment, describe, measure)
    final_state = graph.invoke(make_initial_state())

    assert final_state["detections"] == []
    assert segment.calls == 0
    assert describe.calls == 0
    assert measure.calls == 0


def test_graph_happy_path_runs_all_nodes_once():
    segment = CountingNode(ok_segment)
    describe = CountingDescribe()
    measure = FlakyMeasure(fail_times=0)

    graph = build_graph(ok_detect, segment, describe, measure)
    final_state = graph.invoke(make_initial_state())

    assert final_state["structure"] == {"meal_type_guess": "lunch"}
    assert final_state["measure_attempts"] == 1
    assert final_state["segmentations"] == []
    assert segment.calls == 1
    assert describe.calls == 1
    assert measure.calls == 1


def test_graph_retries_measure_through_describe_until_success():
    segment = CountingNode(ok_segment)
    describe = CountingDescribe()
    measure = FlakyMeasure(fail_times=2)

    graph = build_graph(ok_detect, segment, describe, measure)
    final_state = graph.invoke(make_initial_state())

    assert final_state["structure"] == {"meal_type_guess": "lunch"}
    assert final_state["measure_attempts"] == 3
    assert describe.calls == 3
    assert len(final_state["errors"]) == 2
    # segment es una rama hoja desacoplada del loop de retry: corre una sola
    # vez con la primera detección, no en cada vuelta describe -> measure.
    assert segment.calls == 1


def test_graph_gives_up_after_max_measure_attempts():
    segment = CountingNode(ok_segment)
    describe = CountingDescribe()
    measure = FlakyMeasure(fail_times=999)

    graph = build_graph(ok_detect, segment, describe, measure)
    final_state = graph.invoke(make_initial_state())

    assert "structure" not in final_state
    assert final_state["measure_attempts"] == 3
    assert len(final_state["errors"]) == 3
    assert segment.calls == 1


def test_graph_respects_custom_max_measure_attempts():
    segment = CountingNode(ok_segment)
    describe = CountingDescribe()
    measure = FlakyMeasure(fail_times=999)

    graph = build_graph(
        ok_detect, segment, describe, measure,
        route_after_measure=RouteAfterMeasure(max_attempts=1),
    )
    final_state = graph.invoke(make_initial_state())

    assert "structure" not in final_state
    assert final_state["measure_attempts"] == 1
    assert describe.calls == 1
    assert measure.calls == 1
