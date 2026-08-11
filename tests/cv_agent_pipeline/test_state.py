from cv_agent_pipeline.state import merge_latency


def test_merge_latency_new_value_overrides_old():
    old = {"detect": 10.0}
    new = {"detect": 20.0}
    assert merge_latency(old, new) == {"detect": 20.0}


def test_merge_latency_keeps_keys_from_both_sides():
    old = {"detect": 10.0}
    new = {"segment": 5.0}
    assert merge_latency(old, new) == {"detect": 10.0, "segment": 5.0}


def test_merge_latency_does_not_mutate_original():
    old = {"detect": 10.0}
    new = {"segment": 5.0}
    merge_latency(old, new)
    assert old == {"detect": 10.0}
