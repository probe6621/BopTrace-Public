import numpy as np
import pytest

from boptrace import BopTraceDiagnostics, ModelDiagnosticsHook, TelemetryExporter


def test_model_hook_process_step_emits_step_audit():
    exporter = TelemetryExporter()
    trace = BopTraceDiagnostics()
    hook = ModelDiagnosticsHook(trace, exporter=exporter, model_name="demo-model")

    first = hook.process_step(7, np.array([0.1, 0.2, 0.3]))
    second = hook.process_step(8, np.array([0.15, 0.25, 0.35]), np.array([0.7, 0.2, 0.1]))

    assert first["token_id"] == 7
    assert "alignment_gradient" not in first
    assert second["token_id"] == 8
    assert "alignment_gradient" in second
    assert second["probability_collapse_risk"] is False
    assert exporter.records[-1]["event_type"] == "step_audit"
    assert exporter.records[-1]["metadata"]["model"] == "demo-model"


def test_model_hook_wrap_generation_loop():
    trace = BopTraceDiagnostics()
    hook = ModelDiagnosticsHook(trace)

    token_stream = [
        (1, np.array([0.1, 0.2])),
        (2, np.array([0.2, 0.3])),
    ]
    probability_stream = [
        np.array([0.6, 0.4]),
        np.array([0.9, 0.1]),
    ]

    results = list(hook.wrap_generation_loop(iter(token_stream), iter(probability_stream)))

    assert len(results) == 2
    assert results[1]["alignment_gradient"] > 0
    assert results[1]["probability_collapse_risk"] is False


def test_model_hook_enforces_step_limit():
    trace = BopTraceDiagnostics(step_limit=1)
    hook = ModelDiagnosticsHook(trace)

    hook.process_step(1, np.array([0.1, 0.2]))

    with pytest.raises(PermissionError, match="step limit reached"):
        hook.process_step(2, np.array([0.2, 0.3]))
