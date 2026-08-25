import numpy as np
import pytest

from boptrace import BopTraceDiagnostics, CollapseMonitor, FrictionAnalyzer, TelemetryExporter


def test_alignment_gradient_and_telemetry():
    exporter = TelemetryExporter()
    trace = BopTraceDiagnostics(exporter=exporter)
    prev_state = np.array([0.1, 0.2, 0.3])
    curr_state = np.array([0.15, 0.25, 0.35])

    gradient = trace.measure_alignment_gradient(prev_state, curr_state)

    assert gradient > 0
    assert trace.gradient_history[-1] == gradient
    assert exporter.records[0]["event_type"] == "alignment_gradient"


def test_probability_collapse_detection():
    trace = BopTraceDiagnostics(epsilon_bound=1e-4)
    assert trace.detect_probability_collapse(np.array([0.8, 0.1999, 0.0001])) is False
    assert trace.detect_probability_collapse(np.array([0.9999, 0.00009, 0.00001])) is True


def test_friction_analyzer_requires_matching_shapes():
    analyzer = FrictionAnalyzer()
    with pytest.raises(ValueError, match="same shape"):
        analyzer.measure_friction(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_collapse_monitor_entropy_measures_finite_value():
    monitor = CollapseMonitor()
    entropy = monitor.measure_entropy(np.array([0.6, 0.3, 0.1]))
    assert np.isfinite(entropy)
    assert entropy > 0

