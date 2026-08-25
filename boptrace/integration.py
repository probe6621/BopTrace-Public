from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from .diagnostics import BopTraceDiagnostics
from .telemetry import TelemetryExporter


class ModelDiagnosticsHook:
    """Streams model steps into BopTrace diagnostics and step-level telemetry."""

    def __init__(
        self,
        diagnostics: BopTraceDiagnostics,
        exporter: TelemetryExporter | None = None,
        model_name: str = "unknown-model",
    ) -> None:
        self.diagnostics = diagnostics
        self.exporter = exporter
        self.model_name = model_name
        self.prev_state: np.ndarray | None = None

    def process_step(
        self,
        token_id: int,
        hidden_state: np.ndarray,
        probability_dist: np.ndarray | None = None,
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {}

        self.diagnostics.register_step()

        hidden = np.asarray(hidden_state, dtype=np.float64)
        if self.prev_state is not None:
            gradient = self.diagnostics.measure_alignment_gradient(self.prev_state, hidden)
            friction = self.diagnostics.measure_internal_friction(self.prev_state, hidden)
            metrics["alignment_gradient"] = gradient
            metrics["internal_friction"] = friction

        if probability_dist is not None:
            collapse_risk = self.diagnostics.detect_probability_collapse(probability_dist)
            metrics["probability_collapse_risk"] = collapse_risk

        metrics["token_id"] = int(token_id)
        metrics["hidden_state_shape"] = list(hidden.shape)
        self.prev_state = hidden

        if self.exporter is not None:
            event_type = "collapse_alert" if metrics.get("probability_collapse_risk") else "step_audit"
            self.exporter.emit(event_type, metrics, {"model": self.model_name})

        return metrics

    def wrap_generation_loop(
        self,
        token_stream: Iterator[tuple[int, np.ndarray]],
        probability_stream: Iterator[np.ndarray] | None = None,
    ) -> Iterator[dict[str, Any]]:
        prob_iter = probability_stream if probability_stream is not None else iter(())
        for token_id, hidden_state in token_stream:
            prob_dist = next(prob_iter, None) if probability_stream is not None else None
            yield self.process_step(token_id, hidden_state, prob_dist)
