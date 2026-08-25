from __future__ import annotations

import numpy as np

from .collapse_monitor import CollapseMonitor
from .friction_core import FrictionAnalyzer
from .telemetry import TelemetryExporter


class BopTraceDiagnostics:
    """Main diagnostics interface for alignment gradient and collapse tracking."""

    def __init__(
        self,
        epsilon_bound: float = 1e-5,
        exporter: TelemetryExporter | None = None,
        step_limit: int = 1000,
    ):
        self.epsilon = float(epsilon_bound)
        self.step_limit = int(step_limit)
        self.step_count = 0
        self.gradient_history: list[float] = []
        self.collapse_monitor = CollapseMonitor(epsilon_bound=self.epsilon)
        self.friction_analyzer = FrictionAnalyzer()
        self.exporter = exporter or TelemetryExporter()

    def register_step(self) -> None:
        if self.step_count >= self.step_limit:
            raise PermissionError(
                "BopTrace Community Edition step limit reached (1,000 steps). "
                "Upgrade to Professional for unlimited streaming."
            )
        self.step_count += 1

    def measure_alignment_gradient(self, latent_vector_prev: np.ndarray, latent_vector_curr: np.ndarray) -> float:
        prev = np.asarray(latent_vector_prev, dtype=np.float64)
        curr = np.asarray(latent_vector_curr, dtype=np.float64)
        if prev.shape != curr.shape:
            raise ValueError("latent vectors must share the same shape")
        delta = float(np.linalg.norm(curr - prev))
        stabilized_delta = max(delta, self.epsilon)
        self.gradient_history.append(stabilized_delta)
        self.exporter.emit(
            "alignment_gradient",
            {"delta": stabilized_delta, "shape": list(prev.shape)},
            {"epsilon": self.epsilon},
        )
        return stabilized_delta

    def detect_probability_collapse(self, probability_distribution: np.ndarray) -> bool:
        collapsed = self.collapse_monitor.detect_probability_collapse(probability_distribution)
        entropy = self.collapse_monitor.measure_entropy(probability_distribution)
        self.exporter.emit(
            "probability_collapse",
            {"collapsed": collapsed, "entropy": entropy},
            {"epsilon": self.epsilon},
        )
        return collapsed

    def measure_internal_friction(self, latent_vector_prev: np.ndarray, latent_vector_curr: np.ndarray) -> float:
        friction = self.friction_analyzer.measure_friction(latent_vector_prev, latent_vector_curr)
        self.exporter.emit("internal_friction", {"friction": friction}, {"epsilon": self.epsilon})
        return friction
