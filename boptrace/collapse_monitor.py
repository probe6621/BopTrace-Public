from __future__ import annotations

import numpy as np


class CollapseMonitor:
    """Detects sharp probability concentration and entropy collapse."""

    def __init__(self, epsilon_bound: float = 1e-5):
        self.epsilon = float(epsilon_bound)

    def detect_probability_collapse(self, probability_distribution: np.ndarray) -> bool:
        distribution = np.asarray(probability_distribution, dtype=np.float64)
        if distribution.ndim == 0:
            raise ValueError("probability_distribution must be an array")
        return bool(np.min(distribution) < self.epsilon)

    def measure_entropy(self, probability_distribution: np.ndarray) -> float:
        distribution = np.asarray(probability_distribution, dtype=np.float64)
        clipped = np.clip(distribution, self.epsilon, 1.0)
        clipped = clipped / np.sum(clipped)
        entropy = -np.sum(clipped * np.log(clipped))
        return float(entropy)

