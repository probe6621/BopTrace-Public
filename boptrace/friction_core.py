from __future__ import annotations

import numpy as np


class FrictionAnalyzer:
    """Quantifies internal friction and resonance across latent states."""

    def __init__(self, epsilon_bound: float = 1e-5):
        self.epsilon = float(epsilon_bound)

    def measure_friction(self, latent_prev: np.ndarray, latent_curr: np.ndarray) -> float:
        prev = np.asarray(latent_prev, dtype=np.float64)
        curr = np.asarray(latent_curr, dtype=np.float64)
        if prev.shape != curr.shape:
            raise ValueError("latent states must share the same shape")
        return float(np.mean(np.abs(curr - prev)))

    def measure_resonance(self, probability_path: np.ndarray) -> float:
        path = np.asarray(probability_path, dtype=np.float64)
        if path.ndim == 0:
            raise ValueError("probability_path must have at least one dimension")
        centered = path - np.mean(path)
        return float(np.linalg.norm(centered))
