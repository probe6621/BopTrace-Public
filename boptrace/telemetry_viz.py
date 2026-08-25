from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class TelemetryAnalyzer:
    """Parses JSONL telemetry logs and generates aggregate audit summaries."""

    def __init__(self, jsonl_path: str):
        self.jsonl_path = jsonl_path
        self.events: list[dict[str, Any]] = []
        self._load_logs()

    def _load_logs(self) -> None:
        path = Path(self.jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Telemetry log not found at: {self.jsonl_path}")

        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    def summarize(self) -> dict[str, Any]:
        total_steps = len(self.events)
        if total_steps == 0:
            return {
                "total_steps": 0,
                "collapse_count": 0,
                "collapse_percentage": 0.0,
                "mean_alignment_gradient": 0.0,
                "max_alignment_gradient": 0.0,
                "min_alignment_gradient": 0.0,
            }

        gradients: list[float] = []
        collapse_count = 0

        for event in self.events:
            metrics = event.get("metrics", {})
            if "alignment_gradient" in metrics:
                gradients.append(float(metrics["alignment_gradient"]))
            if metrics.get("probability_collapse_risk", False):
                collapse_count += 1

        mean_grad = float(np.mean(gradients)) if gradients else 0.0
        max_grad = float(np.max(gradients)) if gradients else 0.0
        min_grad = float(np.min(gradients)) if gradients else 0.0

        return {
            "total_steps": total_steps,
            "collapse_count": collapse_count,
            "collapse_percentage": (collapse_count / total_steps) * 100 if total_steps > 0 else 0.0,
            "mean_alignment_gradient": mean_grad,
            "max_alignment_gradient": max_grad,
            "min_alignment_gradient": min_grad,
        }

    def generate_audit_report(self, output_report_path: str | None = None) -> str:
        summary = self.summarize()
        report_lines = [
            "# BopTrace Enterprise Audit Report",
            f"**Log Source:** `{self.jsonl_path}`",
            "",
            "## Session Summary",
            f"- **Total Generation Steps Audited:** {summary['total_steps']}",
            f"- **Probability Collapse Events:** {summary['collapse_count']} ({summary['collapse_percentage']:.2f}% of steps)",
            f"- **Mean Alignment Gradient:** {summary['mean_alignment_gradient']:.6f}",
            f"- **Peak Alignment Gradient (Max):** {summary['max_alignment_gradient']:.6f}",
            f"- **Baseline Alignment Gradient (Min):** {summary['min_alignment_gradient']:.6f}",
        ]
        report_text = "\n".join(report_lines)

        if output_report_path:
            Path(output_report_path).write_text(report_text, encoding="utf-8")

        return report_text

