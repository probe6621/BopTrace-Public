import json
import os
import tempfile

from boptrace.telemetry_viz import TelemetryAnalyzer


def test_telemetry_analyzer_summary_and_report():
    sample_data = [
        {
            "timestamp": "2026-08-25T00:00:00Z",
            "event_type": "step_audit",
            "metrics": {"alignment_gradient": 0.01, "probability_collapse_risk": False, "token_id": 101},
        },
        {
            "timestamp": "2026-08-25T00:00:01Z",
            "event_type": "collapse_alert",
            "metrics": {"alignment_gradient": 0.55, "probability_collapse_risk": True, "token_id": 102},
        },
    ]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl", mode="w", encoding="utf-8") as tmp:
        for item in sample_data:
            tmp.write(json.dumps(item) + "\n")
        temp_path = tmp.name

    try:
        analyzer = TelemetryAnalyzer(temp_path)
        summary = analyzer.summarize()
        assert summary["total_steps"] == 2
        assert summary["collapse_count"] == 1
        assert round(summary["mean_alignment_gradient"], 2) == 0.28

        report = analyzer.generate_audit_report()
        assert "BopTrace Enterprise Audit Report" in report
        assert "- **Probability Collapse Events:** 1" in report

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
