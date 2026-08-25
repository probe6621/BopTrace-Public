import json
import tempfile

from boptrace import TelemetryExporter


def test_telemetry_jsonl_and_stdout(capsys):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
        temp_path = tmp.name

    exporter = TelemetryExporter(output_path=temp_path, stream_to_stdout=True)

    payload = exporter.emit("gradient_check", {"gradient": 0.042}, {"model": "test-model"})
    exporter.close()

    captured = capsys.readouterr()
    assert "gradient_check" in captured.out
    assert payload["metadata"]["model"] == "test-model"

    with open(temp_path, "r", encoding="utf-8") as handle:
        parsed = json.loads(handle.readline())
    assert parsed["metrics"]["gradient"] == 0.042
