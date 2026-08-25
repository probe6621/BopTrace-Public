from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class TelemetryExporter:
    """Structured telemetry exporter with JSONL file and stdout output only."""

    output_path: Optional[str] = None
    stream_to_stdout: bool = False
    records: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._file_handle = open(self.output_path, "a", encoding="utf-8") if self.output_path else None

    def emit(
        self,
        event_type: str,
        metrics: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "metrics": metrics,
            "metadata": metadata or {},
        }
        serialized = json.dumps(payload, sort_keys=True)
        self.records.append(payload)

        if self._file_handle is not None:
            self._file_handle.write(serialized + "\n")
            self._file_handle.flush()

        if self.stream_to_stdout:
            sys.stdout.write(serialized + "\n")
            sys.stdout.flush()

        return payload

    def record(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.emit(event_type, payload)

    def export_json(self) -> str:
        return json.dumps(self.records, indent=2, sort_keys=True)

    def close(self) -> None:
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        self.close()
