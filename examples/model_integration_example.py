from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from boptrace import BopTraceDiagnostics, ModelDiagnosticsHook, TelemetryExporter


def mock_generation_stream(steps: int) -> list[tuple[int, np.ndarray]]:
    stream: list[tuple[int, np.ndarray]] = []
    for token_id in range(steps):
        phase = token_id * 0.2
        hidden_state = np.array(
            [
                np.sin(phase),
                np.cos(phase),
                np.sin(phase * 0.5) + 0.1 * token_id,
            ],
            dtype=np.float64,
        )
        stream.append((token_id, hidden_state))
    return stream


def mock_probability_stream(steps: int) -> list[np.ndarray]:
    stream: list[np.ndarray] = []
    for token_id in range(steps):
        if token_id == steps - 1:
            stream.append(np.array([0.9999, 0.00005, 0.00005], dtype=np.float64))
        else:
            stream.append(np.array([0.6, 0.25, 0.15], dtype=np.float64))
    return stream


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a BopTrace model diagnostics example.")
    parser.add_argument("--steps", type=int, default=5, help="Number of mock generation steps to simulate.")
    parser.add_argument("--model-name", type=str, default="mock-local-model", help="Model name for telemetry.")
    parser.add_argument("--output-path", type=str, default="", help="Optional JSONL telemetry output path.")
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Stream telemetry events to stdout while running.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    exporter = TelemetryExporter(
        output_path=args.output_path or None,
        stream_to_stdout=args.stdout or not args.output_path,
    )
    diagnostics = BopTraceDiagnostics(exporter=exporter)
    hook = ModelDiagnosticsHook(diagnostics, exporter=exporter, model_name=args.model_name)

    token_stream = mock_generation_stream(args.steps)
    probability_stream = mock_probability_stream(args.steps)

    print(f"Running BopTrace example for {args.model_name} with {args.steps} steps...")
    for metrics in hook.wrap_generation_loop(iter(token_stream), iter(probability_stream)):
        print(metrics)

    exporter.close()

    if args.output_path:
        print(f"Telemetry written to {Path(args.output_path).resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

