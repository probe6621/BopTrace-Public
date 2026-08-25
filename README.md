# BopTrace Community Edition

BopTrace Community Edition is a capped public demo for alignment diagnostics and telemetry auditing.

## Included

- alignment gradient tracking
- probability collapse monitoring
- internal friction scoring
- model step integration hook
- structured telemetry export to JSONL or stdout
- telemetry analytics and audit report generation

## Demo limits

- maximum 1,000 generation steps per session
- telemetry sinks are limited to JSONL file output and stdout
- native C++ backend is not included in the public demo build

## Modules

- `boptrace.diagnostics`
- `boptrace.integration`
- `boptrace.telemetry`
- `boptrace.telemetry_viz`
- `boptrace.friction_core`

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Example

```bash
python examples/model_integration_example.py --steps 5 --stdout
```

## Analytics

```python
from boptrace import TelemetryAnalyzer

analyzer = TelemetryAnalyzer("telemetry.jsonl")
print(analyzer.generate_audit_report("audit-report.md"))
```

