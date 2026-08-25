from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from boptrace import BopTraceDiagnostics, ModelDiagnosticsHook, TelemetryAnalyzer, TelemetryExporter


app = FastAPI(title="BopTrace AI Diagnostics")


def run_simulation(steps: int, volatility: float):
    diagnostics = BopTraceDiagnostics()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
        log_path = Path(tmp.name)

    exporter = TelemetryExporter(output_path=str(log_path), stream_to_stdout=False)
    hook = ModelDiagnosticsHook(diagnostics=diagnostics, exporter=exporter, model_name="hf-community-demo")

    gradients: list[float] = []
    collapse_events = 0
    prev_state = np.random.randn(32)

    for i in range(int(steps)):
        curr_state = prev_state + np.random.normal(0.0, float(volatility), size=32)
        prob_dist = np.array([0.5, 0.5], dtype=np.float64)
        metrics = hook.process_step(token_id=1000 + i, hidden_state=curr_state, probability_dist=prob_dist)
        gradients.append(float(metrics.get("alignment_gradient", 0.0)))
        collapse_events += int(bool(metrics.get("probability_collapse_risk", False)))
        prev_state = curr_state

    exporter.close()
    analyzer = TelemetryAnalyzer(str(log_path))
    report = analyzer.generate_audit_report()

    summary = (
        f"### Simulation Complete\n"
        f"- **Steps Processed:** {int(steps)}\n"
        f"- **Collapse Events Detected:** {collapse_events}\n"
        f"- **Mean Gradient:** {np.mean(gradients) if gradients else 0.0:.5f}\n"
        f"- **Community Cap:** 1,000 steps per session"
    )
    plot_data = [{"Step": idx, "Gradient": value} for idx, value in enumerate(gradients)]
    return summary, plot_data, report, collapse_events


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(
        """
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>BopTrace AI Diagnostics</title>
          <style>
            body { font-family: Arial, sans-serif; background:#0b1020; color:#e5e7eb; margin:0; padding:24px; }
            .grid { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
            .card { background:#111827; border:1px solid #263244; border-radius:16px; padding:20px; }
            input { width:100%; }
            button { padding:10px 14px; border:0; border-radius:10px; background:#4f46e5; color:white; font-weight:700; cursor:pointer; }
            pre { white-space:pre-wrap; }
          </style>
        </head>
        <body>
          <h1>⚡ BopTrace AI Diagnostics</h1>
          <p>Real-time AI diagnostics tracking token alignment gradients and preventing model probability collapse.</p>
          <div class="grid">
            <div class="card">
              <label>Simulation Steps (Community Capped)</label>
              <input id="steps" type="range" min="10" max="1000" value="100" step="10" />
              <label>Latent State Volatility</label>
              <input id="volatility" type="range" min="0.01" max="1.0" value="0.1" step="0.01" />
              <button onclick="run()">Run Diagnostics Stream</button>
            </div>
            <div class="card">
              <h3>Session Summary</h3>
              <pre id="summary"></pre>
              <h3>Collapse Risk Meter</h3>
              <pre id="meter"></pre>
            </div>
          </div>
          <div class="grid" style="margin-top:20px;">
            <div class="card">
              <h3>Gradient Plot Data</h3>
              <pre id="plot"></pre>
            </div>
            <div class="card">
              <h3>Audit Report</h3>
              <pre id="report"></pre>
            </div>
          </div>
          <script>
            async function run() {
              const steps = document.getElementById('steps').value;
              const volatility = document.getElementById('volatility').value;
              const response = await fetch('/simulate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({steps: Number(steps), volatility: Number(volatility)})
              });
              const data = await response.json();
              document.getElementById('summary').textContent = data.summary;
              document.getElementById('meter').textContent = data.meter;
              document.getElementById('plot').textContent = JSON.stringify(data.plot_data, null, 2);
              document.getElementById('report').textContent = data.report;
            }
            run();
          </script>
        </body>
        </html>
        """
    )


@app.post("/simulate")
def simulate(payload: dict) -> JSONResponse:
    summary, plot_data, report, collapse_events = run_simulation(
        payload.get("steps", 100), payload.get("volatility", 0.1)
    )
    meter = "🟢 Stable — no collapse events detected" if collapse_events == 0 else "🟡 Watch — collapse signals present"
    return JSONResponse({"summary": summary, "meter": meter, "plot_data": plot_data, "report": report})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
