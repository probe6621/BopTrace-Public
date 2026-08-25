from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from boptrace import BopTraceDiagnostics, ModelDiagnosticsHook, TelemetryAnalyzer, TelemetryExporter


app = FastAPI(title="BopTrace AI Diagnostics")

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stop AI Hallucinations & Probability Collapse | BopTrace AI Diagnostics</title>
  <meta name="description" content="Stop AI hallucinations, prevent probability mass collapse, and ensure model alignment with BopTrace, the enterprise diagnostics SDK for production LLMs. Get real-time telemetry and compliance-ready audit logs. Try the live demo." />
  <meta name="keywords" content="stop llm hallucinations production, ai model drift detection real-time, prevent probability collapse in llms, compliance audit trail for ai models, debug production ai pipelines, llm hallucination detection, real-time ai diagnostics, model drift monitoring, ai compliance reports, production llm telemetry" />
  <meta name="robots" content="index, follow" />
  <meta name="theme-color" content="#0a192f" />
  <style>
    :root {
      color-scheme: dark;
      --bg: #0a192f;
      --panel: #111c33;
      --panel-2: #0f172a;
      --border: rgba(255,255,255,0.08);
      --text: #f5f5f5;
      --muted: #c7d2fe;
      --accent: #00d4ff;
      --accent-2: #3f51b5;
      --success: #4ade80;
      --warn: #fbbf24;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(63,81,181,0.18), transparent 26%),
        radial-gradient(circle at top right, rgba(0,212,255,0.12), transparent 28%),
        var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.6;
    }
    a { color: inherit; }
    .wrap { width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 32px 0 72px; }
    .nav {
      display: flex; align-items: center; justify-content: space-between; gap: 16px;
      padding: 16px 18px; border: 1px solid var(--border); border-radius: 18px;
      background: rgba(15, 23, 42, 0.72); backdrop-filter: blur(12px);
      position: sticky; top: 16px; z-index: 10;
    }
    .brand { display: flex; align-items: center; gap: 12px; font-weight: 800; letter-spacing: -0.03em; }
    .badge {
      width: 40px; height: 40px; border-radius: 12px; display: inline-grid; place-items: center;
      background: linear-gradient(135deg, var(--accent-2), var(--accent));
      color: white; font-size: 1.1rem;
    }
    .nav-links { display: flex; gap: 14px; flex-wrap: wrap; color: #dbeafe; font-size: 0.95rem; }
    .hero, .card, .pricing-card, .split {
      border: 1px solid var(--border); background: rgba(17, 28, 51, 0.88);
      backdrop-filter: blur(10px); border-radius: 24px; box-shadow: 0 20px 60px rgba(0,0,0,.22);
    }
    .hero { display: grid; grid-template-columns: 1.25fr 0.9fr; gap: 24px; padding: 34px; margin-top: 24px; }
    .eyebrow {
      display: inline-flex; align-items: center; gap: 8px; padding: 8px 12px; border-radius: 999px;
      background: rgba(0,212,255,0.12); color: #c7f9ff; font-size: .82rem; letter-spacing: .06em;
      text-transform: uppercase; font-weight: 700;
    }
    h1 {
      margin: 16px 0 12px; font-size: clamp(2.6rem, 5vw, 4.6rem); line-height: .96; letter-spacing: -0.05em;
    }
    .lead { color: #dbeafe; font-size: 1.08rem; max-width: 62ch; }
    .cta-row { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 22px; }
    .btn {
      display: inline-flex; align-items: center; justify-content: center; gap: 8px;
      padding: 14px 18px; border-radius: 14px; border: 1px solid transparent;
      text-decoration: none; font-weight: 800;
    }
    .btn-primary { background: linear-gradient(135deg, var(--accent-2), var(--accent)); color: white; }
    .btn-ghost { background: transparent; border-color: rgba(255,255,255,0.18); color: #e2e8f0; }
    .quote {
      margin-top: 18px; padding: 14px 16px; border-left: 3px solid var(--accent);
      background: rgba(15, 23, 42, 0.68); border-radius: 14px; color: #e0f2fe; font-style: italic;
    }
    .hero-side {
      padding: 24px; background:
        linear-gradient(180deg, rgba(0,212,255,0.08), rgba(63,81,181,0.08)),
        rgba(15,23,42,0.75);
    }
    .stats { display: grid; gap: 12px; }
    .stat {
      padding: 14px 16px; border-radius: 16px; background: rgba(255,255,255,.04); border: 1px solid var(--border);
      display: flex; justify-content: space-between; align-items: center; gap: 12px;
    }
    .stat strong { color: white; }
    section { margin-top: 24px; }
    .section-title { margin: 0 0 12px; font-size: clamp(1.5rem, 2.5vw, 2.3rem); letter-spacing: -0.03em; }
    .section-copy { margin: 0 0 18px; color: #cbd5e1; max-width: 78ch; }
    .cards { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }
    .card { padding: 22px; }
    .card h3 { margin: 0 0 8px; }
    .card p { margin: 0; color: #d1d5db; }
    .split { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; padding: 22px; }
    .wave {
      height: 260px; border-radius: 18px; border: 1px solid var(--border);
      background:
        linear-gradient(135deg, rgba(0,212,255,.12), rgba(63,81,181,.10)),
        radial-gradient(circle at center, rgba(0,212,255,.15), transparent 38%);
      position: relative; overflow: hidden;
    }
    .wave::before, .wave::after {
      content: ""; position: absolute; inset: 0; background: repeating-linear-gradient(
        90deg, transparent 0 16px, rgba(255,255,255,0.04) 16px 17px
      );
      mask: linear-gradient(180deg, transparent, black 25%, black 75%, transparent);
    }
    .wave::after { background: linear-gradient(90deg, transparent, rgba(0,212,255,0.45), transparent); animation: glide 4s linear infinite; }
    @keyframes glide { from { transform: translateX(-50%); } to { transform: translateX(50%); } }
    .logos {
      display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px;
    }
    .logo {
      height: 68px; border-radius: 16px; display: grid; place-items: center; color: #cbd5e1;
      border: 1px dashed rgba(255,255,255,.14); background: rgba(255,255,255,0.03);
      font-weight: 700;
    }
    .pricing {
      display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; padding: 22px;
    }
    .pricing-card { padding: 22px; }
    .price { font-size: 2rem; font-weight: 900; margin: 8px 0 10px; }
    .list { padding-left: 20px; color: #dbeafe; }
    .footer-cta {
      text-align: center; padding: 24px; border-radius: 24px; margin-top: 24px;
      background: linear-gradient(135deg, rgba(63,81,181,0.16), rgba(0,212,255,0.12));
      border: 1px solid var(--border);
    }
    .demo-wrap {
      margin-top: 24px; padding: 22px; border-radius: 24px; border: 1px solid var(--border);
      background: rgba(17, 28, 51, 0.88);
    }
    .grid { display:grid; grid-template-columns: 1fr 1fr; gap:20px; }
    .panel { background: #111827; border:1px solid #263244; border-radius:16px; padding:20px; }
    input { width:100%; }
    button { padding:10px 14px; border:0; border-radius:10px; background:#4f46e5; color:white; font-weight:700; cursor:pointer; }
    pre { white-space:pre-wrap; margin: 0; }
    @media (max-width: 960px) {
      .hero, .split, .pricing, .grid, .cards, .logos { grid-template-columns: 1fr; }
      .nav { position: static; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <nav class="nav">
      <div class="brand">
        <div class="badge">⚡</div>
        <div>
          <div>BopTrace AI Diagnostics</div>
          <div style="font-size:.84rem;color:#94a3b8;">Production LLM diagnostics</div>
        </div>
      </div>
      <div class="nav-links">
        <a href="#problems">Problems</a>
        <a href="#product">Product</a>
        <a href="#pricing">Pricing</a>
        <a href="#demo">Demo</a>
      </div>
    </nav>

    <section class="hero" id="landing">
      <div>
        <div class="eyebrow">Real-time AI model drift detection</div>
        <h1>Stop AI Hallucinations &amp; Probability Collapse. Instantly.</h1>
        <p class="lead">
          Get real-time visibility into internal structural friction and alignment drift across your production LLM streams.
          Transform unpredictable AI into reliable, auditable infrastructure.
        </p>
        <div class="cta-row">
          <a class="btn btn-primary" href="mailto:contact@epsilonframework.org?subject=BopTrace%20Professional%20Inquiry">Secure Your AI Pipeline</a>
          <a class="btn btn-ghost" href="#demo">View Live Demo</a>
          <a class="btn btn-ghost" href="https://epsilonframework.org" target="_blank" rel="noopener noreferrer">Core Framework</a>
        </div>
        <div class="quote">“BopTrace caught gradient collapse in staging that standard logging completely missed.”</div>
      </div>
      <div class="hero-side">
        <div class="stats">
          <div class="stat"><span>LLM hallucinations</span><strong>Detected early</strong></div>
          <div class="stat"><span>Probability collapse</span><strong>Monitored live</strong></div>
          <div class="stat"><span>Audit trails</span><strong>Markdown-ready</strong></div>
          <div class="stat"><span>Telemetry sinks</span><strong>JSONL / stdout</strong></div>
        </div>
      </div>
    </section>

    <section id="demo" class="demo-wrap">
      <h2 class="section-title">Live Community Demo</h2>
      <p class="section-copy">Run a capped generation stream and inspect alignment gradients, collapse risk, and a compliance-ready audit report. For the full landing page, visit the Core Framework link above.</p>
      <div class="cta-row" style="margin-top: 0;">
        <a class="btn btn-ghost" href="#landing">Back to Landing Page</a>
        <a class="btn btn-ghost" href="https://epsilonframework.org" target="_blank" rel="noopener noreferrer">Open Landing Page</a>
      </div>
      <div class="grid">
        <div class="panel">
          <label>Simulation Steps (Community Capped)</label>
          <input id="steps" type="range" min="10" max="1000" value="100" step="10" />
          <label>Latent State Volatility</label>
          <input id="volatility" type="range" min="0.01" max="1.0" value="0.1" step="0.01" />
          <button onclick="run()">Run Diagnostics Stream</button>
        </div>
        <div class="panel">
          <h3>Session Summary</h3>
          <pre id="summary"></pre>
          <h3>Collapse Risk Meter</h3>
          <pre id="meter"></pre>
        </div>
      </div>
      <div class="grid" style="margin-top:20px;">
        <div class="panel">
          <h3>Gradient Plot Data</h3>
          <pre id="plot"></pre>
        </div>
        <div class="panel">
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
    </section>
  </div>
</body>
</html>
"""


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
    return HTMLResponse(INDEX_HTML)


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
