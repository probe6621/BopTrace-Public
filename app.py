from __future__ import annotations

from pathlib import Path
import tempfile

import gradio as gr
import numpy as np

from boptrace import BopTraceDiagnostics, ModelDiagnosticsHook, TelemetryAnalyzer, TelemetryExporter


def run_simulation(num_steps: int, volatility: float):
    diagnostics = BopTraceDiagnostics()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
        log_path = Path(tmp.name)

    exporter = TelemetryExporter(output_path=str(log_path), stream_to_stdout=False)
    hook = ModelDiagnosticsHook(diagnostics=diagnostics, exporter=exporter, model_name="huggingface-demo-model")

    gradients: list[float] = []
    collapse_events = 0
    prev_state = np.random.randn(64)

    for step in range(int(num_steps)):
        curr_state = prev_state + np.random.normal(0.0, volatility, size=64)
        decay = max(0.1, 1.0 - (step / max(num_steps, 1)))
        prob_dist = np.random.dirichlet(np.ones(10) * decay)
        metrics = hook.process_step(token_id=1000 + step, hidden_state=curr_state, probability_dist=prob_dist)
        gradients.append(float(metrics.get("alignment_gradient", 0.0)))
        collapse_events += int(bool(metrics.get("probability_collapse_risk", False)))
        prev_state = curr_state

    exporter.close()
    analyzer = TelemetryAnalyzer(str(log_path))
    report = analyzer.generate_audit_report()

    if collapse_events == 0:
        meter = "🟢 Stable — no collapse events detected"
    elif collapse_events < max(1, num_steps // 10):
        meter = f"🟡 Watch — {collapse_events} collapse event(s) detected"
    else:
        meter = f"🔴 Elevated risk — {collapse_events} collapse event(s) detected"

    summary = (
        f"### Simulation Complete\n"
        f"- **Steps Processed:** {num_steps}\n"
        f"- **Collapse Events Detected:** {collapse_events}\n"
        f"- **Mean Gradient:** {np.mean(gradients) if gradients else 0.0:.5f}\n"
        f"- **Community Cap:** 1,000 steps per session"
    )

    plot_data = [{"Step": idx, "Gradient": value} for idx, value in enumerate(gradients)]
    return summary, meter, plot_data, report


with gr.Blocks() as demo:
    gr.Markdown("# ⚡ BopTrace Live Diagnostics Demo")
    gr.Markdown("Test real-time alignment gradients, structural friction, and probability collapse tracking using the capped BopTrace Community Engine.")

    with gr.Row():
        with gr.Column():
            steps_slider = gr.Slider(minimum=10, maximum=1000, value=100, step=10, label="Simulation Steps (Community Capped)")
            volatility_slider = gr.Slider(minimum=0.01, maximum=1.0, value=0.1, step=0.01, label="Latent State Volatility")
            run_btn = gr.Button("Run Diagnostics Stream", variant="primary")
        with gr.Column():
            output_summary = gr.Markdown(label="Session Summary")
            collapse_meter = gr.Markdown(label="Collapse Risk Meter")
            output_plot = gr.LinePlot(x="Step", y="Gradient", title="Alignment Gradient Velocity Over Time")
            output_report = gr.Markdown(label="Audit Report")

    run_btn.click(
        run_simulation,
        inputs=[steps_slider, volatility_slider],
        outputs=[output_summary, collapse_meter, output_plot, output_report],
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
