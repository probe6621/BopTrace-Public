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
    hook = ModelDiagnosticsHook(diagnostics=diagnostics, exporter=exporter, model_name="boptrace-demo")

    prev_state = np.random.randn(32)
    gradients: list[float] = []
    collapses = 0

    for step in range(int(num_steps)):
        curr_state = prev_state + np.random.normal(0.0, volatility, size=32)
        prob_dist = np.random.dirichlet(np.ones(8) * max(0.1, 1.0 - (step / max(num_steps, 1))))
        metrics = hook.process_step(token_id=1000 + step, hidden_state=curr_state, probability_dist=prob_dist)
        gradients.append(float(metrics.get("alignment_gradient", 0.0)))
        collapses += int(bool(metrics.get("probability_collapse_risk", False)))
        prev_state = curr_state

    exporter.close()
    analyzer = TelemetryAnalyzer(str(log_path))
    report = analyzer.generate_audit_report()

    summary = (
        f"### Simulation Complete\n"
        f"- **Steps Processed:** {num_steps}\n"
        f"- **Collapse Events Detected:** {collapses}\n"
        f"- **Mean Gradient:** {np.mean(gradients) if gradients else 0.0:.5f}\n"
        f"- **Step Limit:** 1000\n"
        f"- **Telemetry Mode:** JSONL + stdout"
    )

    return summary, report, [{"Step": idx, "Gradient": value} for idx, value in enumerate(gradients)]


with gr.Blocks() as demo:
    gr.Markdown("# ⚡ BopTrace Community Demo")
    gr.Markdown("Capped public demo for alignment gradients, friction, and collapse auditing.")

    with gr.Row():
        with gr.Column():
            steps_slider = gr.Slider(minimum=10, maximum=1000, value=100, step=10, label="Simulation Steps")
            volatility_slider = gr.Slider(minimum=0.01, maximum=1.0, value=0.1, step=0.01, label="Latent State Volatility")
            run_btn = gr.Button("Run Diagnostics Stream", variant="primary")
        with gr.Column():
            output_summary = gr.Markdown()
            output_plot = gr.LinePlot(x="Step", y="Gradient", title="Alignment Gradient Velocity Over Time")
            output_report = gr.Markdown(label="Audit Report")

    run_btn.click(run_simulation, inputs=[steps_slider, volatility_slider], outputs=[output_summary, output_report, output_plot])


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
