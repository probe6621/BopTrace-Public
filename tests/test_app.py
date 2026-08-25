from app import run_simulation


def test_run_simulation_returns_summary_and_report():
    summary, meter, points, report = run_simulation(10, 0.05)

    assert "Steps Processed" in summary
    assert "Collapse Risk Meter" not in meter
    assert "BopTrace Enterprise Audit Report" in report
    assert len(points) == 10
