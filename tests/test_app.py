from app import run_simulation


def test_run_simulation_returns_summary_and_report():
    summary, report, points = run_simulation(10, 0.05)

    assert "Steps Processed" in summary
    assert "BopTrace Enterprise Audit Report" in report
    assert len(points) == 10

