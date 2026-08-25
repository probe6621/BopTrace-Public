from app import index, run_simulation


def test_run_simulation_returns_summary_and_report():
    summary, points, report, collapse_events = run_simulation(10, 0.05)

    assert "Steps Processed" in summary
    assert "BopTrace Enterprise Audit Report" in report
    assert len(points) == 10
    assert collapse_events >= 0


def test_index_opens_on_demo_first():
    html = index().body.decode("utf-8")
    assert "Live Community Demo" in html
    assert "Landing + Pricing" in html
    assert "Buy Professional" in html
    assert "Buy Enterprise" in html
    assert "https://epsilonframework.org" in html
    assert "Stop AI Hallucinations &amp; Probability Collapse. Instantly." in html
