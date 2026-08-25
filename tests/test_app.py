import re

from app import index, run_simulation


def test_run_simulation_returns_summary_and_report():
    summary, points, report, collapse_events = run_simulation(10, 0.05)

    assert "Steps Processed" in summary
    assert "BopTrace Enterprise Audit Report" in report
    assert len(points) == 10
    assert collapse_events >= 0


def test_index_opens_on_demo_first():
    first = index().body.decode("utf-8")
    second = index().body.decode("utf-8")
    first_visits = int(re.search(r"Page visits</span><strong>([\d,]+)</strong>", first).group(1).replace(",", ""))
    second_visits = int(re.search(r"Page visits</span><strong>([\d,]+)</strong>", second).group(1).replace(",", ""))
    assert second_visits == first_visits + 1
    assert "Live Community Demo" in second
    assert "Landing + Pricing" in second
    assert "Free Demo Dowload" in second
    assert "Buy Professional" in second
    assert "Buy Enterprise" in second
    assert second.count("$499.95") >= 2
    assert "Open Public Git" in second
    assert "https://epsilonframework.org" in second
    assert "https://github.com/probe6621/BopTrace-Public" in second
    assert "Stop AI Hallucinations &amp; Probability Collapse. Instantly." in second
