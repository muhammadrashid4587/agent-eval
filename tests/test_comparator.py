"""Tests for the model comparison module."""

from __future__ import annotations

import pytest

from agent_eval.comparator import ComparisonResult, ModelComparison, compare_models
from agent_eval.models import BenchmarkReport, ExpectedToolCall, Scenario, ToolDefinition, ToolParameter


def _make_scenarios() -> list[Scenario]:
    """Create a small set of test scenarios."""
    return [
        Scenario(
            name="weather",
            user_message="What is the weather in Paris?",
            available_tools=[
                ToolDefinition(
                    name="get_weather",
                    description="Get weather",
                    parameters=ToolParameter(
                        properties={"city": {"type": "string"}},
                        required=["city"],
                    ),
                ),
            ],
            expected=[ExpectedToolCall(tool="get_weather", args={"city": "Paris"})],
        ),
        Scenario(
            name="calculator",
            user_message="What is 2 + 2?",
            available_tools=[
                ToolDefinition(
                    name="calculate",
                    description="Calculate",
                    parameters=ToolParameter(
                        properties={"expression": {"type": "string"}},
                        required=["expression"],
                    ),
                ),
            ],
            expected=[ExpectedToolCall(tool="calculate", args={"expression": "2 + 2"})],
        ),
    ]


class TestCompareModels:
    """Tests for the compare_models function."""

    def test_dry_run_comparison(self) -> None:
        """Dry-run comparison should succeed for all models."""
        scenarios = _make_scenarios()
        result = compare_models(
            scenarios,
            ["model-a", "model-b"],
            dry_run=True,
        )
        assert len(result.comparisons) == 2
        assert result.comparisons[0].model_name == "model-a"
        assert result.comparisons[1].model_name == "model-b"

    def test_summary_table_populated(self) -> None:
        scenarios = _make_scenarios()
        result = compare_models(scenarios, ["m1", "m2"], dry_run=True)
        assert "weather" in result.summary_table
        assert "calculator" in result.summary_table
        assert "m1" in result.summary_table["weather"]
        assert "m2" in result.summary_table["weather"]

    def test_best_model_selected(self) -> None:
        """With dry-run (perfect scores), both are tied; best_model should be set."""
        scenarios = _make_scenarios()
        result = compare_models(scenarios, ["m1", "m2"], dry_run=True)
        assert result.best_model in ("m1", "m2")

    def test_scenario_names_match(self) -> None:
        scenarios = _make_scenarios()
        result = compare_models(scenarios, ["x"], dry_run=True)
        assert result.scenario_names == ["weather", "calculator"]

    def test_all_scores_perfect_in_dry_run(self) -> None:
        """DryRunProvider echoes expected calls, so all scores should be 1.0."""
        scenarios = _make_scenarios()
        result = compare_models(scenarios, ["dry"], dry_run=True)
        for comp in result.comparisons:
            assert comp.report.avg_score == pytest.approx(1.0)
            for r in comp.report.results:
                assert r.overall_score == pytest.approx(1.0)


class TestComparisonResult:
    """Tests for the ComparisonResult dataclass."""

    def test_compute_summary_empty(self) -> None:
        result = ComparisonResult(scenario_names=[], comparisons=[])
        result.compute_summary()
        assert result.summary_table == {}
        assert result.best_model == ""

    def test_compute_summary_with_data(self) -> None:
        report = BenchmarkReport(model="test", avg_score=0.85)
        comp = ModelComparison(model_name="test", report=report)
        result = ComparisonResult(
            scenario_names=[],
            comparisons=[comp],
        )
        result.compute_summary()
        assert result.best_model == "test"
