"""Unit tests for Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_eval.models import (
    ActualToolCall,
    BenchmarkReport,
    ExpectedToolCall,
    Scenario,
    ScenarioResult,
    ToolDefinition,
    ToolParameter,
)


# ---- ToolParameter --------------------------------------------------------

class TestToolParameter:
    def test_defaults(self) -> None:
        tp = ToolParameter()
        assert tp.type == "object"
        assert tp.properties == {}
        assert tp.required == []

    def test_custom(self) -> None:
        tp = ToolParameter(
            type="object",
            properties={"city": {"type": "string"}},
            required=["city"],
        )
        assert tp.required == ["city"]
        assert "city" in tp.properties


# ---- ToolDefinition -------------------------------------------------------

class TestToolDefinition:
    def test_minimal(self) -> None:
        td = ToolDefinition(name="get_weather")
        assert td.name == "get_weather"
        assert td.description == ""

    def test_with_parameters(self) -> None:
        td = ToolDefinition(
            name="search",
            description="Search the web",
            parameters=ToolParameter(
                properties={"query": {"type": "string"}},
                required=["query"],
            ),
        )
        assert td.parameters.required == ["query"]

    def test_missing_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            ToolDefinition()  # type: ignore[call-arg]


# ---- ExpectedToolCall -----------------------------------------------------

class TestExpectedToolCall:
    def test_basic(self) -> None:
        etc = ExpectedToolCall(tool="get_weather", args={"city": "Tokyo"})
        assert etc.tool == "get_weather"
        assert etc.args["city"] == "Tokyo"

    def test_default_args(self) -> None:
        etc = ExpectedToolCall(tool="noop")
        assert etc.args == {}

    def test_missing_tool_raises(self) -> None:
        with pytest.raises(ValidationError):
            ExpectedToolCall()  # type: ignore[call-arg]


# ---- ActualToolCall -------------------------------------------------------

class TestActualToolCall:
    def test_basic(self) -> None:
        atc = ActualToolCall(tool="get_weather", args={"city": "London"})
        assert atc.tool == "get_weather"

    def test_default_args(self) -> None:
        atc = ActualToolCall(tool="noop")
        assert atc.args == {}


# ---- Scenario -------------------------------------------------------------

class TestScenario:
    def test_minimal(self) -> None:
        s = Scenario(
            name="test",
            user_message="Hello",
            expected=[ExpectedToolCall(tool="greet")],
        )
        assert s.name == "test"
        assert s.system_prompt == "You are a helpful assistant."
        assert len(s.expected) == 1

    def test_full(self) -> None:
        s = Scenario(
            name="Weather lookup",
            description="Check weather tool call",
            system_prompt="You have tools.",
            user_message="Weather in Tokyo?",
            available_tools=[
                ToolDefinition(
                    name="get_weather",
                    description="Get weather",
                    parameters=ToolParameter(
                        properties={"city": {"type": "string"}},
                        required=["city"],
                    ),
                )
            ],
            expected=[ExpectedToolCall(tool="get_weather", args={"city": "Tokyo"})],
        )
        assert len(s.available_tools) == 1
        assert s.expected[0].args["city"] == "Tokyo"

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            Scenario()  # type: ignore[call-arg]


# ---- ScenarioResult -------------------------------------------------------

class TestScenarioResult:
    def test_defaults(self) -> None:
        sr = ScenarioResult(scenario_name="test")
        assert sr.passed is False
        assert sr.overall_score == 0.0
        assert sr.error is None

    def test_score_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ScenarioResult(scenario_name="test", overall_score=1.5)

    def test_negative_latency(self) -> None:
        with pytest.raises(ValidationError):
            ScenarioResult(scenario_name="test", latency_ms=-1.0)


# ---- BenchmarkReport ------------------------------------------------------

class TestBenchmarkReport:
    def test_compute_aggregates(self) -> None:
        r1 = ScenarioResult(
            scenario_name="s1", passed=True, overall_score=1.0, latency_ms=100.0
        )
        r2 = ScenarioResult(
            scenario_name="s2", passed=False, overall_score=0.4, latency_ms=200.0
        )
        report = BenchmarkReport(model="test-model", results=[r1, r2])
        report.compute_aggregates()

        assert report.total_scenarios == 2
        assert report.passed == 1
        assert report.failed == 1
        assert report.avg_score == pytest.approx(0.7)
        assert report.avg_latency_ms == pytest.approx(150.0)

    def test_empty_report(self) -> None:
        report = BenchmarkReport(model="test-model")
        report.compute_aggregates()
        assert report.total_scenarios == 0

    def test_serialization(self) -> None:
        report = BenchmarkReport(model="test-model", total_scenarios=1)
        data = report.model_dump()
        assert data["model"] == "test-model"
        restored = BenchmarkReport(**data)
        assert restored.model == "test-model"
