"""Pydantic models for scenario definitions, tool calls, and evaluation results."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ToolParameter(BaseModel):
    """JSON Schema definition for a tool's parameters."""

    type: str = "object"
    properties: Dict[str, Any] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)


class ToolDefinition(BaseModel):
    """Definition of a tool available to the agent."""

    name: str = Field(..., description="Unique tool name")
    description: str = Field("", description="Human-readable description of the tool")
    parameters: ToolParameter = Field(default_factory=ToolParameter)

    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Tool name must not be blank")
        return v


class ExpectedToolCall(BaseModel):
    """An expected tool invocation with name and arguments."""

    tool: str = Field(..., description="Expected tool name")
    args: Dict[str, Any] = Field(default_factory=dict, description="Expected arguments")

    @field_validator("tool")
    @classmethod
    def _tool_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Expected tool name must not be blank")
        return v


class Scenario(BaseModel):
    """A single evaluation scenario loaded from YAML."""

    name: str = Field(..., description="Short descriptive name for the scenario")
    description: str = Field("", description="Longer description of what is being tested")
    system_prompt: str = Field(
        "You are a helpful assistant.",
        description="System prompt provided to the LLM",
    )
    user_message: str = Field(..., description="The user message that triggers tool use")
    available_tools: List[ToolDefinition] = Field(
        default_factory=list,
        description="Tools the agent can call",
    )
    expected: List[ExpectedToolCall] = Field(
        ...,
        description="Ordered list of expected tool calls",
    )


class ActualToolCall(BaseModel):
    """A tool call actually made by the agent."""

    tool: str = Field(..., description="Tool name the agent invoked")
    args: Dict[str, Any] = Field(default_factory=dict, description="Arguments passed")


class ScenarioResult(BaseModel):
    """Full result of running a single scenario."""

    scenario_name: str
    passed: bool = False
    tool_name_score: float = Field(0.0, ge=0.0, le=1.0, description="Fraction of correct tool names")
    arg_match_score: float = Field(0.0, ge=0.0, le=1.0, description="Average Jaccard similarity of args")
    sequence_score: float = Field(0.0, ge=0.0, le=1.0, description="Sequence ordering score")
    overall_score: float = Field(0.0, ge=0.0, le=1.0, description="Weighted overall score")
    latency_ms: float = Field(0.0, ge=0.0, description="End-to-end latency in milliseconds")
    expected_calls: List[ExpectedToolCall] = Field(default_factory=list)
    actual_calls: List[ActualToolCall] = Field(default_factory=list)
    error: Optional[str] = None


class BenchmarkReport(BaseModel):
    """Aggregated report across all scenarios."""

    model: str
    total_scenarios: int = 0
    passed: int = 0
    failed: int = 0
    avg_score: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    results: List[ScenarioResult] = Field(default_factory=list)

    def compute_aggregates(self) -> None:
        """Recompute aggregate statistics from individual results."""
        if not self.results:
            return

        self.total_scenarios = len(self.results)
        self.passed = sum(1 for r in self.results if r.passed)
        self.failed = self.total_scenarios - self.passed
        self.avg_score = sum(r.overall_score for r in self.results) / self.total_scenarios

        latencies = sorted(r.latency_ms for r in self.results)
        n = len(latencies)
        self.avg_latency_ms = sum(latencies) / n
        self.p50_latency_ms = _percentile(latencies, 0.50)
        self.p95_latency_ms = _percentile(latencies, 0.95)
        self.p99_latency_ms = _percentile(latencies, 0.99)


def _percentile(sorted_data: List[float], pct: float) -> float:
    """Compute a percentile using linear interpolation on sorted data."""
    if not sorted_data:
        return 0.0
    n = len(sorted_data)
    k = (n - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)
