"""agent-eval: A CLI tool for benchmarking LLM agent tool-use accuracy."""

__version__ = "0.1.0"

from agent_eval.models import (
    ActualToolCall,
    BenchmarkReport,
    ExpectedToolCall,
    Scenario,
    ScenarioResult,
    ToolDefinition,
    ToolParameter,
)
from agent_eval.runner import (
    AnthropicProvider,
    BaseProvider,
    DryRunProvider,
    OpenAIProvider,
)

__all__ = [
    "__version__",
    "ActualToolCall",
    "AnthropicProvider",
    "BaseProvider",
    "BenchmarkReport",
    "DryRunProvider",
    "ExpectedToolCall",
    "OpenAIProvider",
    "Scenario",
    "ScenarioResult",
    "ToolDefinition",
    "ToolParameter",
]
