"""Scenario runner: loads YAML scenarios, invokes LLM providers, collects results.

The runner is provider-agnostic.  Concrete providers implement the
``BaseProvider`` interface.  A ``DryRunProvider`` is included for
offline testing without any API keys.
"""

from __future__ import annotations

import abc
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

from agent_eval.models import (
    ActualToolCall,
    BenchmarkReport,
    ExpectedToolCall,
    Scenario,
    ScenarioResult,
    ToolDefinition,
)
from agent_eval.scoring import (
    compute_overall,
    is_pass,
    score_args,
    score_sequence,
    score_tool_names,
)


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------

class BaseProvider(abc.ABC):
    """Abstract base class for LLM providers."""

    @abc.abstractmethod
    def call(
        self,
        system_prompt: str,
        user_message: str,
        tools: List[ToolDefinition],
    ) -> List[ActualToolCall]:
        """Send a prompt to the LLM and return the tool calls it made.

        Args:
            system_prompt: The system message.
            user_message: The user message that should trigger tool use.
            tools: Available tool definitions the agent may invoke.

        Returns:
            An ordered list of tool calls the agent made.
        """
        ...


# ---------------------------------------------------------------------------
# Dry-run provider (deterministic, no API key needed)
# ---------------------------------------------------------------------------

class DryRunProvider(BaseProvider):
    """A mock provider that echoes back the *expected* calls from the scenario.

    Useful for integration tests and CI pipelines where API access is
    unavailable.  The provider is injected with the expected calls before
    each scenario run via :meth:`set_expected`.
    """

    def __init__(self) -> None:
        self._expected: List[ExpectedToolCall] = []

    def set_expected(self, expected: List[ExpectedToolCall]) -> None:
        """Pre-load the expected calls so they are returned verbatim."""
        self._expected = expected

    def call(
        self,
        system_prompt: str,
        user_message: str,
        tools: List[ToolDefinition],
    ) -> List[ActualToolCall]:
        """Return the pre-loaded expected calls as actual calls."""
        return [
            ActualToolCall(tool=e.tool, args=dict(e.args))
            for e in self._expected
        ]


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class OpenAIProvider(BaseProvider):
    """Provider that uses the OpenAI Chat Completions API with tool calling.

    Requires the ``openai`` package and an ``OPENAI_API_KEY`` env var.
    """

    def __init__(self, model: str = "gpt-4") -> None:
        self.model = model
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The openai package is required for OpenAIProvider.  "
                "Install it with:  pip install openai"
            ) from exc

    def _build_tools_payload(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert internal tool definitions to the OpenAI function-calling format."""
        payload: List[Dict[str, Any]] = []
        for t in tools:
            payload.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters.model_dump(),
                    },
                }
            )
        return payload

    def call(
        self,
        system_prompt: str,
        user_message: str,
        tools: List[ToolDefinition],
        max_retries: int = 3,
    ) -> List[ActualToolCall]:
        """Call OpenAI and extract tool calls from the response.

        Includes retry logic with exponential backoff for transient API errors.
        """
        import openai

        client = openai.OpenAI()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        tools_payload = self._build_tools_payload(tools)

        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools_payload if tools_payload else openai.NOT_GIVEN,
                    tool_choice="auto" if tools_payload else openai.NOT_GIVEN,
                    timeout=60.0,
                )
                break
            except openai.RateLimitError as exc:
                last_error = exc
                wait = 2 ** attempt
                logger.warning("Rate limited (attempt %d/%d), retrying in %ds", attempt + 1, max_retries, wait)
                time.sleep(wait)
            except openai.APITimeoutError as exc:
                last_error = exc
                wait = 2 ** attempt
                logger.warning("Request timed out (attempt %d/%d), retrying in %ds", attempt + 1, max_retries, wait)
                time.sleep(wait)
            except openai.AuthenticationError:
                raise
            except openai.APIError as exc:
                last_error = exc
                wait = 2 ** attempt
                logger.warning("API error (attempt %d/%d): %s", attempt + 1, max_retries, exc)
                time.sleep(wait)
        else:
            raise RuntimeError(
                f"OpenAI API call failed after {max_retries} attempts: {last_error}"
            ) from last_error

        if not response.choices:
            logger.warning("OpenAI returned empty choices list")
            return []

        actual_calls: List[ActualToolCall] = []
        choice = response.choices[0]
        if choice.message and choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                actual_calls.append(
                    ActualToolCall(tool=tc.function.name, args=args)
                )
        return actual_calls


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

class AnthropicProvider(BaseProvider):
    """Provider that uses the Anthropic Messages API with tool calling.

    Requires the ``anthropic`` package and an ``ANTHROPIC_API_KEY`` env var.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514") -> None:
        self.model = model
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The anthropic package is required for AnthropicProvider.  "
                "Install it with:  pip install anthropic"
            ) from exc

    def _build_tools_payload(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert internal tool definitions to the Anthropic tool_use format."""
        payload: List[Dict[str, Any]] = []
        for t in tools:
            payload.append(
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": {
                        "type": t.parameters.type,
                        "properties": t.parameters.properties,
                        "required": t.parameters.required,
                    },
                }
            )
        return payload

    def call(
        self,
        system_prompt: str,
        user_message: str,
        tools: List[ToolDefinition],
        max_retries: int = 3,
    ) -> List[ActualToolCall]:
        """Call Anthropic and extract tool calls from the response.

        Includes retry logic with exponential backoff for transient API errors.
        """
        import anthropic

        client = anthropic.Anthropic()
        messages = [{"role": "user", "content": user_message}]
        tools_payload = self._build_tools_payload(tools)

        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system_prompt,
                    messages=messages,
                    tools=tools_payload if tools_payload else [],
                    timeout=60.0,
                )
                break
            except anthropic.RateLimitError as exc:
                last_error = exc
                wait = 2 ** attempt
                logger.warning("Rate limited (attempt %d/%d), retrying in %ds", attempt + 1, max_retries, wait)
                time.sleep(wait)
            except anthropic.APITimeoutError as exc:
                last_error = exc
                wait = 2 ** attempt
                logger.warning("Request timed out (attempt %d/%d), retrying in %ds", attempt + 1, max_retries, wait)
                time.sleep(wait)
            except anthropic.AuthenticationError:
                raise
            except anthropic.APIError as exc:
                last_error = exc
                wait = 2 ** attempt
                logger.warning("API error (attempt %d/%d): %s", attempt + 1, max_retries, exc)
                time.sleep(wait)
        else:
            raise RuntimeError(
                f"Anthropic API call failed after {max_retries} attempts: {last_error}"
            ) from last_error

        actual_calls: List[ActualToolCall] = []
        for block in response.content:
            if block.type == "tool_use":
                actual_calls.append(
                    ActualToolCall(
                        tool=block.name,
                        args=block.input if isinstance(block.input, dict) else {},
                    )
                )
        return actual_calls


# ---------------------------------------------------------------------------
# Scenario loading
# ---------------------------------------------------------------------------

def load_scenarios(path: str | Path) -> List[Scenario]:
    """Load scenario YAML files from a file or directory.

    Args:
        path: A single ``.yaml``/``.yml`` file or a directory containing them.

    Returns:
        A list of validated :class:`Scenario` objects.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If a YAML file cannot be parsed into a Scenario.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scenario path not found: {path}")

    yaml_files: List[Path] = []
    if path.is_file():
        yaml_files.append(path)
    else:
        yaml_files = sorted(path.glob("**/*.yaml")) + sorted(path.glob("**/*.yml"))

    if not yaml_files:
        raise FileNotFoundError(f"No YAML files found in {path}")

    scenarios: List[Scenario] = []
    for yf in yaml_files:
        with open(yf, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if raw is None:
            continue
        # Support files with a single scenario dict or a list of scenarios.
        items = raw if isinstance(raw, list) else [raw]
        for item in items:
            try:
                scenarios.append(Scenario(**item))
            except Exception as exc:
                raise ValueError(f"Invalid scenario in {yf}: {exc}") from exc

    return scenarios


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

def get_provider(model: str, *, dry_run: bool = False) -> BaseProvider:
    """Instantiate the appropriate provider for the given model name.

    Args:
        model: Model identifier (e.g. ``gpt-4``, ``gpt-3.5-turbo``).
        dry_run: If True, return a :class:`DryRunProvider` regardless of model.

    Returns:
        A concrete :class:`BaseProvider` instance.
    """
    if dry_run:
        return DryRunProvider()
    if model.startswith("claude-"):
        return AnthropicProvider(model=model)
    return OpenAIProvider(model=model)


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run_scenarios(
    scenarios: List[Scenario],
    provider: BaseProvider,
    model_name: str = "unknown",
    threshold: float = 0.70,
) -> BenchmarkReport:
    """Execute a list of scenarios against a provider and return a report.

    Args:
        scenarios: The scenarios to evaluate.
        provider: The LLM provider to use.
        model_name: Model identifier for the report.
        threshold: Minimum overall score (0.0-1.0) for a scenario to pass.

    Returns:
        A :class:`BenchmarkReport` with individual and aggregate results.
    """
    report = BenchmarkReport(model=model_name)

    for scenario in scenarios:
        # If using DryRunProvider, inject expected calls.
        if isinstance(provider, DryRunProvider):
            provider.set_expected(scenario.expected)

        result = ScenarioResult(
            scenario_name=scenario.name,
            expected_calls=scenario.expected,
        )

        try:
            start = time.perf_counter()
            actual = provider.call(
                system_prompt=scenario.system_prompt,
                user_message=scenario.user_message,
                tools=scenario.available_tools,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            result.actual_calls = actual
            result.latency_ms = round(elapsed_ms, 2)
            result.tool_name_score = score_tool_names(scenario.expected, actual)
            result.arg_match_score = score_args(scenario.expected, actual)
            result.sequence_score = score_sequence(scenario.expected, actual)
            result.overall_score = compute_overall(
                result.tool_name_score,
                result.arg_match_score,
                result.sequence_score,
            )
            result.passed = is_pass(result.overall_score, threshold=threshold)
        except Exception as exc:
            result.error = str(exc)
            result.passed = False

        report.results.append(result)

    report.compute_aggregates()
    return report
