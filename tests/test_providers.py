"""Tests for provider factory and Anthropic provider."""

from __future__ import annotations

import pytest

from agent_eval.models import ActualToolCall, ToolDefinition, ToolParameter
from agent_eval.runner import (
    AnthropicProvider,
    BaseProvider,
    DryRunProvider,
    OpenAIProvider,
    get_provider,
)


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

class TestGetProvider:
    """Tests for the get_provider factory function."""

    def test_dry_run_returns_dry_run_provider(self) -> None:
        provider = get_provider("anything", dry_run=True)
        assert isinstance(provider, DryRunProvider)

    def test_gpt_model_returns_openai(self) -> None:
        """GPT models should route to OpenAIProvider."""
        try:
            provider = get_provider("gpt-4")
            assert isinstance(provider, OpenAIProvider)
        except ImportError:
            pytest.skip("openai package not installed")

    def test_claude_model_returns_anthropic(self) -> None:
        """Claude models should route to AnthropicProvider."""
        try:
            provider = get_provider("claude-sonnet-4-20250514")
            assert isinstance(provider, AnthropicProvider)
        except ImportError:
            pytest.skip("anthropic package not installed")

    def test_unknown_model_falls_back_to_openai(self) -> None:
        """Unknown model names default to OpenAI."""
        try:
            provider = get_provider("llama-3-70b")
            assert isinstance(provider, OpenAIProvider)
        except ImportError:
            pytest.skip("openai package not installed")


# ---------------------------------------------------------------------------
# Anthropic provider tool payload
# ---------------------------------------------------------------------------

class TestAnthropicToolPayload:
    """Tests for Anthropic tool definition conversion."""

    def _make_provider(self) -> AnthropicProvider:
        try:
            return AnthropicProvider(model="claude-sonnet-4-20250514")
        except ImportError:
            pytest.skip("anthropic package not installed")

    def test_build_tools_payload_format(self) -> None:
        provider = self._make_provider()
        tools = [
            ToolDefinition(
                name="get_weather",
                description="Get weather for a city",
                parameters=ToolParameter(
                    type="object",
                    properties={"city": {"type": "string"}},
                    required=["city"],
                ),
            )
        ]
        payload = provider._build_tools_payload(tools)
        assert len(payload) == 1
        assert payload[0]["name"] == "get_weather"
        assert payload[0]["description"] == "Get weather for a city"
        assert "input_schema" in payload[0]
        schema = payload[0]["input_schema"]
        assert schema["type"] == "object"
        assert schema["properties"] == {"city": {"type": "string"}}
        assert schema["required"] == ["city"]

    def test_build_tools_payload_empty(self) -> None:
        provider = self._make_provider()
        payload = provider._build_tools_payload([])
        assert payload == []

    def test_build_tools_payload_multiple(self) -> None:
        provider = self._make_provider()
        tools = [
            ToolDefinition(name="tool_a", description="A"),
            ToolDefinition(name="tool_b", description="B"),
        ]
        payload = provider._build_tools_payload(tools)
        assert len(payload) == 2
        assert payload[0]["name"] == "tool_a"
        assert payload[1]["name"] == "tool_b"
