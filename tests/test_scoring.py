"""Unit tests for the scoring module."""

from __future__ import annotations

import pytest

from agent_eval.models import ActualToolCall, ExpectedToolCall
from agent_eval.scoring import (
    _flatten_args,
    compute_overall,
    is_pass,
    jaccard_similarity,
    score_args,
    score_sequence,
    score_tool_names,
)


# ---- _flatten_args --------------------------------------------------------

class TestFlattenArgs:
    def test_simple(self) -> None:
        result = _flatten_args({"city": "Tokyo"})
        assert result == {"city=Tokyo"}

    def test_nested(self) -> None:
        result = _flatten_args({"location": {"lat": 35, "lon": 139}})
        assert result == {"location.lat=35", "location.lon=139"}

    def test_list(self) -> None:
        result = _flatten_args({"tags": ["a", "b"]})
        assert result == {"tags[0]=a", "tags[1]=b"}

    def test_empty(self) -> None:
        result = _flatten_args({})
        assert result == set()

    def test_nested_list_of_dicts(self) -> None:
        result = _flatten_args({"items": [{"id": 1}, {"id": 2}]})
        assert result == {"items[0].id=1", "items[1].id=2"}


# ---- jaccard_similarity ---------------------------------------------------

class TestJaccardSimilarity:
    def test_identical(self) -> None:
        assert jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint(self) -> None:
        assert jaccard_similarity({"a"}, {"b"}) == 0.0

    def test_partial_overlap(self) -> None:
        assert jaccard_similarity({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)

    def test_both_empty(self) -> None:
        assert jaccard_similarity(set(), set()) == 1.0

    def test_one_empty(self) -> None:
        assert jaccard_similarity({"a"}, set()) == 0.0


# ---- score_tool_names -----------------------------------------------------

class TestScoreToolNames:
    def test_perfect_match(self) -> None:
        expected = [ExpectedToolCall(tool="get_weather", args={"city": "Tokyo"})]
        actual = [ActualToolCall(tool="get_weather", args={"city": "Tokyo"})]
        assert score_tool_names(expected, actual) == 1.0

    def test_wrong_tool(self) -> None:
        expected = [ExpectedToolCall(tool="get_weather", args={})]
        actual = [ActualToolCall(tool="search_web", args={})]
        assert score_tool_names(expected, actual) == 0.0

    def test_missing_call(self) -> None:
        expected = [
            ExpectedToolCall(tool="a", args={}),
            ExpectedToolCall(tool="b", args={}),
        ]
        actual = [ActualToolCall(tool="a", args={})]
        assert score_tool_names(expected, actual) == pytest.approx(0.5)

    def test_extra_call(self) -> None:
        expected = [ExpectedToolCall(tool="a", args={})]
        actual = [
            ActualToolCall(tool="a", args={}),
            ActualToolCall(tool="b", args={}),
        ]
        assert score_tool_names(expected, actual) == pytest.approx(0.5)

    def test_both_empty(self) -> None:
        assert score_tool_names([], []) == 1.0

    def test_expected_empty_actual_not(self) -> None:
        actual = [ActualToolCall(tool="a", args={})]
        assert score_tool_names([], actual) == 0.0


# ---- score_args -----------------------------------------------------------

class TestScoreArgs:
    def test_perfect_args(self) -> None:
        expected = [ExpectedToolCall(tool="t", args={"city": "Tokyo"})]
        actual = [ActualToolCall(tool="t", args={"city": "Tokyo"})]
        assert score_args(expected, actual) == 1.0

    def test_partial_args(self) -> None:
        expected = [ExpectedToolCall(tool="t", args={"city": "Tokyo", "units": "metric"})]
        actual = [ActualToolCall(tool="t", args={"city": "Tokyo"})]
        # Jaccard: intersection=1, union=2 -> 0.5
        assert score_args(expected, actual) == pytest.approx(0.5)

    def test_empty_args(self) -> None:
        expected = [ExpectedToolCall(tool="t", args={})]
        actual = [ActualToolCall(tool="t", args={})]
        assert score_args(expected, actual) == 1.0

    def test_missing_call_scores_zero(self) -> None:
        expected = [
            ExpectedToolCall(tool="a", args={"x": 1}),
            ExpectedToolCall(tool="b", args={"y": 2}),
        ]
        actual = [ActualToolCall(tool="a", args={"x": 1})]
        # First pair: 1.0, second pair: missing -> 0.0, avg over max_len=2 -> 0.5
        assert score_args(expected, actual) == pytest.approx(0.5)


# ---- score_sequence -------------------------------------------------------

class TestScoreSequence:
    def test_perfect_order(self) -> None:
        expected = [
            ExpectedToolCall(tool="a", args={}),
            ExpectedToolCall(tool="b", args={}),
        ]
        actual = [
            ActualToolCall(tool="a", args={}),
            ActualToolCall(tool="b", args={}),
        ]
        assert score_sequence(expected, actual) == 1.0

    def test_reversed_order(self) -> None:
        expected = [
            ExpectedToolCall(tool="a", args={}),
            ExpectedToolCall(tool="b", args={}),
        ]
        actual = [
            ActualToolCall(tool="b", args={}),
            ActualToolCall(tool="a", args={}),
        ]
        # LCS length=1 out of 2 expected -> 0.5
        assert score_sequence(expected, actual) == pytest.approx(0.5)

    def test_empty_actual(self) -> None:
        expected = [ExpectedToolCall(tool="a", args={})]
        assert score_sequence(expected, []) == 0.0

    def test_both_empty(self) -> None:
        assert score_sequence([], []) == 1.0

    def test_extra_calls_in_order(self) -> None:
        expected = [
            ExpectedToolCall(tool="a", args={}),
            ExpectedToolCall(tool="b", args={}),
        ]
        actual = [
            ActualToolCall(tool="a", args={}),
            ActualToolCall(tool="x", args={}),
            ActualToolCall(tool="b", args={}),
        ]
        # LCS of [a,b] in [a,x,b] = 2, score = 2/2 = 1.0
        assert score_sequence(expected, actual) == 1.0


# ---- compute_overall ------------------------------------------------------

class TestComputeOverall:
    def test_perfect(self) -> None:
        assert compute_overall(1.0, 1.0, 1.0) == pytest.approx(1.0)

    def test_zero(self) -> None:
        assert compute_overall(0.0, 0.0, 0.0) == pytest.approx(0.0)

    def test_mixed(self) -> None:
        # With default weights 0.4, 0.4, 0.2:
        # (0.4*1.0 + 0.4*0.5 + 0.2*0.0) / 1.0 = 0.6
        assert compute_overall(1.0, 0.5, 0.0) == pytest.approx(0.6)


# ---- is_pass --------------------------------------------------------------

class TestIsPass:
    def test_above_threshold(self) -> None:
        assert is_pass(0.8) is True

    def test_at_threshold(self) -> None:
        assert is_pass(0.7) is True

    def test_below_threshold(self) -> None:
        assert is_pass(0.69) is False

    def test_custom_threshold(self) -> None:
        assert is_pass(0.5, threshold=0.5) is True
        assert is_pass(0.49, threshold=0.5) is False
