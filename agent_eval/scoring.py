"""Scoring functions for evaluating agent tool-use accuracy.

Scoring dimensions:
  - tool_name_score:  Exact match on tool name (1.0 or 0.0 per call).
  - arg_match_score:  Jaccard similarity over flattened key=value pairs.
  - sequence_score:   Longest-common-subsequence ratio for call ordering.
  - overall_score:    Weighted combination of the above three.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

from agent_eval.models import ActualToolCall, ExpectedToolCall

# Default weights for the overall score.
WEIGHT_TOOL_NAME = 0.4
WEIGHT_ARG_MATCH = 0.4
WEIGHT_SEQUENCE = 0.2

PASS_THRESHOLD = 0.7


def _flatten_args(args: Dict[str, Any], prefix: str = "") -> Set[str]:
    """Flatten a nested dict into a set of 'key=value' strings for comparison.

    Examples:
        >>> sorted(_flatten_args({"city": "Tokyo", "units": "metric"}))
        ['city=Tokyo', 'units=metric']
        >>> sorted(_flatten_args({"location": {"lat": 35, "lon": 139}}))
        ['location.lat=35', 'location.lon=139']
    """
    items: Set[str] = set()
    for key, value in args.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            items.update(_flatten_args(value, full_key))
        elif isinstance(value, list):
            for i, v in enumerate(value):
                if isinstance(v, dict):
                    items.update(_flatten_args(v, f"{full_key}[{i}]"))
                else:
                    items.add(f"{full_key}[{i}]={v}")
        else:
            items.add(f"{full_key}={value}")
    return items


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute the Jaccard similarity coefficient between two sets.

    Returns 1.0 if both sets are empty (vacuously equal), 0.0 if one is
    empty and the other is not.
    """
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def score_tool_names(
    expected: List[ExpectedToolCall],
    actual: List[ActualToolCall],
) -> float:
    """Score tool-name accuracy across paired expected/actual calls.

    Pairs are matched positionally.  If the actual list is shorter, missing
    calls score 0.  Extra actual calls are penalised.

    Returns a float in [0.0, 1.0].
    """
    if not expected:
        return 1.0 if not actual else 0.0

    max_len = max(len(expected), len(actual))
    matches = 0
    for i in range(max_len):
        if i < len(expected) and i < len(actual):
            if expected[i].tool == actual[i].tool:
                matches += 1
        # else: missing or extra call -> no match

    return matches / max_len


def score_args(
    expected: List[ExpectedToolCall],
    actual: List[ActualToolCall],
) -> float:
    """Score argument accuracy using Jaccard similarity over flattened args.

    Pairs are matched positionally.  Missing or extra calls contribute 0.0.

    Returns a float in [0.0, 1.0].
    """
    if not expected:
        return 1.0 if not actual else 0.0

    max_len = max(len(expected), len(actual))
    total_sim = 0.0
    for i in range(max_len):
        if i < len(expected) and i < len(actual):
            expected_flat = _flatten_args(expected[i].args)
            actual_flat = _flatten_args(actual[i].args)
            total_sim += jaccard_similarity(expected_flat, actual_flat)

    return total_sim / max_len


def _lcs_length(seq_a: List[str], seq_b: List[str]) -> int:
    """Compute the length of the longest common subsequence (LCS)."""
    m, n = len(seq_a), len(seq_b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def score_sequence(
    expected: List[ExpectedToolCall],
    actual: List[ActualToolCall],
) -> float:
    """Score the ordering of tool calls via LCS ratio.

    Returns the length of the longest common subsequence of tool names
    divided by the length of the expected sequence.
    """
    if not expected:
        return 1.0 if not actual else 0.0
    if not actual:
        return 0.0

    expected_names = [e.tool for e in expected]
    actual_names = [a.tool for a in actual]
    lcs = _lcs_length(expected_names, actual_names)
    return lcs / len(expected_names)


def compute_overall(
    tool_name_score: float,
    arg_match_score: float,
    sequence_score: float,
    *,
    w_name: float = WEIGHT_TOOL_NAME,
    w_args: float = WEIGHT_ARG_MATCH,
    w_seq: float = WEIGHT_SEQUENCE,
) -> float:
    """Compute a weighted overall score.

    Args:
        tool_name_score: Score for tool name matching.
        arg_match_score: Score for argument matching.
        sequence_score: Score for sequence ordering.
        w_name: Weight for tool name score.
        w_args: Weight for argument score.
        w_seq: Weight for sequence score.

    Returns:
        A float in [0.0, 1.0].
    """
    total_weight = w_name + w_args + w_seq
    if total_weight == 0:
        return 0.0
    raw = (
        w_name * tool_name_score + w_args * arg_match_score + w_seq * sequence_score
    ) / total_weight
    return max(0.0, min(1.0, raw))


def is_pass(overall_score: float, threshold: float = PASS_THRESHOLD) -> bool:
    """Determine whether a scenario passes based on the overall score."""
    return overall_score >= threshold
