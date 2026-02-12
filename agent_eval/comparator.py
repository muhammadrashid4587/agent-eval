"""Model comparison: run the same scenarios across multiple LLM providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from agent_eval.models import BenchmarkReport, Scenario
from agent_eval.runner import DryRunProvider, get_provider, run_scenarios


@dataclass
class ModelComparison:
    """Results for a single model."""

    model_name: str
    report: BenchmarkReport


@dataclass
class ComparisonResult:
    """Side-by-side comparison across multiple models."""

    scenario_names: List[str]
    comparisons: List[ModelComparison] = field(default_factory=list)
    best_model: str = ""
    summary_table: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def compute_summary(self) -> None:
        """Build the summary table and determine the best model.

        The summary_table maps scenario names to a dict of
        ``{model_name: overall_score}``.
        """
        self.summary_table = {}
        for name in self.scenario_names:
            self.summary_table[name] = {}
            for comp in self.comparisons:
                for r in comp.report.results:
                    if r.scenario_name == name:
                        self.summary_table[name][comp.model_name] = r.overall_score
                        break

        # Determine best model by average score.
        if self.comparisons:
            best_score = -1.0
            for comp in self.comparisons:
                if comp.report.avg_score > best_score:
                    best_score = comp.report.avg_score
                    self.best_model = comp.model_name


def compare_models(
    scenarios: List[Scenario],
    model_names: List[str],
    *,
    dry_run: bool = False,
    threshold: float = 0.70,
) -> ComparisonResult:
    """Run scenarios against every model and return a comparison.

    Args:
        scenarios: The evaluation scenarios.
        model_names: List of model identifiers to compare.
        dry_run: If True, use :class:`DryRunProvider` for all models.
        threshold: Pass/fail threshold forwarded to :func:`run_scenarios`.

    Returns:
        A :class:`ComparisonResult` with per-model reports and a summary.
    """
    scenario_names = [s.name for s in scenarios]
    result = ComparisonResult(scenario_names=scenario_names)

    for model in model_names:
        provider = get_provider(model, dry_run=dry_run)
        report = run_scenarios(
            scenarios, provider, model_name=model, threshold=threshold,
        )
        result.comparisons.append(ModelComparison(model_name=model, report=report))

    result.compute_summary()
    return result
