"""Output reporters: console table (Rich) and JSON."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, TextIO

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from agent_eval.models import BenchmarkReport, ScenarioResult

if TYPE_CHECKING:
    from agent_eval.comparator import ComparisonResult


def _score_colour(score: float) -> str:
    """Return a Rich colour tag based on the score value."""
    if score >= 0.9:
        return "bold green"
    if score >= 0.7:
        return "yellow"
    return "bold red"


def _pass_label(passed: bool) -> Text:
    """Return a styled PASS/FAIL label."""
    if passed:
        return Text("PASS", style="bold green")
    return Text("FAIL", style="bold red")


def report_table(report: BenchmarkReport, file: TextIO | None = None) -> None:
    """Print a styled console table summarising the benchmark report."""
    console = Console(file=file or sys.stdout)

    summary_lines = [
        f"Model:      {report.model}",
        f"Scenarios:  {report.total_scenarios}",
        f"Passed:     {report.passed}",
        f"Failed:     {report.failed}",
        f"Avg score:  {report.avg_score:.2%}",
        f"Avg latency: {report.avg_latency_ms:.1f} ms",
        f"p50 latency: {report.p50_latency_ms:.1f} ms",
        f"p95 latency: {report.p95_latency_ms:.1f} ms",
        f"p99 latency: {report.p99_latency_ms:.1f} ms",
    ]
    console.print(Panel("\n".join(summary_lines), title="agent-eval benchmark", border_style="cyan"))

    table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("Scenario", style="dim", min_width=20)
    table.add_column("Status", justify="center", width=6)
    table.add_column("Tool Name", justify="right", width=10)
    table.add_column("Args", justify="right", width=10)
    table.add_column("Sequence", justify="right", width=10)
    table.add_column("Overall", justify="right", width=10)
    table.add_column("Latency", justify="right", width=12)
    table.add_column("Error", max_width=30)

    for r in report.results:
        table.add_row(
            r.scenario_name,
            _pass_label(r.passed),
            Text(f"{r.tool_name_score:.0%}", style=_score_colour(r.tool_name_score)),
            Text(f"{r.arg_match_score:.0%}", style=_score_colour(r.arg_match_score)),
            Text(f"{r.sequence_score:.0%}", style=_score_colour(r.sequence_score)),
            Text(f"{r.overall_score:.0%}", style=_score_colour(r.overall_score)),
            f"{r.latency_ms:.1f} ms",
            r.error or "",
        )

    console.print(table)


def report_json(report: BenchmarkReport, file: TextIO | None = None) -> None:
    """Emit the benchmark report as pretty-printed JSON."""
    out = file or sys.stdout
    out.write(report.model_dump_json(indent=2))
    out.write("\n")


def report_comparison_table(
    result: ComparisonResult,
    file: TextIO | None = None,
) -> None:
    """Print a side-by-side comparison table across models.

    Rows are scenarios, columns are models.  The best score per scenario
    is highlighted in green, the worst in red.
    """
    console = Console(file=file or sys.stdout)

    model_names = [c.model_name for c in result.comparisons]
    avg_scores = {c.model_name: c.report.avg_score for c in result.comparisons}
    summary_lines = [
        f"Models compared: {', '.join(model_names)}",
        f"Scenarios:       {len(result.scenario_names)}",
        f"Best model:      {result.best_model} ({avg_scores.get(result.best_model, 0):.0%} avg)",
    ]
    console.print(Panel("\n".join(summary_lines), title="agent-eval comparison", border_style="cyan"))

    table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("Scenario", style="dim", min_width=20)
    for model in model_names:
        table.add_column(model, justify="center", min_width=12)

    for scenario_name in result.scenario_names:
        scores = result.summary_table.get(scenario_name, {})
        score_values = [scores.get(m, 0.0) for m in model_names]
        best = max(score_values) if score_values else 0.0
        worst = min(score_values) if score_values else 0.0

        cells: list[Text | str] = [scenario_name]
        for m in model_names:
            s = scores.get(m, 0.0)
            style = _score_colour(s)
            if len(model_names) > 1:
                if s == best and best > worst:
                    style = "bold green"
                elif s == worst and worst < best:
                    style = "bold red"
            cells.append(Text(f"{s:.0%}", style=style))
        table.add_row(*cells)

    avg_cells: list[Text | str] = [Text("Average", style="bold")]
    for m in model_names:
        avg = avg_scores.get(m, 0.0)
        avg_cells.append(Text(f"{avg:.0%}", style="bold " + _score_colour(avg)))
    table.add_row(*avg_cells)

    console.print(table)

    lat_table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    lat_table.add_column("Model", style="dim", min_width=20)
    lat_table.add_column("Avg Latency", justify="right", width=12)
    lat_table.add_column("p50", justify="right", width=10)
    lat_table.add_column("p95", justify="right", width=10)
    lat_table.add_column("Pass Rate", justify="right", width=10)

    for comp in result.comparisons:
        r = comp.report
        pass_rate = r.passed / r.total_scenarios if r.total_scenarios else 0.0
        lat_table.add_row(
            comp.model_name,
            f"{r.avg_latency_ms:.1f} ms",
            f"{r.p50_latency_ms:.1f} ms",
            f"{r.p95_latency_ms:.1f} ms",
            Text(f"{pass_rate:.0%}", style=_score_colour(pass_rate)),
        )
    console.print(lat_table)


def output_report(
    report: BenchmarkReport,
    fmt: str = "table",
    file: TextIO | None = None,
) -> None:
    """Dispatch to the appropriate reporter."""
    if fmt == "table":
        report_table(report, file=file)
    elif fmt == "json":
        report_json(report, file=file)
    else:
        raise ValueError(f"Unknown output format: {fmt!r}. Use 'table' or 'json'.")
