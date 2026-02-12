"""Click CLI entry-point for agent-eval.

Usage examples::

    # Run all scenarios in a directory (dry-run, no API key needed)
    agent-eval run scenarios/ --dry-run

    # Run against GPT-4 and output JSON
    agent-eval run scenarios/ --model gpt-4 --output json

    # Compare models side-by-side
    agent-eval compare scenarios/ --models gpt-4,claude-sonnet-4-20250514
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from agent_eval import __version__
from agent_eval.reporters import output_report, report_comparison_table
from agent_eval.runner import get_provider, load_scenarios, run_scenarios


@click.group()
@click.version_option(version=__version__, prog_name="agent-eval")
def main() -> None:
    """agent-eval -- benchmark LLM agent tool-use accuracy."""


@main.command()
@click.argument("scenarios_path", type=click.Path(exists=True))
@click.option(
    "--model",
    "-m",
    default="gpt-4",
    show_default=True,
    help="Model identifier to evaluate (e.g. gpt-4, claude-sonnet-4-20250514).",
)
@click.option(
    "--output",
    "-o",
    "output_fmt",
    type=click.Choice(["table", "json"], case_sensitive=False),
    default="table",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Use the DryRunProvider (mock responses, no API key needed).",
)
@click.option(
    "--threshold",
    "-t",
    default=0.70,
    show_default=True,
    type=click.FloatRange(0.0, 1.0),
    help="Minimum overall score to pass a scenario (0.0-1.0).",
)
def run(
    scenarios_path: str,
    model: str,
    output_fmt: str,
    dry_run: bool,
    threshold: float,
) -> None:
    """Run evaluation scenarios against an LLM agent.

    SCENARIOS_PATH can be a single YAML file or a directory of YAML files.
    """
    try:
        scenarios = load_scenarios(scenarios_path)
    except (FileNotFoundError, ValueError) as exc:
        click.secho(f"Error loading scenarios: {exc}", fg="red", err=True)
        sys.exit(1)

    if not scenarios:
        click.secho("No scenarios found.", fg="yellow", err=True)
        sys.exit(0)

    click.echo(
        f"Running {len(scenarios)} scenario(s) against "
        f"{'DryRunProvider' if dry_run else model} ...\n"
    )

    provider = get_provider(model, dry_run=dry_run)
    report = run_scenarios(scenarios, provider, model_name=model, threshold=threshold)

    output_report(report, fmt=output_fmt)

    # Exit with non-zero code if any scenario failed.
    if report.failed > 0:
        sys.exit(1)


@main.command()
@click.argument("scenarios_path", type=click.Path(exists=True))
@click.option(
    "--models",
    "-m",
    required=True,
    help="Comma-separated list of models to compare (e.g. gpt-4,claude-sonnet-4-20250514,gpt-4o-mini).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Use the DryRunProvider for all models (mock responses).",
)
@click.option(
    "--threshold",
    "-t",
    default=0.70,
    show_default=True,
    type=click.FloatRange(0.0, 1.0),
    help="Minimum overall score to pass a scenario (0.0-1.0).",
)
@click.option(
    "--output",
    "-o",
    "output_fmt",
    type=click.Choice(["table", "json"], case_sensitive=False),
    default="table",
    show_default=True,
    help="Output format.",
)
def compare(
    scenarios_path: str,
    models: str,
    dry_run: bool,
    threshold: float,
    output_fmt: str,
) -> None:
    """Compare multiple models on the same scenarios side-by-side.

    SCENARIOS_PATH can be a single YAML file or a directory of YAML files.
    """
    from agent_eval.comparator import compare_models

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if len(model_list) < 2:
        click.secho("Provide at least 2 models to compare (comma-separated).", fg="red", err=True)
        sys.exit(1)

    try:
        scenarios = load_scenarios(scenarios_path)
    except (FileNotFoundError, ValueError) as exc:
        click.secho(f"Error loading scenarios: {exc}", fg="red", err=True)
        sys.exit(1)

    if not scenarios:
        click.secho("No scenarios found.", fg="yellow", err=True)
        sys.exit(0)

    click.echo(
        f"Comparing {len(model_list)} models on {len(scenarios)} scenario(s) ...\n"
    )

    result = compare_models(
        scenarios, model_list, dry_run=dry_run, threshold=threshold,
    )

    if output_fmt == "json":
        import json

        data = {
            "best_model": result.best_model,
            "scenario_names": result.scenario_names,
            "summary_table": result.summary_table,
            "models": {
                c.model_name: c.report.model_dump()
                for c in result.comparisons
            },
        }
        click.echo(json.dumps(data, indent=2))
    else:
        report_comparison_table(result)


if __name__ == "__main__":
    main()
