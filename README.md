# agent-eval

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/agent-eval.svg)](https://pypi.org/project/agent-eval/)
[![Tests](https://img.shields.io/github/actions/workflow/status/muhammadrashid4587/agent-eval/tests.yml?label=tests)](https://github.com/muhammadrashid4587/agent-eval/actions)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Benchmark LLM agent tool-use accuracy.** Define test scenarios in YAML, run them against any model, and get a scorecard of correctness, latency, and cost.

---

## Architecture

```
                        +-------------------+
                        |   YAML Scenarios  |
                        | (scenarios/*.yaml)|
                        +---------+---------+
                                  |
                                  v
+-----------+          +----------+----------+
|           |  Click   |                     |
|  CLI      +--------->+     Runner          |
| (cli.py)  |          | (runner.py)         |
|           |          |                     |
+-----------+          |  +---------------+  |
                       |  | BaseProvider   |  |
                       |  |  - DryRun     |  |
                       |  |  - OpenAI     |  |
                       |  +-------+-------+  |
                       +----------+----------+
                                  |
                                  v
                       +----------+----------+
                       |    Scoring Engine   |
                       |   (scoring.py)      |
                       |                     |
                       |  - Tool name match  |
                       |  - Arg similarity   |
                       |  - Sequence order   |
                       +----------+----------+
                                  |
                                  v
                       +----------+----------+
                       |     Reporters       |
                       |  (reporters.py)     |
                       |                     |
                       |  - Rich table       |
                       |  - JSON output      |
                       +---------------------+
```

## Quick Start

### Install

```bash
pip install agent-eval

# Or install from source:
git clone https://github.com/muhammadrashid4587/agent-eval.git
cd agent-eval
pip install -e ".[dev]"
```

### Write a scenario

Create `scenarios/weather.yaml`:

```yaml
name: "Weather lookup"
description: "Agent should call get_weather with correct city"
system_prompt: "You are a helpful assistant with access to weather tools."
user_message: "What's the weather in Tokyo?"
available_tools:
  - name: get_weather
    description: "Get current weather for a city"
    parameters:
      type: object
      properties:
        city:
          type: string
      required: [city]
expected:
  - tool: get_weather
    args:
      city: "Tokyo"
```

### Run it

```bash
# Dry-run (no API key needed -- uses mock provider):
agent-eval run scenarios/ --dry-run

# Against GPT-4:
export OPENAI_API_KEY="sk-..."
agent-eval run scenarios/ --model gpt-4

# JSON output:
agent-eval run scenarios/ --model gpt-4 --output json
```

### Example Output

```
Running 3 scenario(s) against gpt-4 ...

+---------------------------+--------+-----------+------+----------+---------+-----------+-------+
| Scenario                  | Status | Tool Name | Args | Sequence | Overall | Latency   | Error |
+---------------------------+--------+-----------+------+----------+---------+-----------+-------+
| Weather lookup            |  PASS  |      100% | 100% |     100% |    100% |  842.3 ms |       |
| Multi-step booking        |  PASS  |      100% |  75% |     100% |     90% | 1203.1 ms |       |
| Calculator with fallback  |  FAIL  |       50% |  50% |      50% |     50% |  651.7 ms |       |
+---------------------------+--------+-----------+------+----------+---------+-----------+-------+

+-----------------------+
| agent-eval benchmark  |
+-----------------------+
| Model:      gpt-4    |
| Scenarios:  3        |
| Passed:     2        |
| Failed:     1        |
| Avg score:  80.00%   |
| Avg latency: 899.0ms |
| p50 latency: 842.3ms |
| p95 latency: 1203.1ms|
| p99 latency: 1203.1ms|
+-----------------------+
```

## Scoring

Each scenario is scored on three dimensions:

| Dimension       | Method                                        | Weight |
|-----------------|-----------------------------------------------|--------|
| **Tool Name**   | Exact string match (1.0 or 0.0 per call)      | 40%    |
| **Arguments**   | Jaccard similarity over flattened `key=value`  | 40%    |
| **Sequence**    | LCS ratio of tool-call ordering                | 20%    |

A scenario **passes** if the weighted overall score >= **0.70** (configurable).

## Scenario Format

```yaml
name: string            # Required. Short name.
description: string     # Optional. What this tests.
system_prompt: string   # Optional. Defaults to generic assistant prompt.
user_message: string    # Required. The user query.
available_tools:        # List of tool definitions (OpenAI function format).
  - name: string
    description: string
    parameters:
      type: object
      properties: { ... }
      required: [...]
expected:               # Ordered list of expected tool calls.
  - tool: string
    args: { ... }
```

## CLI Reference

```
Usage: agent-eval [OPTIONS] COMMAND [ARGS]...

  agent-eval -- benchmark LLM agent tool-use accuracy.

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  run  Run evaluation scenarios against an LLM agent.
```

### `agent-eval run`

```
Usage: agent-eval run [OPTIONS] SCENARIOS_PATH

Options:
  -m, --model TEXT            Model identifier (default: gpt-4)
  -o, --output [table|json]   Output format (default: table)
  --dry-run                   Use mock provider (no API key)
  --help                      Show this message and exit.
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=agent_eval --cov-report=term-missing

# Type check
mypy agent_eval/

# Lint
ruff check agent_eval/ tests/
```

## Project Structure

```
agent-eval/
├── agent_eval/
│   ├── __init__.py       # Package version
│   ├── cli.py            # Click CLI entry point
│   ├── runner.py         # Scenario loading + LLM provider abstraction
│   ├── scoring.py        # Scoring: tool name, args, sequence, overall
│   ├── models.py         # Pydantic data models
│   └── reporters.py      # Rich table + JSON output
├── scenarios/
│   └── example.yaml      # Sample scenario
├── tests/
│   ├── test_scoring.py   # Scoring unit tests
│   └── test_models.py    # Model validation tests
├── pyproject.toml        # Modern Python packaging
├── LICENSE               # MIT
└── README.md
```

## License

MIT -- see [LICENSE](LICENSE).
