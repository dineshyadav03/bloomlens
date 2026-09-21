"""Print the aggregate scan telemetry (src/telemetry.py) as a readable report.

    uv run python scripts/report_metrics.py          # a table
    uv run python scripts/report_metrics.py --json   # the raw aggregate

Aggregates only: counts, failure rates, p50/p95 latency, means. It never prints a row, a
timestamp or an identifier (there are none to print: see docs/PRIVACY.md). Percentiles are
withheld below 20 scans, and there is deliberately no p99.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import telemetry  # noqa: E402


def _ms(block: dict) -> str:
    if block["p50_ms"] is None:
        return f"n={block['n']} ({block['note']})"
    return f"p50 {block['p50_ms']} ms, p95 {block['p95_ms']} ms (n={block['n']})"


def render(report: dict) -> str:
    scans, latency = report["scans"], report["latency_ms"]
    cost, tokens = report["estimated_cost_usd"], report["tokens"]
    lines = [
        f"Scan telemetry, last {report['window_days']} days",
        f"  scans: {scans['total']} ({scans['ok']} ok, {scans['error']} failed; failure rate {report['failure_rate']})",
        "  failures by category: "
        + (", ".join(f"{k}={v}" for k, v in report["failures_by_category"].items() if v) or "none"),
        f"  latency, warm scans:  {_ms(latency['warm'])}",
        f"  latency, cold scans:  {_ms(latency['cold'])}",
    ]
    for stage, block in latency["warm_by_stage"].items():
        lines.append(f"    {stage:<7} {_ms(block)}")
    lines += [
        f"  retries per scan: {report['retries_per_scan']['mean']} (over {report['retries_per_scan']['n']} scans)",
        f"  tokens: {tokens['scans_with_counts']} scans reported counts (mean in {tokens['mean_input']}, "
        f"out {tokens['mean_output']}); {tokens['scans_without_counts']} did not",
        f"  cost: ${cost['total']} over {cost['scans_priced']} priced scans "
        f"({cost['scans_unpriced']} unpriced); {cost['meaning']}",
        f"  pricing versions: {', '.join(cost['pricing_versions']) or 'none'}",
        f"  BioCLIP revision: {', '.join(report['versions']['bioclip_revision']) or 'n/a'}",
        f"  Gemini model: requested {', '.join(report['versions']['gemini_model_requested']) or 'n/a'}; "
        f"provider-reported {', '.join(report['versions']['gemini_model_reported']) or 'n/a'}",
        f"  measured on: {'; '.join(report['platforms']) or 'n/a'}",
        f"  {latency['note']}",
    ]
    return "\n".join(lines)


def main() -> None:
    report = telemetry.summary()
    print(json.dumps(report, indent=2) if "--json" in sys.argv else render(report))


if __name__ == "__main__":
    main()
