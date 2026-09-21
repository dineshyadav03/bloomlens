"""Audit the locked runtime dependencies against known-vulnerability databases.

Exports the runtime requirements from uv.lock, audits them with pip-audit, and fails
on any advisory that is not listed in docs/security/audit-triage.json. That file holds
the advisories someone has read and judged unreachable, each with its reason; anything
new -- a fresh CVE in an unchanged dependency, or a real regression -- fails loudly.

    uv run --frozen python scripts/audit_dependencies.py

Needs network access (pip-audit queries PyPI's advisory data).
"""

import datetime
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRIAGE = ROOT / "docs" / "security" / "audit-triage.json"
REVIEW_EVERY_DAYS = 90


def load_triage(path: Path = TRIAGE, today: datetime.date | None = None) -> tuple[set[str], list[str]]:
    """(advisory ids judged unreachable, notes to print)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    ignored = {vuln_id for entry in data["entries"] for vuln_id in entry["ids"]}
    notes = []
    age = ((today or datetime.date.today()) - datetime.date.fromisoformat(data["reviewed"])).days
    if age > REVIEW_EVERY_DAYS:
        notes.append(f"triage last reviewed {age} days ago (policy: every {REVIEW_EVERY_DAYS}); please re-review.")
    return ignored, notes


def split_findings(report: dict, ignored: set[str]) -> tuple[list[tuple[str, str, dict]], int]:
    """(untriaged (name, version, vuln) findings, how many advisories were triaged away).
    An advisory counts as triaged if its id *or any alias* is listed."""
    untriaged, triaged = [], 0
    for dep in report["dependencies"]:
        for vuln in dep.get("vulns", []):
            if {vuln["id"], *vuln.get("aliases", [])} & ignored:
                triaged += 1
            else:
                untriaged.append((dep["name"], dep["version"], vuln))
    return untriaged, triaged


def runtime_requirements() -> str:
    exported = subprocess.run(
        ["uv", "export", "--frozen", "--no-dev", "--no-hashes", "--no-emit-project"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    # torch/torchvision are locked as 2.6.0+cpu, a local build PyPI has no record of;
    # the same release's advisories are filed against plain 2.6.0.
    return exported.replace("+cpu", "")


def main() -> int:
    ignored, notes = load_triage()
    with tempfile.TemporaryDirectory() as tmp:
        requirements = Path(tmp) / "requirements.txt"
        requirements.write_text(runtime_requirements(), encoding="utf-8")
        command = [sys.executable, "-m", "pip_audit", "-r", str(requirements), "--no-deps", "--disable-pip"]
        result = subprocess.run([*command, "--progress-spinner", "off", "-f", "json"], capture_output=True, text=True)
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"pip-audit did not produce a report (exit {result.returncode}):\n{result.stderr[-2000:]}")
        return 2

    untriaged, triaged = split_findings(report, ignored)
    for note in notes:
        print(f"note: {note}")
    print(f"audited {len(report['dependencies'])} packages; {triaged} advisories triaged as unreachable")
    if not untriaged:
        print("no untriaged advisories")
        return 0
    print(f"\n{len(untriaged)} UNTRIAGED advisories:")
    for name, version, vuln in untriaged:
        fix = vuln.get("fix_versions") or "none yet"
        print(f"  {name} {version}: {vuln['id']} {vuln.get('aliases', [])[:2]} fix: {fix}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
