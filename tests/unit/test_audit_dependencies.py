"""scripts/audit_dependencies.py: the triage logic (pure), and that the committed triage
file is well-formed. The real audit needs the network and runs in CI, not here."""

import datetime
import importlib.util
import json

import pytest

from tests.conftest import REPO_ROOT

spec = importlib.util.spec_from_file_location("audit_dependencies", REPO_ROOT / "scripts" / "audit_dependencies.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def report(*deps):
    return {"dependencies": [{"name": n, "version": v, "vulns": vulns} for n, v, vulns in deps]}


def vuln(vid, *aliases):
    return {"id": vid, "aliases": list(aliases), "fix_versions": []}


class TestSplitFindings:
    def test_a_listed_advisory_is_triaged_away(self):
        untriaged, triaged = audit.split_findings(report(("torch", "2.6.0", [vuln("PYSEC-1")])), {"PYSEC-1"})
        assert untriaged == [] and triaged == 1

    def test_an_unlisted_advisory_fails(self):
        untriaged, _ = audit.split_findings(report(("pillow", "1.0", [vuln("PYSEC-9")])), {"PYSEC-1"})
        assert [(n, v["id"]) for n, _, v in untriaged] == [("pillow", "PYSEC-9")]

    def test_an_alias_match_counts(self):
        listed = report(("torch", "2.6.0", [vuln("GHSA-x", "CVE-2025-1")]))
        untriaged, triaged = audit.split_findings(listed, {"CVE-2025-1"})
        assert untriaged == [] and triaged == 1

    def test_a_new_advisory_on_an_already_triaged_package_still_fails(self):
        """Triage is per advisory, not per package: a fresh torch CVE is not waved through."""
        untriaged, triaged = audit.split_findings(
            report(("torch", "2.6.0", [vuln("PYSEC-1"), vuln("PYSEC-NEW")])), {"PYSEC-1"}
        )
        assert [v["id"] for _, _, v in untriaged] == ["PYSEC-NEW"] and triaged == 1

    def test_clean_packages_produce_nothing(self):
        assert audit.split_findings(report(("requests", "2.0", [])), set()) == ([], 0)


class TestTriageFile:
    def test_the_committed_file_loads_and_every_entry_explains_itself(self):
        data = json.loads(audit.TRIAGE.read_text(encoding="utf-8"))
        assert data["entries"]
        for entry in data["entries"]:
            assert entry["package"] and entry["version"] and entry["ids"]
            assert len(entry["reason"]) > 200  # a real justification, not a placeholder

    def test_a_stale_review_is_flagged(self, tmp_path):
        path = tmp_path / "t.json"
        path.write_text(json.dumps({"reviewed": "2026-01-01", "entries": [{"ids": ["X"]}]}))
        _, notes = audit.load_triage(path, today=datetime.date(2026, 9, 21))
        assert notes and "re-review" in notes[0]

    def test_a_fresh_review_is_quiet(self, tmp_path):
        path = tmp_path / "t.json"
        path.write_text(json.dumps({"reviewed": "2026-09-01", "entries": [{"ids": ["X"]}]}))
        ignored, notes = audit.load_triage(path, today=datetime.date(2026, 9, 21))
        assert ignored == {"X"} and notes == []

    def test_the_review_date_is_a_real_date(self):
        data = json.loads(audit.TRIAGE.read_text(encoding="utf-8"))
        datetime.date.fromisoformat(data["reviewed"])


def test_the_script_exports_the_locked_runtime_requirements_with_local_versions_normalised():
    text = audit.runtime_requirements()
    assert "torch==2.6.0" in text and "+cpu" not in text
    assert "pytest" not in text  # dev tools are not audited as runtime dependencies


@pytest.mark.parametrize("name", ["pillow", "streamlit", "python-dotenv"])
def test_the_packages_that_had_real_advisories_are_at_patched_versions(name):
    """Regression for the first audit: Pillow 11.0.0 (17 advisories, on the upload-decoding
    path), Streamlit 1.50.0 and python-dotenv 1.0.1 were all reported."""
    minimum = {"pillow": (12, 3, 0), "streamlit": (1, 54, 0), "python-dotenv": (1, 2, 2)}[name]
    from importlib.metadata import version

    installed = tuple(int(part) for part in version(name).split(".")[:3])
    assert installed >= minimum
