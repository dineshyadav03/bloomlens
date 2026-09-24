"""scripts/check_doc_links.py: every internal markdown link and #anchor must resolve.

Built after the checker found six anchors in docs/RESEARCH.md written in kramdown's `{#id}` syntax,
which GitHub does not honour -- so the README's citation links landed at the top of the page.
"""

import importlib.util

import pytest

from tests.conftest import REPO_ROOT

spec = importlib.util.spec_from_file_location("check_doc_links", REPO_ROOT / "scripts" / "check_doc_links.py")
cdl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cdl)


@pytest.fixture
def docs(tmp_path, monkeypatch):
    """A scratch docs tree; ROOT is pointed at it so reported paths are relative to it."""
    monkeypatch.setattr(cdl, "ROOT", tmp_path)
    (tmp_path / "docs").mkdir()

    def write(relative, text):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    return write


class TestSlug:
    @pytest.mark.parametrize(
        ("heading", "expected"),
        [
            ("Quick start", "quick-start"),
            ("Tests — Milestone 12", "tests--milestone-12"),  # the em dash is dropped, both spaces stay
            (
                "Open-world evaluation: protocol, data and the flag — M15a",
                "open-world-evaluation-protocol-data-and-the-flag--m15a",
            ),
            ("`eval/run_e2e.py` results", "evalrun_e2epy-results"),  # backticks go, underscores stay
            ('Museum of the Future, Dubai — "The Library"', "museum-of-the-future-dubai--the-library"),
            ("A [link](https://example.com) in a heading", "a-link-in-a-heading"),
        ],
    )
    def test_follows_githubs_rules(self, heading, expected):
        assert cdl.slug(heading) == expected


class TestHeadings:
    def test_collects_headings_at_every_level(self, docs):
        path = docs("a.md", "# One\n\n## Two words\n\n### Three\n")
        assert cdl.headings(path) == {"one", "two-words", "three"}

    def test_a_repeated_heading_gets_githubs_numeric_suffix(self, docs):
        path = docs("a.md", "## Notes\n\n## Notes\n\n## Notes\n")
        assert cdl.headings(path) == {"notes", "notes-1", "notes-2"}

    def test_an_explicit_html_anchor_counts(self, docs):
        path = docs("a.md", '<a id="cite-1"></a>\n\n## Whatever\n\n<a name="old-style"></a>\n')
        assert {"cite-1", "old-style", "whatever"} <= cdl.headings(path)

    def test_kramdown_custom_ids_do_not_count_because_github_ignores_them(self, docs):
        path = docs("a.md", "## FloraHolland {#floraholland}\n")
        assert "floraholland" not in cdl.headings(path)

    def test_a_comment_line_inside_a_code_fence_is_not_a_heading(self, docs):
        path = docs("a.md", "```bash\n# just a shell comment\n```\n\n## Real\n")
        assert cdl.headings(path) == {"real"}


class TestCheck:
    def test_a_valid_file_and_anchor_link_passes(self, docs):
        docs("docs/other.md", "## Target section\n")
        page = docs("README.md", "See [it](docs/other.md#target-section) and [top](docs/other.md).\n")
        assert cdl.check(page) == []

    def test_a_missing_file_is_reported_with_its_line(self, docs):
        page = docs("README.md", "fine\n\n[gone](docs/missing.md)\n")
        assert cdl.check(page) == ["README.md:3: docs/missing.md -> no such file"]

    def test_a_missing_anchor_in_an_existing_file_is_reported(self, docs):
        docs("docs/other.md", "## Real heading\n")
        page = docs("README.md", "[bad](docs/other.md#not-there)\n")
        (problem,) = cdl.check(page)
        assert "no heading #not-there" in problem and problem.startswith("README.md:1:")

    def test_a_same_page_anchor_is_checked_against_the_page_itself(self, docs):
        page = docs("README.md", "## Status\n\n[ok](#status) and [bad](#nope)\n")
        (problem,) = cdl.check(page)
        assert "#nope" in problem

    def test_links_are_resolved_relative_to_the_file_they_are_in(self, docs):
        docs("eval/results.md", "# Results\n")
        page = docs("docs/DEV.md", "[r](../eval/results.md)\n")
        assert cdl.check(page) == []

    def test_a_link_whose_text_contains_brackets_is_still_checked(self, docs):
        page = docs("README.md", "[[2]](docs/missing.md#x)\n")
        assert len(cdl.check(page)) == 1

    def test_an_image_link_is_checked_too(self, docs):
        page = docs("README.md", "![alt text](docs/img/demo.gif)\n")
        assert cdl.check(page) == ["README.md:1: docs/img/demo.gif -> no such file"]

    def test_external_links_are_never_fetched_or_checked(self, docs):
        page = docs("README.md", "[a](https://example.com/x#y) [b](mailto:me@example.com) [c](//cdn.example.com/z)\n")
        assert cdl.check(page) == []

    def test_links_inside_a_code_fence_are_ignored(self, docs):
        page = docs("README.md", "```md\n[example](docs/nope.md)\n```\n")
        assert cdl.check(page) == []

    def test_a_link_title_does_not_break_the_target(self, docs):
        docs("docs/other.md", "# T\n")
        page = docs("README.md", '[x](docs/other.md "a title")\n')
        assert cdl.check(page) == []


class TestMain:
    def test_it_exits_zero_and_says_ok_when_everything_resolves(self, docs, capsys):
        docs("README.md", "# Hi\n\n[self](#hi)\n")
        assert cdl.main([]) == 0
        assert "OK" in capsys.readouterr().out

    def test_it_exits_one_and_lists_every_break(self, docs, capsys):
        docs("README.md", "[a](x.md) [b](y.md)\n")
        assert cdl.main([]) == 1
        out = capsys.readouterr().out
        assert "x.md" in out and "y.md" in out

    def test_it_checks_every_markdown_file_under_docs_by_default(self, docs, capsys):
        docs("README.md", "# ok\n")
        docs("docs/deep/nested.md", "[broken](nope.md)\n")
        assert cdl.main([]) == 1
        assert "docs/deep/nested.md" in capsys.readouterr().out.replace("\\", "/")

    def test_named_files_only_are_checked_when_given(self, docs, capsys):
        docs("README.md", "# ok\n")
        docs("docs/bad.md", "[broken](nope.md)\n")
        assert cdl.main(["README.md"]) == 0


def test_the_projects_own_docs_have_no_broken_internal_links():
    """The real README and docs/: a moved file or renamed heading fails here, and in CI's lint job."""
    assert cdl.main([]) == 0
