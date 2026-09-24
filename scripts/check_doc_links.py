"""Check that every relative link (and #anchor) in the project's markdown docs resolves.

    uv run python scripts/check_doc_links.py            # README.md and docs/**/*.md
    uv run python scripts/check_doc_links.py README.md  # or just the files named

Only internal links are checked -- a file path that must exist, and a `#heading` that must be one of
that file's headings under GitHub's slug rules (or an explicit `<a id="...">`; kramdown/Pandoc's
`{#custom-id}` heading syntax is NOT honoured by GitHub, so it does not count). External URLs are
not fetched (no network, and a flaky third-party site should not fail a build). Exits 1 and lists
every broken link.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXT = r"(?:[^\[\]]|\[[^\]]*\])*"  # link text, allowing one level of nested brackets: [[2]](url)
LINK = re.compile(r"(?<!\!)\[" + TEXT + r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)|!\[" + TEXT + r"\]\(([^)\s]+)\)")
HTML_ANCHOR = re.compile(r'<a\s+(?:id|name)="([^"]+)"')
FENCE = re.compile(r"^\s*(```|~~~)")


def slug(heading: str) -> str:
    """GitHub's heading anchor: lowercase, drop everything but letters/digits/space/hyphen/underscore,
    spaces to hyphens (so "A — B" becomes "a--b": the dash is dropped, both spaces stay)."""
    heading = re.sub(r"[`*~]|<[^>]+>", "", heading.strip().lower())
    heading = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    return re.sub(r"[^\w\- ]", "", heading).replace(" ", "-")


def headings(path: Path) -> set[str]:
    found: set[str] = set()
    counts: dict[str, int] = {}
    in_fence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
        if in_fence:
            continue
        found.update(HTML_ANCHOR.findall(line))  # <a id="x"></a> works on GitHub; kramdown's {#x} does not
        if not line.startswith("#"):
            continue
        base = slug(line.lstrip("#"))
        seen = counts.get(base, 0)
        counts[base] = seen + 1
        found.add(base if seen == 0 else f"{base}-{seen}")
    return found


def check(path: Path) -> list[str]:
    problems = []
    in_fence = False
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if FENCE.match(line):
            in_fence = not in_fence
        if in_fence:
            continue
        for match in LINK.finditer(line):
            target = match.group(1) or match.group(2)
            if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("//"):
                continue  # http(s), mailto, ...
            file_part, _, anchor = target.partition("#")
            destination = (path.parent / file_part).resolve() if file_part else path
            where = f"{path.relative_to(ROOT)}:{number}"
            if not destination.exists():
                problems.append(f"{where}: {target} -> no such file")
            elif anchor and destination.suffix == ".md" and anchor not in headings(destination):
                problems.append(f"{where}: {target} -> no heading #{anchor} in {destination.relative_to(ROOT)}")
    return problems


def main(argv: list[str]) -> int:
    files = [ROOT / a for a in argv] if argv else [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    problems = [p for f in files for p in check(f)]
    print("\n".join(problems) if problems else f"OK: {len(files)} file(s), no broken internal links")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
