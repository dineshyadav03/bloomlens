"""Assemble and commit the file set the Hugging Face Space needs for
BloomLens' live demo, into a local clone of the Space's own git repo.

Run: python scripts/deploy_hf_space.py <path-to-space-clone>

Deliberately does NOT push -- it prints the push command instead, since
pushing to a public Space is a "publish this" action that should be an
explicit, per-run decision, not something a script does silently.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# (source path relative to PROJECT_ROOT, destination path relative to the
# Space clone). Dockerfile.hf/README.hf.md are renamed on the way in --
# the Space's own Dockerfile/README, distinct from the root Dockerfile/
# README.md used by docker-compose (see docs/ARCHITECTURE.md's "Deployment:
# Hugging Face Spaces" section for why they have to differ).
FILES_TO_COPY = [
    ("src", "src"),
    ("data/species_reference.json", "data/species_reference.json"),
    ("scripts/build_index.py", "scripts/build_index.py"),
    ("app.py", "app.py"),
    ("pyproject.toml", "pyproject.toml"),
    ("uv.lock", "uv.lock"),
    (".python-version", ".python-version"),
    ("Dockerfile.hf", "Dockerfile"),
    ("README.hf.md", "README.md"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("space_dir", type=Path, help="Path to a local clone of the HF Space's git repo")
    args = parser.parse_args()

    space_dir = args.space_dir.resolve()

    if space_dir == PROJECT_ROOT:
        sys.exit("Refusing to deploy into the project's own repo -- pass a separate clone of the Space's git repo.")
    if not (space_dir / ".git").is_dir():
        sys.exit(f"{space_dir} doesn't look like a git repo (no .git/) -- clone the Space's remote there first.")

    # Clear the clone's tracked working tree (but not .git/) so a file
    # removed from the source set doesn't linger from a previous deploy.
    for item in space_dir.iterdir():
        if item.name == ".git":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    for src_rel, dest_rel in FILES_TO_COPY:
        src = PROJECT_ROOT / src_rel
        dest = space_dir / dest_rel
        if not src.exists():
            sys.exit(f"Missing expected source file: {src}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(src, dest)

    subprocess.run(["git", "add", "-A"], cwd=space_dir, check=True)
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=space_dir, capture_output=True, text=True, check=True
    )
    if not status.stdout.strip():
        print("Nothing changed -- the Space clone already matches the current source.")
        return

    subprocess.run(["git", "commit", "-m", "Deploy BloomLens"], cwd=space_dir, check=True)
    print(f"Committed in {space_dir}.")
    print(f"Review it with: git -C \"{space_dir}\" show")
    print(f"Then push it yourself when ready: git -C \"{space_dir}\" push")


if __name__ == "__main__":
    main()
