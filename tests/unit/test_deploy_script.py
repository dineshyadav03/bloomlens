"""scripts/deploy_hf_space.py, run for real (as a subprocess) against throwaway git
repos standing in for a Hugging Face Space clone."""

import re
import shutil
import subprocess
import sys

import pytest

from tests.conftest import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "deploy_hf_space.py"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is required")


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def clone(tmp_path):
    """An empty local repo with one commit, standing in for a cloned Space."""
    path = tmp_path / "space"
    path.mkdir()
    git(path, "init", "-q")
    git(path, "config", "user.email", "test@example.com")
    git(path, "config", "user.name", "test")
    git(path, "commit", "-q", "--allow-empty", "-m", "init")
    return path


def deploy(target):
    return subprocess.run([sys.executable, str(SCRIPT), str(target)], capture_output=True, text=True)


def tracked(clone):
    return set(git(clone, "ls-files").splitlines())


def dockerfile_copy_sources():
    """Source paths of every COPY in Dockerfile.hf (flags and the destination dropped)."""
    sources = []
    for line in (REPO_ROOT / "Dockerfile.hf").read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*COPY\s+(.*)", line)
        if match:
            parts = [p for p in match.group(1).split() if not p.startswith("--")]
            sources.extend(parts[:-1])
    return sources


class TestDeploy:
    def test_it_commits_exactly_the_files_a_space_needs(self, clone):
        result = deploy(clone)
        assert result.returncode == 0, result.stderr
        files = tracked(clone)
        assert {
            "Dockerfile",
            "README.md",
            "app.py",
            "pyproject.toml",
            "uv.lock",
            ".python-version",
            "data/species_reference.json",
            "scripts/build_index.py",
            "src/embeddings.py",
        } <= files
        for unwanted in ("api/main.py", "eval/run_eval.py", "docker-compose.yml", "requirements.txt", ".env"):
            assert unwanted not in files

    def test_every_path_dockerfile_hf_copies_is_actually_deployed(self, clone):
        """Regression: Dockerfile.hf ran `COPY scripts/build_index.py` but the deploy
        script never copied that file, so a real Space build would have failed."""
        assert deploy(clone).returncode == 0
        for source in dockerfile_copy_sources():
            assert (clone / source.rstrip("/")).exists(), f"Dockerfile.hf copies {source!r} but it was not deployed"

    def test_the_space_gets_its_own_dockerfile_and_readme(self, clone):
        deploy(clone)
        assert (clone / "Dockerfile").read_bytes() == (REPO_ROOT / "Dockerfile.hf").read_bytes()
        assert (clone / "README.md").read_bytes() == (REPO_ROOT / "README.hf.md").read_bytes()
        assert (clone / "README.md").read_text(encoding="utf-8").startswith("---\n")  # HF frontmatter

    def test_bytecode_caches_are_never_deployed(self, clone):
        (REPO_ROOT / "src" / "__pycache__").mkdir(exist_ok=True)  # exists after any import; make sure
        deploy(clone)
        assert not [f for f in tracked(clone) if "__pycache__" in f or f.endswith(".pyc")]

    def test_files_that_left_the_source_set_do_not_linger(self, clone):
        (clone / "stale.txt").write_text("old")
        git(clone, "add", "-A")
        git(clone, "commit", "-q", "-m", "stale")
        deploy(clone)
        assert "stale.txt" not in tracked(clone)
        assert not (clone / "stale.txt").exists()

    def test_a_second_run_with_nothing_changed_makes_no_commit(self, clone):
        deploy(clone)
        before = git(clone, "rev-parse", "HEAD")
        result = deploy(clone)
        assert result.returncode == 0
        assert "Nothing changed" in result.stdout
        assert git(clone, "rev-parse", "HEAD") == before

    def test_it_commits_but_never_pushes(self, clone, tmp_path):
        remote = tmp_path / "remote.git"
        subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
        git(clone, "remote", "add", "origin", str(remote))
        assert deploy(clone).returncode == 0
        assert git(clone, "log", "-1", "--format=%s") == "Deploy BloomLens"
        assert git(remote, "rev-list", "--all", "--count") == "0"  # the remote received nothing


class TestRefusals:
    def test_it_refuses_to_deploy_into_the_projects_own_repo(self):
        result = deploy(REPO_ROOT)
        assert result.returncode != 0
        assert "Refusing" in result.stderr + result.stdout

    def test_it_refuses_a_directory_that_is_not_a_git_repo(self, tmp_path):
        result = deploy(tmp_path)
        assert result.returncode != 0
        assert "git repo" in result.stderr + result.stdout
        assert list(tmp_path.iterdir()) == []  # and touched nothing
