# Single image; app.py (Streamlit) vs api/main.py (FastAPI) is chosen per
# docker-compose service via `command:` override, not a second Dockerfile.
FROM python:3.13-slim

WORKDIR /app

# Dependencies come from uv.lock (hash-pinned, CPU-only torch -- no GPU anywhere
# in this project, see docs/RESEARCH.md's Milestone 1 notes). The venv lives
# outside /app so bind mounts / volumes over /app can never shadow it.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_HTTP_TIMEOUT=120 \
    UV_HTTP_CONNECT_TIMEOUT=60 \
    UV_HTTP_RETRIES=8 \
    PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir uv==0.11.29

# Copy only the dependency manifests first so this (slow) layer is cached until
# the lock changes, not on every source edit. uv's wheel cache is a BuildKit
# cache *mount*: it survives failed/retried builds (a flaky download doesn't
# restart from zero) but never becomes part of an image layer -- baking it in
# added 1.2 GB to the image, found by measuring it. Longer timeouts/retries
# because a plain connect timeout to PyPI failed a build once.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

COPY . .

# BioCLIP 2 weights (a real download, a few hundred MB) are fetched at first
# run, not baked in here at build time. This machine's network made a
# build-time download impractical (see docs/ARCHITECTURE.md); docker-compose
# mounts a shared `hf_cache` volume so it's still only a one-time cost across
# container restarts. Revisit baking them in at build time if this ever runs
# somewhere with faster bandwidth (e.g. CI, or a real deploy).

EXPOSE 8501 8000
