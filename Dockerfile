# Single image; app.py (Streamlit) vs api/main.py (FastAPI) is chosen per
# docker-compose service via `command:` override, not a second Dockerfile.
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

# CPU-only torch/torchvision, same index used in local dev -- avoids pulling
# huge CUDA wheels neither service needs (no GPU in this project, see
# docs/RESEARCH.md's Milestone 1 notes on why that's a non-issue here).
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# BioCLIP 2 weights (a real download, a few hundred MB) are fetched at first
# run, not baked in here at build time. This machine's network made a
# build-time download impractical (see docs/ARCHITECTURE.md); docker-compose
# mounts a shared `hf_cache` volume so it's still only a one-time cost across
# container restarts. Revisit baking them in at build time if this ever runs
# somewhere with faster bandwidth (e.g. CI, or a real deploy).

EXPOSE 8501 8000
