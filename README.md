# BloomLens

[![CI](https://github.com/dineshyadav03/bloomlens/actions/workflows/ci.yml/badge.svg)](https://github.com/dineshyadav03/bloomlens/actions/workflows/ci.yml)

Point a camera at a flower and BloomLens proposes a **species**, a visual-condition note, and a **simulated** price. **Only the species step has been measured:** retrieval-stage classification (BioCLIP 2 + Qdrant, zero-shot) scores **87.2% top-1 / 96.1% top-3** on 360 Oxford 102 Flowers test photos covering **18 of the 30** supported species ([eval/results.md](eval/results.md); in-distribution). A separate, pre-registered open-world evaluation ([eval/PROTOCOL.md](eval/PROTOCOL.md), [eval/open_world_results.md](eval/open_world_results.md)) then asked what happens on photos that *aren't* one of those species: on a genuinely unfamiliar Oxford flower category, the system now correctly abstains **53%** of the time (AUROC 0.90); on a different-source non-flower image it abstains **98%** of the time (an easier case by construction). Auction-lot photos are still untested.

**Not measured or not real:** the quality grade is an *unvalidated* LLM heuristic (no expert labels exist), and prices are *simulated* (FloraHolland has no public pricing API). End-to-end latency and behavior *have* been measured, but only on a small, capped pilot — 119 real `identify()` calls (BioCLIP + Qdrant + the live Gemini agent) on a fixed sample: **p50 11.5 s / p95 30.9 s** latency, 0 failures, ≈$0.0021 estimated list-price equivalent per call (free tier: real bill $0) — see [eval/e2e_pilot_results.md](eval/e2e_pilot_results.md). It is a pilot, not a benchmark: no confidence intervals, no full-system accuracy claim (there's no ground truth for what an out-of-distribution photo's "right" answer would be). A per-claim evidence table is planned; until then treat everything beyond the species-retrieval number and this pilot as a demo.

BloomLens is a portfolio project for the cut-flower supply chain, inspired by the scale of the world's largest flower auction and the "scan to learn" interaction from Dubai's Museum of the Future. It contains the core pipeline, an agentic reasoning layer (a LangChain tool-calling agent, not a fixed call order), lot/batch mode, a retrieval evaluation harness, a FastAPI endpoint alongside the Streamlit UI, and a Docker Compose setup with CI — see [Status](#status) below for how each piece was verified.

## The problem

Royal FloraHolland, the flower auction in Aalsmeer, Netherlands, is the world's largest flower market: roughly **43 million flowers and 5 million plants** change hands there on a typical weekday, traded across 35 auction clocks and spanning more than 30,000 varieties, inside a nearly 1 million m² facility. [[1]](docs/RESEARCH.md#floraholland)

At that volume and speed, a buyer looking at a lot on the auction floor has no realistic way to manually verify what species it is, how good its quality is, and what a fair price looks like. BloomLens's premise: let a buyer take one photo and get all three back immediately.

## Inspiration

- **The scan-to-learn interaction** is inspired by "The Library" at Dubai's Museum of the Future — 2,400 laser-etched specimen jars that visitors scan with a handheld device to instantly pull up each species' story. [[2]](docs/RESEARCH.md#museum-of-the-future) BloomLens reproduces that *interaction pattern* (point at something living → get an instant info card) as a web app, not the physical installation.
- **The technical backbone** draws on BioCLIP / BioCLIP 2, real open-source vision foundation models for the tree of life, published at CVPR'24 (Best Student Paper) and NeurIPS'25 (Spotlight). [[3]](docs/RESEARCH.md#bioclip) These make it possible to build the "vector database of botanical species embeddings" for real, rather than mocking it.

See [docs/RESEARCH.md](docs/RESEARCH.md) for the full writeup with sources, and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the planned system design.

## Architecture

```
Flower photo(s) (single, or a "lot" of several)
        │
        ▼
BioCLIP 2 embedding ──► Qdrant vector search ──► top-k candidate species
        │                                             + confidence gate
        ▼
LangChain tool-calling agent (Gemini), deciding which tools to use:
  lookup_taxonomy · assess_quality · check_price
  → species/common name, confidence, quality note, price, summary
        │
        ▼ (lot mode: retrieval per photo, one agent call for the whole lot)
Streamlit UI  /  FastAPI JSON endpoint (API planned)
```

Full breakdown, including the evaluation harness and Docker/CI setup, in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Features

- **Species ID** via BioCLIP 2 zero-shot classification against a curated taxonomy vector index (Qdrant) — **measured 87.2% top-1 / 96.1% top-3 accuracy** on 360 held-out test images across 18 species ([eval/results.md](eval/results.md)), consistent with the 91.4% zero-shot PlantNet number reported in the original paper.
- **Agentic reasoning layer**: a LangChain agent (not a fixed pipeline) — confirmed via its own message trace to genuinely call `lookup_taxonomy`, `assess_quality`, and `check_price` on a normal run, deciding for itself when to use each. Matches the "AI Agent System" framing from the original tutorial this project was inspired by.
- **Confidence handling**: a close call between top candidates surfaces a visible switcher instead of a silently wrong guess. The tier thresholds were checked against real held-out data and confirmed well-calibrated — see [eval/results.md](eval/results.md)'s calibration section. Separately, a machine-readable `abstained` flag (never a keyword search) tells apart a genuine non-match: its threshold was **tuned on a dev set, frozen, and evaluated on a held-out test set exactly once** ([eval/PROTOCOL.md](eval/PROTOCOL.md), [eval/open_world_results.md](eval/open_world_results.md)) — 53% correct-rejection on unfamiliar flower categories, 98% on non-flower images.
- **Lot/batch mode**: scan or upload up to 10 photos as one lot and get a consensus species, agreement fraction, flagged mismatches, and a price trend chart — one agent call for the whole lot, not one per photo.
- **FastAPI endpoint** (`api/main.py`) alongside the Streamlit UI: `POST /identify`, `POST /identify-lot`, `POST /explain`, `GET /inventory`, `GET /health` — sharing `src/identify.py`'s pipeline directly (no second implementation). **Fail-closed**: every route except `/health` needs an `X-API-Key` and answers `503` until keys are configured; uploads are size-, format- and pixel-capped and stripped of EXIF/GPS. See [docs/SECURITY.md](docs/SECURITY.md) (controls, tests, and what is *not* defended) and [docs/PRIVACY.md](docs/PRIVACY.md) (what is sent to Google's Gemini API).
- **One-command setup**: `docker compose up` runs Qdrant (a real server, not local mode — see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for why that distinction matters once two services share one index), the Streamlit UI, and the FastAPI service together. CI (`.github/workflows/ci.yml`) lints, runs the evaluation harness with a real regression gate (fails under 80% top-1 accuracy), and smoke-tests the full Docker stack on every push.
- **"Why this species?" interpretability overlay**: an on-demand heatmap (via `src/interpretability.py`) showing which part of a scanned photo most drove its species match — **Grad-ECLIP**, a real published technique for CLIP-style zero-shot vision-language models (not literal Grad-CAM, which needs a CNN and a classifier head neither of which BioCLIP 2 has). See [docs/RESEARCH.md](docs/RESEARCH.md#grad-eclip) for the real implementation bugs (including a thread-safety one) found and fixed by checking against the paper's own reference code.
- **Persistent inventory log**: every scan (single or lot) is recorded automatically — species, quality, price, timestamp — into a SQLite database (`src/inventory.py`, WAL mode for safe concurrent writes across the `web`/`api` containers). A new "Inventory" tab shows a running per-species count and recent-scan history; `GET /inventory` keeps the API at parity.

## Tech stack

**Built:** Streamlit · FastAPI · BioCLIP 2 (open-source vision foundation model, `open_clip`) · Qdrant (local embedded mode for dev and the live demo, a real server under Docker Compose) · Google Gemini (`gemini-3.1-flash-lite` via `google-genai`) · LangChain (`langchain` + `langchain-google-genai`, agentic tool-calling) · Docker Compose · Hugging Face Spaces (Docker SDK) · GitHub Actions · ruff

## Running it

**Locally:**

Dependencies are locked in `uv.lock` (hash-pinned, CPU-only PyTorch on Linux/Windows) for Python 3.11–3.13. Install [uv](https://docs.astral.sh/uv/), then:

```bash
uv sync                                          # creates .venv from the lockfile
cp .env.example .env   # then add your GEMINI_API_KEY (free tier: https://aistudio.google.com/apikey)
uv run python scripts/build_index.py             # builds the local Qdrant species index + simulated prices
uv run streamlit run app.py
```

Prefer plain pip? `uv export --no-dev --no-hashes -o requirements.txt` writes a pip-compatible file from the same lock (it is generated, not committed, so it can't drift). Only rebuilding the evaluation test set needs the extra `scipy` dependency: `uv sync --extra eval-data`. The quality-label agreement metrics (`eval/quality_agreement.py`, [docs/quality/PROTOCOL.md](docs/quality/PROTOCOL.md)) need `uv sync --extra eval-quality` (krippendorff, scikit-learn, statsmodels) — CI installs this extra automatically when running the test suite.

**Supported environments** — stated from what [CI](.github/workflows/ci.yml) actually runs, not from what should work:

| Platform | Python | Checked on every PR that changes code or dependencies (documentation-only PRs skip these) |
|---|---|---|
| Linux (Ubuntu x86-64) | 3.11, 3.12, 3.13 | locked install, `uv pip check`, importing every project module, and the unit tests (no model weights, loopback-only network) |
| Windows (x86-64) | 3.13 | same |
| macOS (Apple silicon) | 3.13 | same |
| Linux, 3.13 only | — | the full test suite including integration tests against the real BioCLIP 2 weights and a real Qdrant index (98% branch coverage measured; CI fails below 95%); BioCLIP 2 + Qdrant retrieval eval with an 80% top-1 gate; Docker Compose stack `/health`; Hugging Face Space image build |

**Not covered by CI:** model inference on Windows, macOS, or Python 3.11/3.12 (a one-off manual check on Windows with 3.11, 3.12 and 3.13 loaded the pinned model and got identical scores, but nothing re-runs it), live Gemini calls (none, by design: the agent is faked and tests are blocked from reaching the network), the single-scan camera/upload path in the browser (Streamlit's test harness can't drive those widgets; that path is covered at function level and by hand), Intel macOS, Linux ARM, and GPUs (PyTorch is CPU-only). Python 3.14 is unsupported (`torch==2.6.0` has no cp314 wheels); anything below 3.11 is untested.

**Tests:** `uv run pytest -m "not integration"` is the fast unit run (~70 s on the dev machine, of which ~35 s is two tests that start a fresh interpreter to prove tracing stays off; no weights). `uv run pytest` also runs the integration tests, which need the pinned BioCLIP 2 weights (run `uv run python scripts/build_index.py` once to fetch them) and take a few minutes. Tests are hermetic: no real network (only loopback is allowed), a developer's `GEMINI_API_KEY` is stripped, and every database lives in a temp directory.

First run downloads BioCLIP 2 weights (public, no auth needed). If you ever see `Storage folder ... is already accessed by another instance of Qdrant client`, another Python process from a previous run is still holding the local index — close it and retry (this is exactly why Docker Compose uses a real Qdrant server instead, see below).

**The API** needs a key you choose. Add one to `.env` (`BLOOMLENS_API_KEYS=me:<at least 24 characters>`; generate with `python -c "import secrets; print(secrets.token_urlsafe(32))"`), then:

```bash
uv run uvicorn api.main:app --reload
curl -X POST http://127.0.0.1:8000/identify -H "X-API-Key: <the secret>" -F "photo=@your-flower.jpg"
```

Swagger UI is at `http://127.0.0.1:8000/docs` (use *Authorize* to enter the key); it is switched off when `BLOOMLENS_ENV=production`. Without a key the API is closed (503), not open. `GET /metrics` (same key) returns aggregate latency/failure/cost figures only; `uv run python scripts/report_metrics.py` prints them, and `scripts/purge_data.py` enforces retention (see [docs/PRIVACY.md](docs/PRIVACY.md)). The routes that run the pipeline are also rate- and quota-limited per key (429 with `Retry-After`), and the Streamlit app per browser session, with a global daily ceiling on top — shared through the inventory database, so it holds across the `web` and `api` processes on one host; see [docs/SECURITY.md](docs/SECURITY.md) for the limits, their defaults and what they do not cover. The text fields in responses (`summary`, `quality_note`, `confidence_note`) are written by a language model — treat them as untrusted input in whatever consumes them.

**With Docker Compose** (runs Qdrant + the Streamlit app + the API together):

```bash
cp .env.example .env   # add your GEMINI_API_KEY first
docker compose up
```

Streamlit at `http://localhost:8501`, the API at `http://localhost:8000` (both bound to the host's loopback only; the containers run as a non-root user with no extra privileges). Set `BLOOMLENS_API_KEYS` in `.env` first or the API answers 503; compose runs it with `BLOOMLENS_ENV=production`, so `/docs` is off unless you set `BLOOMLENS_ENV=development` in `.env`. First run downloads BioCLIP 2 weights into a shared volume (a real download — this project's dev network measured ~150KB/s, so expect it to take a while the first time; cached for every run after).

**Upgrading from an earlier checkout?** The images used to run as root and mount the model cache at `/root/.cache/huggingface`; they now run as uid 1000 with the cache at `/home/user/.cache/huggingface`. Volumes created by the old images are root-owned, so fix them once (this keeps the downloaded weights; the `--cap-add`s are needed because the service drops all capabilities):

```bash
docker compose run --rm --no-deps --user root --cap-add CHOWN --cap-add DAC_OVERRIDE web sh -c "chown -R 1000:1000 /home/user/.cache/huggingface /app/inventory_data"
```

Note the Streamlit UI has no login of its own — see [docs/SECURITY.md](docs/SECURITY.md) before exposing it beyond your machine.

## Important disclaimers

- **Pricing is simulated.** FloraHolland does not expose a public real-time pricing API. Any price shown by this project is generated demo data, clearly labeled as such — not a real auction price.
- **Quality assessment is a heuristic**, not a calibrated agronomic grading system.
- **Species coverage is a curated demo-sized subset** (~30–50 common cut-flower species), not the full biodiversity scale of the underlying research models.
- This project is not affiliated with Royal FloraHolland, the Museum of the Future, or the Imageomics Institute — all are credited as inspiration/research sources only.

## Future work

All three items originally noted here (live demo, interpretability overlay, inventory log) are now built — see Milestones 8–10 below. Two smaller items surfaced along the way, documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#future-work-not-in-this-build-noted-for-later) rather than fixed immediately: wiring the "why this species?" overlay into lot mode, and a pre-existing pricing-CSV staleness bug found (not caused) while building the inventory log's Docker Compose volume.

## Status

- ✅ **Phase A** — research, architecture, and this documentation.
- ✅ **Milestone 1** — core pipeline: camera scan → BioCLIP 2 species ID → Gemini quality read → simulated price, in a working Streamlit app. Verified end-to-end on real flower photos.
- ✅ **Milestone 2** — confidence gating: three tiers from Qdrant's top-1/top-2 score gap (`high`/`ambiguous`/`low`, thresholds in `src/identify.py`). Ambiguous scans show a visible switcher between the top-2 candidates (updates species/price with no extra Gemini call); low-confidence scans get a hedged message instead of a confident species name.
- ✅ **Milestone 3** — agentic layer: a LangChain (`create_agent`) tool-calling agent over Gemini replaces the fixed call, with `lookup_taxonomy`/`assess_quality`/`check_price` as real tools (`src/tools.py`). Verified via the agent's own message trace, not just its output, that it genuinely calls all three. Found along the way that an agentic `identify()` call costs 4+ Gemini calls instead of 1, which exhausted `gemini-3.6-flash`'s daily free quota — switched to `gemini-3.1-flash-lite`, which tracks separate, more generous quota (see [docs/RESEARCH.md](docs/RESEARCH.md)).
- ✅ **Milestone 4** — lot/batch mode: scan or upload up to 10 photos as one lot in the new "Lot mode" tab. Redesigned from the original plan for a real reason found in Milestone 3 — running the full agent per photo would cost 40+ Gemini calls for a 10-photo lot, so retrieval runs per-photo (free/local) for a majority-vote consensus + agreement fraction, while the agent runs **once** for the whole lot given all photos in one multi-image message. Mismatched photos are flagged individually; a price trend chart uses `src/pricing.price_history`. Verified with a real mixed lot (2 roses + 1 sunflower): correct 2/3 consensus, correct flag, and the agent's own summary independently corroborated the mismatch.
- ✅ **Milestone 5** — evaluation harness: **87.2% top-1 / 96.1% top-3 accuracy** on 360 held-out Oxford 102 Flowers images across 18 of the 30 curated species ([eval/results.md](eval/results.md); species chosen via individually-verified alternate common names, not string matches — see `eval/species_mapping.py`). Zero API cost — evaluates only BioCLIP 2 + Qdrant retrieval, same as `identify()` uses. Also swept alternative confidence-tier thresholds against the real data (the "revisit" Milestone 2 promised) and found the current ones are already well-calibrated — loosening them would trade meaningful precision for coverage, not a free win — so they were kept as-is, backed by data instead of the original guess.
- ✅ **Milestone 6** — FastAPI endpoint: `GET /health`, `POST /identify`, `POST /identify-lot` (`api/main.py`), returning the exact same `IdentifyResult`/`LotResult` Pydantic models the Streamlit app renders — no second implementation. Verified end-to-end with real photos (identical results to calling the pipeline directly), plus the error paths: oversized lot → `400`, missing API key → `503` with a clear message (not a raw traceback), and the auto-generated Swagger UI at `/docs`.
- ✅ **Milestone 7** — Docker Compose + CI, the last originally-planned milestone: `docker compose up` runs Qdrant (a real server), the Streamlit app, and the FastAPI service together. Qdrant had to become a real server specifically because two containers can genuinely run at once — local/embedded mode's file lock is the exact bug hit in Milestone 1; `get_client()` now branches on a `QDRANT_URL` env var, invisibly to every caller. `eval/test_images/` (15MB) is now committed, reversing a Milestone 5 call, so CI's evaluation job needs no external network access. CI runs `ruff` (checked first — nearly clean already, one real line-length fix applied) and the evaluation harness with an actual regression gate (fails under 80% top-1 accuracy, verified by testing it actually trips), plus a Docker smoke test that builds and health-checks the full stack on GitHub's network rather than this machine's slow one.
- 🟡 **Milestone 8** — live demo on Hugging Face Spaces, deploy tooling done, link pending: a separate, HF-specific `Dockerfile.hf` (non-root uid 1000, BioCLIP 2 weights + the Qdrant index baked in at build time for an instant cold start — the opposite call from Milestone 7's local-network-driven decision, and correct here since HF's build servers pull weights from the HF Hub itself), a Space-only `README.hf.md` (so the GitHub README stays free of Hugging Face's required YAML frontmatter, which GitHub doesn't render specially), and `scripts/deploy_hf_space.py` to assemble and commit the Space's file set. Verified for real: built the image locally, ran it as a container, and called `identify()` against a real test photo inside it — correctly returned Amaryllis (species, quality grade, price, and summary all populated), with no first-run delay since the weights/index were already baked in. CI's new `hf-space-build` job builds `Dockerfile.hf` fresh on every push. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#deployment-hugging-face-spaces--implemented-milestone-8) for the full reasoning. **What's left is account-bound, not code**: creating the Space, adding the `GEMINI_API_KEY` secret, and pushing are manual steps only the account owner can do — this line gets replaced with the real link once that's done.
- ✅ **Milestone 9** — interpretability overlay: an on-demand heatmap ("🔍 Why this species?" in the app, `POST /explain` in the API) showing which part of a photo drove its species match, via **Grad-ECLIP** — a real published technique for CLIP-style zero-shot models (not literal Grad-CAM, which needs a CNN and a classifier head, neither of which BioCLIP 2 has). Verified the formula against the paper's own reference implementation and fixed two real bugs a first attempt got wrong (wrong gradient basis producing background-focused heatmaps; a missing normalization step), plus a thread-safety issue caught before shipping (an early version mutated the shared BioCLIP model singleton, which would have corrupted a concurrent scan from another user — rewritten to avoid any shared-state mutation, verified safe with a real concurrent-call test). No ground-truth metric exists for this, unlike the species-ID eval harness — verified by eye on real photos (Amaryllis, Sunflower, Rose, Carnation): heatmaps concentrate on each flower's distinguishing features and shift when explaining a different candidate species. See [docs/RESEARCH.md](docs/RESEARCH.md#grad-eclip) for the full research trail.
- ✅ **Milestone 10** — persistent inventory log, the last originally-requested item: every scan (single or lot) is logged automatically into SQLite (`src/inventory.py`) — species, quality, price, timestamp — with a new "Inventory" tab (running per-species count + recent-scans table) and matching `GET /inventory`/`GET /inventory/species-counts` routes. Uses **WAL mode**, not a new database service — the same class of multi-process concurrency risk Qdrant's local mode hit in Milestone 1 and Docker Compose's separate `web`/`api` containers made real in Milestone 7, solved this time without a server. Verified for real, not just assumed: an 8-threads-×-10-writes concurrency test with no locked-database errors, and a genuine cross-container test — wrote a row from the `web` container, read it back via the `api` container's own HTTP endpoint over the new shared `inventory_data` Docker volume, confirming it actually solves the problem it was designed for. Found (but deliberately didn't fix, to keep scope tight) a pre-existing staleness bug in how `docker-compose.yml` shares — or doesn't — the pricing CSV; see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#future-work-not-in-this-build-noted-for-later).

All 7 originally-planned milestones are complete, plus Milestones 9 and 10; Milestone 8's deploy tooling is built and verified, with the live link itself pending the manual account steps above. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full history.

## License

[MIT](LICENSE)
