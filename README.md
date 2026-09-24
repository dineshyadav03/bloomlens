# BloomLens

[![CI](https://github.com/dineshyadav03/bloomlens/actions/workflows/ci.yml/badge.svg)](https://github.com/dineshyadav03/bloomlens/actions/workflows/ci.yml)

Point a camera at a flower and BloomLens proposes a **species**, a visual-condition note and a **simulated** price. It is a portfolio project for the cut-flower supply chain — a vision-retrieval model (BioCLIP 2 + Qdrant) narrows the candidates, and a Gemini tool-calling agent writes the answer.

**The table below is the claim; nothing beyond it should be read into the demo.** Only the retrieval and abstention steps have been measured against ground truth. Quality is an unvalidated heuristic, prices are simulated, and the whole pipeline has been run on a small capped pilot, not benchmarked.

## What is measured (and what isn't)

| Component | Evidence | n | What it shows — and its limits | Status |
|---|---|---:|---|---|
| **Species retrieval** (BioCLIP 2 + Qdrant, zero-shot) | [open_world_results.md](eval/open_world_results.md) (clean, pre-registered test) · [results.md](eval/results.md) (earlier dev set) | 475 test · 360 dev | **82.7% top-1** on the test set; 87.2% top-1 / 96.1% top-3 on the earlier set. Oxford 102 garden and wild photos, **not auction lots**; only **18 of the 30** species have any test photos; weakest on Gladiolus (30% top-1) and Bearded iris (40%) in the earlier set; BioCLIP may have seen Oxford-style images in pretraining. | measured |
| **Abstention** (declining a flower that isn't one of the 30, or isn't a flower) | [open_world_results.md](eval/open_world_results.md) · [PROTOCOL.md](eval/PROTOCOL.md) | 620 near-OOD · 780 far-OOD · 475 in-set | Threshold **tuned on a dev set, frozen, evaluated on the test set exactly once.** Unfamiliar Oxford flowers: AUROC 0.90 [0.83, 0.96], abstains **53%** [42, 64]. Non-flower images: abstains **98%** (an easier case by construction). Wrongly abstains on 4.0% of in-set photos. Corrupted photos: 70–88% top-1. | measured |
| **Whole pipeline** (retrieval + agent + abstention) | [e2e_pilot_results.md](eval/e2e_pilot_results.md) | 119 real calls | 0 failures. Capped pilot: **no confidence intervals and no accuracy claim** — "agrees with retrieval" (92% on in-set photos) is a consistency check between two stages, not correctness. | pilot |
| **Latency and cost** | pilot above · [telemetry](docs/ARCHITECTURE.md#privacy-safe-telemetry--milestone-14) · [Performance](#performance-and-cold-start) | 119 · 24 | Whole scan **p50 11.5 s / p95 30.9 s**, almost all of it the agent's sequential turns; ≈ $0.002 per scan *estimated list-price equivalent* (free tier: $0 billed). One machine. | measured (one machine) |
| **Quality grade** (A/B/C) | [STANDARD.md](docs/quality/STANDARD.md) · [PROTOCOL.md](docs/quality/PROTOCOL.md) | 0 labels | An LLM's visual read. **A/B/C is the original tutorial's invention**, not a market grade; a photo cannot show stem length, strength or vase life. A labeling protocol, agreement metrics and a blinded labeler are built and tested on synthetic data — **no expert has labeled anything.** | **unvalidated** |
| **Price** | [RESEARCH.md](docs/RESEARCH.md) | — | Generated demo data. FloraHolland has no public pricing API. | **simulated** |
| **Lot mode** (up to 10 photos, one consensus) | [DEVELOPMENT.md](docs/DEVELOPMENT.md) | 1 real lot | Demonstrated on one real mixed lot (2 roses + 1 sunflower). No measured accuracy. | not evaluated |
| **"Why this species?" heatmap** (Grad-ECLIP) | [RESEARCH.md](docs/RESEARCH.md#grad-eclip) | 4 photos | No ground-truth metric exists; checked by eye, plus tests for the invariants (e.g. the shared model is never mutated). | qualitative |
| **Security controls** | [SECURITY.md](docs/SECURITY.md) | — | Fail-closed API, upload limits, quotas, no photos kept — each with a test; the same document lists what is *not* defended (the Streamlit UI has no login). Not independently audited. | tested |
| **Live demo** (Hugging Face Space) | [DEVELOPMENT.md](docs/DEVELOPMENT.md) | — | Deploy tooling is built and its image builds in CI; the Space itself has not been created, so **there is no live link and no cold-start figure for it.** | not done |

## Demo

**No recording yet, on purpose.** A demo that is not a real scan through the real UI would be a mock-up, and the live agent could not be reached when this was written: `gemini-3.1-flash-lite` answered every request with HTTP 503 from about 21:30 to at least 22:15 IST on 2026-09-24 (one failed scan from that outage is described under [Limitations](#limitations)). A GIF of a real scan will be added when the model is back. Until then the fastest way to see the app is the [Quick start](#quick-start) — a photo uploaded in the *Single scan* tab, or several in *Lot mode*.

## Sample output

A real result — the in-set scan whose latency is closest to the pilot's median, from the recorded 119-call run ([e2e_pilot_results.md](eval/e2e_pilot_results.md)). Structured fields only: neither the pilot nor telemetry stores anything a model wrote (by design — [PRIVACY.md](docs/PRIVACY.md)), so no summary text is quoted here.

```jsonc
{
  "label": "Rose",                      // ground truth from the dataset
  "final_species": "Rose",
  "retrieval_top1_species": "Rose",
  "confidence_tier": "ambiguous",       // the close-call display tier; not the abstention rule
  "abstained": false,
  "status": "ok",
  "total_ms": 11439,
  "attempts": 1,
  "input_tokens": 8164,
  "output_tokens": 242,
  "est_cost_usd": 0.002404              // estimated list-price equivalent; the free tier billed $0
}
```

**Where the time goes** — the 116 warm scans of that run (3 more were cold), from the closed-schema telemetry it wrote, on the hardware in [Performance](#performance-and-cold-start):

| Stage | p50 | p95 |
|---|---:|---:|
| Embed the photo (BioCLIP 2, CPU) | 0.92 s | 1.36 s |
| Qdrant search | 1 ms | 2 ms |
| Agent (Gemini: 4 model turns, 3 tool calls) | 10.0 s | 30.0 s |
| **Whole scan** | **11.3 s** | **30.9 s** |

About 8.1k tokens in and 226 out per scan. The agent is nearly all of it, which is why the README does not say "instant".

## How it works

```
Flower photo(s) (single, or a "lot" of several)
        │
        ▼
BioCLIP 2 embedding ──► Qdrant vector search ──► top-k candidate species
        │                                             + abstention gate (frozen threshold)
        ▼
LangChain tool-calling agent (Gemini), deciding which tools to use:
  lookup_taxonomy · assess_quality · check_price
  → species/common name, confidence, quality note, price, summary
        │
        ▼ (lot mode: retrieval per photo, one agent call for the whole lot)
Streamlit UI  /  FastAPI JSON endpoint  ·  SQLite inventory log + aggregate telemetry
```

**Built with:** Streamlit · FastAPI · BioCLIP 2 (`open_clip`) · Qdrant (embedded for local use, a real server under Compose) · Google Gemini (`gemini-3.1-flash-lite` via `google-genai`) · LangChain · Docker Compose · Hugging Face Spaces (Docker SDK) · GitHub Actions · uv · ruff. Design and the reasoning behind it: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). How it was built, milestone by milestone: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

- **Species ID** by zero-shot BioCLIP 2 retrieval against a curated taxonomy index of 30 cut-flower species.
- **Agentic layer:** the agent — not a fixed call order — chooses when to look up taxonomy, record a quality read and check the (simulated) price; confirmed from its own message trace.
- **Confidence and abstention:** a close call between candidates shows a switcher instead of a silent guess; a separate machine-readable `abstained` flag (never a keyword search) declines a genuine non-match.
- **Lot mode:** scan or upload up to 10 photos and get a consensus species, an agreement fraction and flagged mismatches from one agent call.
- **Explainability:** an on-demand Grad-ECLIP heatmap of what drove the species match.
- **Inventory and telemetry:** every scan is logged to SQLite; `/metrics` and the Performance expander expose aggregates only.
- **API:** `POST /identify`, `/identify-lot`, `/explain`, `GET /inventory`, `/metrics`, `/health` — the same pipeline the UI uses.

## Quick start

Dependencies are locked in `uv.lock` (hash-pinned, CPU-only PyTorch on Linux/Windows) for Python 3.11–3.13. Install [uv](https://docs.astral.sh/uv/), then:

```bash
uv sync                                          # creates .venv from the lockfile
cp .env.example .env   # then add your GEMINI_API_KEY (free tier: https://aistudio.google.com/apikey)
uv run python scripts/build_index.py             # fetches the pinned model, builds the local Qdrant index + simulated prices
uv run streamlit run app.py
```

The first run downloads the BioCLIP 2 weights (public, no auth): about **1.6 GB** — see [Performance](#performance-and-cold-start). If you ever see `Storage folder ... is already accessed by another instance of Qdrant client`, another Python process from a previous run is still holding the local index — close it and retry (this is exactly why Docker Compose uses a real Qdrant server).

Prefer plain pip? `uv export --no-dev --no-hashes -o requirements.txt` writes a pip-compatible file from the same lock (generated, not committed, so it can't drift). Rebuilding the evaluation test set needs the extra `scipy` dependency: `uv sync --extra eval-data`. The quality-label agreement metrics ([docs/quality/PROTOCOL.md](docs/quality/PROTOCOL.md)) need `uv sync --extra eval-quality`; CI installs it when running the tests.

**The API** needs a key you choose. Add one to `.env` (`BLOOMLENS_API_KEYS=me:<at least 24 characters>`; generate with `python -c "import secrets; print(secrets.token_urlsafe(32))"`), then:

```bash
uv run uvicorn api.main:app --reload
curl -X POST http://127.0.0.1:8000/identify -H "X-API-Key: <the secret>" -F "photo=@your-flower.jpg"
```

Swagger UI is at `http://127.0.0.1:8000/docs` (use *Authorize*); it is off when `BLOOMLENS_ENV=production`. Without a key the API is closed (503), not open. `GET /metrics` (same key) returns aggregate latency, failure and cost figures only; `scripts/report_metrics.py` prints them and `scripts/purge_data.py` enforces retention ([PRIVACY.md](docs/PRIVACY.md)). The pipeline routes are rate- and quota-limited per key (429 with `Retry-After`), and the Streamlit app per browser session, with a global daily ceiling — shared through the inventory database, so it holds across the `web` and `api` processes on one host ([SECURITY.md](docs/SECURITY.md)). The `summary`, `quality_note` and `confidence_note` fields are written by a language model: treat them as untrusted input.

**With Docker Compose** (Qdrant + the Streamlit app + the API):

```bash
cp .env.example .env   # add your GEMINI_API_KEY first
docker compose up
```

Streamlit at `http://localhost:8501`, the API at `http://localhost:8000` (both bound to the host's loopback; the containers run as a non-root user with no extra privileges). Set `BLOOMLENS_API_KEYS` in `.env` first or the API answers 503; Compose runs it with `BLOOMLENS_ENV=production`, so `/docs` is off unless you set `BLOOMLENS_ENV=development`. The first run downloads the BioCLIP 2 weights into a shared volume and reuses them after that.

**Upgrading from an earlier checkout?** The images used to run as root with the model cache at `/root/.cache/huggingface`; they now run as uid 1000 with it at `/home/user/.cache/huggingface`. Volumes made by the old images are root-owned — fix them once (this keeps the downloaded weights; the `--cap-add`s are needed because the service drops all capabilities):

```bash
docker compose run --rm --no-deps --user root --cap-add CHOWN --cap-add DAC_OVERRIDE web sh -c "chown -R 1000:1000 /home/user/.cache/huggingface /app/inventory_data"
```

## Supported environments

Stated from what [CI](.github/workflows/ci.yml) actually runs, not from what should work:

| Platform | Python | Checked on every PR that changes code or dependencies (documentation-only PRs skip these) |
|---|---|---|
| Linux (Ubuntu x86-64) | 3.11, 3.12, 3.13 | locked install, `uv pip check`, importing every project module, and the unit tests (no model weights, loopback-only network) |
| Windows (x86-64) | 3.13 | same |
| macOS (Apple silicon) | 3.13 | same |
| Linux, 3.13 only | — | the full test suite including integration tests against the real BioCLIP 2 weights and a real Qdrant index (**98.58% coverage** on the last run on `main`; CI fails below 95%); the retrieval eval with an 80% top-1 gate; the Docker Compose stack's `/health`; the Hugging Face Space image build |

**Not covered by CI:** model inference on Windows, macOS, or Python 3.11/3.12 (a one-off manual check on Windows with 3.11, 3.12 and 3.13 loaded the pinned model and got identical scores, but nothing re-runs it), live Gemini calls (none, by design: the agent is faked and tests are blocked from the network), the single-scan camera/upload path in a browser (Streamlit's test harness can't drive those widgets; it is covered at function level and by hand), Intel macOS, Linux ARM, and GPUs (PyTorch is CPU-only). Python 3.14 is unsupported (`torch==2.6.0` has no cp314 wheels); anything below 3.11 is untested.

**Tests:** 1,457 in total. `uv run pytest -m "not integration"` is the fast unit run (1,390 tests; about 5 minutes on the dev machine; no weights); `uv run pytest` also runs the integration tests, which need the pinned weights (`scripts/build_index.py` fetches them). Tests are hermetic: no real network, a developer's `GEMINI_API_KEY` is stripped, and every database lives in a temp directory.

## Performance and cold start

Measured on one machine — an **Acer Swift SFG14-73 laptop: Intel Core Ultra 7 155H (16 cores / 22 threads), 15.7 GB RAM, Windows 11, Python 3.13.4, CPU-only PyTorch 2.6.0** (the Arc iGPU is unused). Nothing here is a benchmark of other hardware.

**Cold start of the retrieval stage** — a fresh process with the weights already on disk, from [`scripts/measure_startup.py`](scripts/measure_startup.py), **six separate runs** (the range is the honest answer: this laptop varies a lot, and the slowest run coincided with Docker Desktop starting in the background):

| Step | Range over 6 runs |
|---|---|
| `import torch` | 5.7 – 8.8 s |
| import the project's modules (`open_clip`, Qdrant client, …) | 6.6 – 16.6 s |
| load BioCLIP 2 onto the CPU | 6.5 – 15.2 s |
| first image embedding | 1.4 – 2.6 s |
| open the local Qdrant index + first search | 0.05 – 0.15 s |
| **process start → first retrieval result** | **21.7 – 43.2 s (median 24.4 s)** |

Once warm, embedding one photo takes **1.0 – 1.5 s** here (per-run medians of ten; an earlier 24-scan telemetry sample on a quieter machine state measured 0.57 s) and a Qdrant search about **1 ms**. The Streamlit process holds about **1.6 GB** of RAM after the model loads.

**A whole scan, with the agent:** warm **p50 11.5 s / p95 30.9 s** over the 119-call pilot ([e2e_pilot_results.md](eval/e2e_pilot_results.md)); an earlier 24-scan telemetry sample measured warm p50 15.1 s / p95 66.5 s and a single cold scan at 28.8 s ([details](docs/ARCHITECTURE.md#privacy-safe-telemetry--milestone-14)). Nearly all of that is the Gemini agent's sequential turns (median ≈ 14.5 s in the telemetry sample) — retrieval is a small part, so a scan is **not instant**. The p95 rests on a few slow scans on one machine.

**What you download and store:**

- BioCLIP 2 weights: **1,631 MB** (`open_clip_model.safetensors`, fetched once, checksum-verified). At the ~150 KB/s this project's development network managed, that is hours; CI caches it.
- Docker images as last built on this machine (`docker images`; built 8–13 days before this measurement and not rebuilt for it): the `web`, `api` and `build-index` images are **2.39 GB** each (shared layers), the Hugging Face Space image **4.87 GB** (weights baked in), and `qdrant/qdrant` 288 MB. Compose builds the first three from source; it does not download them.
- **Hugging Face Space cold start: not measured** — no Space exists yet.

## Security and privacy

- **Fail-closed API:** every route except `/health` needs an `X-API-Key` and answers `503` until keys are configured; uploads are size-, format- and pixel-capped and stripped of EXIF/GPS; per-key and per-session rate limits and daily quotas sit in shared storage. Details, tests and what is *not* defended: [docs/SECURITY.md](docs/SECURITY.md).
- **What leaves the machine:** each photo (re-encoded, metadata removed) and the candidate species names go to Google's Gemini API to write the quality read and summary; nothing else, and BloomLens keeps no photos. On Google's free tier submitted content may be used to improve Google products and reviewed by people, and users in the EEA, UK and Switzerland are not covered. Tracing to third parties is forced off. Exactly what is sent and stored: [docs/PRIVACY.md](docs/PRIVACY.md).
- **Telemetry** is a closed schema enforced by the database — numbers, enums and versions, never a photo, prompt, answer, address or key — exposed as aggregates only, with retention limits.
- **The Streamlit UI has no login of its own** — read [docs/SECURITY.md](docs/SECURITY.md) before exposing it beyond your machine.

## Limitations

- **Pricing is simulated.** Any price shown is generated demo data, clearly labeled — not a real auction price.
- **Quality is an unvalidated heuristic**, not a calibrated grading system, and A/B/C is not a market grade ([STANDARD.md](docs/quality/STANDARD.md)).
- **Coverage is a curated demo set of 30 species**, and only 18 of them have evaluation photos. Every accuracy figure is on Oxford 102 garden and wild photos, never on auction lots.
- **It is slow, and depends on Gemini.** A whole scan takes seconds to tens of seconds because the agent takes several sequential turns; it is not instant. When Gemini is down or rate-limited the scan fails — during a real Gemini 503 outage on 2026-09-24 a scan failed after about a minute — it took 286 s before a retry fix made in this milestone.
- **Model text is untrusted** and shown literally, never as markdown or HTML.
- **One host.** Rate limits and the inventory are correct for processes sharing one volume; there is no multi-replica story.
- **Known follow-ups** (not built): the heatmap in lot mode, and a pricing-CSV staleness bug found while adding the inventory volume — [ARCHITECTURE.md](docs/ARCHITECTURE.md#future-work-not-in-this-build-noted-for-later).
- **No live demo yet** (the Space is not created). This project is not affiliated with Royal FloraHolland, the Museum of the Future, or the Imageomics Institute — all are credited as inspiration and research sources only.

## Background

Royal FloraHolland, the flower auction in Aalsmeer, is the world's largest flower market: roughly **43 million flowers and 5 million plants** change hands on a typical weekday, across 35 auction clocks and more than 30,000 varieties [[1]](docs/RESEARCH.md#floraholland). At that pace a buyer has no realistic way to check species, quality and a fair price by eye — BloomLens's premise is one photo for all three (the measured table above says how far it gets).

The scan-to-learn interaction is inspired by "The Library" at Dubai's Museum of the Future [[2]](docs/RESEARCH.md#museum-of-the-future); the technical backbone is BioCLIP / BioCLIP 2, open-source vision foundation models for the tree of life (CVPR'24 Best Student Paper; NeurIPS'25 Spotlight) [[3]](docs/RESEARCH.md#bioclip). Sources and the research trail: [docs/RESEARCH.md](docs/RESEARCH.md).

## License

[MIT](LICENSE)
