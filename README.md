# BloomLens

Point a camera at a flower, get its **species, quality, and price** back — instantly.

BloomLens is a portfolio project for the cut-flower supply chain, inspired by the scale of the world's largest flower auction and the "scan to learn" interaction from Dubai's Museum of the Future. The core pipeline is built and working — including an agentic reasoning layer (Milestone 3): a LangChain tool-calling agent, not a fixed call order. Lot mode, an evaluation harness, an API, and Docker/CI are planned but not yet built — see [Status](#status) below.

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
        ▼ (lot mode aggregates across photos — planned)
Streamlit UI  /  FastAPI JSON endpoint (API planned)
```

Full breakdown, including the evaluation harness and Docker/CI setup, in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Features

- **Species ID** via BioCLIP 2 zero-shot classification against a curated taxonomy vector index (Qdrant) — 91.4% zero-shot accuracy on the PlantNet benchmark in the original paper.
- **Agentic reasoning layer**: a LangChain agent (not a fixed pipeline) — confirmed via its own message trace to genuinely call `lookup_taxonomy`, `assess_quality`, and `check_price` on a normal run, deciding for itself when to use each. Matches the "AI Agent System" framing from the original tutorial this project was inspired by.
- **Confidence handling**: a close call between top candidates surfaces a visible switcher instead of a silently wrong guess; a clear non-match gets a hedged message instead of a confident species name.
- **Planned**: lot/batch mode with a price trend chart, a measured evaluation harness (top-1/top-3 accuracy + confusion matrix), a FastAPI endpoint alongside the Streamlit UI, and Docker Compose + CI.

## Tech stack

**Built:** Streamlit · BioCLIP 2 (open-source vision foundation model, `open_clip`) · Qdrant (local vector search) · Google Gemini (`gemini-3.1-flash-lite` via `google-genai`) · LangChain (`langchain` + `langchain-google-genai`, agentic tool-calling)

**Planned (later milestones):** FastAPI · Docker Compose · GitHub Actions

## Running it

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt   # Windows; use .venv/bin/pip on macOS/Linux
cp .env.example .env   # then add your GEMINI_API_KEY (free tier: https://aistudio.google.com/apikey)
python scripts/build_index.py                    # builds the local Qdrant species index + simulated prices
streamlit run app.py
```

First run downloads BioCLIP 2 weights (public, no auth needed). If you ever see `Storage folder ... is already accessed by another instance of Qdrant client`, another Python process from a previous run is still holding the local index — close it and retry.

## Important disclaimers

- **Pricing is simulated.** FloraHolland does not expose a public real-time pricing API. Any price shown by this project is generated demo data, clearly labeled as such — not a real auction price.
- **Quality assessment is a heuristic**, not a calibrated agronomic grading system.
- **Species coverage is a curated demo-sized subset** (~30–50 common cut-flower species), not the full biodiversity scale of the underlying research models.
- This project is not affiliated with Royal FloraHolland, the Museum of the Future, or the Imageomics Institute — all are credited as inspiration/research sources only.

## Future work

Not part of this build, but noted for later: a Grad-CAM-style interpretability overlay showing which part of a photo drove the species call, and a persistent inventory/traceability log across scans — closer to the original tutorial's "Inventory Scanner" concept for a greenhouse, deferred because it turns this from a stateless demo into a stateful application.

## Status

- ✅ **Phase A** — research, architecture, and this documentation.
- ✅ **Milestone 1** — core pipeline: camera scan → BioCLIP 2 species ID → Gemini quality read → simulated price, in a working Streamlit app. Verified end-to-end on real flower photos.
- ✅ **Milestone 2** — confidence gating: three tiers from Qdrant's top-1/top-2 score gap (`high`/`ambiguous`/`low`, thresholds in `src/identify.py`). Ambiguous scans show a visible switcher between the top-2 candidates (updates species/price with no extra Gemini call); low-confidence scans get a hedged message instead of a confident species name.
- ✅ **Milestone 3** — agentic layer: a LangChain (`create_agent`) tool-calling agent over Gemini replaces the fixed call, with `lookup_taxonomy`/`assess_quality`/`check_price` as real tools (`src/tools.py`). Verified via the agent's own message trace, not just its output, that it genuinely calls all three. Found along the way that an agentic `identify()` call costs 4+ Gemini calls instead of 1, which exhausted `gemini-3.6-flash`'s daily free quota — switched to `gemini-3.1-flash-lite`, which tracks separate, more generous quota (see [docs/RESEARCH.md](docs/RESEARCH.md)).
- ⬜ Milestone 4 — lot/batch mode + price trend chart
- ⬜ Milestone 5 — evaluation harness
- ⬜ Milestone 6 — FastAPI endpoint
- ⬜ Milestone 7 — Docker Compose + CI

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full plan.

## License

TBD.
