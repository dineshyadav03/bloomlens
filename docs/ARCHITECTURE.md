# Architecture (planned)

Status: **design only** — nothing below is implemented yet. This is the Phase B plan, written up now so it's committed alongside the research, per the project's research-first approach. See [RESEARCH.md](RESEARCH.md) for the sources behind each design choice. Updated after a review pass that added the agentic layer, evaluation harness, lot mode, confidence handling, API layer, and DevOps — this document is kept current as decisions change, not just written once.

## Pipeline

```
Flower photo(s) (Streamlit camera scan — one photo, or several as a "lot"; a file-upload fallback exists for dev/testing without a camera)
        │
        ▼
BioCLIP 2: embed photo (ViT-B/16 vision encoder)
        │
        ▼
Qdrant: cosine similarity vs. taxonomy-string text embeddings
  (one embedding per curated species' flattened taxonomic name,
   e.g. "Plantae Tracheophyta Magnoliopsida Rosales Rosaceae Rosa gallica")
        │
        ▼ top-k candidate species + similarity scores
Confidence gate:
  top-1 clearly ahead  → proceed with that species
  top-1 vs top-2 close → surface both as "did you mean X or Y?"
  all scores low       → "not confidently a known species" (no false match)
        │
        ▼
LangChain **agent** (Gemini as the reasoning model), given tools:
  - lookup_taxonomy(species)   → description/context from species_reference.json
  - assess_quality(image)      → Gemini vision read of bloom stage/wilting/blemishes
  - check_price(species, grade) → simulated pricing module
  The agent decides which tools to call and composes the final answer:
  species/common name, confidence, quality note, price, plain-language summary
        │
        ▼ (lot mode: repeat per photo, then aggregate)
Lot aggregation (if >1 photo): consensus species, avg./range quality,
  lot price estimate + trend chart
        │
        ▼
Streamlit results panel (and/or FastAPI JSON response):
  species · taxonomy breadcrumb · quality note · simulated price + trend ·
  confidence / "did you mean" · "simulated data" disclaimer
```

**Why text embeddings, not reference-image embeddings**: BioCLIP classifies by comparing an image embedding to *taxonomic-name text embeddings* (zero-shot), not to embeddings of other photos — that's how it hit 91.4% zero-shot on the PlantNet benchmark in the original paper. So the Qdrant collection holds one embedding per curated species' taxonomic-name string, not a reference-photo library. See [RESEARCH.md#bioclip](RESEARCH.md#bioclip) for the full reasoning and numbers behind this.

**Why an agent, not a fixed chain**: the original tutorial screenshots named one component an "AI Agent System." Giving Gemini tool-calling access (via a LangChain tool-calling agent) rather than hardcoding "always call these three functions in this order" is both truer to that framing and a materially different, more in-demand skill to demonstrate than a linear RAG chain — the model decides, e.g., whether a clearly-labeled generic photo needs a quality check at all, or whether low BioCLIP confidence means it should ask for taxonomy context on more than one candidate.

## Division of labor

| Layer | Role | Why |
|---|---|---|
| **BioCLIP 2** | Real vector-DB/taxonomy backbone. Embeds the uploaded photo; cosine similarity against a Qdrant collection of curated species' taxonomic-name text embeddings gives top-k candidates — zero-shot, no reference photos needed. | Real implementation of the tutorial's "100K+ species embeddings" idea, backed by a citable CVPR/NeurIPS model (91.4% zero-shot on PlantNet). See [RESEARCH.md#bioclip](RESEARCH.md#bioclip). |
| **Confidence gate** | Compares top-1 vs top-2 similarity scores; routes to a single answer, a "did you mean" choice, or a no-match message. | Prevents confidently-wrong answers — a real trust/UX concern for something buyers might actually act on. |
| **LangChain agent + Gemini** | Reasoning/generation layer with tool access (`lookup_taxonomy`, `assess_quality`, `check_price`), not a fixed pipeline. Produces the final buyer-facing answer. | A vision-language model can read bloom stage, wilting, and blemishes in a way a classifier alone cannot — this is where "quality" comes from — and an agent framing matches the "AI Agent System" the original tutorial named. |
| **Simulated pricing module** | Looks up a generated price table by species + quality grade, returns a price/stem and trend. | FloraHolland's real pricing system isn't public (see [RESEARCH.md#floraholland](RESEARCH.md#floraholland)); this stands in for it, clearly labeled. |
| **Lot/batch mode** | Accepts multiple photos as one "lot," runs the pipeline per photo, aggregates to a consensus species/quality/price with a trend chart. | Real auction buying happens in lots, not single stems — this is what makes it feel like a supply-chain tool rather than a single-flower toy. |
| **Evaluation harness** | Runs the full pipeline against a held-out labeled test set and reports accuracy/confusion matrix. | Turns "a demo that seems to work" into a measured system — the difference between a toy and an evaluated ML pipeline. |
| **Streamlit app + FastAPI endpoint** | Streamlit for the interactive demo UI; a thin FastAPI wrapper around the same core pipeline for a scriptable JSON API. | Shows the model logic isn't wedded to one UI — a real signal of "this could plug into something bigger." |
| **Docker Compose + CI** | One-command local spin-up (app + Qdrant); GitHub Actions runs lint + the evaluation harness on push. | Portfolio polish: anyone can clone and run it, and a green CI badge is a cheap trust signal. |

## Scope for a demo-sized build

- **Reference species set**: ~30–50 common cut-flower/ornamental species (rose, tulip, chrysanthemum, lily, orchid, carnation, sunflower, daisy, iris, etc.) — not the full 200M-taxon BioCLIP 2 corpus. Since classification is zero-shot against taxonomy-name text embeddings (no reference photos required to build the index), this list is just data: one taxonomy string + description per species.
- **Taxonomy/description metadata**: a small hand-curated JSON (kingdom → species, common name, short description) for the fixed species list — accurate at this scale without needing a live GBIF/Wikidata integration.
- **Test/eval images**: sample photos per species from the public **Oxford 102 Flowers** dataset (strong overlap with commercial cut flowers) — a held-out split feeds the evaluation harness; a few more are used to manually exercise the app during development. Not part of the vector index itself.
- **Quality assessment**: a heuristic vision-language read (bloom stage, wilting, visible blemishes) from Gemini — not a calibrated agronomic grading system. Stated plainly to users.
- **Pricing**: a generated CSV (species × grade × date) with a small random walk so it feels "live," explicitly labeled simulated everywhere it's shown.
- **Lot mode**: capped at a small number of photos per lot (e.g. up to 10) for the demo — no need to engineer for real auction lot sizes.

## Agent tools

The LangChain agent gets three tools rather than a hardcoded call order:

- `lookup_taxonomy(species: str) -> str` — pulls the description/taxonomy text for a species out of `species_reference.json`. Used to ground the agent's answer instead of letting the LLM invent botanical facts.
- `assess_quality(image) -> str` — sends the photo to Gemini's vision input with a quality-focused prompt (bloom stage, wilting, blemishes) and returns a short quality note.
- `check_price(species: str, grade: str) -> dict` — looks up the simulated price table, returns price/stem and a short trend description.

The agent's system prompt makes explicit that pricing is simulated and quality is a heuristic read, so those caveats show up in whatever the agent generates — not just in a UI label that a JSON API consumer would never see.

## Confidence handling

- BioCLIP/Qdrant returns similarity scores, not calibrated probabilities — so "confidence" is relative (top-1 vs top-2 gap), not an absolute percentage claim.
- Three tiers: **clear winner** (proceed normally), **close call** (surface top-2 as a "did you mean X or Y?" choice for the user to resolve), **no confident match** (say so, don't force an answer). Thresholds are tuned empirically once the evaluation harness (below) is running, not guessed upfront.

## Lot/batch mode

- User can upload 1–10 photos as a single "lot" instead of one at a time.
- Each photo runs the full pipeline independently; results are aggregated: majority-vote species (flagging any photo that disagrees, since that's a real auction concern — mislabeled lots), quality notes summarized, and a lot-level simulated price with a small trend chart (price over recent simulated dates for that species/grade).
- Chart: a simple line/area chart (Streamlit's built-in charting is enough — no need for a separate charting library) showing the simulated price trend leading up to "today."

## Evaluation harness

- A held-out split of the Oxford 102 Flowers test images (never used to build the Qdrant taxonomy index, since that index doesn't need images at all — see above) with known ground-truth species, mapped onto the curated species list.
- `eval/run_eval.py` runs the identification step (BioCLIP embed → Qdrant search) against every held-out image and reports **top-1 and top-3 accuracy** plus a confusion matrix — this measures the actual classification step, separate from Gemini's free-text quality/summary output which isn't the kind of thing accuracy metrics apply to.
- Results get written to `eval/results.md` (or similar) and referenced from the README, so the portfolio claim is "measured X% top-1 accuracy on N species," not just "it seems to work."
- This same script runs in CI (see DevOps below) so accuracy regressions get caught on every push.

## API layer

- `api/main.py`: a small FastAPI app exposing `POST /identify` (single photo) and `POST /identify-lot` (multiple photos), both returning the same structured JSON the Streamlit app renders.
- Both the Streamlit app and the API import the same core pipeline module (`src/identify.py`) — no duplicated logic between the two front ends.

## DevOps: Docker & CI

- `Dockerfile` for the app; `docker-compose.yml` running the app alongside a local Qdrant container, so `docker compose up` is the entire setup.
- `.github/workflows/ci.yml`: on every push, install dependencies, run linting, and run `eval/run_eval.py` against the held-out set — surfaces a pass/fail + accuracy number as a CI badge on the README.

## Future work (not in this build, noted for later)

- **Interpretability overlay**: a Grad-CAM-style saliency map showing which part of the photo drove BioCLIP's species call — useful trust-building UI, more implementation effort than the items above.
- **Persistent inventory/traceability log**: closer to the original tutorial's "Inventory Scanner — AI Agent System" idea for a greenhouse — logging every scan (species, confidence, quality, timestamp) to build up a running inventory count rather than one-off lookups. Deferred because it turns this from a stateless demo into a small stateful application (needs a real datastore, not just a CSV).

## Planned file layout

```
requirements.txt          streamlit, fastapi, uvicorn, qdrant-client,
                           open-clip-torch/pybioclip, langchain,
                           langchain-google-genai, langchain-qdrant,
                           torch, torchvision, pillow, pytest
data/species_reference.json   curated species + taxonomy + description
data/simulated_prices.csv     generated mock pricing table
eval/test_set.json         held-out image paths + ground-truth species
eval/run_eval.py           runs identification against the held-out set,
                           reports top-1/top-3 accuracy + confusion matrix
src/embeddings.py          load BioCLIP 2, embed an image or a taxonomy string
src/vector_store.py        Qdrant local-mode client: build/query the collection
src/pricing.py             simulated price lookup + generator logic
src/tools.py               the three LangChain tools (lookup_taxonomy,
                           assess_quality, check_price)
src/identify.py            core pipeline: embed → Qdrant search →
                           confidence gate → LangChain agent w/ tools →
                           structured result (Pydantic: species, common_name,
                           confidence, quality_note, price, summary);
                           also exposes a lot-aggregation function
scripts/build_index.py     one-time: embed each curated species' taxonomy
                           string w/ BioCLIP 2's text encoder, upsert into
                           Qdrant (no images needed for this step)
scripts/fetch_test_images.py  pull Oxford 102 Flowers sample photos per
                           species for both eval/ and manual testing
app.py                     Streamlit entrypoint (single photo + lot mode)
api/main.py                FastAPI entrypoint (/identify, /identify-lot)
Dockerfile                 container image for app.py / api/main.py
docker-compose.yml         app + local Qdrant, one-command spin-up
.github/workflows/ci.yml   lint + run eval/run_eval.py on push
.env.example               GEMINI_API_KEY=
```

## Setup this will need (Phase B)

- A free Gemini API key from Google AI Studio, set as `GEMINI_API_KEY` in a local `.env` (never committed).
- BioCLIP 2 weights download on first use (public, no auth required, via `pybioclip`/HuggingFace).
- Docker + Docker Compose, only if using the one-command spin-up path instead of a local Python environment.

## Verification plan (Phase B)

1. `python scripts/build_index.py` — confirm it reports N species embedded into Qdrant without errors.
2. `streamlit run app.py` — upload a real rose or tulip photo, confirm the app returns a plausible species, taxonomy breadcrumb, a quality note, and a simulated price with the disclaimer visible.
3. Upload a clearly non-flower image (e.g. a car) — confirm the app degrades gracefully (no confident match) instead of a false confident match.
4. Upload two visually similar species (e.g. two pink flowers) — confirm the confidence gate surfaces a "did you mean" choice instead of silently picking one.
5. Upload 3–5 photos as a lot — confirm consensus species/quality/price and the trend chart render correctly, and that a deliberately mismatched photo in the lot gets flagged.
6. `python eval/run_eval.py` — confirm it reports top-1/top-3 accuracy and a confusion matrix on the held-out set.
7. `POST /identify` against the FastAPI app with a sample image — confirm the JSON response matches what the Streamlit app shows for the same photo.
8. `docker compose up` — confirm the whole stack comes up and the app is reachable, with no manual Qdrant setup.
9. Push to GitHub — confirm the CI workflow runs and reports the eval accuracy.
