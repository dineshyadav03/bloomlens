# Architecture (planned)

Status: Milestones 1–6 implemented (core pipeline, confidence gating, agentic layer, lot mode, evaluation harness, FastAPI endpoint) — see each section below for what's built vs. still planned. See [RESEARCH.md](RESEARCH.md) for the sources behind each design choice; this document is kept current as decisions change, not just written once.

## Pipeline

```
Flower photo(s) (Streamlit camera scan — one photo, or several as a "lot"; a file-upload fallback exists for dev/testing without a camera)
        │
        ▼
BioCLIP 2: embed photo (ViT-L/14 vision encoder)
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
LangChain **agent** (Gemini as the reasoning model — sees the photo directly
in its own multimodal context), given tools:
  - lookup_taxonomy(species)              → description/context from species_reference.json
  - assess_quality(quality_grade, quality_note) → validates + records the agent's own visual read
  - check_price(species, grade)           → simulated pricing module
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

## Agent tools — implemented (Milestone 3)

`src/tools.py` gives the agent (`src/identify._get_agent`, built with `langchain.agents.create_agent` + `ChatGoogleGenerativeAI`) three tools rather than a hardcoded call order:

- `lookup_taxonomy(species: str) -> str` — pulls the description/taxonomy text for a species out of `species_reference.json`. Used to ground the agent's answer instead of letting the LLM invent botanical facts.
- `assess_quality(quality_grade: str, quality_note: str) -> dict` — **not** a second vision call. The agent already sees the photo directly in its own multimodal message (LangChain's `{"type": "image", "source_type": "base64", ...}` content block, confirmed working alongside tool-calling), so it forms the visual judgment itself; this tool is where it formally records that judgment. It validates the grade is A/B/C and runs a sentence-level negation-aware check flagging e.g. "some browning visible" paired with grade A, while correctly *not* flagging "completely free of wilting, discoloration, or damage" (a naive keyword match without negation handling was tried first and produced exactly that false positive in testing).
- `check_price(species: str, grade: str) -> dict` — thin wrapper around `src/pricing.lookup_price`.

The final answer is still parsed as prompt-requested JSON (not Gemini's `response_schema`/`response_json_schema` structured-output parameter — same reliability issue as Milestone 1) from whichever message the agent graph returns last; that message's `.content` can be a plain string or a list of `{"type": "text", "text": ...}` blocks depending on model/SDK version, so `src/identify._extract_final_text` handles both.

Confirmed by inspecting the agent's actual message trace (not just its final answer) that it genuinely calls all three tools in a normal run, in the order `lookup_taxonomy → assess_quality → check_price`. Retrieval (BioCLIP + Qdrant) and confidence gating stay deterministic pre-steps outside the agent, exactly as originally planned — `identify()` still calls `lookup_price` itself unconditionally after the agent responds, so the displayed price never depends on whether the agent chose to call `check_price`.

**Real operational finding**: an agentic identify() call makes 4+ Gemini calls (one per reasoning/tool step), not one — this multiplies free-tier quota usage severalfold versus Milestone 1's single-call design and was enough to exhaust `gemini-3.6-flash`'s 20-requests/day cap in normal testing. See [RESEARCH.md's Gemini correction section](RESEARCH.md#correction-gemini-model-naming-and-quota-and-api-key-format) — `_GEMINI_MODEL` is now `gemini-3.1-flash-lite`, which tracks separate, more generous quota.

## Confidence handling — implemented (Milestone 2)

- BioCLIP/Qdrant returns similarity scores, not calibrated probabilities — so "confidence" is relative (top-1 vs top-2 gap), not an absolute percentage claim.
- Three tiers, computed purely from Qdrant scores in `src/identify._classify_confidence` (no extra Gemini call): **`high`** (top-1 ≥ 0.55 and gap to top-2 ≥ 0.05) — proceeds normally; **`ambiguous`** (everything in between) — `app.py` shows a visible `st.radio` switcher between the top-2 candidates, defaulting to Gemini's pick, that re-derives species/scientific-name/price via `src/identify.resolve_candidate` on change, no second Gemini call; **`low`** (top-1 < 0.45) — the confident species title is replaced with a hedged "not confidently a known species" message.
- The specific thresholds (0.55/0.05/0.45) are a heuristic starting point from Milestone 1's handful of real test photos, not tuned against a labeled dataset — they're module-level constants specifically so Milestone 5's evaluation harness can revisit them with real accuracy data.

## Lot/batch mode — implemented (Milestone 4)

- User can scan or upload 1–`LOT_MAX_PHOTOS` (10) photos as a single "lot" (`app.py`'s "Lot mode" tab, `src/identify.identify_lot`).
- **Redesigned from the original plan for a real reason**: "each photo runs the full pipeline independently" would mean a 10-photo lot costs 40+ Gemini calls (Milestone 3 found an agentic `identify()` call costs 4+ calls, not 1). Instead: BioCLIP + Qdrant retrieval (free, local) runs **per photo** to get each one's top candidate; a majority vote across those (ties broken by highest summed score, `src/identify._compute_consensus`) gives the lot's consensus species and an agreement fraction (e.g. "2/3 photos agree"); the agent then runs **once for the whole lot**, given all photos together in one multi-image message (confirmed working: `ChatGoogleGenerativeAI` correctly distinguishes multiple images in a single call) plus the consensus species' taxonomy context, and produces one quality_grade/quality_note/summary for the lot. Net effect: a lot costs the same ~4-5 Gemini calls as a single scan, regardless of photo count.
- Photos whose top candidate disagreed with consensus are flagged in the UI with their own detected species (`flagged_photos`) — verified with a real mixed lot (2 roses + 1 sunflower): correctly detected 2/3 agreement, flagged the sunflower, and the agent's own summary independently corroborated the mismatch ("This lot is mixed and highly inconsistent...").
- Below `LOT_LOW_AGREEMENT_THRESHOLD` (0.7), the UI shows an explicit "may contain mixed species" warning rather than presenting the majority vote as a settled answer.
- Chart: `st.line_chart` fed by `src/pricing.price_history` (existing since Milestone 1, unused until now) — no separate charting library needed.
- `identify()`'s retry/parse loop was factored out into a shared `_invoke_agent_with_retries(message)` so both single-scan and lot mode execute through the same tested path; single-scan behavior is unchanged (verified by regression-testing the Milestone 1/2/3 photos after the refactor).

## Evaluation harness — implemented (Milestone 5)

- **Result: 87.2% top-1 / 96.1% top-3 accuracy** on 360 held-out Oxford 102 Flowers test images across 18 of the 30 curated species (`eval/results.md`, `eval/run_eval.py`). Zero API cost — this only exercises BioCLIP 2 + Qdrant retrieval (`embed_image` + `vector_store.search`, the exact functions `identify()` uses), matching the architecture's own scope: measuring the classification step, not Gemini's free-text output, which isn't the kind of thing accuracy metrics apply to.
- **Species coverage is honestly scoped, not inflated**: `eval/species_mapping.py` maps 18 of the 30 curated species to a confident Oxford 102 category — several via well-established alternate common names verified individually (e.g. "barberton daisy" = *Gerbera jamesonii*, "hippeastrum" = the genus commercially sold as "amaryllis," matching our own `species_reference.json`), not string-matched. The other 12 either aren't in Oxford 102 at all, or only have an unconfirmed genus-level match (e.g. Oxford's generic "buttercup" isn't confirmed to be *Ranunculus asiaticus*) — deliberately excluded rather than claimed. `eval/results.md` lists all 12 with the specific reason.
- Test images sampled once (`eval/build_test_set.py`, seeded, up to 20/species) from Oxford's "test" split — deliberately the *largest* split in their benchmark, since their own classifier trains on very few images per class. Verified the well-known 0-index/1-index labeling gotcha for this dataset by reading torchvision's `Flowers102` source directly (it already normalizes to 0-indexed) rather than assuming, then spot-checked sampled images visually before trusting the full run.
- **Confidence-tier thresholds checked against real data, and kept as-is** (the concrete "revisit" the Milestone 2 code comment promised — a genuine check, not a rubber stamp): the `high` tier is 99.1% precise but withholds 105 of 314 correct predictions into `ambiguous` purely because their score/gap fell just under threshold. Swept looser alternatives against the actual data (`eval/run_eval.py`'s threshold sweep) rather than guessing: loosening the gap requirement would gain those correct calls back, but drops precision by over 2 percentage points, because correct and wrong predictions *within* the ambiguous zone have nearly identical score/gap distributions — there's no cleaner cutoff hiding in the data. The `low` threshold couldn't be validated at all here (every test image is a genuine species match, so nothing ever lands there) — that needs deliberately-included non-flower images, noted as future work. Full numbers in `eval/results.md`'s calibration section.
- Confusion matrix confirms the errors are visually sensible, not random noise — e.g. Amaryllis↔Easter lily (both large trumpet-form lilies) and Dahlia↔Sunflower/Gerbera daisy (all radial multi-petal composites) account for most of the misses.

## API layer — implemented (Milestone 6)

- `api/main.py`: FastAPI app exposing `GET /health`, `POST /identify` (single photo), and `POST /identify-lot` (multiple photos) — both return `response_model=IdentifyResult`/`LotResult`, the exact same Pydantic models `identify()`/`identify_lot()` already produced. No reshaping, no second implementation — verified end-to-end with real photos from `eval/test_images/` giving identical results to calling the pipeline directly.
- Route handlers are plain `def`, not `async def`: `identify()`/`identify_lot()` block on CPU inference and Gemini network calls, and FastAPI runs sync handlers in a thread pool automatically — the correct way to serve blocking work without stalling the event loop.
- Error mapping verified against real requests: an oversized lot (>`LOT_MAX_PHOTOS`) → `400` with a clear message; a missing `photos` field entirely → FastAPI's own `422` validation error (idiomatic, not something to override); a pipeline-level `IdentifyError` (missing API key, rate-limited, etc.) → `503` with the error's own message, not a raw traceback — confirmed by temporarily removing `GEMINI_API_KEY` and hitting the endpoint for real.
- Interactive Swagger UI at `/docs` comes free from FastAPI's OpenAPI generation — confirmed it renders correctly for both endpoints and all response schemas.

## DevOps: Docker & CI

## DevOps: Docker & CI — implemented (Milestone 7)

- **Qdrant becomes a real server in Docker, not just local mode with extra steps.** `src/vector_store.py`'s local/embedded mode takes an exclusive file lock — the exact thing that caused a real bug in Milestone 1 ("Storage folder is already accessed by another instance"). Docker Compose runs `web` (Streamlit) and `api` (FastAPI) as separate containers that could genuinely run at once, so local mode's one-process-at-a-time constraint would just reproduce that bug. `get_client()` now branches on a `QDRANT_URL` env var — set in compose, unset (unchanged local-mode behavior) everywhere else — so this is invisible to every caller.
- `docker-compose.yml`: four services — `qdrant` (official image, its own persistent volume), `build-index` (one-shot, populates the *server's* index, `depends_on: qdrant` via a healthcheck), `web`, `api` (both `depends_on: build-index` completing). `web`/`api`/`build-index` share a named `hf_cache` volume for BioCLIP 2's weights, so the download is a one-time cost across container restarts.
- **BioCLIP 2 weights are not baked into the image at build time.** This machine's network measured ~100-200KB/s in Milestone 5 (a 345MB download took 28 minutes) — baking a several-hundred-MB model into every image build would make local iteration impractical. Weights download on first container start instead. Worth revisiting if this ever builds somewhere with real bandwidth (GitHub's CI runners, or a real deploy).
- `Dockerfile`: one image, `python:3.13-slim`, CPU-only torch (same PyTorch CPU wheel index used in local dev — no GPU anywhere in this project, see [RESEARCH.md](RESEARCH.md)'s Milestone 1 notes on why that's genuinely a non-issue here); which service runs is a `command:` override per compose service, not a second Dockerfile.
- `.github/workflows/ci.yml`, two jobs: **`lint-and-eval`** — `ruff check .` (config in `pyproject.toml`: 120-char lines, matching this codebase's actual style, not ruff's 88-char default; `B008` ignored, since it's FastAPI's own required idiom for `= File(...)` parameters — checked before adopting the rule, not assumed) — then `eval/run_eval.py`, which now **fails CI outright if top-1 accuracy drops below 80%** (real headroom under the measured 87.2%, since this pipeline is deterministic and shouldn't drift) — an actual regression gate, not just "the script ran." **`docker-smoke-test`** — builds the real Compose stack and curls both services' `/health` endpoints; deliberately needs no `GEMINI_API_KEY` secret, since health checks don't touch the identify() pipeline. This is the authoritative end-to-end check of the Docker setup, running on GitHub's fast network rather than this machine's slow one.
- **`eval/test_images/` is now committed** (reversing a Milestone 5 call): it was gitignored under the same rule as `qdrant_data/` ("fetched, not source"), but unlike `qdrant_data/` it depends on a slow one-time external download, not instant regeneration from committed source. It's only ~15MB — small enough to commit outright, which is what actually makes the CI eval job possible without a flaky external dependency.

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
