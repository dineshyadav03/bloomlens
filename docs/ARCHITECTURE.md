# Architecture (planned)

Status: Milestones 1–3 implemented (core pipeline, confidence gating, agentic layer) — see each section below for what's built vs. still planned. See [RESEARCH.md](RESEARCH.md) for the sources behind each design choice; this document is kept current as decisions change, not just written once.

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
