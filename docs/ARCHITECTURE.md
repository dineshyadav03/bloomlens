# Architecture (planned)

Status: **design only** — nothing below is implemented yet. This is the Phase B plan, written up now so it's committed alongside the research, per the project's research-first approach. See [RESEARCH.md](RESEARCH.md) for the sources behind each design choice.

## Pipeline

```
Flower photo (Streamlit upload)
        │
        ▼
BioCLIP 2: embed photo (ViT-B/16 vision encoder)
        │
        ▼
Qdrant: cosine similarity vs. taxonomy-string text embeddings
  (one embedding per curated species' flattened taxonomic name,
   e.g. "Plantae Tracheophyta Magnoliopsida Rosales Rosaceae Rosa gallica")
        │
        ▼ top-k candidate species + taxonomy/description context
Gemini (current multimodal Flash model), via LangChain
  prompt = photo + top-k candidates + retrieved taxonomy/description text
  → refined species / common name, confidence, freshness & quality note,
    plain-language buyer summary
        │
        ▼
Simulated pricing lookup (species + quality grade → price/stem, trend)
        │
        ▼
Streamlit results panel:
  species · taxonomy breadcrumb · quality note · simulated price ·
  "simulated data" disclaimer
```

**Why text embeddings, not reference-image embeddings**: BioCLIP classifies by comparing an image embedding to *taxonomic-name text embeddings* (zero-shot), not to embeddings of other photos — that's how it hit 91.4% zero-shot on the PlantNet benchmark in the original paper. So the Qdrant collection holds one embedding per curated species' taxonomic-name string, not a reference-photo library. See [RESEARCH.md#bioclip](RESEARCH.md#bioclip) for the full reasoning and numbers behind this.

## Division of labor

| Layer | Role | Why |
|---|---|---|
| **BioCLIP 2** | Real vector-DB/taxonomy backbone. Embeds the uploaded photo; cosine similarity against a Qdrant collection of curated species' taxonomic-name text embeddings gives top-k candidates — zero-shot, no reference photos needed. | This is the actual implementation of the tutorial's "100K+ species embeddings" idea — using a real, citable CVPR/NeurIPS model (91.4% zero-shot on the PlantNet benchmark) instead of a mocked lookup table. See [RESEARCH.md#bioclip](RESEARCH.md#bioclip). |
| **Gemini + LangChain** | RAG/reasoning layer. Takes the photo plus BioCLIP's candidates plus retrieved taxonomy text, and produces the final buyer-facing answer: refined species guess, a freshness/quality read a pure classifier can't give, and a plain-language summary. | A vision-language model can look at bloom stage, wilting, and blemishes in a way a classifier alone cannot — this is where "quality" comes from. |
| **Simulated pricing module** | Looks up a generated price table by species + quality grade, returns a price/stem and trend note. | FloraHolland's real pricing system isn't public (see [RESEARCH.md#floraholland](RESEARCH.md#floraholland)); this stands in for it, clearly labeled. |
| **Streamlit app** | Upload UI + results panel. | Fastest way to get a working, demoable interface for a portfolio piece. |

## Scope for a demo-sized build

- **Reference species set**: ~30–50 common cut-flower/ornamental species (rose, tulip, chrysanthemum, lily, orchid, carnation, sunflower, daisy, iris, etc.) — not the full 200M-taxon BioCLIP 2 corpus. Since classification is zero-shot against taxonomy-name text embeddings (no reference photos required to build the index), this list is just data: one taxonomy string + description per species.
- **Taxonomy/description metadata**: a small hand-curated JSON (kingdom → species, common name, short description) for the fixed species list — accurate at this scale without needing a live GBIF/Wikidata integration.
- **Test/demo images**: a handful of sample photos per species from the public **Oxford 102 Flowers** dataset (strong overlap with commercial cut flowers), used only to exercise the app during development — not part of the vector index itself.
- **Quality assessment**: a heuristic vision-language read (bloom stage, wilting, visible blemishes) from Gemini — not a calibrated agronomic grading system. Stated plainly to users.
- **Pricing**: a generated CSV (species × grade × date) with a small random walk so it feels "live," explicitly labeled simulated everywhere it's shown.

## Planned file layout

```
requirements.txt          streamlit, qdrant-client, open-clip-torch/pybioclip,
                           langchain, langchain-google-genai, langchain-qdrant,
                           torch, torchvision, pillow
data/species_reference.json   curated species + taxonomy + description
data/simulated_prices.csv     generated mock pricing table
src/embeddings.py         load BioCLIP 2, embed an image or a taxonomy string
src/vector_store.py       Qdrant local-mode client: build/query the collection
src/pricing.py            simulated price lookup + generator logic
src/identify.py           RAG pipeline: embed photo → Qdrant search →
                           LangChain prompt w/ retrieved context → Gemini call →
                           parse structured result (Pydantic: species,
                           common_name, confidence, quality_note, summary)
scripts/build_index.py    one-time: embed each curated species' taxonomy
                           string w/ BioCLIP 2's text encoder, upsert into
                           Qdrant (no images needed for this step)
scripts/fetch_test_images.py  optional: pull a few Oxford 102 Flowers sample
                           photos per species, for manually exercising the app
app.py                    Streamlit entrypoint
.env.example              GEMINI_API_KEY=
```

## Setup this will need (Phase B)

- A free Gemini API key from Google AI Studio, set as `GEMINI_API_KEY` in a local `.env` (never committed).
- BioCLIP 2 weights download on first use (public, no auth required, via `pybioclip`/HuggingFace).

## Verification plan (Phase B)

1. `python scripts/build_index.py` — confirm it reports N species embedded into Qdrant without errors.
2. `streamlit run app.py` — upload a real rose or tulip photo, confirm the app returns a plausible species, taxonomy breadcrumb, a quality note, and a simulated price with the disclaimer visible.
3. Upload a clearly non-flower image (e.g. a car) — confirm the app degrades gracefully (low-confidence message) instead of a false confident match.
4. Spot-check 2–3 more species from the reference set to confirm vector search isn't just returning one dominant class.
