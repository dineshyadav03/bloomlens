# BloomLens

Point a camera at a flower, get its **species, quality, and price** back — instantly.

BloomLens is a portfolio project for the cut-flower supply chain, inspired by the scale of the world's largest flower auction and the "scan to learn" interaction from Dubai's Museum of the Future. It is in the **documentation/planning stage** — no application code has been written yet. This README, and the docs it links to, are the output of that planning pass.

## The problem

Royal FloraHolland, the flower auction in Aalsmeer, Netherlands, is the world's largest flower market: roughly **43 million flowers and 5 million plants** change hands there on a typical weekday, traded across 35 auction clocks and spanning more than 30,000 varieties, inside a nearly 1 million m² facility. [[1]](docs/RESEARCH.md#floraholland)

At that volume and speed, a buyer looking at a lot on the auction floor has no realistic way to manually verify what species it is, how good its quality is, and what a fair price looks like. BloomLens's premise: let a buyer take one photo and get all three back immediately.

## Inspiration

- **The scan-to-learn interaction** is inspired by "The Library" at Dubai's Museum of the Future — 2,400 laser-etched specimen jars that visitors scan with a handheld device to instantly pull up each species' story. [[2]](docs/RESEARCH.md#museum-of-the-future) BloomLens reproduces that *interaction pattern* (point at something living → get an instant info card) as a web app, not the physical installation.
- **The technical backbone** draws on BioCLIP / BioCLIP 2, real open-source vision foundation models for the tree of life, published at CVPR'24 (Best Student Paper) and NeurIPS'25 (Spotlight). [[3]](docs/RESEARCH.md#bioclip) These make it possible to build the "vector database of botanical species embeddings" for real, rather than mocking it.

See [docs/RESEARCH.md](docs/RESEARCH.md) for the full writeup with sources, and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the planned system design.

## Planned architecture

```
Flower photo (Streamlit upload)
        │
        ▼
BioCLIP 2 embedding ──► Qdrant vector search ──► top-k candidate species
        │                                             + taxonomy context
        ▼
Gemini (current multimodal Flash model)
  prompt = photo + candidates + retrieved taxonomy text
  → refined species/common name, confidence, freshness/quality note
        │
        ▼
Simulated pricing lookup (species + grade → price/stem, trend)
        │
        ▼
Streamlit results panel: species, taxonomy, quality note, simulated price
```

Full breakdown in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Tech stack (planned)

Streamlit · BioCLIP 2 (open-source vision foundation model) · Qdrant (vector search) · Google Gemini (current multimodal model) · LangChain (RAG orchestration)

## Important disclaimers

- **Pricing is simulated.** FloraHolland does not expose a public real-time pricing API. Any price shown by this project is generated demo data, clearly labeled as such — not a real auction price.
- **Quality assessment is a heuristic**, not a calibrated agronomic grading system.
- **Species coverage is a curated demo-sized subset** (~30–50 common cut-flower species), not the full biodiversity scale of the underlying research models.
- This project is not affiliated with Royal FloraHolland, the Museum of the Future, or the Imageomics Institute — all are credited as inspiration/research sources only.

## Status

📋 Planning & documentation complete. Implementation (Phase B) is planned but not yet built — see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the file-by-file plan.

## License

TBD.
