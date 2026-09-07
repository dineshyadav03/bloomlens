# Research

This document backs up the claims in the [README](../README.md) and the design choices in [ARCHITECTURE.md](ARCHITECTURE.md). It was written before any code, per the project's own ground rule: research first, build second.

## Origin of the idea

The concept came from a set of tutorial screenshots pitching an AI system for flower-auction buyers: photograph a flower, instantly get its species, quality, and price. The tutorial's own architecture and tech-stack diagrams (a "Multimodal RAG System" using Gemini Pro Vision, a vector database of botanical taxonomy, and a RAG pipeline into "FloraHolland's Real-time Pricing API") are the direct basis for this project's planned pipeline. Everything below is independent verification of the claims and technologies that pitch relied on.

## FloraHolland {#floraholland}

Royal FloraHolland's Aalsmeer auction is the world's largest flower market.

- On a typical weekday, **~43 million flowers and 5 million plants** change hands. [Air Cargo Week](https://aircargoweek.com/aalsmeer-the-beating-heart-of-global-flower-trade/)
- The facility spans **35 auction clocks** and trades **30,000+ species and varieties** of flowers and plants, across a site of nearly **1 million square metres**. [Visit Aalsmeer — official facts page](https://www.visitaalsmeer.nl/en/facts-flower-auction-aalsmeer/)
- **~60% of the world's flower and plant trade** flows through Dutch auctions; Royal FloraHolland alone accounts for ~54% of everything sold in the Netherlands. [Air Cargo Week](https://aircargoweek.com/aalsmeer-the-beating-heart-of-global-flower-trade/)
- The auction traces back to 1910. [Air Cargo Week](https://aircargoweek.com/aalsmeer-the-beating-heart-of-global-flower-trade/)

**Why this matters for the project**: this confirms the "43 million flowers daily" claim from the tutorial screenshots is accurate (not a rounded-up marketing number), and grounds the core problem statement — at that volume and speed, manual species/quality/price verification per lot is not realistic. It also confirms **FloraHolland does not publish a public real-time pricing API** — no such API turned up in search, and their pricing/clock data is understood to be a closed, internal auction system for member growers and buyers. This is why BloomLens uses clearly-labeled **simulated** pricing rather than claiming a live integration.

## Museum of the Future, Dubai — "The Library" {#museum-of-the-future}

The plant-scanner reference in the tutorial screenshots is "The Library," an installation at Dubai's Museum of the Future.

- It holds **2,400 crystal specimen jars**, suspended across a 375 m² space, each laser-etched with an existing or extinct lifeform. [The National](https://www.thenationalnews.com/uae/2022/02/23/five-stunning-displays-inside-dubais-museum-of-the-future/)
- Visitors use a **handheld scanning device** ("bio-synth," designed by the UK experiential-art collective Marshmallow Laser Feast) to scan a jar and pull up that species' history and conservation status on a screen. [Superflux](https://superflux.in/index.php/work/the-vault-of-life-in-museum-of-the-future/), [The National](https://www.thenationalnews.com/uae/2022/02/23/five-stunning-displays-inside-dubais-museum-of-the-future/)
- The catalogue spans a wide range of life forms, from trees to animals, not flowers specifically.

**Why this matters for the project**: this is the direct inspiration for the "photo in → instant info card out" interaction, but it's a physical museum exhibit with custom hardware. BloomLens borrows the *interaction pattern* only — a web app where a photo stands in for the physical scan — and makes no claim of using or replicating the museum's actual hardware or software.

## BioCLIP and BioCLIP 2 {#bioclip}

The tutorial's "Vector Database (Botanical Taxonomy – 100K+ Species Embeddings)" component maps closely onto real, open-source research:

- **BioCLIP** (CVPR 2024, Oral, **Best Student Paper**) is a CLIP-style vision foundation model trained on **TreeOfLife-10M** — 10M+ images spanning **454,000 taxa** — for hierarchical taxonomic classification (Kingdom → Phylum → Class → Order → Family → Genus → Species). It outperformed prior baselines by 16–17 percentage points on fine-grained biological classification. [CVPR Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Stevens_BioCLIP_A_Vision_Foundation_Model_for_the_Tree_of_Life_CVPR_2024_paper.html), [GitHub](https://github.com/Imageomics/bioclip), [paper](https://arxiv.org/abs/2311.18803)
- **BioCLIP 2** (NeurIPS 2025, Spotlight) supersedes it, trained on **TreeOfLife-200M** (214M+ images), with further gains on tasks like habitat classification and trait prediction. [Imageomics announcement](https://imageomics.osu.edu/news/2025/07/branching-out-bioclip-2-pushes-boundaries-ai-nature), [HuggingFace](https://huggingface.co/imageomics/bioclip-2), [paper](https://arxiv.org/pdf/2505.23883)
- Both are freely downloadable: model weights on HuggingFace (`imageomics/bioclip`, `imageomics/bioclip-2`), plus a `pybioclip` Python package for straightforward inference.

**Why this matters for the project**: the tutorial's vector-database idea doesn't have to be a toy lookup table — it can be built on a real, citable, state-of-the-art research model that already does exactly this kind of taxonomic embedding. This is the single biggest upgrade this research pass makes to the original pitch: BioCLIP 2 embeddings + a Qdrant index *are* the "botanical taxonomy vector database," at whatever species-count scope the demo needs.

## Correction: Gemini model naming

The tutorial's tech-stack diagram names `gemini-pro-vision` as the image-recognition model. That model name is long deprecated — Gemini has moved through the 1.5, 2.x, and (as of September 2026) 3.x Flash generations. BloomLens will target whichever current Gemini Flash multimodal model alias is live at build time rather than hardcoding a retired model ID.

## Open questions carried into Phase B

- Exact current Gemini model ID/alias to pin in `requirements.txt` / code (verify at build time — these rotate on the order of months).
- Final curated species list (~30–50) and where each one's short description/taxonomy text is sourced from (plan proposes hand-curated, single-source-of-truth JSON to keep it accurate at small scale).
- Whether Qdrant runs in local/embedded mode (simplest, no server) or via Docker/Cloud free tier for the demo.
