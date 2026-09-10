---
title: BloomLens
emoji: 🌸
colorFrom: pink
colorTo: purple
sdk: docker
pinned: false
---

# 🌸 BloomLens

Point a camera at a flower, get its **species, quality, and price** back — instantly.

This is the live demo for **BloomLens**, a portfolio project for the cut-flower supply chain, inspired by the scale of the world's largest flower auction (Royal FloraHolland — roughly 43 million flowers change hands there on a typical weekday) and the "scan to learn" interaction from Dubai's Museum of the Future. Under the hood: **BioCLIP 2** (a real open-source vision-language model for the tree of life) for zero-shot species ID against a curated taxonomy vector index, a **LangChain** tool-calling agent over **Gemini** for the quality read and reasoning, and a measured **87.2% top-1 / 96.1% top-3** species-ID accuracy on 360 held-out test images.

Try it: allow camera access, point it at a real flower (or upload a photo), and scan.

**Disclaimers:** pricing shown is simulated demo data — FloraHolland doesn't expose a public real-time pricing API. Quality assessment is a heuristic visual read, not a calibrated agronomic grading system.

**Full source, architecture, and research writeup:** [github.com/dineshyadav03/bloomlens](https://github.com/dineshyadav03/bloomlens)
