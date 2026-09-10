# BloomLens evaluation results

Measured against 360 held-out Oxford 102 Flowers test images across 18 of BloomLens's 30 curated species — see `eval/species_mapping.py` for exactly which species and why (several via well-established alternate common names, not string matches). This evaluates **only BioCLIP 2 + Qdrant retrieval** (`embed_image` + `vector_store.search`, the same functions `identify()` uses) — no Gemini call, no API cost, and no bearing on the agent's quality/summary output, which isn't the kind of thing accuracy metrics apply to.

## Headline numbers

- **Top-1 accuracy: 87.2%** (314/360)
- **Top-3 accuracy: 96.1%** (346/360)

## Per-species accuracy

| true_species | n | top1_acc | top3_acc |
| --- | --- | --- | --- |
| Gladiolus | 20.0 | 0.3 | 0.75 |
| Bearded iris | 20.0 | 0.4 | 0.9 |
| Marigold | 20.0 | 0.65 | 0.7 |
| Dahlia | 20.0 | 0.7 | 1.0 |
| Daffodil | 20.0 | 0.8 | 0.95 |
| Amaryllis | 20.0 | 0.85 | 1.0 |
| Calla lily | 20.0 | 1.0 | 1.0 |
| Bird of paradise | 20.0 | 1.0 | 1.0 |
| Gerbera daisy | 20.0 | 1.0 | 1.0 |
| Carnation | 20.0 | 1.0 | 1.0 |
| King protea | 20.0 | 1.0 | 1.0 |
| Moth orchid | 20.0 | 1.0 | 1.0 |
| Oxeye daisy | 20.0 | 1.0 | 1.0 |
| Peruvian lily | 20.0 | 1.0 | 1.0 |
| Rose | 20.0 | 1.0 | 1.0 |
| Snapdragon | 20.0 | 1.0 | 1.0 |
| Sunflower | 20.0 | 1.0 | 1.0 |
| Sweet pea | 20.0 | 1.0 | 1.0 |

## Confusion matrix (rows = true species, columns = predicted)

| true_species | Amaryllis | Bearded iris | Bird of paradise | Calla lily | Carnation | Daffodil | Dahlia | Easter lily | Gerbera daisy | Gladiolus | King protea | Lisianthus | Marigold | Moth orchid | Oxeye daisy | Persian buttercup | Peruvian lily | Rose | Snapdragon | Sunflower | Sweet pea |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Amaryllis | 17 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Bearded iris | 0 | 8 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| Bird of paradise | 0 | 0 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Calla lily | 0 | 0 | 0 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Carnation | 0 | 0 | 0 | 0 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Daffodil | 0 | 0 | 0 | 0 | 4 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Dahlia | 0 | 0 | 0 | 0 | 0 | 0 | 14 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 |
| Gerbera daisy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Gladiolus | 2 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 6 | 0 | 1 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| King protea | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Marigold | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 13 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| Moth orchid | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Oxeye daisy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 0 | 0 | 0 | 0 | 0 | 0 |
| Peruvian lily | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 0 | 0 | 0 | 0 |
| Rose | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 0 | 0 | 0 |
| Snapdragon | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 0 | 0 |
| Sunflower | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 0 |
| Sweet pea | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 |

## Confidence-tier calibration

Current thresholds (`src/identify.py`): high ≥ 0.55 score and ≥ 0.05 gap to runner-up; low < 0.45. These were a heuristic guess from a handful of Milestone 1 photos — here's how they hold up against real held-out data:

| tier | n | precision |
| --- | --- | --- |
| ambiguous | 150.0 | 0.7 |
| high | 210.0 | 0.995 |

Of 314 correct top-1 predictions: 209 landed in **high**, 105 landed in **ambiguous** — i.e. 105 correct calls get a hedge/switcher shown even though they were actually right.

**Could loosening the thresholds fix that without hurting reliability?** Swept a couple of alternatives against this data instead of guessing:

| thresholds | n | precision |
| --- | --- | --- |
| current | 212.0 | 0.9906 |
| looser gap (0.03) | 259.0 | 0.9691 |
| looser score (0.5) | 212.0 | 0.9906 |

Loosening the gap requirement to 0.03 would move 47 more correct predictions into `high`, but drops precision from 99.06% to 96.91% — a real degradation for a tier whose entire point is being trustworthy. Loosening the score floor alone changes nothing (same n, same precision), meaning the gap requirement — not the score floor — is what's actually binding. Why: mean score/gap for correct vs. wrong predictions *within* the ambiguous zone are nearly identical —

| correct_top1 | top1_score | gap |
| --- | --- | --- |
| False | 0.639 | 0.016 |
| True | 0.65 | 0.025 |

— so there's no cleaner cutoff hiding in the data; correct and wrong predictions in that zone are genuinely hard to tell apart by score/gap alone. **Conclusion: thresholds are kept as-is.** The eval validates the original heuristic rather than replacing it — a good outcome for an evaluation to produce, not a null result. Note the `low` tier has no data points here at all: every test image is a genuine match to a known species, so this evaluation can't validate that threshold — doing so would need deliberately-included out-of-distribution/non-flower images, noted here as future work rather than left silently unstated.

## Species not covered by this evaluation

12 of the 30 curated species aren't in Oxford 102 (or only have an unconfirmed genus-level match, deliberately excluded to keep this measurement honest):

- **Chrysanthemum** — not in Oxford 102
- **Easter lily** — not in Oxford 102
- **Peony** — not in Oxford 102
- **Bigleaf hydrangea** — not in Oxford 102
- **Freesia** — not in Oxford 102
- **Poppy anemone** — Oxford has generic 'windflower' -- not confirmed to be Anemone coronaria specifically
- **Persian buttercup** — Oxford has generic 'buttercup' -- not confirmed to be Ranunculus asiaticus specifically
- **Lisianthus** — not in Oxford 102
- **Zinnia** — not in Oxford 102
- **Hyacinth** — Oxford only has 'grape hyacinth' (Muscari, a different genus)
- **Statice** — not in Oxford 102
- **Tulip** — Oxford only has 'siam tulip' (Curcuma alismatifolia, a different genus entirely)