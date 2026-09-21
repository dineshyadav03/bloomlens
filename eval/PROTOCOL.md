# Open-world evaluation protocol — version 1 (pre-registration)

**Status: pre-registered.** This document was committed **before any new test number was computed**. It
fixes the datasets, the splits, the scores, the way thresholds are chosen, the metrics and the rule that the
test set is evaluated **once**. Changing anything below after the test run means a new dataset version
(a new salt, `v2`), not an edit — and the old result stays published as it was.

The existing retrieval evaluation (`eval/results.md`: 87.2 % top-1 / 96.1 % top-3 on 360 images) is a
*closed-world, in-distribution* measurement. It says nothing about what the system does with a photo that is
**not** one of its species. This protocol is for that question.

## 1. The question, and what is not being asked

> When a photo is *not* one of BloomLens's covered species — a different flower, a non-flower object, a
> degraded photo of a covered species — does the **retrieval stage's decision rule** abstain, while it keeps
> answering on genuine covered-species photos?

Measured here: the retrieval stage only (BioCLIP 2 embedding + Qdrant similarities to the 30 species
descriptions), because the abstention rule is built on those scores. **Not** measured here: quality
grading (no labels exist — see `docs/quality/`), price (simulated), and the full agent system beyond the
capped pilot in §9. Nothing in this protocol licenses a claim about those.

## 2. Datasets

Every image is identified by a **source id** (`oxford102:image_05494`, `caltech101:<category>/image_0012`).
A "variant" (a corrupted copy, §2.5) inherits its source's split. Manifests live in `eval/manifest/`
(source id, family, label, group, split, relative path, SHA-256). Images are **not** committed except the
360 that already were; they are fetched by `eval/build_datasets.py` from the public originals.

### 2.1 The legacy 360 — `dev-legacy`

The 360 images in `eval/test_images/` (20 × 18 species) **shaped the current confidence thresholds**
(Milestone 5's sweep) and the tier `low` was never exercised on them. They are therefore *development*
data, never a final test. They keep their existing role in CI (the 80 % retrieval regression gate).

### 2.2 In-distribution pool (`id`)

The 20 Oxford 102 categories that map confidently onto 18 covered species (`eval/species_mapping.py`) —
**all Oxford images of those categories not among the legacy 360**: 1 182 images. *Deviation from the plan,
stated up front:* the plan said "Oxford test split"; that split leaves too few images per species (Moth orchid
would have none), so the pool also uses Oxford's train/val images. This is sound here because BloomLens
trains nothing on images — the index is built from *text* — so Oxford's own partition carries no meaning for
it. Each image records its original Oxford split for transparency.

**12 covered species have no Oxford data** (Chrysanthemum, Easter lily, Peony, Bigleaf hydrangea, Freesia,
Poppy anemone, Persian buttercup, Lisianthus, Zinnia, Hyacinth, Statice, Tulip). They cannot be evaluated as
in-distribution, and a real deployment where *they* appear is **not measured**. Tulip and Hyacinth in
particular are common cut flowers.

### 2.3 Near-OOD (`near_ood`)

Oxford categories that are **not** among the 30 species. Categories with an unresolved taxonomic overlap are
removed from *both* sides — they are neither ID nor near-OOD:

`buttercup` (not confirmed to be *Ranunculus asiaticus*), `windflower` (not confirmed to be *Anemone
coronaria*), `grape hyacinth` (*Muscari*, not *Hyacinthus*), `siam tulip` (*Curcuma*, not *Tulipa*),
`bishop of llandaff` (a *Dahlia* cultivar, and Dahlia is a covered species).

That leaves **77 categories** (6 331 images). **Near-OOD is split by category, not by image**: whole
categories go to dev, test or pilot, so the thresholds are tuned on some unseen categories and tested on
*other* unseen categories — that is what "open world" means. Categories are capped at 20 images (the 20 with
the smallest hash), so no category dominates.

### 2.4 Far-OOD (`far_ood`)

A **different-source** set with no flowers: Caltech-101 (137 MB, CC BY 4.0, Li Fei-Fei et al.), minus
`sunflower`, `water_lilly`, `lotus` (flowers), `Faces`, `Faces_easy` (people — no faces are sent to a third-party
API) and `BACKGROUND_Google` (unlabeled clutter that can contain anything). Split by category like near-OOD,
capped at 20 images per category. Caltech-101 images are not redistributed by this repository.

Far-OOD is *easier* than near-OOD by construction and is reported separately; a good far-OOD number must never
be presented as open-world robustness.

### 2.5 Corrupted in-set (`corrupted_id`) — robustness, **not** OOD

Covariate shift on covered species: for each covered species, the 5 dev and the 5 test ID images with the
smallest hash, each in 8 deterministic variants: `gaussian_blur` (σ = 2 % of the short side), `low_res`
(short side 48 px, upscaled back), `dark` (brightness × 0.25), `occlusion` (black square over 30 % of the
area), `crop` (random 40 %-area crop, resized back), `jpeg` (quality 8), `rotate` (45°), `color_shift` (hue
+40/256, saturation × 0.6). The label stays the species; the right behaviour is "correct or abstain", and
these are reported as accuracy / abstention per corruption, never as an OOD AUROC.

### 2.6 Split assignment (deterministic, before anything is measured)

`bucket(key) = int(sha256("bloomlens-eval-v1|" + key)[:12], 16) / 16^12`.

- **ID** — stratified by species: the species' source ids sorted by bucket; the first 40 % → `dev`, the next
  40 % → `test`, the last 20 % → `pilot` (dev = round(0.4 n), pilot = round(0.2 n), test = the rest).
- **near-OOD / far-OOD** — categories sorted by bucket with the same 40/40/20 rule; then ≤ 20 images per
  category.
- **Leakage rules, enforced by tests on the committed manifests:** every source id has exactly one split;
  no OOD category appears in two splits; no image is in both the legacy 360 and a new split; every variant's
  source is in the same split as its variants; rebuilding gives byte-identical manifests.

### 2.7 Caveats that are part of the protocol

- **Pretraining contamination.** BioCLIP 2 was trained on TreeOfLife-200M, which can contain photos of these
  species and possibly these very images. "Held out" means held out from *our* thresholds and index, not from
  the model's pretraining. Scores may be optimistic for Oxford-like photos.
- **Non-independence.** Oxford photos of one category are often the same plant from several angles; a source
  split does not make them independent. Confidence intervals resample images (ID) or categories (OOD) and are
  still probably optimistic.
- **Domain.** Oxford is garden and wild flowers photographed for a UK dataset — not auction lots, not cut
  stems in buckets. Nothing here generalises to that domain.

## 3. Scores

Each image gets three candidate "in-distribution-ness" scores, computed from the similarities of the image
embedding to **all 30** species vectors (`search(top_k = 30)`). Higher = more in-distribution.

| id | score | formula |
|---|---|---|
| S1 | max cosine | the top-1 similarity |
| S2 | margin | top-1 − top-2 similarity |
| S3 | max-softmax at T | max_j softmax(sim_j / T), T ∈ {0.01, 0.02, 0.05, 0.1} |

## 4. Choosing the rule — on `dev` only

1. **Score and temperature:** the candidate (S1, S2, or S3 at one of the four T) with the highest **dev
   AUROC** separating ID-dev (positive) from near-OOD-dev. Candidates within 0.005 AUROC are tied and broken
   in the order S1, S2, S3 (simplest first).
2. **Threshold τ:** the 5th percentile of the chosen score over ID-dev *original* images
   (`numpy.percentile(scores, 5, method="lower")`), i.e. τ is the largest value that still accepts at least 95 %
   of dev ID images. **Abstain iff score < τ.** No OOD image influences τ.
3. **Baseline for comparison:** the current rule "abstain iff confidence tier is `low`" (top-1 similarity
   < 0.45), evaluated on the same data.
4. **Adoption (decided now, from dev only):** the new rule replaces the tier-`low` gate only if, on **dev**, it
   abstains on at least 10 percentage points more near-OOD images than the baseline **and** abstains on at most
   8 % of ID-dev images. (These two numbers are this protocol's judgement calls, fixed before seeing data.)
   Otherwise the baseline stays and the result is reported as "no improvement found".
5. **Freeze:** the chosen score, T, τ, the adoption decision, the manifest hashes and the code revision are
   written to `eval/frozen/v1.json` with a SHA-256 of that file, and committed **before** the test set is touched.

## 5. Test — once

`eval/run_open_world.py` evaluates `test` exactly once against the frozen file and writes a lock beside it;
it refuses to run again for the same frozen hash. There is no "does it replicate?" run followed by a revision.
If the numbers are poor they are published as they are. Anything else needs `v2`.

## 6. Metrics — per family, never pooled

For **near-OOD**, **far-OOD** and **corrupted-in-set** *separately* (and ID alone where noted):

- **AUROC** (ID positive, higher score = more ID), **AUPR-Out** (OOD positive), and **FPR@95 %TPR** (the share of
  OOD accepted when 95 % of ID is accepted) — threshold-free, on the frozen score.
- At the frozen τ: the **abstention rate** for each family (for ID it is the *false-abstain* rate; for OOD it is
  the *correct-rejection* rate).
- **On ID at τ:** top-1 accuracy of the accepted images (selective accuracy), coverage, and the risk–coverage
  curve with its area.
- **Corrupted-in-set:** top-1 accuracy and abstention rate per corruption type.
- Every figure with a **95 % percentile-bootstrap CI** (10 000 resamples, seed 20260921; resampling images for ID,
  **categories then images** for OOD) and its **n** (images and, for OOD, categories). The baseline tier rule is
  reported next to the frozen rule.
- No per-species claims: 20–30 images per species cannot support them; per-species tables are descriptive only.

## 7. The machine-readable abstention signal

`IdentifyResult.abstained` (bool) and `abstain_source` (`"retrieval"`, `"agent"`, `"both"` or `null`) are computed
from the retrieval policy and the agent's explicit `matches_a_candidate` boolean — **never by looking for words
in text**. Until a rule is frozen and adopted, the policy is the tier-`low` gate.

## 8. Deliberately not done

Tuning on `test`; a second test run; pooling families into one headline; using far-OOD to choose a threshold;
keyword detection of "not a flower"; claiming anything about the 12 uncovered species, about auction-lot photos,
or about quality and price.

## 9. End-to-end pilot (M15c) — a pilot, not a benchmark

On the `pilot` split only: about 90 ID images (5 per species) and about 30 OOD images, a hard `--max-calls`
cap on Gemini requests, a resumable cache keyed by image hash, model and code revision, the frozen thresholds.
It reports agreement of the final species with retrieval top-1, schema-valid and failure rates, latency, tokens
and estimated cost (from telemetry), and the abstention rate by family. It is too small to support a headline
number and will say so.

## 10. Deviations from the milestone plan

1. The ID pool includes Oxford train/val images (§2.2), because the test split alone is too small.
2. Near-OOD is split **by category**, not by image (§2.3) — a stronger open-world test than the plan required.
3. `bishop of llandaff` is also excluded from near-OOD (a *Dahlia* cultivar) beyond the four gray-zone classes
   named in the plan.
4. Far-OOD is Caltech-101 (§2.4), chosen after checking size and licence for this machine's slow link.
