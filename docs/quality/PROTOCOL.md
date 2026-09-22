# Quality-label agreement protocol (v1, pre-registered)

This protocol is written and committed **before any label has been collected**. Nothing in it may
be revised after seeing results; a change requires a new protocol version and a note explaining
why. This mirrors `eval/PROTOCOL.md`'s pre-registration discipline, applied to human labels instead
of model scores.

Scope: this protocol measures **inter-rater agreement** on a *visual condition class* (see
[STANDARD.md](STANDARD.md)) — it does not, by itself, validate BloomLens's `quality_grade` output.
Validating the model against these labels is a separate, later step (M18) that this protocol does
not perform.

## 1. What is being labeled

Each item is one flower photo (or, for a lot, the full set of photos of that lot) drawn from the
existing `eval/manifest/id.json` `pilot`/`dev`/`test` pool, or an equivalent held-out set. Raters
never see the model's output, the retrieval candidates, or each other's labels (see §5, blinding).

Raters assign one of:

- **A** — visually excellent condition (fresh, fully turgid, no visible damage or discoloration)
- **B** — visually good condition (minor, non-disqualifying blemish, slight bloom-stage variance)
- **C** — visually fair condition (visible wilting, browning, blemish, or damage)
- **Cannot grade** — the photo does not show enough of the flower to judge (see §4)

This is deliberately the same three-letter scheme BloomLens's `assess_quality` tool already uses
([tools.py:55-60](../../src/tools.py#L55-L60), [identify.py:57-59](../../src/identify.py#L57-L59))
so that agreement, once measured, is comparable to the model's actual output space — not a
different rating scale translated after the fact. It is still, per STANDARD.md, **the tutorial's
own invented scale**, not a market grade.

## 2. Rater qualifications and training

- At least 2 raters per item (§6 adjudication needs a 3rd on disagreement); raters should have
  practical experience with cut flowers (florists, growers, flower-shop staff, horticulture
  students) — not necessarily professional graders, since BloomLens's grade is explicitly *not*
  a market grade.
- **Calibration round before real labeling**: every rater labels the same fixed set of 15
  practice images (drawn from `dev`, never reused in the measured set) and discusses disagreements
  as a group against the STANDARD.md definitions above. This round's labels are never included in
  the reported agreement statistics — its only purpose is to align raters' mental model of A/B/C
  before labels that count are collected.
- Each rater records, once, their self-described relevant experience (free text) — reported
  descriptively alongside results, never used to exclude or weight a rater's labels after the
  fact.

## 3. Blinding

- Raters see only the photo(s) — never BloomLens's predicted species, quality grade, quality
  note, retrieval candidates, or confidence tier.
- Raters never see another rater's label for the same item until all raters have submitted all
  their labels (enforced by the labeler tool, §7 — it never displays another rater's grade, and
  submissions are append-only).
- Raters are not told which images are "calibration" vs "real" during the calibration round
  itself, only afterward when calibration labels are discarded — this keeps the calibration round
  representative of real labeling behavior.

## 4. "Cannot grade" handling

An item marked "cannot grade" (framing too tight/wide, motion blur, extreme lighting, obstruction)
is:

- **Excluded from the primary agreement calculation** (Krippendorff's α and weighted κ are
  computed only over items where all raters gave a substantive A/B/C label).
- **Reported as a rate**: the fraction of items at least one rater marked "cannot grade", and the
  fraction where *all* raters agreed it was ungradable (a form of agreement in itself, reported
  separately, not folded into the ordinal statistics).
- **Checked with a sensitivity analysis**: agreement is recomputed once treating "cannot grade" as
  its own ordinal category (below C) to confirm the headline exclusion doesn't hide a large
  disagreement driven by borderline-ungradable images. Both numbers are reported side by side —
  neither replaces the other.

## 5. Adjudication rule

- Any item where two raters' grades differ by more than one letter (i.e., A vs. C) is sent to a
  **third, independent rater** who has not seen the first two raters' labels or this item before.
- The adjudicator's label is recorded as a fourth data point, never as a silent replacement for
  the original two — the raw disagreement is kept in the dataset for the agreement calculation
  (adjudication informs a resolved "consensus label" used only if this project later needs one
  ground-truth label per item; it does not change the inter-rater statistics themselves, which
  must reflect what raters actually disagreed on).
- Items where two raters differ by exactly one letter (A vs. B, or B vs. C) are **not**
  adjudicated — a one-letter gap on a 3-point ordinal scale is within-protocol normal
  disagreement, and adjudicating it would bias the reported agreement upward.

## 6. Sample size and what it can and cannot support

- Target: the existing dataset scale referenced elsewhere in this project (~150 images spanning
  the 30 curated species) — i.e., **no new photo collection is assumed by this protocol**; it
  labels a sample drawn from data already gathered for the retrieval evaluation, at roughly 5
  images per species. This scale is intentionally treated as supporting **pooled analysis only,
  not per-species** analysis (justification below).
- **Explicit statement, not a caveat buried in a footnote: with ~150 images across 30 species
  (~5 per species), this study cannot and will not report per-species agreement or per-species
  validation.** Any per-species breakdown shown is descriptive only (e.g., "3 of 5 Rose photos had
  full agreement"), explicitly labeled as too small to generalize, and never presented as a
  per-species accuracy or reliability claim.
- All primary agreement statistics (Krippendorff's α, weighted κ) are computed **pooled across all
  species** — this is what the sample size can actually support.
- Confidence intervals (bootstrap, §8) are reported alongside every pooled statistic so the
  reader can see the real uncertainty from this sample size, rather than a bare point estimate.

## 7. Metrics

Computed by `eval/quality_agreement.py`, which wraps published, vetted libraries — **no
hand-rolled coefficient math**:

| Metric | Library | Role |
|---|---|---|
| **Krippendorff's α (ordinal)** | `krippendorff` | Primary. Handles missing labels (raters who didn't label every item) natively, and the ordinal distance metric matches the A>B>C structure — a A/C disagreement counts as worse than an A/B one. |
| **Weighted (quadratic) Cohen's κ** | `scikit-learn` (`cohen_kappa_score(..., weights="quadratic")`) | Primary, pairwise. Reported for every rater pair, since κ (unlike α) is inherently two-rater; with >2 raters we report the pairwise matrix and its mean, not a single fabricated "overall κ". |
| **Fleiss' κ** | `statsmodels` (`fleiss_kappa`) | Optional/secondary, unweighted multi-rater agreement — included for readers who expect it, explicitly noted as *not* ordinal-aware (treats A-vs-C the same as A-vs-B), so it is reported alongside, never in place of, α. |
| **Bootstrap confidence intervals** | `scipy.stats.bootstrap` | Applied to α and to each pairwise weighted κ, resampling **items** (not label pairs) so the resampling unit matches what varies between hypothetical repeats of this study. |

Every wrapper function in `eval/quality_agreement.py` is tested against a **published worked
example** (a numeric case from a paper or textbook with a known correct α/κ value), not only
synthetic sanity checks — so a library-usage mistake (wrong axis, wrong weighting, wrong missing-
data convention) is caught by comparing against an independently-known-correct number, not just
against another synthetic case built with the same possible misunderstanding.

## 8. What this protocol does not do

- It does not compare rater labels to BloomLens's model output. That comparison (validating
  `quality_grade` against these human labels) is explicitly **out of scope for this document** —
  see M18 in the project plan, which is blocked on recruiting qualified raters and is not started.
- It does not, by itself, make BloomLens's quality grade anything other than an **unvalidated
  heuristic**. Running this protocol and getting a high agreement number would show that *humans*
  can consistently apply the A/B/C visual-condition scale to a photo — it says nothing about
  whether the *model* agrees with humans until that separate comparison is run.
- It makes no claim beyond the pooled, ~150-image, 30-species sample it is run on. It is not
  a benchmark and does not produce a number suitable for a marketing claim about accuracy on any
  individual species.
