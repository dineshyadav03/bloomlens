# What "quality" means in the cut-flower trade, and what a photo can tell you about it

BloomLens's `quality_grade` (A/B/C) is **the original tutorial's invention** — it does not
correspond to any published grading standard. Before validating it against expert labels (M18),
this document first asks the more basic question the earlier plan required: *what does the real
trade actually grade, and how much of that can be judged from a single photo at all?* The answer
reshapes what BloomLens can honestly claim even after expert labels exist.

## Sources, and how they were read

| Source | What it is | How it was consulted | Retrieved |
|---|---|---|---|
| **VBN (Vereniging van Bloemenveilingen in Nederland / Royal FloraHolland) — General Trading Rules and Product Specifications** | The rules actually used at the world's largest cut-flower auctions | Royal FloraHolland's own pages (`royalfloraholland.com/en/buying-2/quality/regulations/product-specifications`, `vbn.nl/en/the-vbn-group-code`) confirm the A1/A2/B1 quality-class system and its role; the specific class wording below is corroborated by an independent wholesale-trade summary (Eagle-Link Flowers) that quotes it near-verbatim. **The full primary specification document (behind VBN's own product-search tool) was not reached** — treat the class wording as a close paraphrase, not a verified quotation, until read directly. |
| **UNECE Standard H-1, "Cut Flowers"** (UN Economic Commission for Europe) | The international reference standard for cut-flower quality classes (Extra / Class I / Class II, the same pattern UNECE uses for fresh produce) | Confirmed to exist via UNECE's own site listing; **the document itself returned an access error (bot-protection) and was not read**. Nothing below is sourced to it beyond the fact of its existence and its three-tier class structure, which is publicly documented by UNECE's produce-standard pattern generally. |
| **"Recommended Grades and Standards for Fresh Cut Flowers"**, Floral Marketing Association & Society of American Florists (1994, expanded 1996 to 21 crops) | A real US/Canada industry grading manual, read directly | Read directly (106 pages, via a public copy hosted by Flowers Canada Growers, viewed page by page). One full crop worked example (Alstroemeria) is reproduced below with page-accurate detail; the manual's structure (repeated per crop) was confirmed by inspection. | 2026-09-22 |
| Wholesale-trade summary of VBN rose grading (Eagle-Link Flowers) | Secondary, but specific and internally consistent with the VBN pages above | Read directly | 2026-09-22 |

This is engineering documentation, not a legal or trade reference — verify against the primary
documents before relying on any of it for a real transaction.

## What the real trade actually grades

### VBN / Royal FloraHolland: three quality classes, not three letters

Cut flowers at Dutch auction trade in **three quality groups: A1, A2, B1** (not A/B/C). Paraphrased
from Royal FloraHolland's own description and corroborated by the trade summary above:

- **A1** must meet *all* minimum requirements: internal quality, freshness, freedom from
  parasites/damage/deficiencies/deviations/contamination, no leaves on the lower 10 cm of the
  stem, a stem straight and sturdy enough to bear the flower, uniformity of colour/thickness/
  sturdiness/bouquet volume, and proper packaging.
- **A2, B1** — the same lot, with one or more of those requirements not met; the specific
  deviation determines which.
- Anything not meeting even B1 is not traded at all.

For roses specifically, growers grade by: stem length (bunched even at the base), ripeness
(bud-opening stage), number of bloomable buds, **flower-bud height in 1 cm classes** (a real
measurement, coded `S19`), and stems per bunch — and the batch must be free of named growth
defects: **flat buds, grass hearts (a green/leafy centre instead of proper petals), and crooked
necks (a bent stem just below the head)**.

### The Floral Marketing Association / SAF manual: numeric grades, physical measurements

Every one of the manual's 21 crops follows the same structure — Product description → About the
crop → **Dimensions** (a grade table) → **Stages of development** (four reference drawings) →
**Characteristics** (Stems/Foliage, Flowers, General) → Terminology. The grades are **numbers**
(1/2/3), each tagged with a colour (a physical tag colour used at packing, not a quality
adjective) — not the letters A/B/C.

**Worked example — Alstroemeria (Peruvian lily, a BloomLens-covered species), reproduced in full:**

| Grade | 1 | 2 | 3 |
|---|---|---|---|
| Colour designation | Blue | Yellow | Red |
| Minimum length | 30 in / 75 cm | 26 in / 65 cm | 24 in / 60 cm |
| Minimum flower diameter | 1½ in / 3.75 cm | 1½ in / 3.75 cm | 1½ in / 3.75 cm |
| Stem strength | 20° | 20° | 20° |
| Stem deviation / curvature | 3 in / 7.5 cm | 3 in / 7.5 cm | 3 in / 7.5 cm |
| Minimum flowers per stem | 3 | 3 | 3 |

("Stem strength" is measured as a deflection angle under a standard load; "stem deviation" is the
gap between the stem and a straightedge laid along it — both are physical bend/rigidity tests, not
visual judgements.)

The manual's **Characteristics** section for Alstroemeria, condensed:

- *Stems/Foliage:* pale-to-medium green, glossy; free from damage by insects or disease, and free
  from physical damage — spots, holes, rotted tissue, wilting, burning, or discolouration.
- *Flowers:* colour and speckling vary by variety; maturity may vary stem-to-stem; free from
  discolouration, burning, insect damage, disease, and chemical residue.
- *General:* storage temperature, ethylene sensitivity and preharvest treatment, and anticipated
  vase life **under specified storage conditions** — none of which a photo can show.

## What can and cannot be judged from a single photo

Cross-referencing the criteria above against what a photo — one angle, one moment, no reference
scale, no physical contact with the stem — can actually show:

| Visible in a photo (with caveats) | Not visible in a photo at all |
|---|---|
| Bloom/development stage (bud vs. half-open vs. fully open) | **Any physical measurement without a reference object**: stem length, flower diameter, stem strength (a bend test), stem deviation (needs a straightedge) |
| Petal browning, wilting, discolouration, visible blemishes | **Vase life** — inherently a multi-day observation, not a single-frame one |
| Visible pest damage or disease spots on petals/leaves | **Cold-chain and storage history** (temperature, humidity, time since harvest) |
| Foliage colour and apparent glossiness | **Preharvest treatment** (e.g. ethylene inhibitors like STS) |
| Obvious growth defects if framed in view (e.g. a visibly bent neck) | **Internal quality** (VBN's own term for defects not visible from outside) |
| Approximate flower/bud count, if all are in frame | **Chemical residue** (not visible without lab testing) |
| Colour and, for multi-flower species, apparent uniformity across visible stems | **Stem sturdiness** as a physical property (only inferable, not measured) |

This is the honest ceiling on what any photo-based system — BloomLens included — can assess, no
matter how well it is validated. **A visual read is a real, useful signal for freshness and
visible damage, and nothing more.**

## What this means for BloomLens's own grade

- The rating target is renamed **visual condition class**, to stop implying it is a market grade.
  It stays A/B/C for now (changing the schema is out of scope for this milestone) but the docs,
  UI copy and this file are explicit that **A/B/C is the tutorial's own invention**, unrelated to
  VBN's A1/A2/B1 or the FMA/SAF manual's 1/2/3.
- A *visual condition class*, honestly scoped, can only ever reflect: bloom stage, visible
  discolouration/browning, visible wilting, and visible pest or mechanical damage. It cannot
  reflect stem length, stem strength, vase life, cold-chain history, or "internal quality" in
  VBN's sense — those require a ruler, a bend test, days of observation, or a lab, respectively.
- **Building a labeler (this milestone) is not validation.** Until blinded expert labels exist
  (M18) and agreement is measured against them (`docs/quality/PROTOCOL.md`), BloomLens's grade
  stays labeled **"unvalidated heuristic"** everywhere it is shown, per the earlier review's
  explicit instruction. Nothing in this file changes that.
