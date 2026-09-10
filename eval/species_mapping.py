"""Oxford 102 Flowers category names -> BloomLens curated species (common_name in
data/species_reference.json). Only species with a confident match are included —
see docs/RESEARCH.md for how each one was verified (several are via well-established
alternate common names, e.g. "barberton daisy" = Gerbera jamesonii, not a string match).

Deliberately excludes loose genus-only matches (e.g. Oxford's generic "buttercup" is
not confirmed to be Ranunculus asiaticus specifically) to keep the measured accuracy
honest about what it covers.
"""

OXFORD_TO_BLOOMLENS = {
    "rose": "Rose",
    "carnation": "Carnation",
    "sunflower": "Sunflower",
    "oxeye daisy": "Oxeye daisy",
    "bearded iris": "Bearded iris",
    "barberton daisy": "Gerbera daisy",
    "sword lily": "Gladiolus",
    "daffodil": "Daffodil",
    "peruvian lily": "Peruvian lily",
    "snapdragon": "Snapdragon",
    "orange dahlia": "Dahlia",
    "pink-yellow dahlia": "Dahlia",
    "marigold": "Marigold",
    "english marigold": "Marigold",
    "hippeastrum": "Amaryllis",
    "king protea": "King protea",
    "giant white arum lily": "Calla lily",
    "sweet pea": "Sweet pea",
    "bird of paradise": "Bird of paradise",
    "moon orchid": "Moth orchid",
}

# BloomLens species NOT covered by this evaluation, and why -- so eval/results.md
# can state its scope honestly instead of implying full 30-species coverage.
UNCOVERED_SPECIES = {
    "Chrysanthemum": "not in Oxford 102",
    "Easter lily": "not in Oxford 102",
    "Peony": "not in Oxford 102",
    "Bigleaf hydrangea": "not in Oxford 102",
    "Freesia": "not in Oxford 102",
    "Poppy anemone": "Oxford has generic 'windflower' -- not confirmed to be Anemone coronaria specifically",
    "Persian buttercup": "Oxford has generic 'buttercup' -- not confirmed to be Ranunculus asiaticus specifically",
    "Lisianthus": "not in Oxford 102",
    "Zinnia": "not in Oxford 102",
    "Hyacinth": "Oxford only has 'grape hyacinth' (Muscari, a different genus)",
    "Statice": "not in Oxford 102",
    "Tulip": "Oxford only has 'siam tulip' (Curcuma alismatifolia, a different genus entirely)",
}
