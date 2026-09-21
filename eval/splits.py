"""Deterministic split assignment for the open-world evaluation (eval/PROTOCOL.md section 2.6).

Pure functions, no I/O: everything is a function of the ids and the salt, so the manifests can
be rebuilt byte-for-byte and the leakage rules can be tested on the committed files.

A "bucket" is a stable pseudo-random number in [0, 1) derived from SHA-256. Sorting by bucket
gives every item a fixed, salt-dependent position; taking prefixes of that order gives the splits.
"""

import hashlib
from collections.abc import Iterable, Mapping
from typing import Literal

SALT = "bloomlens-eval-v1"
Split = Literal["dev", "test", "pilot"]
SPLITS: tuple[Split, ...] = ("dev", "test", "pilot")
DEV_FRACTION, PILOT_FRACTION = 0.4, 0.2  # the test split is whatever is left (about 40 %)
MAX_IMAGES_PER_CATEGORY = 20
CORRUPTION_SOURCES_PER_SPECIES = 5


def bucket(key: str, salt: str = SALT) -> float:
    """A stable number in [0, 1) for `key`: the same on every machine and every run."""
    digest = hashlib.sha256(f"{salt}|{key}".encode()).hexdigest()
    return int(digest[:12], 16) / 16**12


def by_bucket(keys: Iterable[str], salt: str = SALT) -> list[str]:
    """`keys` in bucket order (ties, which need a SHA-256 prefix collision, broken by the key)."""
    return sorted(keys, key=lambda k: (bucket(k, salt), k))


def split_sizes(n: int) -> tuple[int, int, int]:
    """(dev, test, pilot) sizes for `n` items: dev = round(0.4 n), pilot = round(0.2 n), test = the rest."""
    dev, pilot = round(DEV_FRACTION * n), round(PILOT_FRACTION * n)
    return dev, n - dev - pilot, pilot


def assign_splits(keys: Iterable[str], salt: str = SALT) -> dict[str, Split]:
    """Split `keys` 40/40/20 by bucket order: the first `dev` go to dev, the next to test, the rest to pilot."""
    ordered = by_bucket(set(keys), salt)
    dev, test, _ = split_sizes(len(ordered))
    return {key: "dev" if i < dev else "test" if i < dev + test else "pilot" for i, key in enumerate(ordered)}


def assign_stratified(groups: Mapping[str, Iterable[str]], salt: str = SALT) -> dict[str, Split]:
    """Split each group (e.g. each species) on its own, so every group appears in every split."""
    assignment: dict[str, Split] = {}
    for _, keys in sorted(groups.items()):
        assignment.update(assign_splits(keys, salt))
    return assignment


def cap_per_category(
    images_by_category: Mapping[str, Iterable[str]], cap: int = MAX_IMAGES_PER_CATEGORY, salt: str = SALT
) -> dict[str, list[str]]:
    """At most `cap` images per category: those with the smallest bucket (a fixed pseudo-random sample)."""
    return {category: by_bucket(images, salt)[:cap] for category, images in sorted(images_by_category.items())}


def corruption_sources(
    ids_by_species_split: Mapping[tuple[str, str], Iterable[str]],
    per_species: int = CORRUPTION_SOURCES_PER_SPECIES,
    salt: str = SALT,
) -> dict[tuple[str, str], list[str]]:
    """For each (species, split) the `per_species` ID images with the smallest bucket."""
    return {key: by_bucket(ids, salt)[:per_species] for key, ids in sorted(ids_by_species_split.items())}

