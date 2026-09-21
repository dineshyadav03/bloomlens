"""Build the open-world evaluation manifests (eval/PROTOCOL.md section 2) from the raw datasets.

    uv sync --extra eval-data                     # scipy, to read Oxford's .mat files
    uv run python eval/build_datasets.py          # needs eval/raw/flowers-102 and eval/raw/caltech-101.zip

Writes eval/manifest/{legacy,id,near_ood,far_ood,corrupted}.json and SUMMARY.md. The manifests hold
source ids, labels, split, relative path and SHA-256 -- no images, and no model output of any kind (this
script never loads a model, so no result exists to be tuned on). Everything is deterministic: the splits
come from SHA-256 buckets (eval/splits.py), so rebuilding gives the same manifests, which the tests check.

Raw data (git-ignored, under eval/raw/):
- Oxford 102 Flowers: eval/raw/flowers-102/{jpg/, imagelabels.mat, setid.mat} (fetched by build_test_set.py)
- Caltech-101 (CC BY 4.0): eval/raw/caltech-101.zip from https://data.caltech.edu/records/mzrjq-6wc02
"""

import hashlib
import json
import sys
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import corruptions, splits  # noqa: E402
from eval.oxford102 import GRAY_ZONE_CLASSES, OXFORD_DIR, RAW_ROOT, class_name_by_label  # noqa: E402
from eval.species_mapping import OXFORD_TO_BLOOMLENS  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
MANIFEST_DIR = EVAL_DIR / "manifest"
CALTECH_ZIP = RAW_ROOT / "caltech-101.zip"
CALTECH_DIR = RAW_ROOT / "caltech-101"
CALTECH_EXCLUDED = frozenset(
    {
        "sunflower", "water_lilly", "lotus",  # flowers
        "Faces", "Faces_easy",  # people: no faces go to a third-party API
        "BACKGROUND_Google",  # unlabeled clutter, may contain anything
    }
)  # fmt: skip
PROTOCOL_VERSION = "v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --- reading the raw datasets ------------------------------------------------------------


def oxford_records() -> list[dict]:
    """One record per Oxford image: source id, category name, Oxford's own split, path, SHA-256."""
    import scipy.io as sio

    labels = sio.loadmat(str(OXFORD_DIR / "imagelabels.mat"))["labels"][0]
    setid = sio.loadmat(str(OXFORD_DIR / "setid.mat"))
    split_of = {}
    for key, name in (("trnid", "train"), ("valid", "val"), ("tstid", "test")):
        split_of.update({int(image_id): name for image_id in setid[key][0]})
    names = class_name_by_label()
    records = []
    for image_id, label in enumerate(labels, start=1):
        relative = f"flowers-102/jpg/image_{image_id:05d}.jpg"
        records.append(
            {
                "source_id": f"oxford102:image_{image_id:05d}",
                "class": names[int(label)],
                "oxford_split": split_of[image_id],
                "path": relative,
                "sha256": sha256_file(RAW_ROOT / relative),
            }
        )
    return records


def caltech_records() -> list[dict]:
    """One record per Caltech-101 image outside the excluded categories (extracting the archive if needed)."""
    root = CALTECH_DIR / "101_ObjectCategories"
    if not root.is_dir():
        CALTECH_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(CALTECH_ZIP) as archive:
            inner = next(n for n in archive.namelist() if n.endswith("101_ObjectCategories.tar.gz"))
            archive.extract(inner, CALTECH_DIR)
        with tarfile.open(CALTECH_DIR / inner, "r:gz") as tar:
            tar.extractall(CALTECH_DIR, filter="data")
        root = next(CALTECH_DIR.rglob("101_ObjectCategories"))
    records = []
    for category_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if category_dir.name in CALTECH_EXCLUDED:
            continue
        for image in sorted(category_dir.glob("*.jpg")):
            records.append(
                {
                    "source_id": f"caltech101:{category_dir.name}/{image.stem}",
                    "class": category_dir.name,
                    "path": image.relative_to(RAW_ROOT).as_posix(),
                    "sha256": sha256_file(image),
                }
            )
    return records


# --- assembling the manifests (pure) ------------------------------------------------------


def _document(family: str, entries: list[dict], **extra) -> dict:
    ordered = sorted(entries, key=lambda e: (e["split"], e["source_id"], e.get("variant", "")))
    return {"protocol": PROTOCOL_VERSION, "salt": splits.SALT, "family": family, **extra, "entries": ordered}


def unique_by_content(records: list[dict], exclude: frozenset[str] | set[str] = frozenset()) -> list[dict]:
    """One record per distinct image *content* (SHA-256), so a byte-identical copy can never land in a
    different split from its twin. Content in `exclude` is dropped; content that appears under two
    different categories is dropped as ambiguous; otherwise the record with the smallest source id wins.

    Found by the leakage test: two Oxford carnation files (image_08067, image_08077) are the same
    photo, and the split assignment had put them in dev and test."""
    by_content: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record["sha256"] not in exclude:
            by_content[record["sha256"]].append(record)
    kept = []
    for group in by_content.values():
        if len({r["class"] for r in group}) == 1:
            kept.append(min(group, key=lambda r: r["source_id"]))
    return sorted(kept, key=lambda r: r["source_id"])


def assemble(oxford: list[dict], caltech: list[dict], legacy: list[dict]) -> dict[str, dict]:
    """The five manifest documents, from the raw records and the legacy 360 (source id + species)."""
    legacy_ids = {e["source_id"] for e in legacy}
    legacy_content = {e["sha256"] for e in legacy}
    # Content-level hygiene first: no byte-identical twin may sit in another split, and nothing may repeat
    # a legacy image (the legacy records themselves drop out here; the pool excludes them by id anyway).
    oxford = unique_by_content(oxford, exclude=legacy_content)
    caltech = unique_by_content(caltech, exclude={r["sha256"] for r in oxford} | legacy_content)
    by_id = {r["source_id"]: r for r in oxford}

    # In-distribution pool: every Oxford image of a mapped category that is not in the legacy 360.
    pool: dict[str, list[str]] = defaultdict(list)
    for record in oxford:
        species = OXFORD_TO_BLOOMLENS.get(record["class"])
        if species and record["source_id"] not in legacy_ids:
            pool[species].append(record["source_id"])
    assignment = splits.assign_stratified(pool)
    id_entries = [
        {
            "source_id": sid,
            "label": species,
            "group": by_id[sid]["class"],
            "split": assignment[sid],
            "oxford_split": by_id[sid]["oxford_split"],
            "path": by_id[sid]["path"],
            "sha256": by_id[sid]["sha256"],
        }
        for species, ids in sorted(pool.items())
        for sid in ids
    ]

    def ood_entries(records: list[dict], excluded: set[str]) -> list[dict]:
        by_category: dict[str, list[str]] = defaultdict(list)
        record_of = {r["source_id"]: r for r in records}
        for record in records:
            if record["class"] not in excluded:
                by_category[record["class"]].append(record["source_id"])
        category_split = splits.assign_splits(by_category)  # whole categories go to one split
        capped = splits.cap_per_category(by_category)
        return [
            {
                "source_id": sid,
                "label": category,
                "group": category,
                "split": category_split[category],
                "path": record_of[sid]["path"],
                "sha256": record_of[sid]["sha256"],
            }
            for category, ids in capped.items()
            for sid in ids
        ]

    not_near_ood = sorted(set(OXFORD_TO_BLOOMLENS) | set(GRAY_ZONE_CLASSES))
    near = ood_entries(oxford, set(not_near_ood))
    far = ood_entries(caltech, set()) if caltech else []

    # Corrupted-in-set: a few dev and test ID sources per species, each in every variant.
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for entry in id_entries:
        if entry["split"] in ("dev", "test"):
            grouped[(entry["label"], entry["split"])].append(entry["source_id"])
    corrupted = [
        {"source_id": sid, "label": species, "split": split, "variant": variant}
        for (species, split), ids in splits.corruption_sources(grouped).items()
        for sid in ids
        for variant in corruptions.CORRUPTIONS
    ]

    legacy_docs = [{**e, "split": "dev_legacy"} for e in legacy]
    note = "the 360 images that shaped the current thresholds: development data"
    return {
        "legacy": _document("id", legacy_docs, note=note),
        "id": _document("id", id_entries),
        "near_ood": _document("near_ood", near, excluded_classes=not_near_ood),
        "far_ood": _document("far_ood", far, excluded_categories=sorted(CALTECH_EXCLUDED)),
        "corrupted": _document("corrupted_id", corrupted, variants=list(corruptions.CORRUPTIONS)),
    }


def legacy_entries(oxford: list[dict]) -> list[dict]:
    """The 360 legacy images as manifest entries, matched to their Oxford source by CONTENT hash (not by
    the file name's index arithmetic), so a wrong assumption cannot silently mislabel a source."""
    by_hash = {r["sha256"]: r for r in oxford}
    entries = []
    for item in json.loads((EVAL_DIR / "test_set.json").read_text(encoding="utf-8")):
        digest = sha256_file(EVAL_DIR / item["image_path"])
        record = by_hash[digest]  # KeyError = a committed image that is not an Oxford original
        species = OXFORD_TO_BLOOMLENS[record["class"]]
        assert species == item["species"], f"{item['image_path']}: {item['species']} vs Oxford's {species}"
        entries.append(
            {
                "source_id": record["source_id"],
                "label": item["species"],
                "group": record["class"],
                "oxford_split": record["oxford_split"],
                "path": record["path"],
                "sha256": digest,
            }
        )
    return entries


def canonical(document: dict) -> str:
    return json.dumps(document, sort_keys=True, indent=1, ensure_ascii=False) + "\n"


def manifest_hash(document: dict) -> str:
    """Hash of the content, independent of newline style or whitespace."""
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def summary(documents: dict[str, dict]) -> str:
    lines = [
        "# Evaluation manifests (protocol v1)",
        "",
        "Generated by `eval/build_datasets.py`; counts and content hashes only. No model has been run on any of",
        "these images to produce this file. See `eval/PROTOCOL.md`. `groups` = distinct labels: covered species",
        "(id, corrupted), Oxford categories (near_ood), Caltech categories (far_ood).",
        "",
        "| manifest | family | dev | test | pilot | dev_legacy | total | groups | content SHA-256 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, doc in documents.items():
        counts = defaultdict(int)
        for e in doc["entries"]:
            counts[e["split"]] += 1
        groups = len({e["label"] for e in doc["entries"]})
        cells = [name, doc["family"], counts["dev"], counts["test"], counts["pilot"], counts["dev_legacy"]]
        cells += [len(doc["entries"]), groups, f"`{manifest_hash(doc)[:16]}…`"]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    print("hashing Oxford 102 Flowers (8 189 images)...")
    oxford = oxford_records()
    caltech = caltech_records() if CALTECH_ZIP.exists() or CALTECH_DIR.exists() else []
    if not caltech:
        print("WARNING: Caltech-101 not found; far_ood will be empty (see the module docstring)")
    documents = assemble(oxford, caltech, legacy_entries(oxford))
    MANIFEST_DIR.mkdir(exist_ok=True)
    for name, document in documents.items():
        (MANIFEST_DIR / f"{name}.json").write_text(canonical(document), encoding="utf-8", newline="\n")
    (MANIFEST_DIR / "SUMMARY.md").write_text(summary(documents), encoding="utf-8", newline="\n")
    print(summary(documents))


if __name__ == "__main__":
    main()
