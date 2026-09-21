"""BloomLens — scan a flower (or a lot of them), get species, quality, and a simulated price.

Uploads go through src/guard.py (the same validator the API uses), and every piece
of text a language model wrote is shown with st.text -- literally, never as markdown
or HTML -- so nothing an attacker writes inside a photo can format or link anything.

Each scan spends one unit of the browser session's rate/daily quota (and of the global
daily ceiling) from the SQLite file this app shares with the API -- see src/quota.py.
A scan is only run again when the *photo* changes: a widget interaction re-runs this whole
script, and that must not re-bill the model, burn quota or write a duplicate log row.
"""

import hashlib
import logging
import secrets
from contextlib import contextmanager

import streamlit as st
from src import guard, quota
from src.identify import (
    LOT_LOW_AGREEMENT_THRESHOLD,
    LOT_MAX_PHOTOS,
    IdentifyError,
    identify,
    identify_lot,
    resolve_candidate,
)
from src.interpretability import ExplainError, explain_image
from src.inventory import list_recent, log_lot_scan, log_scan, species_counts
from src.pricing import price_history

logger = logging.getLogger("bloomlens.app")

_UNEXPECTED_ERROR = "Something went wrong on our side. Please try again."

st.set_page_config(page_title="BloomLens", page_icon="🌸")

st.title("🌸 BloomLens")
st.caption("Scan a flower to get its species, a quality read, and a simulated auction price.")

st.warning(
    "⚠️ Pricing shown is **simulated demo data** — FloraHolland does not expose a public "
    "real-time pricing API. Quality assessment is a heuristic visual read, not a calibrated "
    "grading system. See [docs/RESEARCH.md](https://github.com/dineshyadav03/bloomlens/blob/main/docs/RESEARCH.md).",
    icon="⚠️",
)

st.info(
    "🔒 **Privacy:** each photo is sent to Google's Gemini API to write the quality read and summary. "
    "Location and camera metadata are removed first, and BloomLens keeps no photos. On Google's free "
    "tier, submitted content may be used to improve Google products and reviewed by people, and users "
    "in the EEA, UK and Switzerland are not covered. Details: "
    "[docs/PRIVACY.md](https://github.com/dineshyadav03/bloomlens/blob/main/docs/PRIVACY.md).",
    icon="🔒",
)


@contextmanager
def _admitted():
    """Hold a concurrency slot and spend one unit of this session's quota for the block.
    On any refusal say why in plain words and stop the script; if the counters can't be
    reached, refuse (fail closed) rather than run unmetered."""
    try:
        with quota.gate().slot():
            try:
                decision = quota.admit(f"ui:{st.session_state.setdefault('session_id', secrets.token_hex(8))}")
            except quota.QuotaUnavailable:
                st.error("Scanning is unavailable right now (the rate limiter can't be reached). Try again soon.")
                st.stop()
            if not decision.allowed:
                st.warning(f"⏳ {decision.message}")
                st.stop()
            yield
    except quota.Busy:
        st.warning("⏳ The service is busy with other scans. Please try again in a few seconds.")
        st.stop()


def _load_photo(file):
    """A validated, metadata-free RGB image from an upload/camera file, or None after
    telling the user why not (fixed wording from src/guard.py -- no filenames, no internals)."""
    try:
        file.seek(0)
        return guard.validate_upload(file)
    except guard.UploadRejected as exc:
        st.error(exc.message)
        return None


_TREND_ICON = {"up": "📈 up", "down": "📉 down", "flat": "➡️ flat", "unknown": "n/a"}

tab_single, tab_lot, tab_inventory = st.tabs(["Single scan", "Lot mode", "Inventory"])

with tab_single:
    st.subheader("Scan")
    photo = st.camera_input("Point your camera at a flower and capture")

    with st.expander("No camera? Upload a photo instead (for testing)"):
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

    image_source = photo or uploaded

    image = _load_photo(image_source) if image_source is not None else None

    if image is not None:
        st.image(image, caption="Scanned flower", width=300)

        digest = hashlib.sha256(image.tobytes() + repr(image.size).encode()).hexdigest()
        cached = st.session_state.get("single_scan")
        if cached is not None and cached[0] == digest:
            result = cached[1]  # a rerun caused by a widget, not a new photo
        else:
            with st.spinner("Identifying species, checking quality, looking up price..."):
                try:
                    with _admitted():
                        result = identify(image)
                except IdentifyError as exc:
                    st.error(str(exc))
                    st.stop()
                except Exception:  # noqa: BLE001 — never show exception text; log it for the operator
                    logger.exception("unexpected error while identifying a photo")
                    st.error(_UNEXPECTED_ERROR)
                    st.stop()

            log_scan(result)
            st.session_state["single_scan"] = (digest, result)

        display_species = result.species
        display_scientific = result.scientific_name
        display_price = result.price_per_stem
        display_trend = result.price_trend

        if result.confidence_tier == "low":
            st.warning(
                f"⚠️ Not confidently any of BloomLens's known species — closest guess: **{result.species}**"
            )
        elif result.confidence_tier == "ambiguous":
            candidate_names = [c["common_name"] for c in result.top_candidates[:2]]
            default_index = candidate_names.index(result.species) if result.species in candidate_names else 0
            st.caption("⚠️ Close call between two visually similar candidates — pick the right one:")
            chosen = st.radio("Which is it?", candidate_names, index=default_index, label_visibility="collapsed")
            if chosen != result.species:
                resolved = resolve_candidate(chosen, result.top_candidates, result.quality_grade)
                display_species = chosen
                display_scientific = resolved["scientific_name"]
                display_price = resolved["price_per_stem"]
                display_trend = resolved["price_trend"]

        if result.confidence_tier != "low":
            st.subheader(f"🌷 {display_species}")
        if display_scientific:
            st.caption(f"*{display_scientific}*")
        st.text(result.summary)

        col1, col2 = st.columns(2)
        col1.metric("Quality grade", result.quality_grade)
        col2.metric(
            "Simulated price/stem",
            f"€{display_price:.2f}" if display_price is not None else "n/a",
        )
        st.caption(f"Price trend: {_TREND_ICON.get(display_trend, display_trend)}")

        st.markdown("**Confidence:**")
        st.text(result.confidence_note)
        st.markdown("**Quality note:**")
        st.text(result.quality_note)

        with st.expander("Other candidates considered"):
            for c in result.top_candidates:
                st.write(f"- {c['common_name']} (similarity {c['score']})")

        with st.expander("🔍 Why this species?"):
            st.caption(
                "Heatmap shows which parts of the photo most influenced this match "
                "(warm = high influence) — a research technique (Grad-ECLIP), not a "
                "certified explanation. Computed only when you open this."
            )
            if st.button("Generate heatmap"):
                with st.spinner("Computing..."):
                    try:
                        overlay = explain_image(image, display_species)
                    except ExplainError as exc:
                        st.error(str(exc))
                    else:
                        st.image(overlay, caption=f"Why: {display_species}", width=300)
    elif image_source is None:
        st.info("Waiting for a scan — use the camera above, or expand the uploader for a test photo.")

with tab_lot:
    st.subheader("Build a lot")
    st.caption(
        f"Scan or upload up to {LOT_MAX_PHOTOS} photos of the same lot, then identify them all "
        "together — one consensus species, one quality read, one price."
    )

    if "lot_photos" not in st.session_state:
        st.session_state.lot_photos = []
    if "lot_camera_key" not in st.session_state:
        st.session_state.lot_camera_key = 0

    # st.rerun() below cuts off rendering immediately, so a warning shown right
    # before it would flash and vanish before the user ever sees it — stash it
    # in session_state and display it after the rerun instead.
    if st.session_state.get("lot_add_warning"):
        st.warning(st.session_state.pop("lot_add_warning"))

    lot_full = len(st.session_state.lot_photos) >= LOT_MAX_PHOTOS

    lot_photo = st.camera_input(
        "Scan a stem to add to the lot",
        key=f"lot_camera_{st.session_state.lot_camera_key}",
        disabled=lot_full,
    )
    add_col, clear_col = st.columns(2)
    if add_col.button("➕ Add to lot", disabled=lot_photo is None or lot_full):
        added = _load_photo(lot_photo)
        if added is not None:
            st.session_state.lot_photos.append(added)
            st.session_state.lot_camera_key += 1  # fresh, empty camera widget next render
            st.rerun()
    if clear_col.button("🗑️ Clear lot", disabled=not st.session_state.lot_photos):
        st.session_state.lot_photos = []
        st.rerun()

    with st.expander("No camera? Upload photos instead (for testing)"):
        lot_uploads = st.file_uploader(
            "Upload images", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True, key="lot_uploader"
        )
        if lot_uploads and st.button("➕ Add uploaded photos to lot"):
            remaining = LOT_MAX_PHOTOS - len(st.session_state.lot_photos)
            accepted = []
            for f in lot_uploads[:remaining]:
                photo_ok = _load_photo(f)  # on a refusal the reason is already on screen
                if photo_ok is None:
                    break
                accepted.append(photo_ok)
            else:  # every file passed: add them (a lot with a bad file adds nothing)
                if len(lot_uploads) > remaining:
                    st.session_state.lot_add_warning = (
                        f"A lot can have at most {LOT_MAX_PHOTOS} photos — only added {remaining} of the "
                        f"{len(lot_uploads)} you uploaded."
                    )
                st.session_state.lot_photos.extend(accepted)
                st.rerun()

    if st.session_state.lot_photos:
        st.write(f"**Lot: {len(st.session_state.lot_photos)}/{LOT_MAX_PHOTOS} photos**")
        thumb_cols = st.columns(min(len(st.session_state.lot_photos), 5))
        for i, lot_img in enumerate(st.session_state.lot_photos):
            with thumb_cols[i % len(thumb_cols)]:
                st.image(lot_img, width=100)
                if st.button("Remove", key=f"remove_lot_photo_{i}"):
                    st.session_state.lot_photos.pop(i)
                    st.rerun()

        if st.button("🔍 Identify Lot", type="primary"):
            with st.spinner(f"Identifying {len(st.session_state.lot_photos)} photos as one lot..."):
                try:
                    with _admitted():
                        lot_result = identify_lot(st.session_state.lot_photos)
                except IdentifyError as exc:
                    st.error(str(exc))
                    st.stop()
                except Exception:  # noqa: BLE001 — see the single-scan handler
                    logger.exception("unexpected error while identifying a lot")
                    st.error(_UNEXPECTED_ERROR)
                    st.stop()

            log_lot_scan(lot_result)

            st.subheader(f"🌷 {lot_result.consensus_species}")
            if lot_result.scientific_name:
                st.caption(f"*{lot_result.scientific_name}*")
            st.text(lot_result.summary)

            agreement_pct = round(lot_result.agreement_fraction * 100)
            agreeing = lot_result.photo_count - len(lot_result.flagged_photos)
            if lot_result.agreement_fraction < LOT_LOW_AGREEMENT_THRESHOLD:
                st.warning(
                    f"⚠️ Only {agreement_pct}% of photos ({agreeing}/{lot_result.photo_count}) agree on "
                    "species — this lot may contain mixed species."
                )
            else:
                st.caption(f"✅ {agreement_pct}% of photos agree on species ({agreeing}/{lot_result.photo_count}).")

            col1, col2 = st.columns(2)
            col1.metric("Quality grade", lot_result.quality_grade)
            col2.metric(
                "Simulated price/stem",
                f"€{lot_result.price_per_stem:.2f}" if lot_result.price_per_stem is not None else "n/a",
            )
            st.caption(f"Price trend: {_TREND_ICON.get(lot_result.price_trend, lot_result.price_trend)}")
            st.markdown("**Quality note:**")
            st.text(lot_result.quality_note)

            history = price_history(lot_result.consensus_species, lot_result.quality_grade)
            if not history.empty:
                st.caption("Simulated price trend (last 30 days):")
                st.line_chart(history.set_index("date"))

            if lot_result.flagged_photos:
                st.write("**Flagged photos** (didn't match the lot's consensus species):")
                for f in lot_result.flagged_photos:
                    st.write(f"- Photo {f['index'] + 1}: detected as {f['top_species']} (similarity {f['score']})")
    else:
        st.info("No photos in the lot yet — scan or upload some above.")

with tab_inventory:
    st.subheader("Inventory log")
    st.caption(
        "Every scan above is logged automatically — species, quality, price, and timestamp — "
        "building a running record across sessions. On the live Hugging Face Spaces demo, this "
        "resets if the demo container restarts (see docs/ARCHITECTURE.md)."
    )

    counts = species_counts()
    if counts:
        st.write("**Scans per species**")
        st.bar_chart(counts)
    else:
        st.info("No scans logged yet — identify a flower in Single scan or Lot mode above.")

    entries = list_recent(100)
    if entries:
        st.write("**Recent scans**")
        st.dataframe(
            [
                {
                    "Scanned at": e.scanned_at,
                    "Mode": e.mode,
                    "Species": e.species,
                    "Scientific name": e.scientific_name,
                    "Quality": e.quality_grade,
                    "Price/stem": e.price_per_stem,
                    "Trend": e.price_trend,
                    "Photos": e.photo_count,
                    "Agreement": e.agreement_fraction,
                }
                for e in entries
            ],
            hide_index=True,
        )
