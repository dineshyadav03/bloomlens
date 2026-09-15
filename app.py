"""BloomLens — scan a flower (or a lot of them), get species, quality, and a simulated price."""

from PIL import Image

import streamlit as st
from src.identify import (
    LOT_LOW_AGREEMENT_THRESHOLD,
    LOT_MAX_PHOTOS,
    IdentifyError,
    identify,
    identify_lot,
    resolve_candidate,
)
from src.interpretability import ExplainError, explain_image
from src.pricing import price_history

st.set_page_config(page_title="BloomLens", page_icon="🌸")

st.title("🌸 BloomLens")
st.caption("Scan a flower to get its species, a quality read, and a simulated auction price.")

st.warning(
    "⚠️ Pricing shown is **simulated demo data** — FloraHolland does not expose a public "
    "real-time pricing API. Quality assessment is a heuristic visual read, not a calibrated "
    "grading system. See [docs/RESEARCH.md](https://github.com/dineshyadav03/bloomlens/blob/main/docs/RESEARCH.md).",
    icon="⚠️",
)

_TREND_ICON = {"up": "📈 up", "down": "📉 down", "flat": "➡️ flat", "unknown": "n/a"}

tab_single, tab_lot = st.tabs(["Single scan", "Lot mode"])

with tab_single:
    st.subheader("Scan")
    photo = st.camera_input("Point your camera at a flower and capture")

    with st.expander("No camera? Upload a photo instead (for testing)"):
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    image_source = photo or uploaded

    if image_source is not None:
        image = Image.open(image_source)
        st.image(image, caption="Scanned flower", width=300)

        with st.spinner("Identifying species, checking quality, looking up price..."):
            try:
                result = identify(image)
            except IdentifyError as exc:
                st.error(str(exc))
                st.stop()
            except Exception as exc:  # noqa: BLE001 — surface unexpected errors plainly in the demo UI
                st.error(f"Something went wrong: {exc}")
                st.stop()

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
        st.write(result.summary)

        col1, col2 = st.columns(2)
        col1.metric("Quality grade", result.quality_grade)
        col2.metric(
            "Simulated price/stem",
            f"€{display_price:.2f}" if display_price is not None else "n/a",
        )
        st.caption(f"Price trend: {_TREND_ICON.get(display_trend, display_trend)}")

        st.write("**Confidence:**", result.confidence_note)
        st.write("**Quality note:**", result.quality_note)

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
    else:
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
        st.session_state.lot_photos.append(Image.open(lot_photo))
        st.session_state.lot_camera_key += 1  # fresh, empty camera widget next render
        st.rerun()
    if clear_col.button("🗑️ Clear lot", disabled=not st.session_state.lot_photos):
        st.session_state.lot_photos = []
        st.rerun()

    with st.expander("No camera? Upload photos instead (for testing)"):
        lot_uploads = st.file_uploader(
            "Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="lot_uploader"
        )
        if lot_uploads and st.button("➕ Add uploaded photos to lot"):
            remaining = LOT_MAX_PHOTOS - len(st.session_state.lot_photos)
            if len(lot_uploads) > remaining:
                st.session_state.lot_add_warning = (
                    f"A lot can have at most {LOT_MAX_PHOTOS} photos — only added {remaining} of the "
                    f"{len(lot_uploads)} you uploaded."
                )
            for f in lot_uploads[:remaining]:
                st.session_state.lot_photos.append(Image.open(f))
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
                    lot_result = identify_lot(st.session_state.lot_photos)
                except IdentifyError as exc:
                    st.error(str(exc))
                    st.stop()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Something went wrong: {exc}")
                    st.stop()

            st.subheader(f"🌷 {lot_result.consensus_species}")
            if lot_result.scientific_name:
                st.caption(f"*{lot_result.scientific_name}*")
            st.write(lot_result.summary)

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
            st.write("**Quality note:**", lot_result.quality_note)

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
