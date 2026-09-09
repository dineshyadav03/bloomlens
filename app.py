"""BloomLens — scan a flower, get species, quality, and a simulated price."""

from PIL import Image

import streamlit as st
from src.identify import IdentifyError, identify, resolve_candidate

st.set_page_config(page_title="BloomLens", page_icon="🌸")

st.title("🌸 BloomLens")
st.caption("Scan a flower to get its species, a quality read, and a simulated auction price.")

st.warning(
    "⚠️ Pricing shown is **simulated demo data** — FloraHolland does not expose a public "
    "real-time pricing API. Quality assessment is a heuristic visual read, not a calibrated "
    "grading system. See [docs/RESEARCH.md](https://github.com/dineshyadav03/bloomlens/blob/main/docs/RESEARCH.md).",
    icon="⚠️",
)

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

    _TREND_ICON = {"up": "📈 up", "down": "📉 down", "flat": "➡️ flat", "unknown": "n/a"}

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
else:
    st.info("Waiting for a scan — use the camera above, or expand the uploader for a test photo.")
