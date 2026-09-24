"""A blinded Streamlit labeler for the quality-agreement study (docs/quality/PROTOCOL.md).

Run with `uv run --extra eval-quality streamlit run tools/label_quality.py`.

Blinding (PROTOCOL.md section 3): a rater sees ONLY the current photo -- never BloomLens's own
species guess or quality grade, never another rater's label for the same item, and never their
own previous grade pre-filled for a new item (each item's grade/note widget is keyed by item id
for exactly that reason). Nor are they told which items are the practice round (section 3): every
item gets the same progress line. Submissions are append-only (eval/quality_labels_db.py) --
resubmitting an item records a new row rather than overwriting the old one.

**Building this tool is not validation.** It produces raw labels for `eval/quality_agreement.py`
to measure agreement on; it says nothing by itself about whether BloomLens's own `quality_grade`
output is any good (docs/quality/PROTOCOL.md section 8). This script is exercised in tests only
against clearly-marked synthetic fixtures -- never real photos or real rater data.
"""

import streamlit as st

from eval import labeling_sample
from eval import quality_labels_db as db

GRADE_OPTIONS = {
    "A — excellent": "A",
    "B — good": "B",
    "C — fair": "C",
    "Cannot grade this photo": "CANNOT_GRADE",
}

st.set_page_config(page_title="BloomLens quality labeler", page_icon="🏷️")
st.title("🏷️ BloomLens quality-label study")
st.caption(
    "Blinded visual-condition-class labeling for docs/quality/PROTOCOL.md. You will see ONLY a "
    "photo — never BloomLens's own species guess, its quality grade, or any other rater's answer."
)

rater_id = st.text_input("Rater ID (as agreed with the study coordinator)", key="rater_id").strip()
if not rater_id:
    st.info("Enter your rater ID to begin.")
    st.stop()

if not db.is_registered_rater(rater_id):
    st.subheader("Before you start")
    st.caption("Recorded once, reported alongside results — never used to exclude or weight your labels.")
    experience = st.text_area("In a sentence or two, describe your relevant experience with cut flowers")
    if st.button("Register and start labeling"):
        db.register_rater(rater_id, experience)
        st.rerun()
    st.stop()

sample = labeling_sample.build_sample(labeling_sample.default_sample_source())
already_done = db.labeled_item_ids(rater_id)
remaining = [item for item in sample if item["id"] not in already_done]

if not remaining:
    st.success(f"You've labeled all {len(sample)} items. Thank you!")
    st.stop()

item = remaining[0]
position = next(i for i, entry in enumerate(sample) if entry["id"] == item["id"])
# One uniform progress line for every item: PROTOCOL.md section 3 says raters are NOT told which
# items are the practice round while they label it (the coordinator runs the group discussion
# afterward), so nothing here may distinguish them. The `calibration` flag only goes to storage.
st.caption(f"Item {position + 1} of {len(sample)}")

st.caption(f"Item ID: {item['id']}")
st.image(item["path"], width="stretch")

grade_label = st.radio("Visual condition class", list(GRADE_OPTIONS), index=None, key=f"grade-{item['id']}")
note = st.text_area("Optional note (what you saw)", key=f"note-{item['id']}")

if st.button("Submit", disabled=grade_label is None):
    db.submit_label(rater_id, item["id"], calibration=item["calibration"], grade=GRADE_OPTIONS[grade_label], note=note)
    st.rerun()
