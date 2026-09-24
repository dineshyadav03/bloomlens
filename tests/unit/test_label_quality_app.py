"""tools/label_quality.py driven by Streamlit's AppTest.

SYNTHETIC FIXTURES ONLY. Every photo here is a tiny generated solid-colour PNG in tmp_path, every
rater id is a made-up string, and the sample source is monkeypatched -- no real photo, rater or
label is ever involved (docs/quality/PROTOCOL.md section 8: building the labeler is not validation).

AppTest cannot inspect st.image, so what is asserted is the surrounding text, the widgets and what
lands in the database.
"""

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from eval import labeling_sample
from eval import quality_labels_db as db
from tests.conftest import REPO_ROOT

APP = str(REPO_ROOT / "tools" / "label_quality.py")

CANARY_SPECIES = "CANARY-SPECIES-DO-NOT-SHOW"
CANARY_MODEL_GRADE = "CANARY-MODEL-GRADE-DO-NOT-SHOW"

A, B, C, CANNOT = "A — excellent", "B — good", "C — fair", "Cannot grade this photo"


def text_of(elements):
    return " | ".join(str(e.value) for e in elements)


def everything_shown(at):
    shown = [*at.markdown, *at.text, *at.error, *at.warning, *at.info, *at.caption, *at.subheader, *at.success]
    return text_of(shown)


@pytest.fixture
def pool(monkeypatch, tmp_path):
    """Five synthetic items; the first two are practice items (calibration_count=2 below).
    Each carries model-output-looking canary fields that a blinded tool must never display."""
    items = []
    for i in range(5):
        path = tmp_path / f"synthetic-{i}.png"
        Image.new("RGB", (16, 16), (40 * i, 100, 60)).save(path)
        items.append(
            {
                "id": f"synthetic-{i}",
                "path": str(path),
                "species": CANARY_SPECIES,
                "quality_grade": CANARY_MODEL_GRADE,
            }
        )
    monkeypatch.setattr(labeling_sample, "default_sample_source", lambda: items)
    monkeypatch.setattr(labeling_sample, "DEFAULT_CALIBRATION_COUNT", 2)
    real_build = labeling_sample.build_sample
    monkeypatch.setattr(labeling_sample, "build_sample", lambda its, **kw: real_build(its, calibration_count=2))
    return items


def new_app():
    return AppTest.from_file(APP, default_timeout=30)


def as_rater(at, rater_id):
    at.text_input(key="rater_id").set_value(rater_id)
    return at.run()


def registered(rater_id="rater-1"):
    db.register_rater(rater_id, "synthetic test rater")
    return as_rater(new_app().run(), rater_id)


def submit(at, grade_label, note=None):
    at.radio[0].set_value(grade_label)
    if note is not None:
        at.text_area[0].set_value(note)
    at = at.run()
    return next(b for b in at.button if b.label == "Submit").click().run()


class TestGettingStarted:
    def test_it_asks_for_a_rater_id_first(self, pool):
        at = new_app().run()
        assert not at.exception
        assert "Enter your rater ID" in text_of(at.info)
        assert not at.radio  # no item is shown yet

    def test_an_unregistered_rater_must_register_before_seeing_any_item(self, pool):
        at = as_rater(new_app().run(), "rater-1")
        assert "Before you start" in text_of(at.subheader)
        assert not at.radio
        assert "Item ID" not in everything_shown(at)

    def test_registering_records_the_experience_note_once_and_shows_the_first_item(self, pool):
        at = as_rater(new_app().run(), "rater-1")
        at.text_area[0].set_value("five years in a flower shop")
        at = next(b for b in at.button if b.label == "Register and start labeling").click().run()
        assert db.is_registered_rater("rater-1")
        assert "Item ID: synthetic-0" in text_of(at.caption)

    def test_a_returning_rater_skips_registration(self, pool):
        at = registered()
        assert "Before you start" not in text_of(at.subheader)
        assert "Item ID: synthetic-0" in text_of(at.caption)


class TestBlinding:
    def test_nothing_the_pipeline_or_metadata_knows_is_ever_shown(self, pool):
        at = registered()
        shown = everything_shown(at)
        assert CANARY_SPECIES not in shown
        assert CANARY_MODEL_GRADE not in shown

    def test_the_same_holds_after_submitting_and_on_the_next_item(self, pool):
        at = submit(registered(), A)
        shown = everything_shown(at)
        assert CANARY_SPECIES not in shown and CANARY_MODEL_GRADE not in shown

    def test_another_raters_label_is_never_displayed(self, pool):
        db.register_rater("rater-2", "synthetic")
        db.submit_label("rater-2", "synthetic-0", calibration=True, grade="C", note="OTHER-RATER-NOTE-CANARY")
        at = registered("rater-1")
        shown = everything_shown(at)
        assert "OTHER-RATER-NOTE-CANARY" not in shown
        assert "Item ID: synthetic-0" in text_of(at.caption)  # rater-1 still starts at the first item
        assert at.radio[0].value is None  # and nothing is pre-selected from rater-2's answer

    def test_the_previous_grade_is_not_carried_over_to_the_next_item(self, pool):
        at = submit(registered(), B)
        assert "Item ID: synthetic-1" in text_of(at.caption)
        assert at.radio[0].value is None


class TestLabeling:
    def test_submit_is_disabled_until_a_grade_is_chosen(self, pool):
        at = registered()
        submit_button = next(b for b in at.button if b.label == "Submit")
        assert submit_button.disabled
        at.radio[0].set_value(A)
        at = at.run()
        assert not next(b for b in at.button if b.label == "Submit").disabled

    def test_submitting_stores_the_grade_note_and_calibration_flag(self, pool):
        submit(registered(), C, note="visible browning")
        import contextlib

        with contextlib.closing(db.connect()) as conn:
            row = conn.execute("SELECT rater_id, item_id, calibration, grade, note FROM quality_labels").fetchone()
        assert row == ("rater-1", "synthetic-0", 1, "C", "visible browning")

    def test_cannot_grade_is_stored_as_its_own_value(self, pool):
        submit(registered(), CANNOT)
        assert db.latest_labels(include_calibration=True) == {"synthetic-0": {"rater-1": "CANNOT_GRADE"}}

    def test_submitting_advances_to_the_next_item(self, pool):
        at = submit(registered(), A)
        assert "Item ID: synthetic-1" in text_of(at.caption)

    def test_a_calibration_item_is_marked_as_practice(self, pool):
        at = registered()
        assert "Practice item 1 of 2" in text_of(at.warning)
        assert "never included in the measured results" in text_of(at.warning)

    def test_after_the_practice_round_items_are_counted_as_the_study(self, pool):
        at = submit(submit(registered(), A), B)  # both practice items
        assert "Item ID: synthetic-2" in text_of(at.caption)
        assert "Item 1 of 3" in text_of(at.caption)
        assert not at.warning

    def test_calibration_labels_are_stored_but_excluded_from_the_measured_labels(self, pool):
        at = submit(submit(registered(), A), B)
        submit(at, C)  # first study item
        assert db.latest_labels() == {"synthetic-2": {"rater-1": "C"}}
        assert set(db.latest_labels(include_calibration=True)) == {"synthetic-0", "synthetic-1", "synthetic-2"}

    def test_finishing_every_item_says_thank_you_and_shows_no_item(self, pool):
        at = registered()
        for grade in (A, B, C, A, B):
            at = submit(at, grade)
        assert not at.exception
        assert "labeled all 5 items" in text_of(at.success)
        # AppTest keeps elements from before an st.rerun() that the new run didn't overwrite, so
        # "no item is shown" is asserted on a fresh session, which has no earlier run to inherit.
        fresh = as_rater(new_app().run(), "rater-1")
        assert "labeled all 5 items" in text_of(fresh.success)
        assert not fresh.radio and "Item ID" not in everything_shown(fresh)

    def test_a_returning_rater_resumes_where_they_left_off(self, pool):
        submit(submit(registered(), A), B)
        at = as_rater(new_app().run(), "rater-1")  # a fresh browser session, same rater
        assert "Item ID: synthetic-2" in text_of(at.caption)

    def test_two_raters_progress_independently(self, pool):
        submit(registered("rater-1"), A)
        db.register_rater("rater-2", "synthetic")
        at = as_rater(new_app().run(), "rater-2")
        assert "Item ID: synthetic-0" in text_of(at.caption)  # rater-2 hasn't labeled anything
