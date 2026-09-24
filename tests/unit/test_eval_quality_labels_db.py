"""eval/quality_labels_db.py: the blinded labeler's append-only SQLite store.

QUALITY_LABELS_DB_PATH is pointed at a fresh tmp_path file for every test by the autouse
`hermetic_env` fixture (tests/conftest.py), so these tests never touch a real database.
"""

from contextlib import closing

import pytest

from eval import quality_labels_db as db


class TestRaterRegistration:
    def test_a_new_rater_is_not_registered_until_registered(self):
        assert db.is_registered_rater("r1") is False
        db.register_rater("r1", "10 years as a florist")
        assert db.is_registered_rater("r1") is True

    def test_registering_twice_keeps_the_first_experience_note(self):
        db.register_rater("r1", "first note")
        db.register_rater("r1", "second note, should be ignored")
        with closing(db.connect()) as conn:
            (note,) = conn.execute("SELECT experience_note FROM raters WHERE rater_id = ?", ("r1",)).fetchone()
        assert note == "first note"

    def test_different_raters_are_independent(self):
        db.register_rater("r1", "a")
        assert db.is_registered_rater("r2") is False


class TestSubmitLabel:
    def test_rejects_an_invalid_grade(self):
        with pytest.raises(ValueError, match="grade must be one of"):
            db.submit_label("r1", "item-1", calibration=False, grade="D")

    def test_a_valid_submission_is_recorded(self):
        db.submit_label("r1", "item-1", calibration=False, grade="A", note="looks great")
        assert db.labeled_item_ids("r1") == {"item-1"}

    def test_resubmitting_the_same_item_adds_a_row_rather_than_overwriting(self):
        db.submit_label("r1", "item-1", calibration=False, grade="A")
        db.submit_label("r1", "item-1", calibration=False, grade="B")  # a correction
        with db.connect() as conn:
            (count,) = conn.execute(
                "SELECT COUNT(*) FROM quality_labels WHERE rater_id = ? AND item_id = ?", ("r1", "item-1")
            ).fetchone()
        assert count == 2  # both submissions survive -- append-only


class TestLabeledItemIds:
    def test_reflects_only_that_raters_own_submissions(self):
        db.submit_label("r1", "item-1", calibration=False, grade="A")
        db.submit_label("r2", "item-2", calibration=False, grade="B")
        assert db.labeled_item_ids("r1") == {"item-1"}
        assert db.labeled_item_ids("r2") == {"item-2"}

    def test_an_unknown_rater_has_labeled_nothing(self):
        assert db.labeled_item_ids("nobody") == set()


class TestLatestLabels:
    def test_excludes_calibration_labels_by_default(self):
        db.submit_label("r1", "practice-1", calibration=True, grade="A")
        db.submit_label("r1", "item-1", calibration=False, grade="B")
        assert db.latest_labels() == {"item-1": {"r1": "B"}}

    def test_can_include_calibration_labels_on_request(self):
        db.submit_label("r1", "practice-1", calibration=True, grade="A")
        result = db.latest_labels(include_calibration=True)
        assert result == {"practice-1": {"r1": "A"}}

    def test_a_correction_resolves_to_the_most_recent_submission(self):
        db.submit_label("r1", "item-1", calibration=False, grade="A")
        db.submit_label("r1", "item-1", calibration=False, grade="C")  # the rater changed their mind
        assert db.latest_labels() == {"item-1": {"r1": "C"}}

    def test_multiple_raters_on_the_same_item_are_both_kept(self):
        db.submit_label("r1", "item-1", calibration=False, grade="A")
        db.submit_label("r2", "item-1", calibration=False, grade="B")
        assert db.latest_labels() == {"item-1": {"r1": "A", "r2": "B"}}

    def test_empty_store_returns_an_empty_dict(self):
        assert db.latest_labels() == {}

    def test_round_trips_into_the_agreement_wrappers(self):
        # A small end-to-end sanity check: labels collected here are directly usable by
        # eval/quality_agreement.py without any reshaping.
        from eval import quality_agreement as qa

        for item_id, (grade_a, grade_b) in {
            "i1": ("A", "A"),
            "i2": ("B", "C"),
            "i3": ("C", "C"),
        }.items():
            db.submit_label("r1", item_id, calibration=False, grade=grade_a)
            db.submit_label("r2", item_id, calibration=False, grade=grade_b)
        latest = db.latest_labels()
        rater_a = [latest[item]["r1"] for item in ("i1", "i2", "i3")]
        rater_b = [latest[item]["r2"] for item in ("i1", "i2", "i3")]
        kappa = qa.weighted_cohen_kappa(rater_a, rater_b)
        assert -1.0 <= kappa <= 1.0
