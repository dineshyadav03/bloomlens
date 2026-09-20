"""The Streamlit app (app.py) driven by Streamlit's AppTest harness, with the
pipeline faked so only the UI logic is under test.

Limit, stated plainly: AppTest cannot drive st.camera_input / st.file_uploader, so
the single-scan happy path (photo in -> result out) is NOT covered here; it is
covered at the function level (test_identify_pipeline) and by hand. Lot mode is
driven through session_state, which is how the app itself stores its photos."""

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from src import identify as ident
from src import inventory
from src.identify import IdentifyError
from tests.conftest import REPO_ROOT

APP = str(REPO_ROOT / "app.py")


def new_app():
    return AppTest.from_file(APP, default_timeout=120)


def lot_photos(n=3):
    return [Image.new("RGB", (24, 24), (200, 20 * i, 40)) for i in range(n)]


def click(at, label):
    button = next(b for b in at.button if b.label == label)
    return button.click().run()


def text_of(elements):
    return " | ".join(str(e.value) for e in elements)


def test_it_renders_three_tabs_without_an_exception():
    at = new_app().run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Single scan", "Lot mode", "Inventory"]
    assert at.title[0].value == "🌸 BloomLens"


def test_the_simulated_data_disclaimer_is_always_visible():
    at = new_app().run()
    assert "simulated demo data" in text_of(at.warning)


def test_single_scan_waits_for_a_photo():
    at = new_app().run()
    assert "Waiting for a scan" in text_of(at.info)


def test_the_inventory_starts_empty():
    at = new_app().run()
    assert "No scans logged yet" in text_of(at.info)


class TestLotMode:
    @pytest.fixture
    def fake_lot(self, monkeypatch, make_lot_result):
        calls = []

        def fake(images):
            calls.append(len(images))
            return make_lot_result(photo_count=len(images))

        monkeypatch.setattr(ident, "identify_lot", fake)
        return calls

    def app_with_photos(self, n=3):
        at = new_app()
        at.session_state["lot_photos"] = lot_photos(n)
        return at.run()

    def test_identifying_a_lot_shows_the_result_and_logs_one_row(self, fake_lot):
        at = click(self.app_with_photos(3), "🔍 Identify Lot")
        assert not at.exception
        assert fake_lot == [3]  # the whole lot goes through one call
        assert "Sunflower" in text_of(at.subheader)
        assert [m.value for m in at.metric if m.label == "Quality grade"] == ["B"]
        (row,) = inventory.list_recent()
        assert (row.mode, row.species, row.photo_count) == ("lot", "Sunflower", 3)

    def test_low_agreement_is_called_out_and_the_flagged_photo_is_listed(self, fake_lot):
        at = click(self.app_with_photos(3), "🔍 Identify Lot")
        assert "may contain mixed species" in text_of(at.warning)
        assert "Photo 3: detected as Rose" in text_of(at.markdown)

    def test_the_inventory_tab_reflects_the_scan_in_the_same_run(self, fake_lot):
        at = click(self.app_with_photos(3), "🔍 Identify Lot")
        assert "No scans logged yet" not in text_of(at.info)
        assert len(at.dataframe) == 1

    def test_a_pipeline_error_is_shown_and_nothing_is_logged(self, monkeypatch):
        def failing(_images):
            raise IdentifyError("Gemini is unavailable right now")

        monkeypatch.setattr(ident, "identify_lot", failing)
        at = click(self.app_with_photos(2), "🔍 Identify Lot")
        assert "Gemini is unavailable right now" in text_of(at.error)
        assert inventory.list_recent() == []

    def test_clearing_the_lot_empties_it(self, fake_lot):
        at = click(self.app_with_photos(3), "🗑️ Clear lot")
        assert at.session_state["lot_photos"] == []
        assert "No photos in the lot yet" in text_of(at.info)
