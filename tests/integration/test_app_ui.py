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


# --- uploads and model text ------------------------------------------------------------
# AppTest cannot drive the camera/uploader widgets, but app.py runs in this process, so the
# widget *functions* can be replaced: the app then receives exactly what Streamlit would
# hand it (a file-like UploadedFile), and everything downstream is the real code.

import io  # noqa: E402

import streamlit as st  # noqa: E402

from tests.unit.test_guard import encode, png_bomb  # noqa: E402

CANARY = "CANARY-77-DO-NOT-SHOW"


def all_rendered_text(at):
    """Every string the page shows, whichever element type carries it."""
    shown = [*at.markdown, *at.text, *at.error, *at.warning, *at.info, *at.caption, *at.subheader, *at.exception]
    return text_of(shown)


@pytest.fixture
def camera(monkeypatch):
    """camera(data) makes the single-scan camera 'capture' those bytes."""

    def _set(data):
        monkeypatch.setattr(
            st, "camera_input", lambda *a, **kw: io.BytesIO(data) if "key" not in kw and data is not None else None
        )

    return _set


def photo_bytes(fmt="JPEG", **kwargs):
    return encode(Image.new("RGB", (40, 30), (10, 160, 60)), fmt, **kwargs)


class TestSingleScanUpload:
    def test_a_non_image_is_refused_in_plain_words_and_never_reaches_the_pipeline(self, camera, monkeypatch):
        monkeypatch.setattr(ident, "identify", lambda _i: pytest.fail("must not run"))
        camera(b"%PDF-1.7 " + CANARY.encode())
        at = new_app().run()
        assert "isn't a readable image" in text_of(at.error)
        for leak in ("PIL", "Traceback", "cannot identify", CANARY):
            assert leak not in all_rendered_text(at)
        assert inventory.list_recent() == []

    def test_a_decompression_bomb_is_refused_before_decoding(self, camera, monkeypatch):
        monkeypatch.setattr(ident, "identify", lambda _i: pytest.fail("must not run"))
        camera(png_bomb(20_000, 20_000))
        at = new_app().run()
        assert "too many pixels" in text_of(at.error)

    def test_the_pipeline_receives_a_photo_stripped_of_its_metadata(self, camera, monkeypatch, make_identify_result):
        seen = {}

        def fake(image):
            seen["exif"], seen["info"] = len(image.getexif()), dict(image.info)
            return make_identify_result()

        exif = Image.Exif()
        exif[0x0110] = "SecretCameraModel"
        monkeypatch.setattr(ident, "identify", fake)
        camera(photo_bytes(exif=exif))
        at = new_app().run()
        assert not at.exception and seen == {"exif": 0, "info": {}}

    def test_model_text_is_shown_literally_never_as_markdown(self, camera, monkeypatch, make_identify_result):
        nasty = "<img src=x onerror=alert(1)> **bold** [click](https://evil.example) # heading"
        monkeypatch.setattr(
            ident,
            "identify",
            lambda _i: make_identify_result(summary=nasty, confidence_note=nasty, quality_note=nasty),
        )
        camera(photo_bytes())
        at = new_app().run()
        assert not at.exception
        literal = [t.value for t in at.text]
        assert literal.count(nasty) == 3  # summary, confidence note, quality note
        assert nasty not in text_of(at.markdown)  # and none of it went through the markdown renderer

    def test_an_unexpected_error_shows_no_exception_text(self, camera, monkeypatch):
        def crash(_image):
            raise RuntimeError(f"secret {CANARY}")

        monkeypatch.setattr(ident, "identify", crash)
        camera(photo_bytes())
        at = new_app().run()
        assert "Something went wrong on our side" in text_of(at.error)
        assert CANARY not in all_rendered_text(at)
        assert inventory.list_recent() == []


class TestLotUpload:
    @pytest.fixture
    def uploads(self, monkeypatch):
        def _set(*datas):
            files = [io.BytesIO(d) for d in datas]
            monkeypatch.setattr(
                st, "file_uploader", lambda *a, **kw: files if kw.get("key") == "lot_uploader" else None
            )

        return _set

    def test_good_uploads_join_the_lot_as_clean_images(self, uploads):
        uploads(photo_bytes(), photo_bytes("PNG"))
        at = click(new_app().run(), "➕ Add uploaded photos to lot")
        assert [i.size for i in at.session_state["lot_photos"]] == [(40, 30)] * 2

    def test_one_bad_file_adds_nothing_and_says_why(self, uploads):
        uploads(photo_bytes(), b"not an image " + CANARY.encode())
        at = click(new_app().run(), "➕ Add uploaded photos to lot")
        assert at.session_state["lot_photos"] == []
        assert "isn't a readable image" in text_of(at.error)
        assert CANARY not in all_rendered_text(at)

    def test_the_lot_result_text_is_literal_too(self, monkeypatch, make_lot_result):
        nasty = "**not bold** <b>x</b>"
        monkeypatch.setattr(ident, "identify_lot", lambda images: make_lot_result(summary=nasty, quality_note=nasty))
        at = new_app()
        at.session_state["lot_photos"] = lot_photos(2)
        at = click(at.run(), "🔍 Identify Lot")
        assert [t.value for t in at.text].count(nasty) == 2
        assert nasty not in text_of(at.markdown)


def test_the_privacy_notice_says_where_photos_go():
    at = new_app().run()
    notice = text_of(at.info)
    assert "Gemini" in notice and "free tier" in notice and "docs/PRIVACY.md" in notice
