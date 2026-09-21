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


# --- rate limits, quotas, and not re-running a scan on every widget click ---------------

from contextlib import closing  # noqa: E402
from datetime import UTC, datetime  # noqa: E402

from src import db, quota  # noqa: E402


@pytest.fixture
def fixed_clock(monkeypatch):
    monkeypatch.setattr(quota, "_utc", lambda now: datetime(2026, 9, 21, 12, 0, 30, tzinfo=UTC) if now is None else now)


def quota_identities():
    with closing(db.connect()) as conn:
        return sorted({row[0] for row in conn.execute("SELECT identity FROM quota_counters")})


def photo_variant(n):
    """A distinct photo per n (the app re-runs the model only when the photo changes)."""
    return encode(Image.new("RGB", (40, 30), (n * 20 % 255, 100, 60)), "PNG")


class TestSingleScanLimits:
    @pytest.fixture
    def scans(self, monkeypatch, make_identify_result):
        calls = []

        def fake(image):
            calls.append(image.size)
            return make_identify_result()

        monkeypatch.setattr(ident, "identify", fake)
        return calls

    def test_a_widget_rerun_does_not_rescan_rebill_or_duplicate_the_log_row(self, camera, scans, fixed_clock):
        camera(photo_variant(1))
        at = new_app().run()
        at = at.run()  # what any button/radio interaction causes
        at = at.run()
        assert len(scans) == 1
        assert len(inventory.list_recent()) == 1
        assert not at.exception and "Rose" in text_of(at.subheader)

    def test_a_new_photo_is_a_new_scan(self, camera, scans, fixed_clock):
        camera(photo_variant(1))
        at = new_app().run()
        camera(photo_variant(2))
        at.run()
        assert len(scans) == 2 and len(inventory.list_recent()) == 2

    def test_over_the_rate_limit_the_user_is_told_and_nothing_runs(self, camera, scans, fixed_clock, monkeypatch):
        monkeypatch.setenv("BLOOMLENS_RATE_PER_MINUTE", "1")
        camera(photo_variant(1))
        at = new_app().run()
        camera(photo_variant(2))
        at.run()
        assert len(scans) == 1  # the second photo was refused before the pipeline
        assert "Too many requests" in text_of(at.warning)
        assert len(inventory.list_recent()) == 1

    def test_a_refused_scan_leaves_the_page_usable(self, camera, scans, fixed_clock, monkeypatch):
        monkeypatch.setenv("BLOOMLENS_QUOTA_PER_DAY", "1")
        camera(photo_variant(1))
        at = new_app().run()
        camera(photo_variant(2))
        at.run()
        assert "daily quota" in text_of(at.warning) and "00:00 UTC" in text_of(at.warning)
        assert not at.exception

    def test_each_browser_session_has_its_own_allowance_but_they_share_the_global_ceiling(
        self, camera, scans, fixed_clock, monkeypatch
    ):
        monkeypatch.setenv("BLOOMLENS_RATE_PER_MINUTE", "1")
        monkeypatch.setenv("BLOOMLENS_GLOBAL_QUOTA_PER_DAY", "2")
        camera(photo_variant(1))
        first, second, third = new_app().run(), new_app().run(), new_app().run()
        assert len(scans) == 2  # two sessions each got their one scan
        assert "capacity" in text_of(third.warning)  # the third hit the shared daily ceiling
        assert not first.exception and not second.exception

    def test_identity_is_a_random_session_id_never_an_address_or_a_secret(self, camera, scans, fixed_clock):
        camera(photo_variant(1))
        new_app().run()
        identities = quota_identities()
        assert "*" in identities
        (session,) = [i for i in identities if i != "*"]
        assert session.startswith("ui:") and len(session) == len("ui:") + 16

    def test_when_every_slot_is_busy_the_scan_waits_and_costs_nothing(self, camera, scans, fixed_clock, monkeypatch):
        monkeypatch.setenv("BLOOMLENS_MAX_CONCURRENT", "1")
        quota.reset_gate()
        camera(photo_variant(1))
        with quota.gate().slot():
            at = new_app().run()
        assert "busy" in text_of(at.warning) and scans == []
        assert quota_identities() == []

    def test_an_unreachable_counter_store_refuses_the_scan(self, camera, scans, fixed_clock, tmp_path, monkeypatch):
        blocker = tmp_path / "a-file"
        blocker.write_text("x")
        monkeypatch.setenv("INVENTORY_DB_PATH", str(blocker / "shared.db"))
        camera(photo_variant(1))
        at = new_app().run()
        assert "rate limiter can't be reached" in text_of(at.error)
        assert scans == []
        assert str(blocker) not in all_rendered_text(at)

    def test_an_invalid_photo_costs_no_quota(self, camera, scans, fixed_clock, monkeypatch):
        monkeypatch.setenv("BLOOMLENS_RATE_PER_MINUTE", "1")
        camera(b"not an image")
        new_app().run()
        assert quota_identities() == []
        camera(photo_variant(1))
        new_app().run()
        assert len(scans) == 1


class TestLotLimits:
    def test_a_lot_spends_one_unit_and_a_refused_lot_runs_nothing(self, monkeypatch, make_lot_result, fixed_clock):
        calls = []
        monkeypatch.setattr(ident, "identify_lot", lambda images: calls.append(len(images)) or make_lot_result())
        monkeypatch.setenv("BLOOMLENS_RATE_PER_MINUTE", "1")
        at = new_app()
        at.session_state["lot_photos"] = lot_photos(3)
        at = click(at.run(), "🔍 Identify Lot")
        assert calls == [3]  # three photos, one unit
        at = click(at, "🔍 Identify Lot")
        assert calls == [3] and "Too many requests" in text_of(at.warning)


# --- the Performance expander: aggregates only ------------------------------------------

from tests.unit.test_telemetry import insert as insert_metric  # noqa: E402


def seed_metrics(n, **overrides):
    stamp = datetime.now(UTC).isoformat()
    with closing(db.connect()) as conn, db.transaction(conn):
        for i in range(n):
            insert_metric(conn, recorded_at=stamp, total_ms=1500 + i * 10, **overrides)


class TestPerformanceExpander:
    def test_with_no_scans_it_says_so(self):
        at = new_app().run()
        assert "No scans recorded yet" in text_of(at.caption)

    def test_a_small_sample_explains_why_there_are_no_percentiles(self):
        seed_metrics(5)
        at = new_app().run()
        assert "at least 20 successful warm scans (have 5)" in text_of(at.caption)
        assert "p50" not in text_of(at.caption)

    def test_enough_scans_show_p50_and_p95_and_the_counts(self):
        seed_metrics(30)
        at = new_app().run()
        assert "p50" in text_of(at.caption) and "p95" in text_of(at.caption)
        labels = {m.label: m.value for m in at.metric}
        assert labels["Scans"] == "30" and labels["Failed"] == "0%"

    def test_it_states_the_privacy_rule(self):
        at = new_app().run()
        assert "No photo, prompt, answer, address or identity is ever recorded" in text_of(at.caption)

    def test_it_shows_no_row_level_text(self):
        seed_metrics(30, bioclip_revision="REV-CANARY-ROW")
        at = new_app().run()
        assert "REV-CANARY-ROW" not in all_rendered_text(at)
