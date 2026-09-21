"""The API perimeter (api/security.py + api/main.py): who gets in, how big a request may
be, what leaves in an error, and what headers ride on every response."""

import asyncio
import io
import logging

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import api.main as api
from api import security
from api.security import SecurityMiddleware, match_key, parse_keys
from src import guard
from src.identify import IdentifyConfigError, IdentifyError
from tests.conftest import TEST_API_KEY
from tests.unit.test_guard import png_bomb

SECRET_A = "alpha-secret-0123456789-abcdefgh"
SECRET_B = "bravo-secret-9876543210-zyxwvuts"

# The served surface, read from the app's own OpenAPI document (FastAPI wraps included
# routers, so app.routes no longer lists the individual operations).
PROTECTED = [
    (method.upper(), path)
    for path, operations in api.app.openapi()["paths"].items()
    if path not in security.PUBLIC_PATHS
    for method in operations
]


@pytest.fixture
def client(api_headers):
    return TestClient(api.app, headers=api_headers)


@pytest.fixture
def anonymous():
    """A client that presents no key."""
    return TestClient(api.app)


def png_upload(name="flower.png", size=(20, 20)):
    buf = io.BytesIO()
    Image.new("RGB", size, (9, 99, 199)).save(buf, format="PNG")
    return name, buf.getvalue(), "image/png"


def call_asgi(app, path, *, method="POST", headers=(), chunks=(b"",)):
    """Drive an ASGI app directly; returns (status, headers, body, chunks_the_app_pulled)."""
    sent, pulled = [], []
    queue = [{"type": "http.request", "body": c, "more_body": i < len(chunks) - 1} for i, c in enumerate(chunks)]

    async def receive():
        pulled.append(1)
        return queue.pop(0) if queue else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": method, "path": path, "headers": list(headers), "query_string": b""}
    asyncio.run(app(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return start["status"], dict(start["headers"]), body, len(pulled)


async def drain_body_app(scope, receive, send):
    """Reads its whole body, then answers 200."""
    while (await receive()).get("more_body"):
        pass
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


class TestFailClosed:
    def test_with_no_keys_configured_every_protected_route_is_a_503_never_a_200(self):
        client = TestClient(api.app)
        assert len(PROTECTED) >= 5
        for method, path in PROTECTED:
            response = client.request(method, path)
            assert response.status_code == 503, path
            assert "not configured" in response.json()["detail"]

    def test_health_stays_public_even_when_nothing_is_configured(self):
        assert TestClient(api.app).get("/health").json() == {"status": "ok"}

    def test_a_key_that_is_too_short_leaves_the_api_closed_and_is_not_logged(self, monkeypatch, caplog):
        monkeypatch.setenv("BLOOMLENS_API_KEYS", "tester:short-secret")
        with caplog.at_level(logging.ERROR, logger="bloomlens.security"):
            response = TestClient(api.app).get("/inventory", headers={"X-API-Key": "short-secret"})
        assert response.status_code == 503
        assert "short-secret" not in caplog.text and "shorter than" in caplog.text

    @pytest.mark.parametrize(
        "raw",
        [
            "no-colon-here",
            f"bad label:{SECRET_A}",
            f":{SECRET_A}",
            f"a:{SECRET_A},a:{SECRET_B}",  # duplicate label
            f"a:{SECRET_A},b:{SECRET_A}",  # duplicate secret
        ],
    )
    def test_malformed_configuration_fails_closed(self, monkeypatch, raw):
        monkeypatch.setenv("BLOOMLENS_API_KEYS", raw)
        assert TestClient(api.app).get("/inventory", headers={"X-API-Key": SECRET_A}).status_code == 503

    def test_the_auth_off_switch_is_explicit_and_loud(self, monkeypatch, caplog):
        monkeypatch.setenv("BLOOMLENS_AUTH", "disabled")
        with caplog.at_level(logging.WARNING, logger="bloomlens.security"):
            response = TestClient(api.app).get("/inventory")
        assert response.status_code == 200
        assert "DISABLED" in caplog.text

    def test_the_auth_off_switch_is_refused_in_production(self, monkeypatch):
        monkeypatch.setenv("BLOOMLENS_AUTH", "disabled")
        monkeypatch.setenv("BLOOMLENS_ENV", "production")
        assert TestClient(api.app).get("/inventory").status_code == 503

    def test_any_other_value_of_the_switch_does_not_open_the_api(self, monkeypatch):
        monkeypatch.setenv("BLOOMLENS_AUTH", "off")
        assert TestClient(api.app).get("/inventory").status_code == 503


class TestAuthentication:
    def test_a_missing_key_is_a_401(self, api_headers, anonymous):
        assert anonymous.get("/inventory").status_code == 401

    def test_a_wrong_key_is_the_same_401_as_a_missing_one(self, api_headers, anonymous):
        missing = anonymous.get("/inventory")
        wrong = anonymous.get("/inventory", headers={"X-API-Key": TEST_API_KEY[:-1] + "!"})
        assert wrong.status_code == 401 and wrong.json() == missing.json()

    def test_the_key_must_be_in_the_header_not_the_query_string(self, api_headers, anonymous):
        assert anonymous.get(f"/inventory?api_key={TEST_API_KEY}").status_code == 401

    def test_a_valid_key_is_let_through(self, client):
        assert client.get("/inventory").status_code == 200

    def test_every_protected_route_rejects_an_unauthenticated_request(self, api_headers, anonymous):
        """Enumerated from the app's own route table, so a route added later without
        protection can't slip through untested."""
        assert len(PROTECTED) >= 5
        for method, path in PROTECTED:
            assert anonymous.request(method, path).status_code == 401, path

    def test_a_401_never_touches_the_pipeline_or_the_inventory(self, api_headers, anonymous, monkeypatch):
        monkeypatch.setattr(api, "identify", lambda _i: pytest.fail("must not run unauthenticated"))
        response = anonymous.post("/identify", files={"photo": png_upload()})
        assert response.status_code == 401

    def test_several_labeled_keys_each_work_and_are_told_apart(self):
        config = security.AuthConfig(keys=parse_keys(f"alice:{SECRET_A},bob:{SECRET_B}"))
        assert match_key(config, SECRET_A.encode()) == "alice"
        assert match_key(config, SECRET_B.encode()) == "bob"
        assert match_key(config, b"someone-else-entirely-0123456789") is None
        assert match_key(config, b"") is None and match_key(config, None) is None

    def test_the_label_reaches_the_route_and_the_secret_does_not(self, monkeypatch):
        monkeypatch.setenv("BLOOMLENS_API_KEYS", f"alice:{SECRET_A},bob:{SECRET_B}")
        seen = {}

        async def probe(scope, receive, send):
            seen.update(scope["state"])
            await drain_body_app(scope, receive, send)

        status, *_ = call_asgi(
            SecurityMiddleware(probe, docs_enabled=False), "/x", headers=[(b"x-api-key", SECRET_B.encode())]
        )
        assert status == 200
        assert seen == {"client_label": "bob"}

    def test_every_configured_key_is_compared_so_timing_does_not_reveal_which_matched(self, monkeypatch):
        monkeypatch.setenv("BLOOMLENS_API_KEYS", f"a:{SECRET_A},b:{SECRET_B},c:c-secret-0123456789-abcdefghij")
        calls = []
        real = security.hmac.compare_digest
        monkeypatch.setattr(security.hmac, "compare_digest", lambda x, y: calls.append(1) or real(x, y))
        match_key(security.auth_config(), SECRET_A.encode())  # the FIRST key matches...
        assert len(calls) == 3  # ...and all three were still compared

    def test_secrets_are_not_retained_in_the_clear(self):
        keys = parse_keys(f"alice:{SECRET_A}")
        assert SECRET_A.encode() not in b"".join(digest for _, digest in keys)
        assert all(len(digest) == 32 for _, digest in keys)

    def test_a_route_served_without_the_middleware_still_fails_closed(self, api_headers):
        """Defense in depth: if someone builds the app and forgets the middleware, the
        per-route dependency refuses rather than serving."""
        bare = api.FastAPI()
        bare.include_router(api.protected)
        assert TestClient(bare).get("/inventory", headers=api_headers).status_code == 401


class TestRequestSize:
    def test_a_declared_oversize_body_is_refused_without_reading_any_of_it(self, api_headers):
        app = SecurityMiddleware(drain_body_app, docs_enabled=False)
        headers = [(b"x-api-key", TEST_API_KEY.encode()), (b"content-length", b"999999999")]
        status, _, body, pulled = call_asgi(app, "/identify", headers=headers, chunks=(b"x" * 10,))
        assert status == 413 and pulled == 0
        assert b"too large" in body

    def test_an_undeclared_oversize_body_is_cut_off_as_it_streams_in(self, api_headers):
        """Chunked upload, no Content-Length: refused once the running total passes the cap."""
        app = SecurityMiddleware(drain_body_app, docs_enabled=False)
        headers = [(b"x-api-key", TEST_API_KEY.encode())]
        chunk = b"x" * 40_000  # the default cap for an unlisted route is 64 KiB
        status, _, _, pulled = call_asgi(app, "/other", headers=headers, chunks=(chunk,) * 50)
        assert status == 413
        assert pulled <= 3  # it did not read all 50 chunks

    def test_the_cap_is_exact_at_the_boundary(self, api_headers):
        app = SecurityMiddleware(drain_body_app, docs_enabled=False)
        headers = [(b"x-api-key", TEST_API_KEY.encode())]
        limit = SecurityMiddleware.body_limit("/other")
        assert call_asgi(app, "/other", headers=headers, chunks=(b"x" * limit,))[0] == 200
        assert call_asgi(app, "/other", headers=headers, chunks=(b"x" * (limit + 1),))[0] == 413

    def test_a_single_photo_route_has_a_smaller_cap_than_the_lot_route(self):
        assert SecurityMiddleware.body_limit("/identify") < SecurityMiddleware.body_limit("/identify-lot")

    def test_an_oversize_photo_through_the_real_app_is_a_413_and_nothing_runs(self, client, monkeypatch):
        monkeypatch.setattr(guard, "MAX_UPLOAD_BYTES", 50_000)
        monkeypatch.setattr(api, "identify", lambda _i: pytest.fail("must not run"))
        response = client.post("/identify", files={"photo": ("big.png", b"\x89PNG" + b"x" * 300_000, "image/png")})
        assert response.status_code == 413
        assert client.get("/inventory").json() == []

    def test_a_chunked_oversize_body_through_the_real_app_is_a_413(self, client, monkeypatch):
        """A well-formed multipart upload whose one file never ends, sent chunked (so no
        Content-Length): the running total, not a header, is what stops it."""
        monkeypatch.setattr(guard, "MAX_UPLOAD_BYTES", 50_000)

        def endless():
            yield b'--x\r\nContent-Disposition: form-data; name="photo"; filename="a.png"\r\n\r\n'
            for _ in range(200):
                yield b"x" * 20_000

        response = client.post(
            "/identify", content=endless(), headers={"Content-Type": "multipart/form-data; boundary=x"}
        )
        assert response.status_code == 413
        assert "content-length" not in response.request.headers

    def test_a_photo_under_the_body_cap_but_over_the_photo_cap_is_refused_by_the_guard(self, client, monkeypatch):
        monkeypatch.setattr(guard, "MAX_UPLOAD_BYTES", 1_000)
        response = client.post("/identify", files={"photo": ("a.png", b"x" * 2_000, "image/png")})
        assert response.status_code == 413
        assert "too large" in response.json()["detail"]

    def test_a_lot_over_its_total_cap_is_a_413(self, client, monkeypatch):
        name, data, mime = png_upload()
        monkeypatch.setattr(guard, "MAX_LOT_BYTES", len(data) * 2)
        monkeypatch.setattr(api, "identify_lot", lambda _i: pytest.fail("must not run"))
        files = [("photos", (f"{i}.png", data, mime)) for i in range(3)]
        assert client.post("/identify-lot", files=files).status_code == 413


class TestUploadContent:
    def test_a_decompression_bomb_is_a_413_and_never_reaches_the_pipeline(self, client, monkeypatch):
        monkeypatch.setattr(api, "identify", lambda _i: pytest.fail("must not run"))
        response = client.post("/identify", files={"photo": ("bomb.png", png_bomb(20_000, 20_000), "image/png")})
        assert response.status_code == 413
        assert "pixels" in response.json()["detail"]

    def test_a_text_file_named_png_is_a_400(self, client):
        response = client.post("/identify", files={"photo": ("evil.png", b"<script>alert(1)</script>", "image/png")})
        assert response.status_code == 400

    def test_a_gif_named_jpg_is_a_415_by_content_not_name(self, client):
        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, format="GIF")
        response = client.post("/identify", files={"photo": ("cute.jpg", buf.getvalue(), "image/jpeg")})
        assert response.status_code == 415

    def test_the_explain_route_is_guarded_too(self, client):
        response = client.post("/explain", files={"photo": ("bomb.png", png_bomb(20_000, 20_000), "image/png")},
                               data={"species": "Rose"})
        assert response.status_code == 413

    def test_the_photo_the_pipeline_receives_carries_no_metadata(self, client, monkeypatch, make_identify_result):
        seen = {}

        def fake(image):
            seen["info"], seen["exif"] = dict(image.info), len(image.getexif())
            return make_identify_result()

        exif = Image.Exif()
        exif[0x0110] = "CameraModel"
        buf = io.BytesIO()
        Image.new("RGB", (30, 30)).save(buf, format="JPEG", exif=exif)
        monkeypatch.setattr(api, "identify", fake)
        assert client.post("/identify", files={"photo": ("a.jpg", buf.getvalue(), "image/jpeg")}).status_code == 200
        assert seen == {"info": {}, "exif": 0}


class TestErrorsRevealNothing:
    def test_an_upload_error_never_echoes_the_filename_or_library_text(self, client):
        response = client.post("/identify", files={"photo": ("private-name-7431.png", b"junk", "image/png")})
        text = response.text
        assert response.status_code == 400
        for leak in ("private-name-7431", "PIL", "cannot identify", "Traceback", "BytesIO"):
            assert leak not in text

    def test_an_unexpected_crash_is_a_bare_500(self, api_headers, monkeypatch):
        def boom(_image):
            raise RuntimeError("secret-path C:/Users/x/.env and a token sk-123456")

        monkeypatch.setattr(api, "identify", boom)
        quiet = TestClient(api.app, headers=api_headers, raise_server_exceptions=False)
        response = quiet.post("/identify", files={"photo": png_upload()})
        assert response.status_code == 500
        assert "secret-path" not in response.text and "sk-123456" not in response.text

    def test_a_pipeline_error_reaches_the_caller_as_its_fixed_message_only(self, client, monkeypatch):
        def failing(_image):
            raise IdentifyError("The identification service didn't return a usable answer.")

        monkeypatch.setattr(api, "identify", failing)
        response = client.post("/identify", files={"photo": png_upload()})
        assert response.status_code == 503
        assert response.json() == {"detail": "The identification service didn't return a usable answer."}

    def test_a_misconfiguration_is_reported_generically(self, client, monkeypatch):
        def unconfigured(_image):
            raise IdentifyConfigError("GEMINI_API_KEY is not set. Copy .env.example to .env")

        monkeypatch.setattr(api, "identify", unconfigured)
        response = client.post("/identify", files={"photo": png_upload()})
        assert response.status_code == 503
        assert "GEMINI" not in response.text and ".env" not in response.text

    def test_an_unknown_species_to_explain_does_not_echo_the_input(self, client):
        response = client.post("/explain", files={"photo": png_upload()}, data={"species": "Zq-echo-canary-91"})
        assert response.status_code == 400
        assert "echo-canary" not in response.text

    def test_a_validation_error_says_where_not_what(self, client):
        response = client.get("/inventory?limit=echo-canary-42")
        assert response.status_code == 422
        assert "echo-canary" not in response.text
        assert response.json()["detail"][0]["loc"] == ["query", "limit"]


class TestInputBounds:
    @pytest.mark.parametrize("limit", [0, -1, 501, 10**9])
    def test_inventory_limit_is_bounded(self, client, limit):
        assert client.get(f"/inventory?limit={limit}").status_code == 422

    @pytest.mark.parametrize("limit", [1, 500])
    def test_the_bounds_themselves_are_allowed(self, client, limit):
        assert client.get(f"/inventory?limit={limit}").status_code == 200

    def test_an_absurdly_long_species_name_is_refused_before_any_work(self, client):
        response = client.post("/explain", files={"photo": png_upload()}, data={"species": "x" * 5000})
        assert response.status_code == 422


class TestResponseHeaders:
    EXPECTED = {"x-content-type-options": "nosniff", "cache-control": "no-store", "referrer-policy": "no-referrer"}

    def test_a_normal_response_carries_them(self, client):
        response = client.get("/inventory")
        for name, value in self.EXPECTED.items():
            assert response.headers[name] == value
        assert "frame-ancestors 'none'" in response.headers["content-security-policy"]

    def test_so_do_the_early_refusals(self, api_headers, anonymous):
        for response in (anonymous.get("/inventory"), TestClient(api.app).get("/health")):
            assert response.headers["x-content-type-options"] == "nosniff"
            assert response.headers["cache-control"] == "no-store"

    def test_and_a_503_from_a_missing_configuration(self):
        assert TestClient(api.app).get("/inventory").headers["x-content-type-options"] == "nosniff"

    def test_no_cors_headers_are_ever_sent(self, client):
        for request in (
            client.get("/inventory", headers={"Origin": "https://evil.example"}),
            client.options(
                "/inventory",
                headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"},
            ),
        ):
            assert not any(name.lower().startswith("access-control-") for name in request.headers)


class TestDocs:
    def test_development_serves_the_docs_and_they_document_the_key(self, anonymous):
        assert anonymous.get("/docs").status_code == 200
        spec = anonymous.get("/openapi.json").json()
        schemes = spec["components"]["securitySchemes"]
        assert any(s.get("name") == "X-API-Key" and s["in"] == "header" for s in schemes.values())
        assert "security" not in spec["paths"]["/health"]["get"]
        assert spec["paths"]["/inventory"]["get"]["security"]

    def test_production_serves_no_docs_and_no_schema(self, api_headers):
        prod = TestClient(api.create_app(production=True), headers=api_headers)
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert prod.get(path).status_code == 404, path
        assert prod.get("/health").status_code == 200
        assert prod.get("/inventory").status_code == 200

    def test_production_docs_paths_are_not_reachable_anonymously_either(self, api_headers):
        assert TestClient(api.create_app(production=True)).get("/openapi.json").status_code == 401

    def test_the_environment_variable_selects_production(self, monkeypatch, api_headers):
        monkeypatch.setenv("BLOOMLENS_ENV", "production")
        assert TestClient(api.create_app(), headers=api_headers).get("/docs").status_code == 404
