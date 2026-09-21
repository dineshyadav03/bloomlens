# Security model

BloomLens is a portfolio project, not a hardened product. This page says what it
defends, against whom, how each defence is tested, and — just as important — what
it does **not** defend. Privacy (what leaves the app, what Google keeps) has its own
page: [PRIVACY.md](PRIVACY.md).

## What is being protected

| Asset | Why it matters |
|---|---|
| The Gemini API key | Spending and quota; lives only in `GEMINI_API_KEY` (never in the image, logs or responses). |
| Uploaded photos | May carry a location and a camera identity; may be someone's property. |
| The host's CPU, memory and disk | Inference is heavy; uploads are attacker-sized. |
| The API's callers' keys | Labeled `X-API-Key` secrets; identity for later quota accounting. |
| The inventory log | Low value (species, grade, simulated price), but integrity matters for the demo. |

## Who it is defended against

- **An anonymous network client** who can reach the API or Streamlit port: should get nothing
  useful from the API without a key, and should not be able to exhaust the machine with one request.
- **A malicious uploader** (authenticated or, for Streamlit, anyone): hostile files — decompression
  bombs, disguised formats, oversized bodies, metadata carrying personal data.
- **Hostile content inside a photo** (text written on a sign, a label): an attempt to steer the language model.
- **A careless operator**: forgets to set keys, leaves debug switches on, exposes a port by accident.

Out of scope: a compromised host, a malicious dependency (only audited, see below), a
determined attacker with a valid key (until per-key quotas land in M13b), and anything
that needs a multi-tenant boundary.

## Controls, and the test that proves each

| Control | Where | Proven by |
|---|---|---|
| API is **fail-closed**: no configured key ⇒ every route but `/health` answers 503, never 200 | `api/security.py` | `test_api_security.py::TestFailClosed` |
| Labeled keys (`label:secret`, ≥ 24 characters), compared in constant time against **every** key, stored only as SHA-256 | `api/security.py` | `TestAuthentication` |
| Missing and wrong keys are the same 401 (no oracle); the key is read from a header, never the URL | same | same |
| `BLOOMLENS_AUTH=disabled` is the only way to run without keys, logs loudly and is **refused** when `BLOOMLENS_ENV=production` | same | `TestFailClosed` |
| Auth and size checks run **before** the body is parsed, in one pure-ASGI middleware | same | `TestRequestSize` (asserts the body is never read on a declared oversize) |
| Per-route body caps, enforced from `Content-Length` *and* while streaming (chunked bodies too) | same | `TestRequestSize` |
| Upload validation shared by API and Streamlit: byte cap while reading, format decided from **content** (JPEG/PNG/WebP), 40-megapixel cap from the header **before** decoding, animated images refused | `src/guard.py` | `test_guard.py` (real 400-megapixel PNG bomb stays under 50 MB peak) |
| EXIF / GPS / XMP / ICC stripped by rebuilding the image from pixels; orientation applied first | same | `TestMetadataIsStripped` (GPS-tagged fixture → absent from the bytes sent) |
| Errors carry fixed wording only: no file names, library text or provider text | `api/main.py`, `src/identify.py` | `TestErrorsRevealNothing`, `test_identify_validation.py` (canary strings never appear) |
| Validation errors say *where*, not *what* (the default 422 echoes input) | `api/main.py` | `TestErrorsRevealNothing` |
| Model output checked for shape: grade ∈ {A, B, C}, per-field length caps; a violation is a failed parse | `src/identify.py` | `test_identify_validation.py` |
| Model text rendered **literally** (`st.text`, JSON strings) rather than sanitised | `app.py` | `test_app_ui.py::TestSingleScanUpload` |
| Tracing to third parties forced off | `src/privacy.py` | `test_privacy.py` (fresh interpreter with tracing switched on) |
| Security headers on every response, early refusals included: `nosniff`, `no-store`, `no-referrer`, CSP `default-src 'none'` | `api/security.py` | `TestResponseHeaders` |
| No CORS, ever | — | `TestResponseHeaders` |
| Interactive docs and the OpenAPI document are off in production | `api/main.py` | `TestDocs` |
| Bounded inputs: `/inventory?limit` 1–500, `species` ≤ 100 characters | `api/main.py` | `TestInputBounds` |
| Streamlit's own fence: `maxUploadSize`, XSRF on, CORS on, usage stats off, no error details in the browser | `.streamlit/config.toml` | asserted with `streamlit config show` |
| Containers run as a non-root user, ports bound to `127.0.0.1`, `no-new-privileges`, all capabilities dropped | `Dockerfile`, `docker-compose.yml` | `docker inspect` (recorded in the PR) |

The security tests were **mutation-checked**: removing or weakening each control (fail-open,
early-exit key compare, no streaming cap, no declared-length cap, a dropped header, docs left
on, echoing 422s, an unbounded limit, bypassing the guard, quoting provider text, logging
exception text …) turns the suite red. One survivor was investigated and was an equivalent
mutation (FastAPI serves `/docs` only if the OpenAPI URL is set too).

## Configuration reference

| Variable | Meaning | Default |
|---|---|---|
| `BLOOMLENS_API_KEYS` | `label:secret,label2:secret2`; secrets ≥ 24 characters, unique | none ⇒ API closed |
| `BLOOMLENS_AUTH` | `disabled` turns auth off (development only; refused in production) | unset |
| `BLOOMLENS_ENV` | `production` turns off `/docs` and `/openapi.json` and forbids `BLOOMLENS_AUTH=disabled` | unset (`docker-compose.yml` sets `production` for the API) |
| `BLOOMLENS_MAX_UPLOAD_MB` | per-photo byte cap | 8 |
| `BLOOMLENS_MAX_LOT_MB` | whole-lot byte cap | 40 |

Generate a key with `python -c "import secrets; print(secrets.token_urlsafe(32))"`.

## Residual risk — read this before exposing anything

1. **Streamlit has no authentication.** Anyone who can reach its port can scan photos and spend the
   Gemini quota, and the inventory tab is visible to them. It is a demo UI. Compose binds it to
   `127.0.0.1`; put a reverse proxy with auth in front of it, or accept the exposure, before
   publishing. The public Hugging Face Space accepts this knowingly (see PRIVACY.md).
2. **No rate or quota limits yet.** A valid API key can send requests as fast as the host allows,
   and each accepted scan costs Gemini calls. Per-key quotas and rate limits are milestone **M13b**.
3. **Prompt injection cannot be prevented, only contained.** Text inside a photo may still bias the
   model's *words*. What the design guarantees is that the result cannot exceed its shape: the grade is
   A/B/C, lengths are capped, `species` is always one of the retrieved candidates (the model's text only
   chooses among them), and nothing the model writes is rendered as markup or written anywhere but the
   response. A consumer of the API must still treat those strings as untrusted.
4. **Memory per Streamlit session.** A lot holds up to 10 decoded photos in the session; at the
   40-megapixel cap each is up to ~120 MB. Fine for a demo, a resource-exhaustion risk for a public host.
5. **The Qdrant container has no authentication.** It is reachable only on the compose network (no
   published port), and its image is not pinned to a digest.
6. **TLS is not terminated here.** Use a proxy for HTTPS; add HSTS there.
7. **Hugging Face Spaces and XSRF.** Streamlit's XSRF protection is on. Some deployments inside an
   iframe report upload failures with it on; that has **not been tested** here because no Space exists
   yet. If uploads fail there, the known workaround is to disable XSRF for that deployment only, which is
   a real security trade-off and should be a conscious decision.
8. **Dependencies.** `uv.lock` pins and hashes everything; `pip-audit` runs in CI against the lock and
   `gitleaks` scans history. An audit finding in a transitive dependency is reported, not auto-fixed.
9. **Logs.** Upload rejections log a reason code only; identification failures log the attempt and
   exception *type*. Unexpected-error logs include a traceback (which can contain exception text) but
   never a photo.

## Reporting a problem

Open a GitHub issue, or — for anything that shouldn't be public first — a private security advisory on
the repository.
