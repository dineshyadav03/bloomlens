"""The API's perimeter: authentication, request-size limits and response headers.

One pure-ASGI middleware does all of it *before* the framework parses a body, so an
unauthenticated or oversized request is refused without its upload ever being read
(FastAPI would otherwise parse the multipart body before running any dependency).

Authentication is fail-closed. `X-API-Key` is compared against labeled keys from
`BLOOMLENS_API_KEYS="label:secret,label2:secret2"`. With no keys configured every
route except /health answers 503, never 200. The only way to run without keys is
the explicit `BLOOMLENS_AUTH=disabled` (refused when `BLOOMLENS_ENV=production`),
which logs a warning. The label -- never the secret, never an IP address -- is the
caller's identity for everything downstream.
"""

import hashlib
import hmac
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache

from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader
from starlette.responses import JSONResponse

from src import guard

logger = logging.getLogger("bloomlens.security")

API_KEY_HEADER = "X-API-Key"
MIN_SECRET_LENGTH = 24
ANONYMOUS_LABEL = "anonymous"
STATE_KEY = "client_label"

PUBLIC_PATHS = frozenset({"/health"})
DOC_PATHS = frozenset({"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"})

_LABEL_RE = re.compile(r"[A-Za-z0-9_.-]{1,32}")
TOO_LARGE = "That request is too large."
_MULTIPART_OVERHEAD = 64 * 1024  # boundaries, field names and headers around the files
_DEFAULT_BODY_LIMIT = 64 * 1024  # any other route: a few form fields at most


def is_production() -> bool:
    return os.environ.get("BLOOMLENS_ENV", "").strip().lower() == "production"


class AuthConfigError(ValueError):
    """The key configuration is unusable. Never carries a secret in its message."""


@dataclass(frozen=True)
class AuthConfig:
    disabled: bool = False
    keys: tuple[tuple[str, bytes], ...] = ()  # (label, sha256(secret)) -- secrets aren't kept in the clear

    @property
    def usable(self) -> bool:
        return self.disabled or bool(self.keys)


def parse_keys(raw: str) -> tuple[tuple[str, bytes], ...]:
    entries, labels, digests = [], set(), set()
    for item in filter(None, (part.strip() for part in raw.split(","))):
        label, sep, secret = item.partition(":")
        if not sep or not _LABEL_RE.fullmatch(label):
            raise AuthConfigError("each key must look like label:secret (label: letters, digits, _ . - up to 32)")
        if len(secret) < MIN_SECRET_LENGTH:
            raise AuthConfigError(f"the secret for '{label}' is shorter than {MIN_SECRET_LENGTH} characters")
        digest = hashlib.sha256(secret.encode()).digest()
        if label in labels or digest in digests:
            raise AuthConfigError("labels and secrets must each be unique")
        labels.add(label)
        digests.add(digest)
        entries.append((label, digest))
    return tuple(entries)


@lru_cache(maxsize=8)
def _load(raw_keys: str, auth_flag: str, production: bool) -> AuthConfig:
    if auth_flag == "disabled":
        if production:
            logger.error("BLOOMLENS_AUTH=disabled is refused when BLOOMLENS_ENV=production; the API stays closed.")
            return AuthConfig()
        logger.warning("API authentication is DISABLED (BLOOMLENS_AUTH=disabled). Do not expose this instance.")
        return AuthConfig(disabled=True)
    try:
        return AuthConfig(keys=parse_keys(raw_keys))
    except AuthConfigError as exc:
        logger.error("BLOOMLENS_API_KEYS is invalid, so the API stays closed: %s", exc)
        return AuthConfig()


def auth_config() -> AuthConfig:
    """Read from the environment on every call (cached per value) so a test or an
    operator's restart picks up changes with no import-time state."""
    return _load(
        os.environ.get("BLOOMLENS_API_KEYS", ""),
        os.environ.get("BLOOMLENS_AUTH", "").strip().lower(),
        is_production(),
    )


def match_key(config: AuthConfig, presented: bytes | None) -> str | None:
    """The label whose secret was presented, or None. Every configured key is
    compared (no early exit) on fixed-length digests, in constant time."""
    if not presented:
        return None
    digest = hashlib.sha256(presented).digest()
    found = None
    for label, expected in config.keys:
        if hmac.compare_digest(digest, expected):
            found = label
    return found


_api_key_scheme = APIKeyHeader(
    name=API_KEY_HEADER, auto_error=False, description="A labeled key from BLOOMLENS_API_KEYS."
)


def current_client(request: Request, _documented_key: str | None = Depends(_api_key_scheme)) -> str:
    """Route dependency: the authenticated caller's label. The middleware has already
    decided; this refuses (fail closed) if a route is ever served without it having
    run, and puts the key scheme into the OpenAPI document."""
    label = getattr(request.state, STATE_KEY, None)
    if label is None:
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")
    return label


_SECURITY_HEADERS = (
    (b"x-content-type-options", b"nosniff"),
    (b"cache-control", b"no-store"),
    (b"referrer-policy", b"no-referrer"),
)
_API_ONLY_HEADERS = ((b"content-security-policy", b"default-src 'none'; frame-ancestors 'none'"),)


class BodyTooLarge(HTTPException):
    """An HTTPException, not a private type: FastAPI turns any *other* exception raised
    while it reads a body into a 400 "error parsing the body". The framework renders this
    one as a 413; if nothing inside does (a bare ASGI app), the middleware answers itself."""

    def __init__(self):
        super().__init__(status_code=413, detail=TOO_LARGE)


class SecurityMiddleware:
    def __init__(self, app, *, docs_enabled: bool):
        self.app = app
        self.public = PUBLIC_PATHS | (DOC_PATHS if docs_enabled else frozenset())
        self.docs = DOC_PATHS if docs_enabled else frozenset()

    @staticmethod
    def body_limit(path: str) -> int:
        if path in ("/identify", "/explain"):
            return guard.MAX_UPLOAD_BYTES + _MULTIPART_OVERHEAD
        if path == "/identify-lot":
            return guard.MAX_LOT_BYTES + _MULTIPART_OVERHEAD
        return _DEFAULT_BODY_LIMIT

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope["path"]
        started = False
        extra = _SECURITY_HEADERS if path in self.docs else _SECURITY_HEADERS + _API_ONLY_HEADERS

        async def send_with_headers(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                present = {name.lower() for name, _ in message.get("headers", [])}
                message = {
                    **message,
                    "headers": [*message.get("headers", []), *(h for h in extra if h[0] not in present)],
                }
            await send(message)

        async def refuse(status: int, detail: str):
            await JSONResponse({"detail": detail}, status_code=status)(scope, receive, send_with_headers)

        if path not in self.public:
            config = auth_config()
            if not config.usable:
                await refuse(503, "API authentication is not configured on this server.")
                return
            if config.disabled:
                label = ANONYMOUS_LABEL
            else:
                headers = {name.lower(): value for name, value in scope["headers"]}
                label = match_key(config, headers.get(API_KEY_HEADER.lower().encode()))
                if label is None:
                    logger.warning("rejected a request with a missing or invalid API key")
                    await refuse(401, "Missing or invalid API key.")
                    return
            scope.setdefault("state", {})[STATE_KEY] = label

        limit = self.body_limit(path)
        declared = next((v for k, v in scope["headers"] if k == b"content-length"), None)
        if declared is not None and declared.isdigit() and int(declared) > limit:
            await refuse(413, TOO_LARGE)
            return

        received = 0

        async def counting_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise BodyTooLarge
            return message

        try:
            await self.app(scope, counting_receive, send_with_headers)
        except BodyTooLarge:
            if started:
                raise
            await refuse(413, TOO_LARGE)
