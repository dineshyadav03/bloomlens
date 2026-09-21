"""BloomLens FastAPI endpoint -- a thin translation layer over src/identify.py,
not a second implementation. Both this and app.py (Streamlit) call the exact
same identify()/identify_lot() and render the exact same IdentifyResult/
LotResult Pydantic models; nothing about the pipeline lives here.

Every route except /health needs an `X-API-Key` (api/security.py); uploads go
through src/guard.py, the same validator the Streamlit app uses. The routes that run
the pipeline are rate- and quota-limited per key label (src/quota.py, counters in the
shared SQLite file), after the upload has validated -- a bad upload costs nothing.
Text fields in the responses are written by a language model: treat them as untrusted.

Run: BLOOMLENS_API_KEYS="me:<24+ char secret>" uvicorn api.main:app --reload
(see .claude/launch.json for a preset). Swagger UI is at /docs unless
BLOOMLENS_ENV=production.
"""

import io
import logging
from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from PIL import Image

from api.security import SecurityMiddleware, current_client, is_production
from src import guard, quota
from src.identify import (
    LOT_MAX_PHOTOS,
    IdentifyConfigError,
    IdentifyError,
    IdentifyResult,
    LotResult,
    identify,
    identify_lot,
)
from src.interpretability import ExplainError, explain_image, is_known_species
from src.inventory import InventoryEntry, list_recent, log_lot_scan, log_scan, species_counts

logger = logging.getLogger("bloomlens.api")

INVENTORY_MAX_LIMIT = 500
SPECIES_FIELD_MAX_LENGTH = 100

public = APIRouter()
protected = APIRouter(dependencies=[Depends(current_client)])


def _read_image(upload: UploadFile) -> Image.Image:
    try:
        return guard.validate_upload(upload.file)
    except guard.UploadRejected as exc:
        raise HTTPException(status_code=exc.status, detail=exc.message) from exc


def _read_lot(uploads: list[UploadFile]) -> list[Image.Image]:
    try:
        return guard.validate_lot([u.file for u in uploads])
    except guard.UploadRejected as exc:
        raise HTTPException(status_code=exc.status, detail=exc.message) from exc


BUSY_RETRY_AFTER_SECONDS = 5


@contextmanager
def _admitted(client: str) -> Iterator[None]:
    """Hold a concurrency slot and spend one unit of `client`'s quota for the block.
    Called only after the upload has validated. Refusals are 429 (rate/quota, with
    Retry-After), 503 (all slots busy, or the counters are unreachable: fail closed).
    A request that is admitted and then fails at the provider is *not* refunded."""
    try:
        with quota.gate().slot():
            try:
                decision = quota.admit(f"api:{client}")
            except quota.QuotaUnavailable as exc:
                raise HTTPException(
                    status_code=503, detail="Rate limiting is unavailable, so the request was not accepted."
                ) from exc
            if not decision.allowed:
                raise HTTPException(
                    status_code=429, detail=decision.message, headers={"Retry-After": str(decision.retry_after)}
                )
            yield
    except quota.Busy as exc:
        raise HTTPException(
            status_code=503,
            detail="The service is busy. Try again shortly.",
            headers={"Retry-After": str(BUSY_RETRY_AFTER_SECONDS)},
        ) from exc


def _service_unavailable(exc: IdentifyError) -> HTTPException:
    """IdentifyError text is fixed wording (src/identify.py), never provider output.
    Misconfiguration is the operator's business, so callers get a generic line."""
    logger.warning("identification failed: %s", exc)
    if isinstance(exc, IdentifyConfigError):
        return HTTPException(status_code=503, detail="The service is not configured correctly.")
    return HTTPException(status_code=503, detail=str(exc))


@public.get("/health")
def health() -> dict:
    return {"status": "ok"}


@protected.post("/identify", response_model=IdentifyResult)
def identify_endpoint(photo: UploadFile = File(...), client: str = Depends(current_client)) -> IdentifyResult:
    """Identify a single scanned flower photo. Runs synchronously (not `async def`)
    because identify() blocks on CPU inference and Gemini network calls -- FastAPI
    offloads sync route handlers to a thread pool automatically, which is the
    correct way to serve blocking work without stalling the event loop."""
    image = _read_image(photo)
    with _admitted(client):
        try:
            result = identify(image)
        except IdentifyError as exc:
            raise _service_unavailable(exc) from exc
    log_scan(result)
    return result


@protected.post("/identify-lot", response_model=LotResult)
def identify_lot_endpoint(
    photos: list[UploadFile] = File(...), client: str = Depends(current_client)
) -> LotResult:
    """Identify a lot of up to LOT_MAX_PHOTOS photos as one consensus result."""
    if not photos:
        raise HTTPException(status_code=400, detail="No photos provided.")
    if len(photos) > LOT_MAX_PHOTOS:
        raise HTTPException(
            status_code=400, detail=f"A lot can have at most {LOT_MAX_PHOTOS} photos (got {len(photos)})."
        )

    images = _read_lot(photos)
    with _admitted(client):
        try:
            result = identify_lot(images)
        except IdentifyError as exc:
            raise _service_unavailable(exc) from exc
    log_lot_scan(result)
    return result


@protected.post("/explain")
def explain_endpoint(
    photo: UploadFile = File(...),
    species: str = Form(..., max_length=SPECIES_FIELD_MAX_LENGTH),
    client: str = Depends(current_client),
) -> Response:
    """Returns a PNG of `photo` with a heatmap overlay showing which regions most
    drove its match to `species` (see src/interpretability.py) -- an on-demand
    trust/debug aid, not part of the core /identify result."""
    image = _read_image(photo)
    if not is_known_species(species):  # a validation failure: refuse it before it can cost quota
        raise HTTPException(status_code=400, detail="That species isn't in BloomLens's curated species list.")
    with _admitted(client):
        try:
            overlay = explain_image(image, species)
        except ExplainError as exc:
            raise HTTPException(
                status_code=400, detail="That species isn't in BloomLens's curated species list."
            ) from exc

    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@protected.get("/inventory", response_model=list[InventoryEntry])
def inventory_endpoint(limit: int = Query(100, ge=1, le=INVENTORY_MAX_LIMIT)) -> list[InventoryEntry]:
    """Most recent logged scans (newest first) -- see src/inventory.py."""
    return list_recent(limit)


@protected.get("/inventory/species-counts")
def inventory_species_counts_endpoint() -> dict[str, int]:
    """Running count of logged scans per species."""
    return species_counts()


async def _validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """FastAPI's default 422 echoes the offending input back; say where and why, not what."""
    detail = [{"loc": list(e.get("loc", ())), "msg": e.get("msg", "invalid")} for e in exc.errors()]
    return JSONResponse({"detail": detail}, status_code=422)


def create_app(*, production: bool | None = None) -> FastAPI:
    """`production` (default: BLOOMLENS_ENV=production) turns off the interactive docs
    and the OpenAPI document, which describe every route to whoever asks."""
    production = is_production() if production is None else production
    app = FastAPI(
        title="BloomLens API",
        description="Scan a flower (or a lot of them) and get species, quality, and a simulated price.",
        version="0.1.0",
        docs_url=None if production else "/docs",
        redoc_url=None,
        openapi_url=None if production else "/openapi.json",
    )
    app.include_router(public)
    app.include_router(protected)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_middleware(SecurityMiddleware, docs_enabled=not production)
    return app


app = create_app()
