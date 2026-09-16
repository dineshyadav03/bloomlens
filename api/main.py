"""BloomLens FastAPI endpoint -- a thin translation layer over src/identify.py,
not a second implementation. Both this and app.py (Streamlit) call the exact
same identify()/identify_lot() and render the exact same IdentifyResult/
LotResult Pydantic models; nothing about the pipeline lives here.

Run: uvicorn api.main:app --reload   (see .claude/launch.json for a preset)
Then browse http://localhost:8000/docs for interactive Swagger UI.
"""

import io

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from src.identify import (
    LOT_MAX_PHOTOS,
    IdentifyError,
    IdentifyResult,
    LotResult,
    identify,
    identify_lot,
)
from src.interpretability import ExplainError, explain_image
from src.inventory import InventoryEntry, list_recent, log_lot_scan, log_scan, species_counts

app = FastAPI(
    title="BloomLens API",
    description="Scan a flower (or a lot of them) and get species, quality, and a simulated price.",
    version="0.1.0",
)


def _read_image(upload: UploadFile) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(upload.file.read()))
        image.load()  # force full decode now, so a truncated/corrupt file fails here (400) not mid-pipeline (500)
        return image
    except Exception as exc:  # noqa: BLE001 -- any unreadable upload is a client error, not a service error
        raise HTTPException(status_code=400, detail=f"'{upload.filename}' isn't a readable image: {exc}") from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/identify", response_model=IdentifyResult)
def identify_endpoint(photo: UploadFile = File(...)) -> IdentifyResult:
    """Identify a single scanned flower photo. Runs synchronously (not `async def`)
    because identify() blocks on CPU inference and Gemini network calls -- FastAPI
    offloads sync route handlers to a thread pool automatically, which is the
    correct way to serve blocking work without stalling the event loop."""
    image = _read_image(photo)
    try:
        result = identify(image)
    except IdentifyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    log_scan(result)
    return result


@app.post("/identify-lot", response_model=LotResult)
def identify_lot_endpoint(photos: list[UploadFile] = File(...)) -> LotResult:
    """Identify a lot of up to LOT_MAX_PHOTOS photos as one consensus result."""
    if not photos:
        raise HTTPException(status_code=400, detail="No photos provided.")
    if len(photos) > LOT_MAX_PHOTOS:
        raise HTTPException(
            status_code=400, detail=f"A lot can have at most {LOT_MAX_PHOTOS} photos (got {len(photos)})."
        )

    images = [_read_image(p) for p in photos]
    try:
        result = identify_lot(images)
    except IdentifyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    log_lot_scan(result)
    return result


@app.post("/explain")
def explain_endpoint(photo: UploadFile = File(...), species: str = Form(...)) -> Response:
    """Returns a PNG of `photo` with a heatmap overlay showing which regions most
    drove its match to `species` (see src/interpretability.py) -- an on-demand
    trust/debug aid, not part of the core /identify result."""
    image = _read_image(photo)
    try:
        overlay = explain_image(image, species)
    except ExplainError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/inventory", response_model=list[InventoryEntry])
def inventory_endpoint(limit: int = 100) -> list[InventoryEntry]:
    """Most recent logged scans (newest first) -- see src/inventory.py."""
    return list_recent(limit)


@app.get("/inventory/species-counts")
def inventory_species_counts_endpoint() -> dict[str, int]:
    """Running count of logged scans per species."""
    return species_counts()
