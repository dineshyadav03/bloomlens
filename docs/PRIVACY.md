# Privacy: what leaves BloomLens, what is kept, and what Google does with it

BloomLens sends each photo you scan to Google's Gemini API. That is the one
place data leaves the app, and this page states exactly what goes, what never
does, what BloomLens itself keeps, and which parts of Google's handling are
outside this project's control. Anything here that is an *assumption* about a
third party is labelled as one.

This is engineering documentation, not legal advice. If you operate BloomLens
for other people, you are the one who has to decide whether these flows are
acceptable for them.

## What leaves the app

Per scan, one request to the Gemini API carries:

| Sent | Detail |
|---|---|
| The photo | A JPEG **re-encoded from the decoded pixels** (`src/guard.py`, then `src/identify.py::_image_to_jpeg_bytes`). Lot mode sends up to 10. |
| The system prompt | Fixed text in `src/identify.py`. |
| Candidate species text | Common name, scientific name and a description for the top-3 retrieval matches, from `data/species_reference.json`. |
| Tool results | The agent calls three local tools; their outputs go back to the model: botanical text for a species, the quality grade/note the model itself wrote, and a *simulated* price row. |
| Ordinary request metadata | The API key you configured, and the network address of the machine running BloomLens (that is how any HTTP call works; Google sees the *server's* address, never your users'). |

## What never leaves the app

- **The original upload bytes.** The image is decoded and rebuilt from pixels alone.
- **EXIF, GPS position, camera make/model/serial, timestamps, XMP and ICC profiles.**
  These are removed before anything else happens, and the camera's orientation is
  applied first so the photo is not sent sideways. Enforced by tests:
  `tests/unit/test_guard.py::TestMetadataIsStripped` builds a JPEG carrying a GPS block and a
  camera serial, and asserts none of it is present in the bytes that would be sent to Gemini.
- **File names** and anything else about the upload's origin.
- **The identity of whoever is using BloomLens**: no account, key label, session id or IP
  address is put in a Gemini request.
- **The inventory database** (see below) and any keys other than the Gemini key.
- **Anything to a tracing service.** LangChain/LangSmith tracing would ship prompts and
  images to LangSmith; `src/privacy.py` forces it off at import time, whatever the
  environment says, and a test starts a subprocess with tracing switched on to prove it.

## What BloomLens itself keeps

- **Photos: nothing.** They live in memory for the length of one request. They are never
  written to disk, never logged and never put in a database.
- **The inventory log** (`data/inventory.db`): per scan, a UTC timestamp, single/lot, the
  species, scientific name, confidence tier, quality grade, price fields, photo count and lot
  agreement. No image, no free text from the model, no user identifier.
- **Logs** record *that* an upload was rejected and a short reason code, never the file,
  its name or its contents.
- Planned (milestone M14, **not built yet**): request telemetry with a closed schema.
  The rule it will be built to is fixed now: it may store timings, counts, versions and a
  failure category, and it must **never** store images, prompts, IP addresses, API keys,
  raw provider error messages or model responses. Retention (default 30 days) and a purge
  job come with it.

## What Google does with it: assumptions, tier-dependent

> **Source:** Google's *Gemini API Additional Terms of Service*
> (<https://ai.google.dev/gemini-api/terms>), read on **2026-09-21**; the page was marked
> "Last updated April 28, 2026". What follows is a **paraphrase**, taken through a
> summarising tool, not a quotation. Terms change: read the current page before relying on
> any of it.

- **Unpaid ("free tier") services.** Content submitted may be used to provide, improve and
  develop Google products and may be reviewed by human reviewers. Do not send anything
  through the free tier that you would not want a person at Google to see.
- **Paid services.** Content is not used to improve Google's products; Google logs
  prompts/outputs for a limited period for policy and legal purposes. The retention period
  was not stated precisely in the text read — **treat it as unknown**, not as zero.
- **Users in the EEA, the UK or Switzerland.** The terms require the *paid* tier when your
  API client is made available to users located there. A public demo running on a free key
  therefore should not be offered to those users.

What this means for BloomLens today:

| Deployment | Assessment |
|---|---|
| You, alone, with your own photos, free key | Fine if you accept the free-tier terms above. |
| A public demo (e.g. a Hugging Face Space) on a free key | Visitors' photos may be used to improve Google's products and seen by reviewers, and users in the EEA/UK/CH are not permitted to use it. Say so on the page (the app shows a notice) or use a paid key. |
| Anything handling photos you don't own or that show people/places | Use a paid key, get consent, and read the terms yourself first. |

The in-app notice (`app.py`) says photos are sent to Google's Gemini API; the free-tier
caveat is added to it.

## Model output is untrusted text

The species/quality text comes from a language model that has looked at an image an
outsider chose. Text written *in* a photo can try to steer it. BloomLens therefore:

- tells the model in the system prompt that text inside a photo is data, not instructions;
- validates the shape of the answer (a grade must be exactly `A`, `B` or `C`, every field
  has a length cap) and treats a violation as a failed parse;
- displays model text **literally** in the Streamlit UI (`st.text`, so no markdown or HTML
  is interpreted) rather than trying to sanitise it with patterns; and
- returns it from the API as plain JSON strings. **Anything that consumes the API should
  treat those strings as untrusted input** and escape them for wherever they are shown.

## Deleting data

There is no per-user data to delete. The inventory log is a local file: delete
`data/inventory.db` (or the Docker volume it lives in). Anything Google retained is governed
by their terms above and is not reachable from here.
