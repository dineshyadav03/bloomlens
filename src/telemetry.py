"""Minimal, privacy-safe per-scan telemetry, and aggregate-only reporting.

What is recorded, per scan, in a CLOSED set of columns (src/db.py `scan_metrics`):
a UTC timestamp, single/lot, photo count, ok/error, a failure *category* (an enum, never
text), stage timings, agent attempts, model turns, tool-call count, summed token counts,
the versions in play (BioCLIP revision, requested Gemini model, the model name the
provider echoed), the price version and the estimated cost, and a coarse hardware string.

What is NEVER recorded -- no column can hold it, and a test writes canary strings through
every path to prove none reaches the file: images, prompts, model answers or tool
arguments, exception messages (provider or otherwise), IP addresses, API keys, caller
identities (key labels, session ids). Token, turn and tool columns are nullable: they
exist only when the provider reported them (docs/telemetry_probe.md), and the aggregates
say how many scans actually had a value.

Reporting is aggregate-only: counts, failure rates, p50/p95, means. No row is ever
returned. Percentiles are withheld until there are MIN_SAMPLES_FOR_PERCENTILES scans, and
p99 is not offered at all: at this scale it would be one or two samples.

Recording never breaks a scan: any failure to record is logged by exception *type* and
swallowed.
"""

import contextvars
import logging
import os
import platform
import re
import statistics
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Literal, get_args

from src import db, llm_cost, retention
from src.versions import BIOCLIP_REVISION, GEMINI_MODEL

logger = logging.getLogger("bloomlens.telemetry")

FailureCategory = Literal["rate_limit", "server_5xx", "timeout", "parse_error", "auth", "empty_index", "other"]
FAILURE_CATEGORIES: tuple[str, ...] = get_args(FailureCategory)

MIN_SAMPLES_FOR_PERCENTILES = 20
_MAX_COUNT = 10**9  # a sanity bound on any number taken from a provider response
_MODEL_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def _platform_summary() -> str:
    return f"{platform.system()} {platform.machine()}, {os.cpu_count()} CPUs"


def _count(value) -> int | None:
    """A provider-reported number, or None if it is anything else."""
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_COUNT:
        return None
    return value


@dataclass
class ScanRecorder:
    mode: Literal["single", "lot"]
    photo_count: int
    cold_start: bool
    started: float = field(default_factory=time.perf_counter)
    stage_seconds: dict[str, float] = field(default_factory=dict)
    attempts: int = 0
    model_turns: int | None = None
    tool_calls: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    model_reported: str | None = None


_current: contextvars.ContextVar[ScanRecorder | None] = contextvars.ContextVar("bloomlens_scan", default=None)


@contextmanager
def timed(stage: Literal["embed", "search", "agent"]) -> Iterator[None]:
    """Add the block's duration to the active scan's `stage` (no-op outside a scan)."""
    recorder = _current.get()
    started = time.perf_counter()
    try:
        yield
    finally:
        if recorder is not None:
            recorder.stage_seconds[stage] = recorder.stage_seconds.get(stage, 0.0) + time.perf_counter() - started


def note_attempt() -> None:
    """One agent invocation (a first try or a retry). No-op outside a scan."""
    recorder = _current.get()
    if recorder is not None:
        recorder.attempts += 1


def note_agent_result(result) -> None:
    """Fold one agent run's token usage, model turns and tool calls into the active scan.
    Reads only numbers (and the provider's model *name*, if it is a plain identifier)
    from the messages -- never their content. Anything unexpected is ignored."""
    recorder = _current.get()
    if recorder is None:
        return
    try:
        for message in result["messages"]:
            usage = getattr(message, "usage_metadata", None)
            if isinstance(usage, dict):
                tokens_in, tokens_out = _count(usage.get("input_tokens")), _count(usage.get("output_tokens"))
                if tokens_in is not None and tokens_out is not None:
                    recorder.input_tokens = (recorder.input_tokens or 0) + tokens_in
                    recorder.output_tokens = (recorder.output_tokens or 0) + tokens_out
                    recorder.model_turns = (recorder.model_turns or 0) + 1
                metadata = getattr(message, "response_metadata", None)
                reported = metadata.get("model_name") if isinstance(metadata, dict) else None
                if isinstance(reported, str) and _MODEL_NAME_RE.fullmatch(reported):
                    recorder.model_reported = reported
            calls = getattr(message, "tool_calls", None)
            if isinstance(calls, list):
                recorder.tool_calls = (recorder.tool_calls or 0) + len(calls)
    except (KeyError, TypeError, AttributeError):
        # Not an agent result. Telemetry must never turn a good scan into a failed one.
        logger.warning("telemetry: ignoring an agent result of an unexpected shape")


def category_of(exc: BaseException) -> str:
    """A failure's category: what the pipeline attached to it, else 'other'. Never text."""
    category = getattr(exc, "category", None)
    return category if category in FAILURE_CATEGORIES else "other"


@contextmanager
def scan(mode: Literal["single", "lot"], photo_count: int, *, cold_start: bool) -> Iterator[ScanRecorder]:
    """Bracket one scan. On exit -- success or failure -- exactly one row is recorded
    (except for KeyboardInterrupt and friends, which are not scan outcomes)."""
    recorder = ScanRecorder(mode, photo_count, cold_start)
    token = _current.set(recorder)
    failure: str | None = None
    skip = False
    try:
        yield recorder
    except Exception as exc:
        failure = category_of(exc)
        raise
    except BaseException:
        skip = True
        raise
    finally:
        _current.reset(token)
        if not skip:
            record(recorder, failure)


def _ms(recorder: ScanRecorder, stage: str) -> int | None:
    seconds = recorder.stage_seconds.get(stage)
    return None if seconds is None else round(seconds * 1000)


def _ensure_pricing(conn) -> None:
    """Copy the code's price versions into the immutable table, and refuse if one was
    edited in place (a price change is a new version)."""
    for v in llm_cost.PRICE_VERSIONS:
        values = (
            v.version,
            v.model,
            v.effective_from,
            v.input_usd_per_mtok,
            v.output_usd_per_mtok,
            v.source_url,
            v.retrieved_at,
        )
        conn.execute("INSERT OR IGNORE INTO pricing_versions VALUES (?, ?, ?, ?, ?, ?, ?)", values)
        stored = conn.execute("SELECT * FROM pricing_versions WHERE version = ?", (v.version,)).fetchone()
        if tuple(stored) != values:
            raise RuntimeError(f"pricing version {v.version} differs from the stored one; add a new version")


def record(recorder: ScanRecorder, failure_category: str | None, *, now: datetime | None = None) -> None:
    """Write one row. Never raises."""
    try:
        now = now or datetime.now(UTC)
        version = llm_cost.version_in_force(GEMINI_MODEL, now.date())
        cost = None
        if version is not None and recorder.input_tokens is not None and recorder.output_tokens is not None:
            cost = llm_cost.estimate_cost_usd(version, recorder.input_tokens, recorder.output_tokens)
        row = (
            now.isoformat(),
            recorder.mode,
            recorder.photo_count,
            "ok" if failure_category is None else "error",
            failure_category,
            int(recorder.cold_start),
            _ms(recorder, "embed"),
            _ms(recorder, "search"),
            _ms(recorder, "agent"),
            round((time.perf_counter() - recorder.started) * 1000),
            recorder.attempts or None,
            recorder.model_turns,
            recorder.tool_calls,
            recorder.input_tokens,
            recorder.output_tokens,
            BIOCLIP_REVISION,
            GEMINI_MODEL,
            recorder.model_reported,
            version.version if cost is not None else None,  # the price a cost was estimated with
            cost,
            _platform_summary(),
        )
        with closing(db.connect()) as conn, db.transaction(conn):
            _ensure_pricing(conn)
            conn.execute(
                "INSERT INTO scan_metrics (recorded_at, mode, photo_count, status, failure_category, cold_start, "
                "embed_ms, search_ms, agent_ms, total_ms, attempts, model_turns, tool_calls, input_tokens, "
                "output_tokens, bioclip_revision, gemini_model, model_reported, pricing_version, est_cost_usd, "
                "platform) VALUES (" + ", ".join("?" * len(row)) + ")",
                row,
            )
            retention.purge_if_due(conn, now)
    except Exception as exc:  # noqa: BLE001 -- telemetry must never break a scan
        logger.warning("telemetry: could not record a scan (%s)", type(exc).__name__)


# --- aggregate-only reporting -------------------------------------------------------------


def _percentiles(values: list[int]) -> dict:
    n = len(values)
    if n < MIN_SAMPLES_FOR_PERCENTILES:
        note = f"withheld: fewer than {MIN_SAMPLES_FOR_PERCENTILES} scans"
        return {"n": n, "p50_ms": None, "p95_ms": None, "note": note}
    cuts = statistics.quantiles(values, n=100, method="inclusive")  # linear interpolation, like numpy's default
    return {"n": n, "p50_ms": round(cuts[49]), "p95_ms": round(cuts[94])}


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 3) if values else None


def summary(*, now: datetime | None = None, window_days: int | None = None) -> dict:
    """Aggregates over the retention window. Contains no row, timestamp or identifier."""
    now = now or datetime.now(UTC)
    window_days = window_days or retention.settings().metrics_days
    since = (now - timedelta(days=window_days)).isoformat()
    with closing(db.connect(busy_timeout=5)) as conn:
        rows = conn.execute(
            "SELECT status, failure_category, cold_start, embed_ms, search_ms, agent_ms, total_ms, attempts, "
            "input_tokens, output_tokens, est_cost_usd, pricing_version, bioclip_revision, gemini_model, "
            "model_reported, platform FROM scan_metrics WHERE recorded_at >= ?",
            (since,),
        ).fetchall()

    total = len(rows)
    errors = [r for r in rows if r[0] == "error"]
    ok = [r for r in rows if r[0] == "ok"]
    warm = [r for r in ok if r[2] == 0]
    cold = [r for r in ok if r[2] == 1]
    by_category = {c: sum(1 for r in errors if r[1] == c) for c in FAILURE_CATEGORIES}
    attempted = [r[7] - 1 for r in rows if r[7] is not None]
    with_tokens = [r for r in rows if r[8] is not None and r[9] is not None]
    priced = [r[10] for r in rows if r[10] is not None]

    def stage(index: int) -> dict:
        return _percentiles([r[index] for r in warm if r[index] is not None])

    return {
        "window_days": window_days,
        "scans": {"total": total, "ok": len(ok), "error": len(errors)},
        "failure_rate": round(len(errors) / total, 4) if total else None,
        "failures_by_category": by_category,
        "latency_ms": {
            "warm": _percentiles([r[6] for r in warm]),
            "cold": _percentiles([r[6] for r in cold]),
            "warm_by_stage": {"embed": stage(3), "search": stage(4), "agent": stage(5)},
            "note": "successful scans only; p50/p95 only (p99 is not reported); 'cold' includes loading the model",
        },
        "retries_per_scan": {"n": len(attempted), "mean": _mean(attempted)},
        "tokens": {
            "scans_with_counts": len(with_tokens),
            "scans_without_counts": total - len(with_tokens),
            "mean_input": _mean([r[8] for r in with_tokens]),
            "mean_output": _mean([r[9] for r in with_tokens]),
        },
        "estimated_cost_usd": {
            "total": round(sum(priced), 6) if priced else None,
            "scans_priced": len(priced),
            "scans_unpriced": total - len(priced),
            "pricing_versions": sorted({r[11] for r in rows if r[11] is not None}),
            "meaning": llm_cost.COST_LABEL,
        },
        "versions": {
            "bioclip_revision": sorted({r[12] for r in rows}),
            "gemini_model_requested": sorted({r[13] for r in rows}),
            "gemini_model_reported": sorted({r[14] for r in rows if r[14] is not None}),
        },
        "platforms": sorted({r[15] for r in rows}),
    }
