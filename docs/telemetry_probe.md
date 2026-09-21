# Telemetry probe: what a real agent run actually returns

Telemetry columns for token counts, the provider's model version and tool-call counts were
specified as **nullable until a real trace proved they exist**. This is that trace.

**How:** `scripts/probe_agent_trace.py` made one real `identify()`-equivalent call on a committed
test photo (`eval/test_images/Amaryllis_5455.jpg`, an Oxford 102 Flowers sample) through the same
agent the app uses, on **2026-09-21**, against `gemini-3.1-flash-lite` on the free tier. It prints
*structure only*: key names, numeric values, an allow-list of non-content strings (model name,
finish reason), tool names and argument **names**. No prompt, image, message content or model
answer was printed or saved, and tracing was off (`src/privacy.py`).

## What one scan returned

One agent run = **8 messages**: the user message, then four model turns interleaved with three
tool results.

| Question | Finding | Consequence for the schema |
|---|---|---|
| Token counts? | **Yes**, on every `AIMessage`: `usage_metadata.input_tokens`, `output_tokens`, `total_tokens`. This run: 1854 + 1931 + 2025 + 2098 = **7 908 in**, 18 + 48 + 24 + 123 = **213 out**, 8 121 total. | `input_tokens`, `output_tokens` recorded as the **sum over the run's model turns**. Still nullable: an error before any model turn has none. |
| Cached tokens? | `usage_metadata.input_token_details.cache_read` = 0 (present). | Not stored (always 0 here; a cache-aware price would need it — noted, not built). |
| Reasoning/"thinking" tokens? | **Not reported** in this run (no `output_token_details`). | Not stored; cost is estimated from input/output tokens only. If they exist but are folded into `output_tokens`, the estimate is still right for billing; if they are separate and unreported, it is an underestimate. **Unknown.** |
| Provider-reported model version? | **No.** `response_metadata.model_name` is `gemini-3.1-flash-lite`, identical to the name the app requested; there is no `model_version` and no dated/numbered revision. | Recorded as `model_reported` (what the provider echoed). **Nothing more precise exists**, so a silent server-side model change would be invisible in telemetry. Stated as a limit. |
| Tool-call counts? | **Yes.** `tool_calls` on the `AIMessage`s: `lookup_taxonomy`, `assess_quality`, `check_price`, one each = **3 tool calls, 4 model turns**. | `model_turns` and `tool_calls` are recorded as counts (no names, no arguments). |
| Finish reason | `STOP` on every turn. | Not stored. |
| Wall time | 21.2 s for the agent step alone (network included). | Real timing data is captured per scan (below) rather than assumed. |

## Things the probe deliberately did not do

- One run is a *shape* check, not a distribution: another prompt, a lot (several images) or an
  error path may return different fields (e.g. `usage_metadata` absent). Every token/turn/tool
  column stays nullable, and aggregates report how many rows actually had a value.
- It does not validate cost: a token count times a list price is an estimate (see the pricing table
  and the "estimated list-price equivalent" wording), never a bill.
