# inferopt

Find out where your Anthropic API spend is being wasted, with proof.

A local logging proxy + analyzer. It sits between your code and
`api.anthropic.com`, records per-request cost metadata into a **local**
SQLite file, and turns it into ranked, dollar-quantified findings across
three levers:

1. **Prompt caching** - detects call sites with big stable prefixes that are
   never cached, and caches that are *silently broken* (it localizes the
   exact byte where your prefix diverges, e.g. a timestamp in the system
   prompt). Cache reads bill at 10% of input price. Zero quality risk.
2. **Batch API** - flags non-streamed call sites (cron jobs, pipelines,
   bulk work) that would get identical responses at a flat 50% off.
   Zero quality risk if nothing is waiting on the response.
3. **Tier / effort arbitrage** - what-if table for cheaper models
   (Opus -> Sonnet is 2.5x cheaper, -> Haiku 5x) and lower
   `output_config.effort`, plus a `replay` command that re-runs real logged
   requests on the cheaper config and (optionally) LLM-judges the outputs
   side by side. This lever needs validation; the tool makes that cheap.

## Quickstart (3 steps)

```bash
git clone <this repo> && cd inferopt        # or just copy the folder
python3 -m venv .venv && .venv/bin/pip install -e .

# terminal 1: start the proxy
.venv/bin/inferopt proxy

# terminal 2: run your normal workload through it
export ANTHROPIC_BASE_URL=http://127.0.0.1:8484
python your_script.py   # any Anthropic SDK, any language - just the env var

# then:
.venv/bin/inferopt report
```

Works with every official Anthropic SDK (they all honor
`ANTHROPIC_BASE_URL`) and with Claude Code when authenticated with an API
key. Streaming is fully supported.

## AWS Bedrock

The proxy has a second rail for Bedrock. Because Bedrock requests are
SigV4-signed over the Host header, the proxy re-signs each request with
your local AWS credential chain (env vars / profile / SSO) before
forwarding - so your normal `aws` login must work in the shell running
the proxy.

```bash
# terminal 2 (instead of / alongside ANTHROPIC_BASE_URL):
export AWS_ENDPOINT_URL_BEDROCK_RUNTIME=http://127.0.0.1:8484
python your_bedrock_script.py     # boto3 - zero code changes
```

For the anthropic SDK's Bedrock client, pass the URL explicitly:
`AnthropicBedrock(base_url="http://127.0.0.1:8484", ...)`.

Supported and instrumented: `invoke_model`, `converse`, and both streaming
variants (AWS eventstream parsing included). Region comes from
`INFEROPT_BEDROCK_REGION` / `AWS_REGION` / `AWS_DEFAULT_REGION`
(default us-east-1). `invoke` and `converse` calls with the same prompt
structure map to the same call site, so findings are rail-agnostic.
Costs use Anthropic first-party rates - Bedrock list prices can differ
slightly (regional endpoints carry a 10% premium).

## Commands

| Command | What it does |
|---|---|
| `inferopt proxy [--port 8484]` | transparent logging proxy for api.anthropic.com |
| `inferopt report [--days 30] [--json]` | ranked findings + call-site table + tier what-if |
| `inferopt callsites` | list observed call sites (to pick one for replay) |
| `inferopt replay --callsite <fp> --model claude-sonnet-5 [--effort low] [--judge]` | re-run logged requests on a cheaper config, write side-by-side markdown, optional LLM judge. Prints a cost estimate and asks before spending. |
| `inferopt purge` | delete the local database |

## Privacy

- Everything is stored locally in `~/.inferopt/inferopt.db`. Nothing is sent
  anywhere except your original API call, unmodified, to Anthropic.
- Request bodies are stored so `replay` and invalidator localization work.
  Set `INFEROPT_STORE_BODIES=0` to keep metadata + prefixes only
  (disables `replay`).
- Your API key passes through the proxy but is never stored.

## How the math works

- Prices verified against the official pricing docs (2026-08-27):
  cache read = 0.1x input, 5-min cache write = 1.25x, batch = 50% off both
  input and output. Sonnet 5 is $2/$10, Haiku 4.5 $1/$5, Opus 5 $5/$25.
- A "call site" = requests sharing the same tool set + normalized system
  prompt (timestamps/UUIDs/numbers masked). All findings are per call site.
- Monthly figures are extrapolated from the observed window and labeled
  with the extrapolation factor. Short windows = directional only.
- Cache-break localization: the serialized prefix (tools -> system, the
  API's render order) is diffed byte-by-byte between consecutive requests;
  the first differing byte is the invalidator.

## Replay on Bedrock

`inferopt replay` works on both rails. For Bedrock rows it re-signs with
the local AWS credential chain (same as the proxy - run it in a
creds-loaded shell, e.g. `AWS_PROFILE=my-aws-profile`). Model overrides must be
the full cross-region inference-profile id (`us.anthropic.claude-...`),
not the bare on-demand id. The judge defaults to the baseline row's own
model on Bedrock (the incumbent sets the quality bar) - override with
`--judge-model`.

**Body-storage policy that makes validation work:** run synthetic-input
sessions with bodies ON (default) so replay/judge have material, and
client-data sessions with `INFEROPT_STORE_BODIES=0` (metadata-only).
Validate on synthetic, observe production metadata-only.

## Structured outputs / tool calling

Tool-calling responses (LangChain `.with_structured_output()`, function
calling, etc.) contain no text blocks - the payload is in
`tool_use.input`. inferopt serializes those as
`[tool_use <name>] <sorted-json>` on both rails (stream and non-stream),
so structured call sites are fully loggable and replayable; the judge is
instructed to grade tool-call field values on substance. The judge is
blind (not told which side is the incumbent) and A/B order alternates per
pair to cancel position and verbosity bias.

## Profiling for free (v0.9)

Free models price at $0, which makes every savings figure $0 and the
report useless as evidence. But the *waste* is real and is borne by
whoever runs the same traffic on a paid model. So:

```bash
inferopt report --price-as claude-sonnet-5
```

re-prices the observed token counts at that model's rates. The header
states plainly that the traffic actually ran elsewhere. This makes
zero-budget profiling produce quotable numbers without pretending you
measured spend you didn't.

## OpenAI-compatible rail (v0.8)

Third rail for any OpenAI-compatible endpoint - OpenRouter, Together,
vLLM, LiteLLM, local servers. Auth is a bearer header, so requests
forward verbatim with nothing to re-sign:

```bash
export INFEROPT_OPENAI_UPSTREAM=https://openrouter.ai/api   # default
# point your client's base_url at:  http://127.0.0.1:8484/v1
```

Pricing for ~400 models comes live from OpenRouter's public catalog
(cached 24h in `~/.inferopt/`), including per-model cache read/write
rates, so cost math works without hardcoding anything. Free models
(`:free`) price at zero, which makes zero-budget profiling possible.

Rail-aware analysis: OpenAI-style providers cache **automatically** on a
byte-stable prefix (no `cache_control` field), so the advice changes
accordingly, and the Batch API finding is suppressed on this rail since
there is no equivalent 50%-off tier. Tier what-if is limited to
first-party/Bedrock ids - suggesting `claude-haiku-4-5` to a caller
using `anthropic/claude-sonnet-5` would emit a broken model string, and
cross-format tier mapping isn't wired up yet.

**New finding type - `unstable-prefix`:** a call site with a big
reusable prefix that nonetheless differs between calls, so caching can
never engage. Previously such sites fell through every branch (the
enable-caching check needs a *stable* prefix; broken-cache needs caching
already active). This is the classic coding-agent failure: a timestamp
or session id injected near the front of the system prompt re-bills
thousands of tokens every single turn. The report localizes the exact
diverging byte.

## The validation ledger (v0.7)

Replay evidence is recorded in a local `validations` table and fed back
into the report. A replay with any behavioral mismatch is auto-recorded
NO-GO; otherwise it records the judge counts and asks you to make the
call after spot-checking:

```bash
inferopt decide --callsite <fp> --model <model> --go     # or --no-go
inferopt ledger                                          # what's decided
```

The report honors it: the COMBINED section features a **validated GO**
tier when one exists, never features a validated NO-GO (it lists those as
excluded), and the tier what-if table is annotated `[VALIDATED GO]` /
`[VALIDATED NO-GO]` with the judge/mismatch evidence inline.

Three totals, deliberately: **zero-quality-risk** (arithmetic levers
only), **defensible today** (zero-risk + validated tier swaps - this is
the number you quote), and **potential if every untested swap passed**
(marked do-not-quote). Without this, the report defaults to whichever
tier is cheapest, which is exactly the model most likely to fail
validation.

## Savings compose, they do not add (v0.6)

Per-lever figures in the FINDINGS list are **standalone** - what that one
lever saves if applied alone. Batching a cheaper model saves 50% of the
*already reduced* bill, so summing levers overstates savings badly (on one
real call site: $340 summed vs $247 actual, a 37% overstatement). The
report's COMBINED section composes levers multiplicatively and splits the
total two ways: **zero-quality-risk** (caching + batch, arithmetic only)
and **if validated tier swaps also applied** (needs replay evidence).
Quote the first number to a customer; quote the second only with replay
artifacts attached.

## Behavioral checks (v0.5)

Before the LLM judge runs, replay does a deterministic tool-usage
comparison per pair: a candidate that narrates intent ("I'll research...")
instead of invoking the tools the baseline invoked is flagged as a
BEHAVIORAL MISMATCH - a functional regression for agentic loops that text
judges reliably miss. Mismatches are counted separately and should be
treated as candidate_worse regardless of judge verdicts. The judge is also
explicitly told never to assume content beyond a truncation point.

## Known limitations (v0.4)

- Rails: Anthropic API + Bedrock. Vertex AI not yet supported.
- Replay calls go direct (bypass the proxy), so replay spend isn't logged
  in the db - the printed estimate is your record.
- Rows logged before v0.4 with NULL response_text (tool-calling responses)
  stay unreplayable; re-run those workloads to capture them properly.
- `--effort` override not yet supported for converse-logged rows.
- Bedrock re-signing validated against real AWS (us-east-1).
- Claude Code on OAuth (subscription) auth: not the target; instrument
  API-key workloads.
- `usage` fields are read from responses; requests that fail mid-stream may
  undercount output tokens.
- The judge sees A/B in a fixed order (baseline first) - spot-check pairs
  yourself before trusting a verdict enough to change prod.
