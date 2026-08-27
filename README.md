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

## Known limitations (v0.3)

- Rails: Anthropic API + Bedrock. Vertex AI not yet supported.
- `--effort` override not yet supported for converse-logged rows.
- Bedrock re-signing validated against real AWS (us-east-1).
- Claude Code on OAuth (subscription) auth: not the target; instrument
  API-key workloads.
- `usage` fields are read from responses; requests that fail mid-stream may
  undercount output tokens.
- The judge sees A/B in a fixed order (baseline first) - spot-check pairs
  yourself before trusting a verdict enough to change prod.
