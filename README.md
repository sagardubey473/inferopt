# inferopt

**Find where your LLM spend is being wasted, with proof, without sending anyone your data.**

Point it at logs you already have. It groups your traffic by call site,
finds the waste, and tells you exactly what to change and what it is
worth per month. Everything runs locally.

## Try it in 10 seconds

No install, no account, no API key, no data of your own. A synthetic week
of traffic for a fictional support product ships inside the package:

```bash
uvx inferopt demo
```

(Or `pipx run inferopt demo`, or `pip install inferopt && inferopt demo`.)
The analyze path has **zero dependencies** on purpose, so that command
finishes in about a second.

Abbreviated output:

```
requests: 630 ok / 630 total   observed window: 6.57 days
spend (observed): $16.92   extrapolated monthly: $77.24 (x4.6)

COMBINED SAVINGS PER CALL SITE (levers compose, they do NOT add)
  a783d0d52b42  baseline $50.72/mo
      zero-quality-risk (caching+batch): -> $14.89/mo  save $35.83/mo
      + tier swap to claude-sonnet-5 (NEEDS replay validation): -> $5.95/mo

  TOTAL zero-quality-risk savings: $41.41/mo
  TOTAL defensible today (zero-risk + VALIDATED tiers): $41.41/mo
  TOTAL potential if every untested tier swap passed: $59.45/mo  <- do not quote this

CLEAN CALL SITES (1) - analyzed, nothing to fix
  c01722375ce4  Tag the sentiment of this support message as one of: positiv

FINDINGS (ranked by estimated monthly savings)
  1. [batch-candidate] site a783d0d52b42 - est. $28.20/mo  (certain IF latency-tolerant)
     182 non-streamed calls, avg latency 2391ms already tolerated. If no user is
     waiting on these, the Batch API gives identical responses at 50% off.

  2. [broken-cache] site a783d0d52b42 - est. $20.94/mo  (high (arithmetic))
     cache_control is set but hit rate is 0.0% across 182 calls.
     Prefix diverges at byte 373 - this is the invalidator:
       A: ..."text": "Current time: 2026-08-10T|>>09:00:00Z.\nYou are the triage...
       B: ..."text": "Current time: 2026-08-10T|>>10:00:00Z.\nYou are the triage...
     Move volatile content below the last cache breakpoint.
```

That second finding is the one people tend to care about: prompt caching
is a strict prefix match, so a timestamp near the front of a system
prompt silently re-bills the entire prefix on every single call. It
throws no error. The bill just goes up. inferopt diffs consecutive
requests and points at the exact byte.

## Run it on your own logs

```bash
uvx inferopt analyze your-logs.jsonl        # .jsonl, .json, .csv, .gz
```

The canonical format is one JSON object per line:

```json
{"request":  {"...the exact body you sent to the API..."},
 "response": {"...the exact body you got back..."},
 "timestamp": 1756400000,
 "latency_ms": 1234}
```

Common exporter shapes are auto-detected, including Helicone
(`request_body` / `response_body`), Langfuse (`input` / `output` /
`usage`), and bare response objects. Anthropic, OpenAI-compatible, and
Bedrock (invoke and converse) request shapes are all recognized.

Responses alone are enough for token accounting, tier comparison, and
cache hit rates. Including the **requests** additionally unlocks
prefix-divergence localization, which is usually the finding that
matters most.

## What it looks for

| Finding | What it means | Quality risk |
|---|---|---|
| `broken-cache` | caching is on but the prefix keeps changing | none, arithmetic |
| `unstable-prefix` | a large reusable prefix that differs every call | none, arithmetic |
| `enable-caching` | a stable prefix that is never cached | none, arithmetic |
| `batch-candidate` | non-streamed traffic nothing is waiting on, at 50% off | none, if latency-tolerant |
| tier what-if | a cheaper model that may hold quality | **needs validation** |

The first four are arithmetic on your own token counts. The last one is
a hypothesis, and inferopt is deliberately pedantic about the difference:
the report gives you a **zero-quality-risk** total, a **defensible
today** total, and a **potential** total marked do-not-quote.

Savings compose multiplicatively rather than adding. Batching a cheaper
model saves 50% of the already reduced bill, so summing per-lever figures
overstates badly. The COMBINED section does this correctly.

## Privacy

- Runs entirely on your machine. Nothing is uploaded, no account, no telemetry.
- The only network calls are the ones your own code already makes, plus a
  daily fetch of OpenRouter's public model catalog for pricing.
- `INFEROPT_STORE_BODIES=0` keeps token metadata and prefixes only.
- Data lands in a local SQLite file at `~/.inferopt/`. `inferopt purge` deletes it.

## Live monitoring (optional)

If you want continuous measurement rather than a one-off analysis, there
is a local proxy that logs traffic as it happens. It speaks three rails:

```bash
pip install 'inferopt[proxy,bedrock]'    # the proxy needs extra packages
inferopt proxy
export ANTHROPIC_BASE_URL=http://127.0.0.1:8484              # Anthropic API
export AWS_ENDPOINT_URL_BEDROCK_RUNTIME=http://127.0.0.1:8484 # Bedrock (re-signs SigV4)
# OpenAI-compatible clients: base_url = http://127.0.0.1:8484/v1
```

It forwards requests unmodified and adds nothing to the response path.
For Bedrock it re-signs with your local AWS credential chain, since
SigV4 covers the Host header.

## Validating a tier swap

Cheaper-model findings are hypotheses until tested. `replay` re-runs real
logged requests against a candidate model and compares them:

```bash
inferopt replay --callsite <fp> --model claude-sonnet-5 --judge
inferopt decide --callsite <fp> --model claude-sonnet-5 --go
```

Two things worth knowing about how it judges. Before any LLM sees the
output, a deterministic check compares tool usage: a candidate that
narrates intent instead of invoking the tools the baseline invoked is
flagged as a behavioral mismatch and treated as worse regardless of what
a judge says. That failure mode is invisible to text-quality scoring and
we have watched it happen. Second, the judge is blind and A/B order
alternates per pair, because an earlier non-blind version demonstrably
rewarded verbosity.

Decisions are recorded in a local ledger, and the report honors it: it
features a validated GO, never features a validated NO-GO, and counts
only proven savings in the defensible total.

## Commands

| Command | What it does | Needs |
|---|---|---|
| `demo` | run the bundled sample and print a full report | nothing |
| `analyze <file>` | one-shot analysis of a log file | nothing |
| `ingest <file>` | load a log file into the persistent database | nothing |
| `report [--days N] [--price-as MODEL]` | analyze everything collected | nothing |
| `callsites` | list observed call sites | nothing |
| `proxy [--port 8484]` | live logging proxy | `[proxy]` |
| `replay --callsite <fp> --model <m> [--judge]` | test a cheaper model on real requests | `[replay]` |
| `decide --callsite <fp> --model <m> --go\|--no-go` | record a validation outcome | nothing |
| `ledger` | show recorded decisions | nothing |
| `purge` | delete the local database | nothing |

`--price-as MODEL` re-prices observed tokens at another model's rates,
which is useful when traffic ran on free or self-hosted models but you
want the real-world cost of the same waste.

## Limitations

- Monthly figures extrapolate the observed window. A short window is
  directional only, and the report prints the multiplier so you can judge.
- Cache math assumes the 5-minute TTL.
- `replay` on Bedrock needs AWS credentials in the shell; `--effort`
  overrides are not yet supported for converse-shaped rows.
- Vertex AI is not supported yet.
- Batch findings assume nothing is waiting on the response. inferopt
  infers that from streaming and timing, but you know your system.

## License

MIT.
