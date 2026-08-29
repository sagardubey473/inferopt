"""Generate examples/sample-log.jsonl - a synthetic but realistic week of
LLM traffic for a fictional support-automation product. Deliberately
includes one call site with NOTHING wrong, so the demo shows the tool
staying quiet when there is nothing to find."""
import json
import random

random.seed(7)
DAY = 86400
T0 = 1756000000  # fixed start so the file is reproducible
OUT = "examples/sample-log.jsonl"

TOOLS = [{"name": "lookup_account", "description": "Look up a customer account",
          "input_schema": {"type": "object",
                           "properties": {"id": {"type": "string"}}}},
         {"name": "escalate", "description": "Escalate to a human agent",
          "input_schema": {"type": "object",
                           "properties": {"reason": {"type": "string"}}}}]

TRIAGE_SYS = ("You are the triage assistant for Northwind Support. Classify "
              "each inbound ticket by product area, severity, and required "
              "team. Follow the routing rules below exactly.\n" +
              "\n".join(f"Rule {i}: tickets mentioning {w} route to the "
                        f"{w}-team with severity weighted accordingly, and "
                        f"must include a justification citing the rule id."
                        for i, w in enumerate(
                            ["billing", "auth", "api", "mobile", "webhooks",
                             "exports", "sso", "quotas", "latency", "uptime",
                             "onboarding", "integrations"] * 12)))

DRAFT_SYS = ("You are the reply-drafting assistant for Northwind Support. "
             "Write a concise, warm, factually grounded reply.\n" +
             "\n".join(f"Style note {i}: {s}" for i, s in enumerate(
                 ["never promise a refund", "never invent SLAs",
                  "always name the next step", "use the customer's name"] * 40)))

SENTIMENT_SYS = ("Tag the sentiment of this support message as one of: "
                 "positive, neutral, frustrated, angry. Reply with the tag "
                 "only, no explanation or punctuation.")

DIGEST_SYS = ("You are the nightly analytics summarizer for Northwind "
              "Support. Given the day's resolved tickets, produce an "
              "executive digest with volume, themes, and anomalies.\n" +
              "\n".join(f"Section {i}: include a short paragraph covering "
                        f"{s} with concrete counts." for i, s in enumerate(
                            ["volume by area", "escalation rate", "top themes",
                             "regressions", "notable outliers"] * 30)))


def rec(ts, model, system, user, out_tok, cached, cache_ctl=True,
        tools=None, stream=False, latency=None, cache_write=0):
    sys_block = ([{"type": "text", "text": system,
                   "cache_control": {"type": "ephemeral"}}]
                 if cache_ctl else system)
    req = {"model": model, "max_tokens": 4096, "system": sys_block,
           "messages": [{"role": "user", "content": user}]}
    if tools:
        req["tools"] = tools
    if stream:
        req["stream"] = True
    total_in = len(system) // 4 + len(user) // 4
    uncached = max(total_in - cached, 40)
    return {
        "timestamp": ts,
        "latency_ms": latency or random.randint(900, 4200),
        "request": req,
        "response": {
            "id": f"msg_{random.randint(10**9, 10**10)}",
            "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": "..."}],
            "usage": {"input_tokens": uncached, "output_tokens": out_tok,
                      "cache_read_input_tokens": cached,
                      "cache_creation_input_tokens": cache_write},
        },
    }


rows = []

# 1. ticket-classify - caching IS enabled but a timestamp in the system
#    prompt invalidates the prefix on every call. The flagship finding.
for d in range(7):
    for i in range(26):
        ts = T0 + d * DAY + 9 * 3600 + i * 900
        stamped = f"Current time: 2026-08-{10+d:02d}T{9+i//4:02d}:00:00Z.\n" \
                  + TRIAGE_SYS
        rows.append(rec(ts, "claude-opus-5", stamped,
                        f"Ticket #{d}{i}: customer cannot log in after SSO "
                        f"migration, tried twice, sev unclear.",
                        out_tok=random.randint(180, 320), cached=0,
                        cache_ctl=True, tools=TOOLS,
                        cache_write=len(stamped) // 4))

# 2. reply-draft - user-facing, streamed, prefix is stable and caching
#    works. Only a tier question here, no cache waste.
for d in range(7):
    for i in range(18):
        ts = T0 + d * DAY + 10 * 3600 + i * 1200
        rows.append(rec(ts, "claude-opus-5", DRAFT_SYS,
                        f"Draft a reply to ticket #{d}{i}. Customer is "
                        f"frustrated about a delayed export.",
                        out_tok=random.randint(400, 900),
                        cached=len(DRAFT_SYS) // 4, stream=True,
                        latency=random.randint(2000, 6000)))

# 3. nightly-digest - a cron job at 02:00, nobody waiting. Batch-eligible.
for d in range(7):
    for i in range(6):
        ts = T0 + d * DAY + 2 * 3600 + i * 300
        rows.append(rec(ts, "claude-opus-5", DIGEST_SYS,
                        f"Summarize resolved tickets for 2026-08-{10+d:02d}, "
                        f"batch {i}. " + "ticket data; " * 400,
                        out_tok=random.randint(1400, 2600),
                        cached=len(DIGEST_SYS) // 4,
                        latency=random.randint(8000, 20000)))

# 4. sentiment-tag - CONTROL. Already on Haiku, tiny prompt, well cached,
#    latency-sensitive. Nothing to find here, and the report should say so.
for d in range(7):
    for i in range(40):
        ts = T0 + d * DAY + 8 * 3600 + i * 400
        rows.append(rec(ts, "claude-haiku-4-5", SENTIMENT_SYS,
                        f"Message {d}-{i}: thanks, that fixed it!",
                        out_tok=6, cached=0, cache_ctl=False, stream=True,
                        latency=random.randint(180, 400)))

rows.sort(key=lambda r: r["timestamp"])
with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {OUT}: {len(rows)} records over 7 days")
