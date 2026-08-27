"""Replay logged requests at a cheaper tier / lower effort and compare.

This is the seed of the shadow-mode prover: baseline output is what you
already got (and paid for); the candidate is re-run live. Optional LLM judge
gives a per-pair verdict. Always prints a cost estimate and asks before
spending tokens.
"""
import json
import os
import re
import sys
import time

import httpx

from . import db, pricing

API_VERSION = "2023-06-01"
JUDGE_MODEL = "claude-opus-5"


def _auth_headers():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return {"x-api-key": key, "anthropic-version": API_VERSION}
    tok = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if tok:
        return {"Authorization": f"Bearer {tok}",
                "anthropic-version": API_VERSION,
                "anthropic-beta": "oauth-2025-04-20"}
    sys.exit("inferopt replay: set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN)")


def _sanitize(body):
    """Make a stored request valid for the (possibly different) target model."""
    key = pricing.resolve(body.get("model")) or ""
    if key in pricing.NO_SAMPLING:
        for f in ("temperature", "top_p", "top_k"):
            body.pop(f, None)
    th = body.get("thinking")
    if key in ("claude-fable-5", "claude-mythos-5"):
        body.pop("thinking", None)  # always-on; explicit config rejected
    elif isinstance(th, dict) and th.get("type") == "enabled":
        body["thinking"] = {"type": "adaptive"}  # budget_tokens removed
    return body


def _first_user_text(body):
    for m in body.get("messages") or []:
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                for blk in c:
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        return blk.get("text", "")
    return ""


def _judge(client, headers, task, baseline, candidate):
    prompt = (
        "You are grading two AI responses to the same task. Response A is the "
        "baseline (expensive model); Response B is a cheaper candidate. Judge "
        "whether B is an acceptable substitute for A for this task.\n\n"
        f"TASK (truncated):\n{task[:2000]}\n\n"
        f"RESPONSE A (baseline, truncated):\n{baseline[:4000]}\n\n"
        f"RESPONSE B (candidate, truncated):\n{candidate[:4000]}\n\n"
        "Reply with exactly one line first: "
        "VERDICT: equivalent | candidate_worse | candidate_better\n"
        "Then one sentence of justification."
    )
    r = client.post("/v1/messages", headers=headers, json={
        "model": JUDGE_MODEL, "max_tokens": 300,
        "messages": [{"role": "user", "content": prompt}],
    })
    r.raise_for_status()
    text = "\n".join(b.get("text", "") for b in r.json().get("content", [])
                     if b.get("type") == "text")
    m = re.search(r"VERDICT:\s*(\w+)", text)
    return (m.group(1) if m else "unparsed"), text.strip()


def replay(callsite, n=5, model=None, effort=None, judge=False, yes=False,
           db_path=None):
    if not model and not effort:
        sys.exit("replay: pass --model and/or --effort (nothing to change)")
    con = db.connect(db_path)
    rows = con.execute(
        "SELECT * FROM requests WHERE callsite=? AND body_json IS NOT NULL "
        "AND response_text IS NOT NULL AND status BETWEEN 200 AND 299 "
        "ORDER BY ts DESC LIMIT ?", (callsite, n)).fetchall()
    if not rows:
        sys.exit(f"replay: no stored request bodies for site {callsite} "
                 "(bodies stored? INFEROPT_STORE_BODIES=0 disables replay)")

    # ---- cost estimate (upper-bound-ish; excludes thinking variation) ----
    est = 0.0
    for r in rows:
        tgt = model or r["model"]
        rates = pricing.rates(tgt) or (5.0, 25.0)
        tot_in = (r["input_tokens"] or 0) + (r["cache_read_tokens"] or 0) \
            + (r["cache_write_tokens"] or 0)
        est += (tot_in * rates[0] + (r["output_tokens"] or 0) * rates[1]) / 1e6
    if judge:
        est += len(rows) * 0.03  # rough judge cost bound
    print(f"replaying {len(rows)} requests from site {callsite}"
          f"{' on ' + model if model else ''}"
          f"{' at effort=' + effort if effort else ''}"
          f"{' with judge' if judge else ''}")
    print(f"estimated replay cost: ~${est:.2f}")
    if not yes:
        if input("proceed? [y/N] ").strip().lower() != "y":
            sys.exit("aborted")

    headers = _auth_headers()
    base = os.environ.get("INFEROPT_REPLAY_UPSTREAM",
                          "https://api.anthropic.com")
    client = httpx.Client(base_url=base, timeout=600.0)

    pairs, verdicts = [], []
    for i, r in enumerate(rows, 1):
        body = json.loads(r["body_json"])
        body.pop("stream", None)
        if model:
            body["model"] = model
        if effort:
            body.setdefault("output_config", {})["effort"] = effort
        _sanitize(body)
        t0 = time.time()
        resp = client.post("/v1/messages", headers=headers, json=body)
        lat = time.time() - t0
        if resp.status_code != 200:
            print(f"  [{i}/{len(rows)}] HTTP {resp.status_code}: "
                  f"{resp.text[:200]}")
            continue
        data = resp.json()
        cand = "\n".join(b.get("text", "") for b in data.get("content", [])
                         if b.get("type") == "text")
        verdict, reason = ("", "")
        if judge:
            try:
                verdict, reason = _judge(client, headers,
                                         _first_user_text(body),
                                         r["response_text"], cand)
                verdicts.append(verdict)
            except Exception as e:
                verdict, reason = "judge_error", str(e)[:200]
        pairs.append((r, cand, verdict, reason, lat))
        print(f"  [{i}/{len(rows)}] ok {lat:.1f}s"
              + (f"  verdict={verdict}" if judge else ""))

    tag = (model or "") + (("-" + effort) if effort else "")
    path = f"inferopt-replay-{callsite}-{tag.strip('-') or 'same'}.md"
    with open(path, "w") as f:
        f.write(f"# Replay: site {callsite} -> "
                f"{model or 'same model'}"
                f"{', effort=' + effort if effort else ''}\n\n")
        if verdicts:
            from collections import Counter
            f.write(f"**Judge summary:** {dict(Counter(verdicts))} "
                    f"(judge: {JUDGE_MODEL}; fixed A/B order - spot-check "
                    f"a few pairs yourself)\n\n")
        for r, cand, verdict, reason, lat in pairs:
            f.write(f"## request {r['id']} ({r['model']})\n\n")
            f.write(f"**task (first user msg):**\n\n```\n"
                    f"{_first_user_text(json.loads(r['body_json']))[:1500]}"
                    f"\n```\n\n")
            f.write(f"**baseline ({r['model']}):**\n\n```\n"
                    f"{(r['response_text'] or '')[:2500]}\n```\n\n")
            f.write(f"**candidate ({model or r['model']}"
                    f"{', effort=' + effort if effort else ''}, "
                    f"{lat:.1f}s):**\n\n```\n{cand[:2500]}\n```\n\n")
            if verdict:
                f.write(f"**judge:** {verdict}\n\n> {reason}\n\n")
    print(f"\nwrote {path}")
    if verdicts:
        from collections import Counter
        print(f"judge summary: {dict(Counter(verdicts))}")
    print("spot-check the pairs yourself before changing anything in prod.")
