"""Replay logged requests at a cheaper tier / lower effort and compare.

Baseline output = what you already got (and paid for); candidate = re-run
live. Works on both rails: anthropic (needs ANTHROPIC_API_KEY) and bedrock
(re-signs with the local AWS credential chain, same as the proxy). Optional
LLM judge gives a per-pair verdict. Always prints a cost estimate and asks
before spending.

Bedrock model overrides must be the full inference-profile id
(e.g. us.anthropic.claude-haiku-4-5-...), not the bare on-demand id.
"""
import json
import os
import re
import sys
import time
import urllib.parse

import httpx

from . import capture, db, pricing

API_VERSION = "2023-06-01"
_TOOL_RE = re.compile(r"\[tool_use ([^\]]+)\]")


def _tool_names(text):
    return set(_TOOL_RE.findall(text or ""))


def _behavioral_check(baseline, candidate):
    """Deterministic tool-usage comparison - catches what a text judge
    can't: a candidate that narrates intent instead of invoking tools."""
    bt, ct = _tool_names(baseline), _tool_names(candidate)
    if bt and not ct:
        return (f"baseline invoked tools ({', '.join(sorted(bt))}) but "
                f"candidate answered in prose without any tool call - "
                f"functional regression for agentic loops")
    if bt and ct and not (bt & ct):
        return (f"tool mismatch: baseline used {sorted(bt)}, candidate "
                f"used {sorted(ct)}")
    return None
DEFAULT_JUDGE_ANTHROPIC = "claude-opus-5"


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


def _bedrock_region():
    return (os.environ.get("INFEROPT_BEDROCK_REGION")
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1")


def _bedrock_base():
    return os.environ.get(
        "INFEROPT_BEDROCK_UPSTREAM",
        f"https://bedrock-runtime.{_bedrock_region()}.amazonaws.com")


def _sanitize(body, target_model):
    """Make an anthropic-shaped body valid for the target model."""
    key = pricing.resolve(target_model) or ""
    if key in pricing.NO_SAMPLING:
        for f in ("temperature", "top_p", "top_k"):
            body.pop(f, None)
    th = body.get("thinking")
    if key in ("claude-fable-5", "claude-mythos-5"):
        body.pop("thinking", None)
    elif isinstance(th, dict) and th.get("type") == "enabled":
        body["thinking"] = {"type": "adaptive"}
    return body


def _first_user_text(body):
    for m in body.get("messages") or []:
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                for blk in c:
                    if isinstance(blk, dict):
                        if blk.get("type") == "text":
                            return blk.get("text", "")
                        if "text" in blk and "type" not in blk:
                            return blk["text"]  # converse shape
    return ""


def _call_anthropic(client, headers, body):
    r = client.post("/v1/messages", headers=headers, json=body)
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}: {r.text[:200]}"
    data = r.json()
    return capture.anthropic_content_text(data.get("content")), None


def _call_bedrock(client, action, model_id, body):
    from . import bedrock as br
    raw = json.dumps(body).encode()
    path = f"/model/{urllib.parse.quote(model_id, safe='')}/{action}"
    url = _bedrock_base() + path
    try:
        headers = br.sign("POST", url,
                          {"content-type": "application/json",
                           "accept": "application/json"},
                          raw, _bedrock_region())
    except Exception as e:
        return None, f"signing failed: {e}"
    r = client.post(url, headers=headers, content=raw)
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}: {r.text[:300]}"
    _i, _o, _cr, _cw, text = br.extract_response(action, r.json())
    return text, None


def _judge_prompt(task, resp_a, resp_b):
    return (
        "You are grading two AI responses to the same task, labeled A and "
        "B. One is from the current production model and one from a "
        "candidate replacement; you are NOT told which is which.\n"
        "Judge SUBSTANCE only: factual and technical accuracy, completeness "
        "of the required content, and instruction adherence. Do NOT reward "
        "length, verbosity, bullet density, or formatting - more detailed "
        "does not mean better unless the task demands it. A response may be "
        "a serialized tool call ('[tool_use <name>] {json}'); in that case "
        "judge the correctness and completeness of the field values. A "
        "response may be TRUNCATED at the character limit: judge only what "
        "is visible and never assume, infer, or invent content beyond the "
        "truncation point.\n\n"
        f"TASK (truncated):\n{task[:2000]}\n\n"
        f"RESPONSE A (truncated):\n{resp_a[:6000]}\n\n"
        f"RESPONSE B (truncated):\n{resp_b[:6000]}\n\n"
        "Reply with exactly one line first: "
        "VERDICT: equivalent | a_better | b_better\n"
        "Then one sentence of justification."
    )


def replay(callsite, n=5, model=None, effort=None, judge=False, yes=False,
           judge_model=None, db_path=None):
    if not model and not effort:
        sys.exit("replay: pass --model and/or --effort (nothing to change)")
    con = db.connect(db_path)
    rows = con.execute(
        "SELECT * FROM requests WHERE callsite=? AND body_json IS NOT NULL "
        "AND response_text IS NOT NULL AND status BETWEEN 200 AND 299 "
        "ORDER BY ts DESC LIMIT ?", (callsite, n)).fetchall()
    if not rows:
        sys.exit(f"replay: no stored request bodies for site {callsite}. "
                 "Bodies are required - if the proxy ran with "
                 "INFEROPT_STORE_BODIES=0, re-run a few requests with bodies "
                 "on (synthetic inputs recommended) and try again.")

    def _rail(r):
        try:
            return r["rail"] or "anthropic"
        except (KeyError, IndexError):
            return "anthropic"

    rails = {_rail(r) for r in rows}
    bedrock_any = any(x.startswith("bedrock") for x in rails)
    if effort and "bedrock:converse" in rails:
        sys.exit("replay: --effort override isn't supported for converse-"
                 "logged rows yet; use --model, or replay an invoke site")

    # ---- cost estimate ----
    est = 0.0
    for r in rows:
        tgt = model or r["model"]
        rates = pricing.rates(tgt) or (5.0, 25.0)
        tot_in = (r["input_tokens"] or 0) + (r["cache_read_tokens"] or 0) \
            + (r["cache_write_tokens"] or 0)
        est += (tot_in * rates[0] + (r["output_tokens"] or 0) * rates[1]) / 1e6
    if judge:
        est *= 1.6  # judge reads both outputs; rough bound
    print(f"replaying {len(rows)} requests from site {callsite}"
          f"{' on ' + model if model else ''}"
          f"{' at effort=' + effort if effort else ''}"
          f"{' with judge' if judge else ''}  [rails: {', '.join(sorted(rails))}]")
    print(f"estimated replay cost: ~${est:.2f}")
    if not yes:
        if input("proceed? [y/N] ").strip().lower() != "y":
            sys.exit("aborted")

    anthropic_headers = None
    if any(x == "anthropic" for x in rails):
        anthropic_headers = _auth_headers()
    a_client = httpx.Client(
        base_url=os.environ.get("INFEROPT_REPLAY_UPSTREAM",
                                "https://api.anthropic.com"), timeout=600.0)
    b_client = httpx.Client(timeout=600.0)

    def run_candidate(row, body):
        rail = _rail(row)
        if rail.startswith("bedrock"):
            action = "invoke" if rail == "bedrock:invoke" else "converse"
            if action == "invoke":
                _sanitize(body, model or row["model"])
                if effort:
                    body.setdefault("output_config", {})["effort"] = effort
            return _call_bedrock(b_client, action, model or row["model"], body)
        body.pop("stream", None)
        if model:
            body["model"] = model
        if effort:
            body.setdefault("output_config", {})["effort"] = effort
        _sanitize(body, body.get("model"))
        return _call_anthropic(a_client, anthropic_headers, body)

    def run_judge(row, task, baseline, candidate, idx):
        rail = _rail(row)
        swapped = idx % 2 == 1  # alternate A/B order to cancel position bias
        a, b = (candidate, baseline) if swapped else (baseline, candidate)
        prompt = _judge_prompt(task, a or "", b or "")
        if rail.startswith("bedrock"):
            jm = judge_model or row["model"]  # incumbent model as judge
            body = {"anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 300,
                    "messages": [{"role": "user", "content": prompt}]}
            text, err = _call_bedrock(b_client, "invoke", jm, body)
        else:
            jm = judge_model or DEFAULT_JUDGE_ANTHROPIC
            body = {"model": jm, "max_tokens": 300,
                    "messages": [{"role": "user", "content": prompt}]}
            text, err = _call_anthropic(a_client, anthropic_headers, body)
        if err:
            return "judge_error", err
        m = re.search(r"VERDICT:\s*(\w+)", text or "")
        raw = m.group(1).lower() if m else "unparsed"
        cand_side = "a" if swapped else "b"
        if raw == "equivalent":
            verdict = "equivalent"
        elif raw in ("a_better", "b_better"):
            verdict = ("candidate_better" if raw[0] == cand_side
                       else "candidate_worse")
        else:
            verdict = raw
        return verdict, (text or "").strip()

    pairs, verdicts, mismatches = [], [], 0
    for i, r in enumerate(rows, 1):
        body = json.loads(r["body_json"])
        t0 = time.time()
        cand, err = run_candidate(r, body)
        lat = time.time() - t0
        if err:
            print(f"  [{i}/{len(rows)}] FAILED: {err}")
            continue
        behavioral = _behavioral_check(r["response_text"], cand)
        if behavioral:
            mismatches += 1
        verdict, reason = "", ""
        if judge:
            verdict, reason = run_judge(r, _first_user_text(body),
                                        r["response_text"], cand, i)
            verdicts.append(verdict)
        pairs.append((r, cand, verdict, reason, lat, behavioral))
        print(f"  [{i}/{len(rows)}] ok {lat:.1f}s"
              + (f"  verdict={verdict}" if judge else "")
              + (f"  BEHAVIORAL MISMATCH: {behavioral}" if behavioral else ""))

    tag = (model or "") + (("-" + effort) if effort else "")
    tag = tag.replace("/", "_").replace(":", "_").strip("-") or "same"
    path = f"inferopt-replay-{callsite}-{tag}.md"
    with open(path, "w") as f:
        f.write(f"# Replay: site {callsite} -> {model or 'same model'}"
                f"{', effort=' + effort if effort else ''}\n\n")
        if verdicts:
            from collections import Counter
            f.write(f"**Judge summary:** {dict(Counter(verdicts))} "
                    f"(blind judge, A/B order alternated per pair; "
                    f"spot-check pairs yourself)\n\n")
        if mismatches:
            f.write(f"**BEHAVIORAL MISMATCHES: {mismatches}/{len(pairs)} "
                    f"pairs** - candidate's tool usage diverges from "
                    f"baseline (see per-pair notes). Treat these as "
                    f"candidate_worse regardless of judge verdicts: a text "
                    f"judge cannot fully weigh a skipped tool call.\n\n")
        for r, cand, verdict, reason, lat, behavioral in pairs:
            f.write(f"## request {r['id']} ({r['model']}, {_rail(r)})\n\n")
            f.write(f"**task (first user msg):**\n\n```\n"
                    f"{_first_user_text(json.loads(r['body_json']))[:1500]}"
                    f"\n```\n\n")
            f.write(f"**baseline ({r['model']}):**\n\n```\n"
                    f"{(r['response_text'] or '')[:2500]}\n```\n\n")
            f.write(f"**candidate ({model or r['model']}"
                    f"{', effort=' + effort if effort else ''}, "
                    f"{lat:.1f}s):**\n\n```\n{(cand or '')[:2500]}\n```\n\n")
            if behavioral:
                f.write(f"**BEHAVIORAL MISMATCH:** {behavioral}\n\n")
            if verdict:
                f.write(f"**judge:** {verdict}\n\n> {reason}\n\n")
    print(f"\nwrote {path}")
    if verdicts:
        from collections import Counter
        print(f"judge summary: {dict(Counter(verdicts))}")
    if mismatches:
        print(f"BEHAVIORAL MISMATCHES: {mismatches}/{len(pairs)} pairs - "
              f"treat as candidate_worse regardless of judge verdicts")

    if model:
        from collections import Counter
        c = Counter(verdicts)
        auto = "no-go" if mismatches else "untested"
        con.execute(
            "INSERT INTO validations (ts,callsite,alt_model,n,equivalent,"
            "better,worse,mismatches,decision,note) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (time.time(), callsite, model, len(pairs),
             c.get("equivalent", 0), c.get("candidate_better", 0),
             c.get("candidate_worse", 0), mismatches, auto,
             "auto: behavioral mismatch" if mismatches else
             "evidence recorded; run `inferopt decide` after spot-check"))
        con.commit()
        if mismatches:
            print(f"\nledger: recorded {model} on {callsite} as NO-GO "
                  f"(behavioral mismatch is disqualifying)")
        else:
            print(f"\nledger: evidence recorded. After you spot-check the "
                  f"pairs, record the call:\n  inferopt decide --callsite "
                  f"{callsite} --model {model} --go   (or --no-go)")
    print("spot-check the pairs yourself before changing anything in prod.")
