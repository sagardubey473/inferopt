"""The report: turn logged traffic into ranked, dollar-quantified findings.

Three lever families:
  1. Prompt caching  - uncached stable prefixes; broken caches w/ byte-level
                       invalidator localization (arithmetic, no eval needed)
  2. Batch API       - non-streamed call sites, flat 50% (needs latency OK)
  3. Tier/effort     - what-if costs at cheaper tiers (needs `replay` to
                       validate quality before acting)
"""
import json
import time
from collections import Counter, defaultdict

from . import pricing

MIN_CACHEABLE_TOKENS = 1024  # minimum cacheable prefix on most models


def _est_tokens(s):
    return len(s or "") // 4


def first_divergence(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else None


def _ctx(s, i, span=45):
    lo = max(0, i - span)
    seg = s[lo:i] + "|>>" + s[i:i + span]
    return ("..." if lo else "") + seg.replace("\n", " ") + "..."


def load_validations(con):
    """Latest decision per (callsite, alt_model)."""
    out = {}
    try:
        rows = con.execute(
            "SELECT * FROM validations ORDER BY ts").fetchall()
    except Exception:
        return out
    for r in rows:
        out[(r["callsite"], r["alt_model"])] = dict(r)
    return out


def _val_status(vals, site, alt):
    """Match a ledger entry to a pricing key (ledger stores full profile ids
    like us.anthropic.claude-haiku-4-5-...; findings use pricing keys)."""
    for (s, m), v in vals.items():
        if s == site and pricing.resolve(m) == pricing.resolve(alt):
            return v
    return None


def load(con, days):
    cutoff = time.time() - days * 86400
    rows = [dict(r) for r in con.execute(
        "SELECT * FROM requests WHERE ts>=? ORDER BY ts", (cutoff,))]
    ok = [r for r in rows if r["status"] and 200 <= r["status"] < 300]
    groups = defaultdict(list)
    for r in ok:
        groups[r["callsite"]].append(r)
    return rows, ok, groups


def summarize_group(rs):
    g = {}
    g["n"] = len(rs)
    g["hint"] = rs[-1]["hint"]
    g["models"] = Counter(r["model"] for r in rs)
    g["model"] = g["models"].most_common(1)[0][0]
    for k, col in [("in_t", "input_tokens"), ("out_t", "output_tokens"),
                   ("cr", "cache_read_tokens"), ("cw", "cache_write_tokens")]:
        g[k] = sum(r[col] or 0 for r in rs)
    g["cost"] = sum(r["cost_usd"] or 0 for r in rs)
    g["unknown_cost"] = sum(1 for r in rs if r["cost_usd"] is None)
    g["stream_share"] = sum(r["stream"] or 0 for r in rs) / len(rs)
    g["uses_cache"] = (any(r["uses_cache_control"] for r in rs)
                       or any((r["cache_read_tokens"] or 0)
                              or (r["cache_write_tokens"] or 0) for r in rs))
    g["efforts"] = Counter(r["effort"] or "default(high)" for r in rs)
    prefixes = Counter(r["prefix"] or "" for r in rs)
    mc_prefix, mc_n = prefixes.most_common(1)[0]
    g["prefix"] = mc_prefix
    g["prefix_share"] = mc_n / len(rs)
    g["prefix_tokens"] = _est_tokens(mc_prefix)
    total_in = g["in_t"] + g["cr"] + g["cw"]
    g["hit_rate"] = g["cr"] / total_in if total_in else 0.0
    g["avg_latency"] = sum(r["latency_ms"] or 0 for r in rs) / len(rs)
    try:
        g["rail"] = rs[-1]["rail"] or "anthropic"
    except (KeyError, IndexError):
        g["rail"] = "anthropic"
    return g


def _config_monthly(g, factor, model=None, cache_fix=False, batch=False):
    """Monthly cost for a call site under a combination of levers.

    Levers compose multiplicatively, not additively: batching a cheaper
    model saves 50% of the *already reduced* bill.
    """
    rates = pricing.rates(model or g["model"])
    if rates is None:
        return None
    p_in, p_out = rates
    n = g["n"]
    billable_in = g["in_t"] + g["cw"]          # full-price input tokens
    cached_in = g["cr"]                        # already at 0.1x
    if cache_fix:
        movable = min(g["prefix_tokens"] * n, billable_in)
        billable_in -= movable
        cached_in += movable
    cost = (billable_in * p_in
            + cached_in * p_in * pricing.CACHE_READ_MULT
            + g["out_t"] * p_out) / 1e6
    if batch:
        cost *= pricing.BATCH_MULT
    return cost * factor


def analyze(con, days=30, price_as=None):
    rows, ok, groups = load(con, days)
    out = {"days": days, "n_total": len(rows), "n_ok": len(ok),
           "groups": {}, "findings": [], "tier_whatif": [], "notes": []}
    if not ok:
        return out

    first_ts = min(r["ts"] for r in ok)
    last_ts = max(r["ts"] for r in ok)
    window_days = max((last_ts - first_ts) / 86400, 1 / 24)
    factor = 30.0 / window_days  # extrapolation to a 30-day month
    out["window_days"] = window_days
    out["factor"] = factor
    out["spend"] = sum(r["cost_usd"] or 0 for r in ok)
    if price_as:
        out["spend"] = sum(
            pricing.cost_usd(price_as, r["input_tokens"] or 0,
                             r["output_tokens"] or 0,
                             r["cache_read_tokens"] or 0,
                             r["cache_write_tokens"] or 0) or 0 for r in ok)
    out["monthly_spend"] = out["spend"] * factor

    levers = {}
    out["price_as"] = price_as
    for fp, rs in groups.items():
        g = summarize_group(rs)
        if price_as:
            g["real_model"] = g["model"]
            g["model"] = price_as
            g["cost"] = sum(
                pricing.cost_usd(price_as, r["input_tokens"] or 0,
                                 r["output_tokens"] or 0,
                                 r["cache_read_tokens"] or 0,
                                 r["cache_write_tokens"] or 0) or 0
                for r in rs)
        out["groups"][fp] = g
        levers[fp] = {"cache": False, "batch": False, "tier": None}
        rates = pricing.rates(g["model"])
        if rates is None:
            out["notes"].append(
                f"site {fp}: unknown model '{g['model']}' - costs not computed")
            continue
        price_in, _ = rates
        monthly_reqs = g["n"] * factor
        monthly_cost = g["cost"] * factor

        # ---- Lever 1a: stable prefix, caching never enabled ----
        if (not g["uses_cache"] and g["n"] >= 3
                and g["prefix_share"] >= 0.9
                and g["prefix_tokens"] >= MIN_CACHEABLE_TOKENS):
            save = 0.9 * g["prefix_tokens"] * price_in * monthly_reqs / 1e6
            levers[fp]["cache"] = True
            out["findings"].append({
                "kind": "enable-caching", "site": fp, "hint": g["hint"],
                "monthly_savings": save, "confidence": "high (arithmetic)",
                "detail": (
                    f"~{g['prefix_tokens']:,} stable prefix tokens re-billed at "
                    f"full price on every call ({g['n']} calls observed, prefix "
                    f"identical in {g['prefix_share']:.0%} of them). "
                    + ("This provider caches automatically on a byte-stable "
                       "prefix, but no cached tokens were reported - check "
                       "that the prefix meets the minimum length and that "
                       "nothing volatile precedes it."
                       if g.get("rail") == "openai" else
                       "Add cache_control to the system prompt / tools "
                       "block; cache reads bill at 10% of input.")
                    + " No output changes at all."),
            })

        # ---- Lever 1b: caching on but broken ----
        elif g["uses_cache"] and g["n"] >= 3 and g["hit_rate"] < 0.5:
            div = None
            for a, b in zip(rs, rs[1:]):
                if (a["prefix"] or "") != (b["prefix"] or ""):
                    i = first_divergence(a["prefix"] or "", b["prefix"] or "")
                    if i is not None:
                        div = (i, a["prefix"], b["prefix"])
                        break
            cacheable = min(g["prefix_tokens"],
                            (g["in_t"] + g["cr"] + g["cw"]) // max(g["n"], 1))
            save = 0.9 * cacheable * price_in * monthly_reqs / 1e6 \
                - g["cr"] * factor * price_in * pricing.CACHE_READ_MULT / 1e6
            levers[fp]["cache"] = True
            f = {"kind": "broken-cache", "site": fp, "hint": g["hint"],
                 "monthly_savings": max(save, 0),
                 "confidence": "high (arithmetic)",
                 "detail": (
                     ("prompt caching is active on this provider but the hit "
                      "rate is only "
                      if g.get("rail") == "openai" else
                      "cache_control is set but hit rate is ")
                     + f"{g['hit_rate']:.1%} across {g['n']} calls.")}
            if div:
                i, pa, pb = div
                f["detail"] += (
                    f" Prefix diverges at byte {i:,} - this is the "
                    f"invalidator:\n      A: {_ctx(pa, i)}\n"
                    f"      B: {_ctx(pb, i)}\n"
                    f"    Move volatile content below the last cache "
                    f"breakpoint (render order: tools -> system -> messages).")
            out["findings"].append(f)

        # ---- Lever 1c: unstable prefix (never caches, and can't) ----
        elif (g["n"] >= 3 and g["prefix_share"] < 0.9
              and g["prefix_tokens"] >= MIN_CACHEABLE_TOKENS
              and g["hit_rate"] < 0.2):
            div = None
            for x, y in zip(rs, rs[1:]):
                if (x["prefix"] or "") != (y["prefix"] or ""):
                    i = first_divergence(x["prefix"] or "", y["prefix"] or "")
                    if i is not None:
                        div = (i, x["prefix"], y["prefix"])
                        break
            save = 0.9 * g["prefix_tokens"] * price_in * monthly_reqs / 1e6
            levers[fp]["cache"] = True
            f = {"kind": "unstable-prefix", "site": fp, "hint": g["hint"],
                 "monthly_savings": save, "confidence": "high (arithmetic)",
                 "detail": (
                     f"~{g['prefix_tokens']:,} tokens of otherwise-reusable "
                     f"prefix, but it differs between calls "
                     f"({g['prefix_share']:.0%} identical) and the observed "
                     f"cache hit rate is {g['hit_rate']:.1%}. Prompt caching "
                     f"is a strict prefix match, so a single varying byte "
                     f"near the front re-bills the whole prefix every call.")}
            if div:
                i, pa, pb = div
                f["detail"] += (
                    f"\n    First divergence at byte {i:,}:"
                    f"\n      A: {_ctx(pa, i)}"
                    f"\n      B: {_ctx(pb, i)}"
                    f"\n    Move that value out of the prefix (below the "
                    f"last cache breakpoint, or into a later message).")
            out["findings"].append(f)

        # ---- Lever 2: batch candidates (first-party API / Bedrock only;
        # OpenAI-compatible gateways have no equivalent 50%-off batch tier)
        if (g.get("rail") != "openai" and g["stream_share"] <= 0.1
                and g["n"] >= 3 and monthly_cost > 0):
            levers[fp]["batch"] = True
            out["findings"].append({
                "kind": "batch-candidate", "site": fp, "hint": g["hint"],
                "monthly_savings": monthly_cost * pricing.BATCH_MULT,
                "confidence": "certain IF latency-tolerant (flat 50%)",
                "detail": (
                    f"{g['n']} non-streamed calls, avg latency "
                    f"{g['avg_latency']:.0f}ms already tolerated. If no user "
                    f"is waiting on these (cron / pipeline / bulk job), the "
                    f"Batch API gives identical responses at 50% off both "
                    f"input and output."),
            })

        # ---- Lever 3: tier what-if (needs replay validation) ----
        # Only for first-party/Bedrock ids - suggesting "claude-haiku-4-5"
        # to a caller using "anthropic/claude-sonnet-5" would be a broken
        # model string. Cross-format tier mapping is not wired up yet.
        cur_key = pricing.resolve(g["model"])
        if not pricing.is_first_party(g["model"]):
            continue
        for alt in pricing.CHEAPER_TIERS:
            if alt == cur_key:
                continue
            alt_cost = pricing.cost_usd(alt, g["in_t"], g["out_t"],
                                        g["cr"], g["cw"]) * factor
            if monthly_cost - alt_cost > 0.01 * factor:
                note = ""
                if (cur_key and cur_key.startswith("claude-sonnet")
                        and alt == "claude-sonnet-5"):
                    note = (" - newer generation of the same family "
                            "(cheaper, often better): best candidate to TEST "
                            "first, but validate with replay before shipping "
                            "- generation upgrades can still regress on "
                            "structured-extraction tasks")
                out["tier_whatif"].append({
                    "site": fp, "hint": g["hint"], "model": g["model"],
                    "alt": alt, "monthly_cost": monthly_cost,
                    "alt_monthly_cost": alt_cost,
                    "monthly_savings": monthly_cost - alt_cost,
                    "note": note,
                })
                cur_best = levers[fp]["tier"]
                if cur_best is None or (monthly_cost - alt_cost) > cur_best[1]:
                    levers[fp]["tier"] = (alt, monthly_cost - alt_cost)

    # ---- effort observation ----
    if all(set(g["efforts"]) == {"default(high)"}
           for g in out["groups"].values()):
        out["notes"].append(
            "No request sets output_config.effort - everything runs at the "
            "default ('high'). Routine call sites may hold quality at "
            "'medium'/'low' with fewer thinking+output tokens; validate with "
            "`inferopt replay --effort low`.")

    vals = load_validations(con)
    out["validations"] = [
        {"site": s, "model": m, **{k: v[k] for k in
         ("n", "equivalent", "better", "worse", "mismatches", "decision")}}
        for (s, m), v in vals.items()]

    for w in out["tier_whatif"]:
        v = _val_status(vals, w["site"], w["alt"])
        if v is None:
            w["validation"] = "untested"
        else:
            w["validation"] = v["decision"]
            w["evidence"] = (f"n={v['n']} judge {v['equivalent']}e/"
                             f"{v['better']}b/{v['worse']}w"
                             + (f", {v['mismatches']} behavioral mismatches"
                                if v["mismatches"] else ""))

    out["combined"] = []
    for fp, lv in levers.items():
        g = out["groups"][fp]
        if not (lv["cache"] or lv["batch"] or lv["tier"]):
            continue
        base = _config_monthly(g, factor)
        if base is None:
            continue
        applied = []
        if lv["cache"]:
            applied.append("caching")
        if lv["batch"]:
            applied.append("batch")
        # safe = levers with no quality risk; tier needs replay validation
        safe = _config_monthly(g, factor, cache_fix=lv["cache"],
                               batch=lv["batch"])
        row = {"site": fp, "hint": g["hint"], "baseline": base,
               "safe_levers": applied, "safe_monthly": safe,
               "safe_savings": base - (safe if safe is not None else base)}
        # pick the tier to feature: a validated GO wins; never feature a
        # NO-GO just because it is cheapest.
        candidates = [w for w in out["tier_whatif"] if w["site"] == fp]
        go = [w for w in candidates if w.get("validation") == "go"]
        untested = [w for w in candidates
                    if w.get("validation") == "untested"]
        excluded = [w["alt"] for w in candidates
                    if w.get("validation") == "no-go"]
        pick = (go[0] if go else (untested[0] if untested else None))
        row["excluded_tiers"] = excluded
        if pick:
            alt = pick["alt"]
            full = _config_monthly(g, factor, model=alt,
                                   cache_fix=lv["cache"], batch=lv["batch"])
            row["tier_alt"] = alt
            row["tier_validation"] = pick.get("validation", "untested")
            row["full_monthly"] = full
            row["full_savings"] = base - (full if full is not None else base)
        out["combined"].append(row)
    out["combined"].sort(key=lambda r: -(r.get("full_savings")
                                         or r["safe_savings"]))
    out["total_safe_savings"] = sum(r["safe_savings"] for r in out["combined"])
    out["total_validated_savings"] = sum(
        r["full_savings"] if r.get("tier_validation") == "go"
        else r["safe_savings"] for r in out["combined"])
    out["total_potential_savings"] = sum(
        r.get("full_savings", r["safe_savings"]) for r in out["combined"])
    out["findings"].sort(key=lambda f: -f["monthly_savings"])
    out["tier_whatif"].sort(key=lambda f: -f["monthly_savings"])
    return out


def _money(x):
    return f"${x:,.2f}" if abs(x) >= 0.01 else f"${x:.4f}"


def render(out):
    L = []
    add = L.append
    add(f"inferopt report - last {out['days']:g} days")
    add("=" * 60)
    if out.get("price_as"):
        add(f"PRICED AS: {out['price_as']} - observed token counts re-priced")
        add(f"at that model's rates (actual traffic ran on other/free models).")
        add("")
    if not out["n_ok"]:
        add("No successful requests logged yet. Start `inferopt proxy`, set")
        add("ANTHROPIC_BASE_URL=http://127.0.0.1:8484 and run your workload.")
        return "\n".join(L)
    add(f"requests: {out['n_ok']} ok / {out['n_total']} total   "
        f"observed window: {out['window_days']:.2f} days")
    add(f"spend (observed): {_money(out['spend'])}   "
        f"extrapolated monthly: {_money(out['monthly_spend'])} "
        f"(x{out['factor']:.1f})")
    add("")
    add("CALL SITES")
    add("-" * 60)
    for fp, g in sorted(out["groups"].items(), key=lambda kv: -kv[1]["cost"]):
        eff = ",".join(f"{k}x{v}" for k, v in g["efforts"].most_common())
        add(f"  {fp}  n={g['n']:<4} {g['model']:<22} [{g['rail']}] "
            f"cost={_money(g['cost']):<9} cache_hit={g['hit_rate']:.0%} "
            f"stream={g['stream_share']:.0%} effort={eff}")
        add(f"      {g['hint']}")
    add("")
    if out.get("combined"):
        add("COMBINED SAVINGS PER CALL SITE (levers compose, they do NOT add)")
        add("-" * 60)
        for r in out["combined"]:
            lv = "+".join(r["safe_levers"]) or "none"
            add(f"  {r['site']}  baseline {_money(r['baseline'])}/mo")
            add(f"      zero-quality-risk ({lv}): "
                f"-> {_money(r['safe_monthly'])}/mo  "
                f"save {_money(r['safe_savings'])}/mo")
            if "full_savings" in r:
                vs = r.get("tier_validation", "untested")
                label = {"go": "VALIDATED GO",
                         "untested": "NEEDS replay validation"}.get(vs, vs)
                add(f"      + tier swap to {r['tier_alt']} ({label}): "
                    f"-> {_money(r['full_monthly'])}/mo  "
                    f"save {_money(r['full_savings'])}/mo")
            if r.get("excluded_tiers"):
                add(f"      (excluded, validated NO-GO: "
                    f"{', '.join(r['excluded_tiers'])})")
        add("")
        add(f"  TOTAL zero-quality-risk savings: "
            f"{_money(out['total_safe_savings'])}/mo")
        add(f"  TOTAL defensible today (zero-risk + VALIDATED tiers): "
            f"{_money(out['total_validated_savings'])}/mo")
        add(f"  TOTAL potential if every untested tier swap passed: "
            f"{_money(out['total_potential_savings'])}/mo  "
            f"<- do not quote this to anyone")
        add("")
    add("FINDINGS (ranked by estimated monthly savings)")
    add("  NOTE: each figure below is STANDALONE - what that one lever saves")
    add("  if applied alone. Do NOT sum them; see COMBINED above.")
    add("-" * 60)
    if not out["findings"]:
        add("  none yet - need >=3 requests per call site to fire heuristics")
    for i, f in enumerate(out["findings"], 1):
        add(f"  {i}. [{f['kind']}] site {f['site']} - "
            f"est. {_money(f['monthly_savings'])}/mo  ({f['confidence']})")
        add(f"     {f['hint']}")
        add(f"     {f['detail']}")
        add("")
    if out["tier_whatif"]:
        add("TIER WHAT-IF (requires quality validation via `inferopt replay`)")
        add("-" * 60)
        for w in out["tier_whatif"]:
            vs = w.get("validation", "untested")
            mark = {"go": "  [VALIDATED GO]", "no-go": "  [VALIDATED NO-GO]",
                    "untested": ""}.get(vs, f"  [{vs}]")
            add(f"  site {w['site']}: {w['model']} {_money(w['monthly_cost'])}/mo"
                f" -> {w['alt']} {_money(w['alt_monthly_cost'])}/mo"
                f"  (save {_money(w['monthly_savings'])}/mo){mark}"
                + ("" if vs != "untested" else w.get("note", "")))
            if w.get("evidence"):
                add(f"     evidence: {w['evidence']}")
            add(f"     validate: inferopt replay --callsite {w['site']} "
                f"--model {w['alt']} --judge")
        add("")
    for n in out["notes"]:
        add(f"NOTE: {n}")
    add("")
    add("Savings are extrapolations from the observed window; treat small")
    add("windows as directional. Cache math assumes the 5-minute TTL.")
    return "\n".join(L)


def report(con, days=30, as_json=False, price_as=None):
    out = analyze(con, days, price_as=price_as)
    if as_json:
        return json.dumps(out, indent=2, default=str)
    return render(out)
