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
    g["uses_cache"] = any(r["uses_cache_control"] for r in rs)
    g["efforts"] = Counter(r["effort"] or "default(high)" for r in rs)
    prefixes = Counter(r["prefix"] or "" for r in rs)
    mc_prefix, mc_n = prefixes.most_common(1)[0]
    g["prefix"] = mc_prefix
    g["prefix_share"] = mc_n / len(rs)
    g["prefix_tokens"] = _est_tokens(mc_prefix)
    total_in = g["in_t"] + g["cr"] + g["cw"]
    g["hit_rate"] = g["cr"] / total_in if total_in else 0.0
    g["avg_latency"] = sum(r["latency_ms"] or 0 for r in rs) / len(rs)
    return g


def analyze(con, days=30):
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
    out["monthly_spend"] = out["spend"] * factor

    for fp, rs in groups.items():
        g = summarize_group(rs)
        out["groups"][fp] = g
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
            out["findings"].append({
                "kind": "enable-caching", "site": fp, "hint": g["hint"],
                "monthly_savings": save, "confidence": "high (arithmetic)",
                "detail": (
                    f"~{g['prefix_tokens']:,} stable prefix tokens re-billed at "
                    f"full price on every call ({g['n']} calls observed, prefix "
                    f"identical in {g['prefix_share']:.0%} of them). Add "
                    f"cache_control to the system prompt / tools block; cache "
                    f"reads bill at 10% of input. No output changes at all."),
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
            f = {"kind": "broken-cache", "site": fp, "hint": g["hint"],
                 "monthly_savings": max(save, 0),
                 "confidence": "high (arithmetic)",
                 "detail": (
                     f"cache_control is set but hit rate is "
                     f"{g['hit_rate']:.1%} across {g['n']} calls.")}
            if div:
                i, pa, pb = div
                f["detail"] += (
                    f" Prefix diverges at byte {i:,} - this is the "
                    f"invalidator:\n      A: {_ctx(pa, i)}\n"
                    f"      B: {_ctx(pb, i)}\n"
                    f"    Move volatile content below the last cache "
                    f"breakpoint (render order: tools -> system -> messages).")
            out["findings"].append(f)

        # ---- Lever 2: batch candidates ----
        if g["stream_share"] <= 0.1 and g["n"] >= 3 and monthly_cost > 0:
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
        cur_key = pricing.resolve(g["model"])
        for alt in pricing.CHEAPER_TIERS:
            if alt == cur_key:
                continue
            alt_cost = pricing.cost_usd(alt, g["in_t"], g["out_t"],
                                        g["cr"], g["cw"]) * factor
            if monthly_cost - alt_cost > 0.01 * factor:
                out["tier_whatif"].append({
                    "site": fp, "hint": g["hint"], "model": g["model"],
                    "alt": alt, "monthly_cost": monthly_cost,
                    "alt_monthly_cost": alt_cost,
                    "monthly_savings": monthly_cost - alt_cost,
                })

    # ---- effort observation ----
    if all(set(g["efforts"]) == {"default(high)"}
           for g in out["groups"].values()):
        out["notes"].append(
            "No request sets output_config.effort - everything runs at the "
            "default ('high'). Routine call sites may hold quality at "
            "'medium'/'low' with fewer thinking+output tokens; validate with "
            "`inferopt replay --effort low`.")

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
        add(f"  {fp}  n={g['n']:<4} {g['model']:<22} "
            f"cost={_money(g['cost']):<9} cache_hit={g['hit_rate']:.0%} "
            f"stream={g['stream_share']:.0%} effort={eff}")
        add(f"      {g['hint']}")
    add("")
    add("FINDINGS (ranked by estimated monthly savings)")
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
            add(f"  site {w['site']}: {w['model']} {_money(w['monthly_cost'])}/mo"
                f" -> {w['alt']} {_money(w['alt_monthly_cost'])}/mo"
                f"  (save {_money(w['monthly_savings'])}/mo)")
            add(f"     validate: inferopt replay --callsite {w['site']} "
                f"--model {w['alt']} --judge")
        add("")
    for n in out["notes"]:
        add(f"NOTE: {n}")
    add("")
    add("Savings are extrapolations from the observed window; treat small")
    add("windows as directional. Cache math assumes the 5-minute TTL.")
    return "\n".join(L)


def report(con, days=30, as_json=False):
    out = analyze(con, days)
    if as_json:
        return json.dumps(out, indent=2, default=str)
    return render(out)
