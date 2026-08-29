import argparse
import os
import sys


def main():
    p = argparse.ArgumentParser(
        prog="inferopt",
        description="Local inference cost optimizer for the Anthropic API: "
                    "logging proxy + savings report + replay prover.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("proxy", help="run the logging proxy")
    sp.add_argument("--port", type=int, default=8484)

    sr = sub.add_parser("report", help="analyze logged traffic, print findings")
    sr.add_argument("--days", type=float, default=30)
    sr.add_argument("--json", action="store_true")
    sr.add_argument("--price-as", dest="price_as", metavar="MODEL",
                    help="re-price observed tokens at MODEL's rates - use "
                         "when the traffic ran on free models but you want "
                         "the real-world cost of the same waste")

    sa = sub.add_parser("analyze",
                        help="analyze an existing log file (no proxy, no "
                             "config change) and print the report")
    sa.add_argument("logfile")
    sa.add_argument("--days", type=float, default=3650)
    sa.add_argument("--json", action="store_true")
    sa.add_argument("--price-as", dest="price_as", metavar="MODEL",
                    help="re-price observed tokens at MODEL's rates")

    si = sub.add_parser("ingest",
                        help="load a log file into the persistent database")
    si.add_argument("logfile")

    sub.add_parser("demo",
                   help="run the bundled sample log and print a full report "
                        "(no API key, no data of your own, no setup)")

    sub.add_parser("callsites", help="list observed call sites")

    sx = sub.add_parser("replay",
                        help="re-run logged requests at a cheaper tier/effort")
    sx.add_argument("--callsite", required=True)
    sx.add_argument("--n", type=int, default=5)
    sx.add_argument("--model")
    sx.add_argument("--effort",
                    choices=["low", "medium", "high", "xhigh", "max"])
    sx.add_argument("--judge", action="store_true",
                    help="LLM-judge each baseline/candidate pair (costs tokens)")
    sx.add_argument("--judge-model",
                    help="judge model override (bedrock default: the baseline "
                         "row's own model; anthropic default: claude-opus-5)")
    sx.add_argument("--yes", action="store_true",
                    help="skip the cost confirmation prompt")

    sd = sub.add_parser("decide",
                        help="record a GO/NO-GO on a validated tier swap")
    sd.add_argument("--callsite", required=True)
    sd.add_argument("--model", required=True)
    g = sd.add_mutually_exclusive_group(required=True)
    g.add_argument("--go", action="store_true")
    g.add_argument("--no-go", dest="no_go", action="store_true")
    sd.add_argument("--note", default="")

    sub.add_parser("ledger", help="show recorded validation decisions")

    sub.add_parser("purge", help="delete the local database")

    args = p.parse_args()

    if args.cmd == "demo":
        import sqlite3
        from importlib import resources
        from . import analyze, db, ingest
        with resources.as_file(
                resources.files("inferopt").joinpath(
                    "data/sample-log.jsonl.gz")) as p:
            con = sqlite3.connect(":memory:")
            con.executescript(db.SCHEMA)
            con.row_factory = sqlite3.Row
            print("Running the bundled sample: one synthetic week of traffic "
                  "for a fictional\nsupport product. Nothing here is your "
                  "data and no network calls are made.\n")
            ingest.ingest(con, str(p))
            print()
            print(analyze.report(con, days=3650))
        print("\nRun it on your own logs:  inferopt analyze <your-log-file>")
        print("Format notes:             https://github.com/sagardubey473/"
              "inferopt#run-it-on-your-own-logs")
    elif args.cmd == "analyze":
        import sqlite3
        from . import analyze, db, ingest
        con = sqlite3.connect(":memory:")
        con.executescript(db.SCHEMA)
        con.row_factory = sqlite3.Row
        n, _ = ingest.ingest(con, args.logfile)
        if not n:
            print("nothing usable in that file - see the format notes in "
                  "the README (`inferopt analyze --help`)")
            return
        print()
        print(analyze.report(con, days=args.days, as_json=args.json,
                             price_as=args.price_as))
    elif args.cmd == "ingest":
        from . import db, ingest
        ingest.ingest(db.connect(), args.logfile)
        print("run `inferopt report` to analyze it")
    elif args.cmd == "proxy":
        try:
            from . import proxy
        except ImportError as e:
            sys.exit(f"the live proxy needs extra packages ({e.name}). "
                     f"Install with:  pip install 'inferopt[proxy]'  "
                     f"(add [bedrock] for the AWS rail). The `analyze` and "
                     f"`report` commands need nothing extra.")
        proxy.run(args.port)
    elif args.cmd == "report":
        from . import analyze, db
        print(analyze.report(db.connect(), days=args.days, as_json=args.json,
                             price_as=args.price_as))
    elif args.cmd == "callsites":
        from . import analyze, db
        _, _, groups = analyze.load(db.connect(), days=3650)
        if not groups:
            print("no traffic logged yet")
            return
        for fp, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            g = analyze.summarize_group(rs)
            print(f"{fp}  n={g['n']:<5} {g['model']:<24} "
                  f"cost=${g['cost']:.4f}  {g['hint']}")
    elif args.cmd == "replay":
        try:
            from . import replay
        except ImportError as e:
            sys.exit(f"replay needs extra packages ({e.name}). Install with: "
                     f"pip install 'inferopt[replay]'")
        replay.replay(args.callsite, n=args.n, model=args.model,
                      effort=args.effort, judge=args.judge, yes=args.yes,
                      judge_model=args.judge_model)
    elif args.cmd == "decide":
        import time
        from . import db
        con = db.connect()
        decision = "go" if args.go else "no-go"
        row = con.execute(
            "SELECT * FROM validations WHERE callsite=? AND alt_model=? "
            "ORDER BY ts DESC LIMIT 1", (args.callsite, args.model)).fetchone()
        if row is None:
            print(f"warning: no replay evidence on record for {args.model} "
                  f"at {args.callsite}. Recording the decision anyway, but "
                  f"run `inferopt replay --judge` first if you haven't.")
            con.execute(
                "INSERT INTO validations (ts,callsite,alt_model,n,decision,"
                "note) VALUES (?,?,?,0,?,?)",
                (time.time(), args.callsite, args.model, decision,
                 args.note or "decision recorded without replay evidence"))
        else:
            con.execute("UPDATE validations SET decision=?, note=? WHERE id=?",
                        (decision, args.note or row["note"], row["id"]))
        con.commit()
        print(f"recorded {decision.upper()}: {args.model} at {args.callsite}")
        print("the report's COMBINED section will now honor this.")
    elif args.cmd == "ledger":
        from . import db
        con = db.connect()
        rows = con.execute(
            "SELECT * FROM validations ORDER BY ts DESC").fetchall()
        if not rows:
            print("no validations recorded yet")
            return
        for r in rows:
            print(f"{(r['decision'] or '?').upper():<8} {r['callsite']}  "
                  f"{r['alt_model']}  n={r['n']} "
                  f"judge {r['equivalent']}e/{r['better']}b/{r['worse']}w"
                  + (f"  MISMATCHES={r['mismatches']}" if r["mismatches"]
                     else "") + (f"  ({r['note']})" if r["note"] else ""))
    elif args.cmd == "purge":
        from . import db
        path = db.DEFAULT_DB
        if os.path.exists(path):
            if input(f"delete {path}? [y/N] ").strip().lower() == "y":
                for suffix in ("", "-wal", "-shm"):
                    try:
                        os.remove(path + suffix)
                    except FileNotFoundError:
                        pass
                print("deleted")
        else:
            print("nothing to delete")


if __name__ == "__main__":
    sys.exit(main())
