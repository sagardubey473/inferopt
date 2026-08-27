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
    sx.add_argument("--yes", action="store_true",
                    help="skip the cost confirmation prompt")

    sub.add_parser("purge", help="delete the local database")

    args = p.parse_args()

    if args.cmd == "proxy":
        from . import proxy
        proxy.run(args.port)
    elif args.cmd == "report":
        from . import analyze, db
        print(analyze.report(db.connect(), days=args.days, as_json=args.json))
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
        from . import replay
        replay.replay(args.callsite, n=args.n, model=args.model,
                      effort=args.effort, judge=args.judge, yes=args.yes)
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
