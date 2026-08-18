"""
nfl_sweep.py
Runs a whole feature sweep unattended and prints one table at the end.

For each spec it trains the WP model, runs the backtest, reads the per-bet CSV
and records the dollars. Nothing to babysit, nothing to paste between steps.

Resumable: a spec whose per-bet CSV already exists is scored from that file and
not retrained, so a killed run picks up where it stopped. --force retrains
everything.

Usage:
    python nfl_sweep.py --trades <path to kalshi csv>
    python nfl_sweep.py --trades ... --preset pairs
    python nfl_sweep.py --trades ... --specs "third_conv" "third_conv,turnover"
    python nfl_sweep.py --trades ... --preset groups --baseline wp_edge_bets_14yr_ep.csv
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

SEASONS = [str(y) for y in range(2011, 2025)]

# every saber feature short name
ALL = ["success", "epa_dropback", "epa_rush", "early_epa", "explosive", "cpoe",
       "series", "pass_oe", "disruption", "third_conv", "turnover", "penalty",
       "plays"]

# The five partners that did least badly in the pair sweep, ranked by ROI:
# epa_rush 6.02, plays 5.70, turnover 5.60, pass_oe 5.37, disruption 5.36.
# Triples are built from these rather than all 12, since all pairs of 12 is 66 runs.
TOP_PARTNERS = ["epa_rush", "plays", "turnover", "pass_oe", "disruption"]

PRESETS = {
    # each family on its own
    "groups": ["efficiency", "explosive", "tendency", "disruption",
               "situational", "volume"],
    # the one that worked, plus each other feature one at a time
    "pairs": ["third_conv"] + [f"third_conv,{f}" for f in ALL if f != "third_conv"],
    # hand-picked combinations
    "combos": ["situational,tendency,disruption",
               "situational,tendency",
               "situational,tendency,disruption,explosive",
               "third_conv,pass_oe,turnover,cpoe"],
    "all": ["" ],   # empty spec = all 13
}

from itertools import combinations  # noqa: E402

# third_conv plus every pair drawn from the five best partners: 10 runs
PRESETS["triples"] = [f"third_conv,{a},{b}"
                      for a, b in combinations(TOP_PARTNERS, 2)]

_REST = [f for f in ALL if f != "third_conv"]

# third_conv plus every pair drawn from all twelve partners: 66 runs
PRESETS["triples_all"] = [f"third_conv,{a},{b}" for a, b in combinations(_REST, 2)]

# third_conv plus every trio: C(12,3) = 220 runs
PRESETS["quads_all"] = ["third_conv," + ",".join(c) for c in combinations(_REST, 3)]

# third_conv plus every quartet: C(12,4) = 495 runs
PRESETS["quints_all"] = ["third_conv," + ",".join(c) for c in combinations(_REST, 4)]

# both, 715 runs
PRESETS["quads_and_quints"] = PRESETS["quads_all"] + PRESETS["quints_all"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True, help="path to the Kalshi trade CSV")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="pairs")
    ap.add_argument("--specs", nargs="+", default=None,
                    help="explicit --saber-only strings, overrides --preset")
    ap.add_argument("--baseline", default=None,
                    help="optional external per-bet CSV to score alongside the sweep. "
                         "Off by default — the reference is the first spec you list")
    ap.add_argument("--reference", default=None,
                    help="exact row label to measure everything against. Defaults to "
                         "the first row scored, so list your current model first")
    ap.add_argument("--seasons", nargs="+", default=SEASONS)
    ap.add_argument("--eval-season", default="2025")
    ap.add_argument("--skill-source", default="spread")
    ap.add_argument("--train-script", default="nfl_train_v3.py")
    ap.add_argument("--backtest-script", default="nfl_backtest_v5.py")
    ap.add_argument("--offsets", nargs="+", type=float, default=None,
                    help="entry lag in seconds after the snap. Given several, each "
                         "spec is trained ONCE and backtested at every offset, since "
                         "the model does not depend on the lag")
    ap.add_argument("--backtest-args", default=None,
                    help="extra flags passed straight to the backtest, e.g. "
                         "--backtest-args \"--min-edge 0.03\". Use a separate "
                         "--out-dir per setting so the resume logic keeps them apart")
    ap.add_argument("--train-args", default=None,
                    help="extra flags passed straight to the trainer")
    ap.add_argument("--out-dir", default="sweep")
    ap.add_argument("--summary", default="sweep_results.csv")
    ap.add_argument("--force", action="store_true",
                    help="retrain specs whose per-bet CSV already exists")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    return ap.parse_args()


def slug(spec: str) -> str:
    return spec.replace(",", "_").replace(" ", "") or "all13"


def score(path: Path, label: str) -> dict | None:
    if not path.exists():
        return None
    d = pd.read_csv(path)
    if d.empty:
        return None
    st, net = d["staked"].sum(), d["net"].sum()
    g = d.groupby("game_id")["net"].sum()
    row = {
        "spec": label,
        "bets": len(d),
        "games": d["game_id"].nunique(),
        "avg_cost": d["cost_cents"].mean(),
        "staked": st,
        "fees": d["fee"].sum(),
        "net": net,
        "roi": net / st if st else float("nan"),
        "win": d["won"].mean(),
        "per_game": g.mean(),
        "ahead_pct": (g > 0).mean(),
    }
    if "backed_is_fav" in d.columns:
        fav, dog = d[d.backed_is_fav == 1.0], d[d.backed_is_fav == 0.0]
        row["fav_roi"] = fav.net.sum() / fav.staked.sum() if len(fav) else float("nan")
        row["dog_roi"] = dog.net.sum() / dog.staked.sum() if len(dog) else float("nan")
    return row


def run(cmd: list[str]) -> bool:
    print("    $ " + " ".join(cmd[1:]), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-6:]
        print("    FAILED:")
        for line in tail:
            print("      " + line)
        return False
    return True


def main() -> None:
    args = parse_args()
    specs = args.specs if args.specs is not None else PRESETS[args.preset]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"specs ({len(specs)}): {specs}")
    print(f"per-bet CSVs -> {out}/  summary -> {args.summary}")
    if args.backtest_args:
        print(f"extra backtest flags: {args.backtest_args}")
    if args.train_args:
        print(f"extra trainer flags: {args.train_args}")
    if args.dry_run:
        return

    rows, failed = [], []
    base = None
    if args.baseline:
        base = score(Path(args.baseline), f"EXTERNAL {Path(args.baseline).name}")
        if base:
            rows.append(base)
            print(f"external {args.baseline}: ${base['net']:+,.0f}  {base['roi']:+.2%}")
        else:
            print(f"external {args.baseline} not found, skipping it")

    def ref_net() -> float | None:
        """First scored row, unless --reference names one."""
        if args.reference:
            for r in rows:
                if r["spec"] == args.reference:
                    return r["net"]
            return None
        return rows[0]["net"] if rows else None

    t0 = time.time()
    for i, spec in enumerate(specs, 1):
        label = spec or "all13"
        csv = out / f"bets_{slug(spec)}.csv"
        done = i - 1
        eta = ""
        if done:
            per = (time.time() - t0) / done
            eta = f"   eta {per * (len(specs) - done) / 3600:.1f}h"
        print(f"\n[{i}/{len(specs)}] {label}"
              f"   elapsed {(time.time() - t0) / 60:.0f}m{eta}", flush=True)

        offsets = args.offsets
        if offsets:
            # one training, then a backtest per lag
            targets = [(o, out / f"bets_{slug(spec)}_off{o:g}.csv") for o in offsets]
            todo = [(o, c) for o, c in targets if not c.exists() or args.force]
            if todo:
                train = [sys.executable, args.train_script,
                         "--seasons", *args.seasons,
                         "--eval-season", args.eval_season,
                         "--calib-bins", "0",
                         "--skill-source", args.skill_source]
                if spec is not None:
                    train.append("--saber")
                    if spec:
                        train += ["--saber-only", spec]
                if args.train_args:
                    train += shlex.split(args.train_args, posix=False)
                if not run(train):
                    failed.append(label)
                    continue
            for o, c in targets:
                if c.exists() and not args.force:
                    print(f"    {o:g}s already there")
                else:
                    back = [sys.executable, args.backtest_script,
                            "--trades", args.trades, "--per-bet-out", str(c),
                            "--offset", str(o)]
                    if args.backtest_args:
                        back += shlex.split(args.backtest_args, posix=False)
                    if not run(back):
                        failed.append(f"{label} @ {o:g}s")
                        continue
                r = score(c, f"{label} @ {o:g}s")
                if r is None:
                    failed.append(f"{label} @ {o:g}s")
                    continue
                rows.append(r)
                rn = ref_net()
                d = (f"   vs ref ${r['net'] - rn:+,.0f}"
                     if rn is not None and r["net"] != rn else "")
                print(f"    {o:>4g}s  {r['bets']:,} bets   ${r['net']:+,.0f}   "
                      f"{r['roi']:+.2%}{d}", flush=True)
            continue

        if csv.exists() and not args.force:
            print("    per-bet CSV already there, scoring it and moving on")
        else:
            train = [sys.executable, args.train_script,
                     "--seasons", *args.seasons,
                     "--eval-season", args.eval_season,
                     "--calib-bins", "0",
                     "--skill-source", args.skill_source]
            if spec is not None:
                train.append("--saber")
                if spec:
                    train += ["--saber-only", spec]
            if args.train_args:
                train += shlex.split(args.train_args, posix=False)
            if not run(train):
                failed.append(label)
                continue
            back = [sys.executable, args.backtest_script,
                    "--trades", args.trades, "--per-bet-out", str(csv)]
            if args.backtest_args:
                back += shlex.split(args.backtest_args, posix=False)
            if not run(back):
                failed.append(label)
                continue

        r = score(csv, label)
        if r is None:
            print("    no bets produced")
            failed.append(label)
            continue
        rows.append(r)
        best = max(rows, key=lambda x: x["net"])
        if best["spec"] != r["spec"]:
            print(f"    best so far: {best['spec']} ${best['net']:+,.0f}", flush=True)
        rn = ref_net()
        delta = (f"   vs ref ${r['net'] - rn:+,.0f}"
                 if rn is not None and r["net"] != rn else "")
        print(f"    {r['bets']:,} bets   ${r['net']:+,.0f}   {r['roi']:+.2%}{delta}",
              flush=True)

    if not rows:
        print("\nnothing scored")
        return

    t = pd.DataFrame(rows).sort_values("net", ascending=False)
    rn = ref_net()
    if rn is not None:
        t["vs_ref"] = t["net"] - rn
        ref_label = args.reference or rows[0]["spec"]
        print(f"\nreference row: {ref_label}  ${rn:+,.0f}")
    t.to_csv(args.summary, index=False)

    show = t.copy()
    show["bets"] = show.bets.map("{:,}".format)
    show["avg_cost"] = show.avg_cost.map("{:.1f}c".format)
    for c in ("staked", "fees", "net", "per_game"):
        show[c] = show[c].map("${:,.0f}".format)
    for c in ("roi", "win", "ahead_pct", "fav_roi", "dog_roi"):
        if c in show:
            show[c] = show[c].map(lambda v: f"{v:+.2%}" if pd.notna(v) else "")
    if "vs_ref" in show:
        show["vs_ref"] = show["vs_ref"].map("${:+,.0f}".format)

    cols = [c for c in ["spec", "bets", "avg_cost", "staked", "net", "roi", "win",
                        "per_game", "fav_roi", "dog_roi", "vs_ref"]
            if c in show.columns]
    print(f"\n{'=' * 70}\nSWEEP RESULTS  ({(time.time() - t0) / 60:.0f} minutes)")
    print(show[cols].to_string(index=False))
    print(f"\nwrote {args.summary}")
    if failed:
        print(f"failed specs: {failed}")
    print("NOTE: the model dir now holds whichever spec ran last. Retrain the one "
          "you want to keep before using nfl_wp_predict.py.")


if __name__ == "__main__":
    main()
