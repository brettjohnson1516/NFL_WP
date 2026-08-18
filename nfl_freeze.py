"""
nfl_freeze.py
Package the current model as a numbered release.

Copies the model artifacts, the exact scripts that produced them, and the
backtest's per-bet CSV into one folder, records the spec that built it, and
writes a manifest with a SHA256 for every file. Later you can prove which
model produced a given result, and --verify tells you whether anything has
drifted since.

    python nfl_freeze.py --name v1 --spec "third_conv,epa_rush,turnover" \
        --bets bets_check.csv --notes "12s lag, 14 seasons"

    python nfl_freeze.py --verify releases/v1
    python nfl_freeze.py --list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nfl_common import COMMON_VERSION, cache_dir, model_dir

# artifacts the live predictor needs
MODEL_FILES = ["nfl_wp_model.json", "nfl_wp_meta.json", "nfl_wp_calibration.json",
               "nfl_ep_model.json", "nfl_ep_meta.json"]

# the code that produced them, so a rebuild is possible from the folder alone
CODE_FILES = ["nfl_common.py", "nfl_train_v3.py", "nfl_backtest_v5.py",
              "nfl_wp_predict.py", "nfl_ep_v2.py", "nfl_fetch_v2.py",
              "nfl_sweep6.py", "nfl_compare_v2.py"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="release name, e.g. v1")
    ap.add_argument("--spec", default=None,
                    help="the --saber-only string this model was trained with")
    ap.add_argument("--offset", type=float, default=12.0,
                    help="entry lag in seconds the headline backtest used")
    ap.add_argument("--bets", default=None,
                    help="per-bet CSV from the backtest of THIS model")
    ap.add_argument("--notes", default="")
    ap.add_argument("--releases-dir", default="releases")
    ap.add_argument("--verify", default=None,
                    help="path to a release folder; re-hash and report drift")
    ap.add_argument("--list", action="store_true", help="list existing releases")
    return ap.parse_args()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize_bets(p: Path) -> dict:
    d = pd.read_csv(p)
    st, net = float(d["staked"].sum()), float(d["net"].sum())
    g = d.groupby("game_id")["net"].sum()
    out = {
        "bets": int(len(d)),
        "games": int(d["game_id"].nunique()),
        "avg_cost_cents": round(float(d["cost_cents"].mean()), 2),
        "staked": round(st, 2),
        "fees": round(float(d["fee"].sum()), 2),
        "net": round(net, 2),
        "roi": round(net / st, 6) if st else None,
        "win_rate": round(float(d["won"].mean()), 6),
        "net_per_game": round(float(g.mean()), 2),
        "games_ahead_pct": round(float((g > 0).mean()), 6),
    }
    if "backed_is_fav" in d.columns:
        for lab, m in (("favorite", d.backed_is_fav == 1.0),
                       ("underdog", d.backed_is_fav == 0.0)):
            s = d[m]
            if len(s):
                out[f"{lab}_roi"] = round(float(s.net.sum() / s.staked.sum()), 6)
    return out


def git_rev() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except Exception:                                # noqa: BLE001
        return None


def do_verify(folder: Path) -> None:
    man_path = folder / "manifest.json"
    if not man_path.exists():
        raise SystemExit(f"no manifest.json in {folder}")
    man = json.loads(man_path.read_text())
    print(f"release: {man.get('name')}   frozen {man.get('frozen_utc')}")
    print(f"spec: {man.get('spec')}   offset: {man.get('offset')}s")
    if man.get("backtest"):
        b = man["backtest"]
        print(f"recorded backtest: {b['bets']:,} bets  ${b['net']:+,.0f}  "
              f"{b['roi']:+.2%}")

    bad, missing = [], []
    for rel, want in man["files"].items():
        p = folder / rel
        if not p.exists():
            missing.append(rel)
        elif sha256(p) != want:
            bad.append(rel)
    print(f"\n{len(man['files'])} files recorded")
    if missing:
        print(f"MISSING: {missing}")
    if bad:
        print(f"CHANGED: {bad}")
    if not missing and not bad:
        print("all files match the manifest")

    # compare the frozen copies against what is live in the working folders
    print("\nvs your current working copies:")
    live_model = model_dir(create=False)
    drift = []
    for rel in man["files"]:
        if "/" in rel:
            sub, name = rel.split("/", 1)
            live = (live_model / name) if sub == "model" else (Path.cwd() / name)
        else:
            # top-level entries (bets.csv) have no working-copy counterpart
            continue
        if not live.exists():
            drift.append(f"  {name}: not in {live.parent}")
        elif sha256(live) != man["files"][rel]:
            drift.append(f"  {name}: DIFFERS from the frozen copy")
    print("\n".join(drift) if drift else "  identical")


def main() -> None:
    args = parse_args()
    root = Path(args.releases_dir)

    if args.list:
        if not root.exists():
            print(f"no {root}/ yet")
            return
        for f in sorted(root.iterdir()):
            m = f / "manifest.json"
            if m.exists():
                d = json.loads(m.read_text())
                b = d.get("backtest") or {}
                line = f"{f.name:<12} {d.get('frozen_utc', '')[:19]}  {d.get('spec')}"
                if b:
                    line += f"   ${b['net']:+,.0f}  {b['roi']:+.2%}"
                print(line)
        return

    if args.verify:
        do_verify(Path(args.verify))
        return

    if not args.name:
        raise SystemExit("--name is required (or use --verify / --list)")

    md = model_dir(create=False)
    dest = root / args.name
    if dest.exists():
        raise SystemExit(f"{dest} already exists; pick another --name")
    (dest / "model").mkdir(parents=True)
    (dest / "code").mkdir()

    files: dict[str, str] = {}

    print(f"model dir: {md}")
    for name in MODEL_FILES:
        src = md / name
        if not src.exists():
            print(f"  {name}: NOT FOUND, skipping")
            continue
        shutil.copy2(src, dest / "model" / name)
        files[f"model/{name}"] = sha256(src)
        print(f"  {name}")

    print("code:")
    for name in CODE_FILES:
        src = Path(name)
        if not src.exists():
            print(f"  {name}: NOT FOUND, skipping")
            continue
        shutil.copy2(src, dest / "code" / name)
        files[f"code/{name}"] = sha256(src)
        print(f"  {name}")

    backtest = None
    if args.bets:
        bp = Path(args.bets)
        if bp.exists():
            shutil.copy2(bp, dest / "bets.csv")
            files["bets.csv"] = sha256(bp)
            backtest = summarize_bets(bp)
            print(f"bets: {bp.name} -> {backtest['bets']:,} bets, "
                  f"${backtest['net']:+,.0f}, {backtest['roi']:+.2%}")
        else:
            print(f"bets: {bp} NOT FOUND")

    # the meta the model was actually written with, so the spec is not just my word
    meta = None
    mp = md / "nfl_wp_meta.json"
    if mp.exists():
        meta = json.loads(mp.read_text())

    manifest = {
        "name": args.name,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "spec": args.spec,
        "offset": args.offset,
        "notes": args.notes,
        "common_version": COMMON_VERSION,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git": git_rev(),
        "cache_dir": str(cache_dir()),
        "model_dir": str(md),
        "wp_meta": meta,
        "backtest": backtest,
        "files": files,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))

    lines = [f"# NFL win-probability model — {args.name}", ""]
    if args.notes:
        lines += [args.notes, ""]
    if meta:
        lines += [
            f"- trained on: {meta.get('train_seasons')}",
            f"- eval season: {meta.get('eval_seasons')}",
            f"- rounds: {meta.get('final_rounds')}",
            f"- skill source: {meta.get('skill_source')}",
            f"- saber spec: {meta.get('saber_only') or ('all' if meta.get('saber') else 'off')}",
            f"- features ({len(meta.get('features', []))}): {', '.join(meta.get('features', []))}",
            "",
        ]
    lines += [f"- entry lag: {args.offset:g}s after the snap", ""]
    if backtest:
        lines += [
            "## Backtest",
            f"- {backtest['bets']:,} bets across {backtest['games']} games",
            f"- staked ${backtest['staked']:,.0f}, fees ${backtest['fees']:,.0f}",
            f"- net ${backtest['net']:+,.0f}  ROI {backtest['roi']:+.2%}",
            f"- win {backtest['win_rate']:.1%} vs {backtest['avg_cost_cents'] / 100:.1%} break-even",
            f"- ${backtest['net_per_game']:+,.0f} per game, ahead in "
            f"{backtest['games_ahead_pct']:.1%} of games",
            "",
        ]
    lines += [
        "## Reproduce",
        "```powershell",
        "$env:NFL_MODEL_DIR = \"<your model dir>\"",
        f"python nfl_train_v3.py --seasons {' '.join(str(s) for s in (meta or {}).get('train_seasons', []))}"
        f" --eval-season {' '.join(str(s) for s in (meta or {}).get('eval_seasons', []))}"
        f" --calib-bins 0 --skill-source {(meta or {}).get('skill_source', 'spread')}"
        + (f" --saber --saber-only {args.spec}" if args.spec else ""),
        f"python nfl_backtest_v5.py --trades <kalshi csv> --offset {args.offset:g}"
        " --per-bet-out bets.csv",
        "```",
        "",
        f"Verify: `python nfl_freeze.py --verify {dest.as_posix()}`",
    ]
    (dest / "README.md").write_text("\n".join(lines))

    print(f"\nfroze {len(files)} files -> {dest}")
    print(f"verify with: python nfl_freeze.py --verify {dest.as_posix()}")


if __name__ == "__main__":
    main()
