"""
nfl_bet_compare.py
Joins two per-bet CSVs from nfl_wp_edge_backtest.py on (game_id, play_id) and
decomposes the P&L difference between them.

Why the decomposition is exact
------------------------------
Both runs read the same trade tape with the same entry rule, so a play bet on
the SAME side in both runs fills at the same print, costs the same, and settles
the same. Its P&L is identical in both files and cancels.

Everything else is therefore accounted for by three buckets:

    SHARED, SIDE FLIPPED   both bet the play, on opposite teams
    ONLY IN A             A bet it, B passed
    ONLY IN B             B bet it, A passed

and  net(B) - net(A)  =  [flipped B - flipped A] + [only B] - [only A].

The script prints that identity and checks it closes.

Usage:
    python nfl_bet_compare.py wp_edge_bets.csv wp_edge_bets_ytd.csv
    python nfl_bet_compare.py A.csv B.csv --label-a baseline --label-b ytd --out merged.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

KEY = ["game_id", "play_id"]
CARRY = ["bet_side", "model_p", "cost_cents", "edge", "state_wp_home",
         "staked", "fee", "net", "won", "qtr", "backed_event", "score_bucket"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("file_a")
    ap.add_argument("file_b")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--out", default=None, help="write the joined frame to CSV")
    return ap.parse_args()


def load(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in KEY if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing {missing}")
    dup = df.duplicated(subset=KEY).sum()
    if dup:
        print(f"  WARNING: {path} has {dup:,} duplicate (game_id, play_id) rows")
    keep = KEY + [c for c in CARRY if c in df.columns]
    print(f"{label}: {path}  {len(df):,} bets  staked ${df['staked'].sum():,.0f}  "
          f"net ${df['net'].sum():+,.0f}  ROI {df['net'].sum() / df['staked'].sum():+.2%}  "
          f"win {df['won'].mean():.1%}")
    return df[keep]


def block(df: pd.DataFrame, suffix: str) -> dict:
    if df.empty:
        return {"bets": 0, "staked": 0.0, "net": 0.0, "roi": np.nan,
                "win": np.nan, "cost": np.nan, "edge": np.nan}
    st = float(df["staked" + suffix].sum())
    return {
        "bets": len(df),
        "staked": st,
        "net": float(df["net" + suffix].sum()),
        "roi": float(df["net" + suffix].sum() / st) if st else np.nan,
        "win": float(df["won" + suffix].mean()),
        "cost": float(df["cost_cents" + suffix].mean()),
        "edge": float(df["edge" + suffix].mean()),
    }


def fmt(rows: list[dict]) -> str:
    t = pd.DataFrame(rows)
    t = t[t["bets"] > 0].copy()
    if t.empty:
        return "  (none)"
    t["staked"] = t["staked"].map(lambda v: f"{v:,.0f}")
    t["net"] = t["net"].map(lambda v: f"{v:+,.0f}")
    t["roi"] = t["roi"].map(lambda v: f"{v:+.2%}" if np.isfinite(v) else "")
    t["win"] = t["win"].map(lambda v: f"{v:.1%}" if np.isfinite(v) else "")
    t["cost"] = t["cost"].map(lambda v: f"{v:.1f}" if np.isfinite(v) else "")
    t["edge"] = t["edge"].map(lambda v: f"{v:+.3f}" if np.isfinite(v) else "")
    t["bets"] = t["bets"].map(lambda v: f"{v:,}")
    return t.to_string(index=False)


def main() -> None:
    args = parse_args()

    a = load(args.file_a, args.label_a)
    b = load(args.file_b, args.label_b)

    net_a_total = None
    net_b_total = None

    m = a.merge(b, on=KEY, how="outer", suffixes=("_a", "_b"), indicator=True)
    in_a = m["_merge"].isin(["left_only", "both"])
    in_b = m["_merge"].isin(["right_only", "both"])
    net_a_total = float(m.loc[in_a, "net_a"].sum())
    net_b_total = float(m.loc[in_b, "net_b"].sum())

    both = m[m["_merge"] == "both"].copy()
    only_a = m[m["_merge"] == "left_only"].copy()
    only_b = m[m["_merge"] == "right_only"].copy()

    same_side = both[both["bet_side_a"] == both["bet_side_b"]].copy()
    flipped = both[both["bet_side_a"] != both["bet_side_b"]].copy()

    print(f"\nplays bet by both: {len(both):,}   "
          f"only {args.label_a}: {len(only_a):,}   only {args.label_b}: {len(only_b):,}")
    print(f"  of the shared plays, same side: {len(same_side):,}   "
          f"side flipped: {len(flipped):,}")

    # the same-side block must be identical in both files
    if len(same_side):
        d_net = (same_side["net_a"] - same_side["net_b"]).abs().max()
        d_cost = (same_side["cost_cents_a"] - same_side["cost_cents_b"]).abs().max()
        print(f"  same-side block: net ${same_side['net_a'].sum():+,.0f}, "
              f"max |net_a - net_b| = {d_net:.4f}, max |cost_a - cost_b| = {d_cost:.4f}")
        if d_net > 0.01:
            print("  WARNING: same-side shared bets do NOT match; the two runs did not "
                  "use the same entry rule or the same tape")

    print(f"\n{args.label_a} view")
    print(fmt([
        {"group": "shared, same side", **block(same_side, "_a")},
        {"group": "shared, side flipped", **block(flipped, "_a")},
        {"group": f"only in {args.label_a}", **block(only_a, "_a")},
    ]))

    print(f"\n{args.label_b} view")
    print(fmt([
        {"group": "shared, same side", **block(same_side, "_b")},
        {"group": "shared, side flipped", **block(flipped, "_b")},
        {"group": f"only in {args.label_b}", **block(only_b, "_b")},
    ]))

    # ---- the identity ------------------------------------------------------
    flip_a = float(flipped["net_a"].sum()) if len(flipped) else 0.0
    flip_b = float(flipped["net_b"].sum()) if len(flipped) else 0.0
    oa = float(only_a["net_a"].sum()) if len(only_a) else 0.0
    ob = float(only_b["net_b"].sum()) if len(only_b) else 0.0
    delta = net_b_total - net_a_total
    parts = (flip_b - flip_a) + ob - oa

    print("\nWHERE THE DIFFERENCE COMES FROM")
    print(f"  net {args.label_b} - net {args.label_a}          = ${delta:+,.0f}")
    print(f"    side flipped:  {args.label_b} ${flip_b:+,.0f} - "
          f"{args.label_a} ${flip_a:+,.0f} = ${flip_b - flip_a:+,.0f}")
    print(f"    only in {args.label_b}: ${ob:+,.0f}")
    print(f"    only in {args.label_a}: ${oa:+,.0f}  (given up, so contributes ${-oa:+,.0f})")
    print(f"    sum of parts                = ${parts:+,.0f}")
    print(f"    unexplained                 = ${delta - parts:+,.0f}")

    # ---- paired per-game test: is B really better in DOLLARS? --------------
    # Comparing two headline ROIs throws away the pairing. Both runs bet the
    # same games, and the bets they share settle identically, so the per-game
    # DIFFERENCE has far less noise than either run's own per-game P&L.
    ga = m.groupby("game_id")["net_a"].sum()
    gb = m.groupby("game_id")["net_b"].sum()
    games = ga.index.union(gb.index)
    a = ga.reindex(games).fillna(0.0).to_numpy(dtype=float)
    b = gb.reindex(games).fillna(0.0).to_numpy(dtype=float)
    d = b - a
    n = len(d)

    print(f"\nPER-GAME DIFFERENCE over {n:,} games "
          f"({args.label_b} minus {args.label_a})")
    print(f"  total difference   ${d.sum():+,.0f}")
    print(f"  mean per game      ${d.mean():+,.0f}")
    print(f"  {args.label_b} ahead in {float((d > 0).mean()):.1%} of games")

    # ---- how far apart are the two models on shared plays ------------------
    if len(both):
        pa = both["state_wp_home_a"].to_numpy(dtype=float)
        pb = both["state_wp_home_b"].to_numpy(dtype=float)
        ok = np.isfinite(pa) & np.isfinite(pb)
        if ok.sum() > 1:
            print(f"\nmodel win probability on shared plays: "
                  f"mean abs difference {np.mean(np.abs(pa[ok] - pb[ok])):.4f}, "
                  f"correlation {np.corrcoef(pa[ok], pb[ok])[0, 1]:.4f}, "
                  f"max {np.max(np.abs(pa[ok] - pb[ok])):.4f}")

    # ---- what the disagreement looks like ----------------------------------
    for name, blk, suf in ((f"only in {args.label_b}", only_b, "_b"),
                           (f"only in {args.label_a}", only_a, "_a")):
        if blk.empty:
            continue
        bands = [(11, 30), (31, 50), (51, 70), (71, 90)]
        rows = [{"group": f"{lo}-{hi}c",
                 **block(blk[(blk["cost_cents" + suf] >= lo)
                             & (blk["cost_cents" + suf] <= hi)], suf)}
                for lo, hi in bands]
        print(f"\n{name}, by entry cost")
        print(fmt(rows))

        if ("qtr" + suf) in blk.columns:
            rows = [{"group": f"Q{q}", **block(blk[blk["qtr" + suf] == q], suf)}
                    for q in (1, 2, 3, 4)]
            print(f"{name}, by quarter")
            print(fmt(rows))

    if args.out:
        m.drop(columns=["_merge"]).to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
