"""
nfl_wp_edge_backtest.py
Backtests the two-stage EP -> WP model against the Kalshi live trade tape.

Rule
----
For every snap N in the 2025 regular season:
  * entry time  = time_of_day(N) + --offset seconds (default 12)
  * game state  = the NEXT snap's pre-snap row. By T+12 the play has resolved,
    so the next snap's down/distance/field position/score IS the state at entry.
    This is the same convention nfl_playtype_backtest.py uses and is leak-free.
  * model WP    = the trained WP booster on that state (calibration applied if
    nfl_wp_calibration.json has knots)
  * fill        = the NEXT print at or after entry, within --max-stale seconds
  * cost        = if we buy the side the print is on, pay the print price;
                  otherwise 101 - print price (complement + 1c to cross)
  * bet         = placed only when model_prob(side) - cost/100 >= --min-edge
                  on either side. At most one side can qualify, since the two
                  costs sum to 101c.
  * size        = edge-scaled. The edge of every qualifying bet is converted to
                  a z-score across the bet book, size multiplier = 1 + slope*z
                  clipped to [--size-min, --size-max], then the whole book is
                  rescaled so the AVERAGE bet is exactly --contracts contracts.
                  Bigger edge -> bigger bet, same average size as flat sizing.
  * settle      = 100c per winning contract, Kalshi taker fee on entry

Requires the models produced by nfl_ep_train.py and nfl_wp_train.py, and
imports nfl_common so the features are built exactly as they were in training.

Usage:
    python nfl_wp_edge_backtest.py --trades <path to kalshi_nfl_2025_regular.csv>
    python nfl_wp_edge_backtest.py --trades ... --min-edge 0.03 --contracts 50
    python nfl_wp_edge_backtest.py --trades ... --per-bet-out bets.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

from nfl_common import (
    COMMON_VERSION,
    EP_FEATURES,
    NEXT_SCORE_VALUES,
    V2_FEATURES,
    add_wp_features,
    cache_dir,
    load_pbp,
    model_dir,
    model_rows,
    prepare,
)

# Kalshi / nflverse abbreviation drift
TEAM_ALIASES = {
    "JAC": "JAX", "LAR": "LA", "WSH": "WAS", "ARZ": "ARI", "BLT": "BAL",
    "CLV": "CLE", "HST": "HOU", "SD": "LAC", "SL": "LA", "STL": "LA",
    "OAK": "LV", "LVR": "LV", "NOR": "NO", "TAM": "TB", "KAN": "KC",
    "SFO": "SF", "GNB": "GB", "NWE": "NE",
}

TRADE_COLS = ["home_team", "away_team", "game_date", "timestamp",
              "team_yes", "yes_price_cents", "taker_side"]


SCRIPT_VERSION = "2026-08-26a"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", type=str, required=True,
                    help="path to the Kalshi live trade CSV")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--offset", type=float, default=12.0,
                    help="seconds after the snap at which the bet is entered")
    ap.add_argument("--min-edge", type=float, default=0.02,
                    help="required model probability minus cost, in probability units")
    ap.add_argument("--contracts", type=int, default=100,
                    help="AVERAGE contracts per bet; individual bets are scaled by edge")
    ap.add_argument("--size-slope", type=float, default=1.0,
                    help="contracts multiplier = 1 + slope * z(edge). 0 = flat sizing")
    ap.add_argument("--size-min", type=float, default=0.20,
                    help="floor on the size multiplier before rescaling")
    ap.add_argument("--size-max", type=float, default=3.00,
                    help="cap on the size multiplier before rescaling")
    ap.add_argument("--max-stale", type=float, default=120.0,
                    help="max seconds between entry and the filling print")
    ap.add_argument("--min-cost", type=int, default=11, help="min entry cost in cents")
    ap.add_argument("--max-cost", type=int, default=90, help="max entry cost in cents")
    ap.add_argument("--fee-coeff", type=float, default=0.07, help="Kalshi taker fee coefficient")
    ap.add_argument("--no-synthetic-score-state", action="store_true",
                    help="drop scoring plays instead of modelling the post-score state")
    ap.add_argument("--pregame-role", choices=["kalshi", "spread"], default="kalshi",
                    help="how the pregame favorite is defined: the last Kalshi print "
                         "at or before the first snap, or spread_line from schedules. "
                         "kalshi falls back to spread per game when no pre-kickoff print exists")
    ap.add_argument("--include-playoffs", action="store_true")
    ap.add_argument("--no-calibration", action="store_true")
    ap.add_argument("--model-dir", type=str, default=None)
    ap.add_argument("--per-bet-out", type=str, default=None)
    return ap.parse_args()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def norm_team(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip().str.upper()
    return out.replace(TEAM_ALIASES)


def as_mask(x) -> np.ndarray:
    """Comparison result -> plain numpy bool array, NA -> False."""
    if isinstance(x, pd.Series):
        return x.fillna(False).to_numpy(dtype=bool)
    a = np.asarray(x)
    if a.dtype == object:
        return np.array([bool(v) if v is not None and v is not pd.NA else False for v in a])
    return a.astype(bool)


ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
HMS_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d{1,6}))?")


def parse_time_of_day(series: pd.Series, gameday: pd.Series) -> pd.Series:
    """
    Return a tz-aware UTC timestamp per play.

    nflverse time_of_day is either a full UTC ISO stamp or a bare HH:MM:SS.
    Detect which, and for the bare form attach the gameday, rolling to the next
    date for anything before 10:00 UTC (night games crossing midnight).
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        s = pd.to_datetime(series, utc=True, errors="coerce")
        print("  time_of_day: datetime dtype, used directly")
        return s

    raw = series.astype(str)
    iso_share = float(raw.str.match(ISO_DATE_RE).mean())
    if iso_share > 0.5:
        print(f"  time_of_day: absolute ISO route ({iso_share:.1%} dated)")
        return to_utc(series, label="time_of_day ")

    print(f"  time_of_day: bare-time route ({iso_share:.1%} dated)")
    if gameday is None:
        raise SystemExit("time_of_day is a bare clock time and game_date is absent; "
                         "cannot build an absolute wall clock")
    ext = raw.str.extract(HMS_RE)
    hh = pd.to_numeric(ext[0], errors="coerce")
    mm = pd.to_numeric(ext[1], errors="coerce")
    ss = pd.to_numeric(ext[2], errors="coerce")
    frac = pd.to_numeric(ext[3], errors="coerce").fillna(0) / 1e6
    secs = hh * 3600 + mm * 60 + ss + frac

    day = pd.to_datetime(gameday, errors="coerce", utc=True)
    day = day + pd.to_timedelta((secs < 10 * 3600).astype(float).fillna(0), unit="D")
    return day + pd.to_timedelta(secs, unit="s")


ISO_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
)


def to_utc(series: pd.Series, label: str = "") -> pd.Series:
    """
    Parse a column of ISO timestamps to tz-aware UTC.

    pandas infers ONE format from the first non-null value and coerces every
    value that does not match it to NaT. nflverse time_of_day mixes stamps with
    and without fractional seconds, so a single to_datetime call silently drops
    a large minority of rows. Each format is tried in turn against whatever is
    still unparsed.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, utc=True, errors="coerce")

    raw = series.astype("string")
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")
    todo = raw.notna()
    counts: dict[str, int] = {}

    for fmt in ISO_FORMATS:
        if not todo.any():
            break
        parsed = pd.to_datetime(raw[todo], utc=True, format=fmt, errors="coerce")
        ok = parsed.notna()
        if ok.any():
            out.loc[parsed.index[ok]] = parsed[ok]
            counts[fmt] = int(ok.sum())
            todo.loc[parsed.index[ok]] = False

    if todo.any():
        parsed = pd.to_datetime(raw[todo], utc=True, errors="coerce")
        ok = parsed.notna()
        if ok.any():
            out.loc[parsed.index[ok]] = parsed[ok]
            counts["generic"] = int(ok.sum())
            todo.loc[parsed.index[ok]] = False

    detail = ", ".join(f"{k}={v:,}" for k, v in counts.items()) or "none"
    print(f"  {label}parsed: {detail}; unparsed {int(todo.sum()):,}")
    return out


def kalshi_fee(contracts: np.ndarray, cost_cents: np.ndarray, coeff: float) -> np.ndarray:
    """Kalshi taker fee in dollars: ceil(coeff * C * P * (1-P)) to the next cent."""
    p = np.asarray(cost_cents, dtype=float) / 100.0
    raw = coeff * np.asarray(contracts, dtype=float) * p * (1.0 - p)
    return np.ceil(raw * 100.0) / 100.0


def edge_scaled_size(edge: np.ndarray, target_avg: float, slope: float,
                     lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Contracts per bet, scaled by how large the edge is relative to the rest of
    the bet book, with the mean pinned to target_avg.

    z          = (edge - mean(edge)) / sd(edge)   across all qualifying bets
    multiplier = clip(1 + slope * z, lo, hi)
    contracts  = round(target_avg * multiplier / mean(multiplier)), floor of 1

    Rounding and the floor of 1 both push the realised mean off target, so the
    scale factor is solved for in a short loop instead of applied once.
    Returns (contracts, z).
    """
    e = np.asarray(edge, dtype=float)
    sd = float(np.std(e))
    if slope == 0.0 or not np.isfinite(sd) or sd <= 0.0:
        z = np.zeros_like(e)
    else:
        z = (e - float(np.mean(e))) / sd

    mult = np.clip(1.0 + slope * z, lo, hi)
    m = float(np.mean(mult))
    if not np.isfinite(m) or m <= 0.0:
        mult = np.ones_like(e)
        m = 1.0
    mult = mult / m

    scale = target_avg
    contracts = np.maximum(1, np.rint(scale * mult)).astype(int)
    for _ in range(40):
        realised = float(np.mean(contracts))
        if realised <= 0 or abs(realised - target_avg) < 1e-9:
            break
        scale *= target_avg / realised
        new = np.maximum(1, np.rint(scale * mult)).astype(int)
        if np.array_equal(new, contracts):
            break
        contracts = new
    return contracts, z


BACKED_EVENTS = [
    "backed converted 1st down",
    "backed made FG",
    "backed scored TD",
    "backed punted",
    "backed turnover",
    "backed 2nd and 8+",
    "backed 2nd and 4 or less",
    "backed 3rd and 5+",
    "backed 3rd and 2 or less",
    "opponent converted 1st down",
    "opponent made FG",
    "opponent scored TD",
    "opponent punted",
    "opponent turnover",
    "opponent 2nd and 8+",
    "opponent 2nd and 4 or less",
    "opponent 3rd and 5+",
    "opponent 3rd and 2 or less",
    "other",
]


def classify_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label what happened on each play, and which team it happened for.

    Derived from the trimmed 45-column pbp cache, which has no penalty,
    interception, fumble_lost, fourth_down_failed, sp, touchdown or
    field_goal_result columns. Points on a play come from the running-score
    delta over the FULL play sequence, before any row is dropped.

    interception / fumble / failed-4th cannot be separated without those
    columns, so they collapse into one 'turnover' label - which is how the
    combined bucket was asked for anyway. Penalties are not derivable at all.
    """
    g = df.groupby("game_id", sort=False)
    dh = g["total_home_score"].diff()
    da = g["total_away_score"].diff()
    first = ~df["game_id"].duplicated()
    dh[first] = df.loc[first, "total_home_score"]
    da[first] = df.loc[first, "total_away_score"]
    dh = dh.fillna(0.0).to_numpy(dtype=float)
    da = da.fillna(0.0).to_numpy(dtype=float)

    home = df["home_team"].to_numpy().astype(object)
    away = df["away_team"].to_numpy().astype(object)
    pos = df["posteam"].to_numpy().astype(object)
    has_pos = df["posteam"].notna().to_numpy()

    home_pos = as_mask(df["posteam"] == df["home_team"])
    pos_pts = np.where(home_pos, dh, da)
    def_pts = np.where(home_pos, da, dh)
    opp = np.where(home_pos, away, home)

    pt = df["play_type"].astype(str)
    nxt_pos = g["posteam"].shift(-1)
    nxt_down = pd.to_numeric(g["down"].shift(-1), errors="coerce").to_numpy(dtype=float)
    nxt_pt = g["play_type"].shift(-1).astype(str)

    scrimmage = pt.isin(["run", "pass"]).to_numpy()
    same_pos = as_mask(nxt_pos == df["posteam"])
    other_pos = as_mask(nxt_pos.notna() & (nxt_pos != df["posteam"]))
    not_ko = (nxt_pt != "kickoff").to_numpy()

    ev = np.full(len(df), "other", dtype=object)
    owner = pos.copy()

    # lowest priority first; later assignments win
    fd = has_pos & same_pos & (nxt_down == 1) & not_ko & scrimmage
    ev[fd] = "first_down"

    to = has_pos & other_pos & not_ko & scrimmage & (pos_pts == 0)
    ev[to] = "turnover"

    punt = has_pos & (pt.to_numpy() == "punt")
    ev[punt] = "punt"

    fg = has_pos & (pos_pts == 3)
    ev[fg] = "field_goal"

    td = has_pos & (pos_pts >= 6)
    ev[td] = "touchdown"

    dtd = has_pos & (def_pts >= 6)          # pick-six, fumble return
    ev[dtd] = "touchdown"
    owner[dtd] = opp[dtd]

    ev[~has_pos] = "other"

    df = df.copy()
    df["ev_type"] = ev
    df["ev_team"] = owner
    return df


def next_eligible_index(gid: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Position of the next model-eligible row in the same game, or -1."""
    pos = np.arange(len(gid))
    elig_pos = pos[eligible]
    if len(elig_pos) == 0:
        return np.full(len(gid), -1)
    idx = np.searchsorted(elig_pos, pos, side="right")
    ok = idx < len(elig_pos)
    nxt = np.where(ok, elig_pos[np.clip(idx, 0, len(elig_pos) - 1)], -1)
    same = ok & (gid[np.clip(nxt, 0, len(gid) - 1)] == gid)
    return np.where(same, nxt, -1)


SCORE_BUCKETS = ["trailing 8+", "trailing 4-7", "trailing 1-3", "tied",
                 "leading 1-3", "leading 4-7", "leading 8+"]


def score_bucket(diff: np.ndarray) -> np.ndarray:
    d = np.asarray(diff, dtype=float)
    out = np.full(len(d), "", dtype=object)
    out[d <= -8] = "trailing 8+"
    out[(d >= -7) & (d <= -4)] = "trailing 4-7"
    out[(d >= -3) & (d <= -1)] = "trailing 1-3"
    out[d == 0] = "tied"
    out[(d >= 1) & (d <= 3)] = "leading 1-3"
    out[(d >= 4) & (d <= 7)] = "leading 4-7"
    out[d >= 8] = "leading 8+"
    return out


def summarize(df: pd.DataFrame, label: str) -> dict:
    n = len(df)
    if n == 0:
        return {"group": label, "bets": 0}
    staked = float(df["staked"].sum())
    return {
        "group": label,
        "bets": n,
        "avg_cost": float(df["cost_cents"].mean()),
        "avg_edge": float(df["edge"].mean()),
        "avg_size": float(df["contracts"].mean()),
        "staked": staked,
        "fees": float(df["fee"].sum()),
        "net": float(df["net"].sum()),
        "roi": float(df["net"].sum() / staked) if staked else float("nan"),
        "win_pct": float(df["won"].mean()),
    }


def print_table(rows: list[dict], title: str) -> None:
    t = pd.DataFrame([r for r in rows if r.get("bets", 0) > 0])
    if t.empty:
        print(f"\n{title}\n  (no bets)")
        return
    t = t.copy()

    # TOTAL row: sums the rows actually shown. avg_cost, avg_edge and win_pct are
    # means over bets, so weighting them by "bets" reproduces the true overall
    # figure exactly rather than averaging the group averages.
    b = t["bets"].to_numpy(dtype=float)
    tot_staked = float(t["staked"].sum())
    total = {
        "group": "TOTAL",
        "bets": int(b.sum()),
        "avg_cost": float((t["avg_cost"] * b).sum() / b.sum()) if b.sum() else 0.0,
        "avg_edge": float((t["avg_edge"] * b).sum() / b.sum()) if b.sum() else 0.0,
        "avg_size": float((t["avg_size"] * b).sum() / b.sum()) if b.sum() else 0.0,
        "staked": tot_staked,
        "fees": float(t["fees"].sum()),
        "net": float(t["net"].sum()),
        "roi": float(t["net"].sum() / tot_staked) if tot_staked else float("nan"),
        "win_pct": float((t["win_pct"] * b).sum() / b.sum()) if b.sum() else 0.0,
    }
    t = pd.concat([t, pd.DataFrame([total])], ignore_index=True)

    for c in ("staked", "fees", "net"):
        t[c] = t[c].map(lambda v: f"{v:,.0f}")
    t["avg_cost"] = t["avg_cost"].map(lambda v: f"{v:.1f}")
    t["avg_edge"] = t["avg_edge"].map(lambda v: f"{v:.3f}")
    t["avg_size"] = t["avg_size"].map(lambda v: f"{v:.0f}")
    t["roi"] = t["roi"].map(lambda v: f"{v:+.2%}")
    t["win_pct"] = t["win_pct"].map(lambda v: f"{v:.1%}")
    t["bets"] = t["bets"].map(lambda v: f"{v:,}")
    print(f"\n{title}")
    body = t.to_string(index=False).splitlines()
    width = max(len(x) for x in body)
    # rule above the TOTAL row so it reads as a sum, not another group
    print("\n".join(body[:-1] + ["-" * width, body[-1]]))


# --------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    md = Path(args.model_dir) if args.model_dir else model_dir(create=False)
    print(f"script version: {SCRIPT_VERSION}")
    print(f"nfl_common version: {COMMON_VERSION}")
    print(f"cache dir: {cache_dir()}")
    print(f"model dir: {md}")

    trades_path = Path(args.trades).expanduser()
    if not trades_path.exists():
        raise SystemExit(f"trade file not found: {trades_path}")

    # ---- models -----------------------------------------------------------
    wp_meta_path = md / "nfl_wp_meta.json"
    if not wp_meta_path.exists():
        raise SystemExit(f"{wp_meta_path} not found; run nfl_wp_train.py first")
    wp_meta = json.loads(wp_meta_path.read_text())
    feats = wp_meta["features"]
    skill_source = wp_meta.get("skill_source", "both")
    wp_model_path = md / "nfl_wp_model.json"
    print(f"WP model skill source: {skill_source}")
    print(f"WP model trained on: {wp_meta.get('train_seasons')}   "
          f"rounds: {wp_meta.get('final_rounds')}   ytd: {bool(wp_meta.get('ytd'))}")
    print(f"WP model features ({len(feats)}): {feats}")
    try:
        import datetime as _dt
        mt = _dt.datetime.fromtimestamp(wp_model_path.stat().st_mtime)
        print(f"WP model file written: {mt:%Y-%m-%d %H:%M:%S}")
    except OSError:
        pass

    ep_booster = xgb.Booster()
    ep_booster.load_model(str(md / "nfl_ep_model.json"))
    wp_booster = xgb.Booster()
    wp_booster.load_model(str(md / "nfl_wp_model.json"))

    ep_meta_path = md / "nfl_ep_meta.json"
    ep_best = None
    if ep_meta_path.exists():
        ep_best = json.loads(ep_meta_path.read_text()).get("best_iteration")

    x_knots, y_knots = [], []
    calib_path = md / "nfl_wp_calibration.json"
    if calib_path.exists() and not args.no_calibration:
        cal = json.loads(calib_path.read_text())
        x_knots = cal.get("x_knots") or []
        y_knots = cal.get("y_knots") or []
    print(f"calibration: {'on, ' + str(len(x_knots)) + ' knots' if x_knots else 'off'}")

    if args.season in (wp_meta.get("train_seasons") or []):
        print(f"WARNING: season {args.season} was in the WP model's training set - "
              f"these results are in-sample")

    # ---- play-by-play + features -----------------------------------------
    use_v2 = bool(wp_meta.get("v2")) or any(f in feats for f in V2_FEATURES)
    need_lines = skill_source in ("skill_diff", "both") or use_v2
    raw = load_pbp([args.season], with_lines=need_lines)
    print(f"raw plays: {len(raw):,}")

    sched_path = cache_dir() / "schedules.parquet"
    sched_required = ["game_id", "season", "game_type", "gameday",
                      "home_team", "away_team", "result"]
    sched_optional = ["week", "spread_line"]
    sched_names = set(pq.ParquetFile(sched_path).schema_arrow.names)
    sched_missing = [c for c in sched_required if c not in sched_names]
    if sched_missing:
        raise SystemExit(f"schedules.parquet is missing column(s): {sched_missing}")
    sched = pd.read_parquet(
        sched_path,
        columns=sched_required + [c for c in sched_optional if c in sched_names],
    )
    for c in sched_optional:
        if c not in sched.columns:
            print(f"  schedules.parquet has no {c}; dependent breakdowns will be empty")
            sched[c] = np.nan
    sched = sched[as_mask(sched["season"] == args.season)].copy()
    if not args.include_playoffs:
        sched = sched[as_mask(sched["game_type"] == "REG")].copy()
    sched["home_n"] = norm_team(sched["home_team"])
    sched["away_n"] = norm_team(sched["away_team"])
    sched["gameday"] = pd.to_datetime(sched["gameday"], errors="coerce")
    print(f"schedule games in scope: {len(sched):,}")

    keep_games = set(sched["game_id"])
    df = raw[as_mask(raw["game_id"].isin(keep_games))].copy()
    print(f"plays in scope: {len(df):,}")

    df = prepare(df)
    df = classify_events(df)
    df["_row"] = np.arange(len(df))
    print("play events: " + ", ".join(
        f"{k}={v:,}" for k, v in df["ev_type"].value_counts().items()))
    print("  NOTE: penalties are not derivable from this pbp cache "
          "(no penalty/penalty_team column), so 'opponent penalty' has no row")

    elig = model_rows(df)
    print(f"model-eligible rows: {len(elig):,}")

    dep = xgb.DMatrix(elig[EP_FEATURES].astype(float), feature_names=EP_FEATURES)
    it = (0, ep_best + 1) if ep_best is not None else None
    ep = (ep_booster.predict(dep, iteration_range=it) if it
          else ep_booster.predict(dep)) @ NEXT_SCORE_VALUES
    elig = add_wp_features(elig, ep, v2=use_v2)

    missing_feats = [f for f in feats if f not in elig.columns]
    if missing_feats:
        raise SystemExit(f"cannot build WP features {missing_feats} for {args.season}")

    dwp = xgb.DMatrix(elig[feats].astype(float), feature_names=feats)
    wp_raw = wp_booster.predict(dwp)
    wp = (np.clip(np.interp(wp_raw, x_knots, y_knots), 0.0, 1.0)
          if x_knots else wp_raw)

    df["wp_posteam"] = np.nan
    df.loc[elig["_row"].to_numpy(), "wp_posteam"] = wp
    df["wp_home"] = np.where(as_mask(df["posteam_is_home"] == 1),
                             df["wp_posteam"], 1.0 - df["wp_posteam"])

    # ---- post-score state -------------------------------------------------
    # A scoring play's next row is a PAT or kickoff, which has no down and
    # distance, so those plays would otherwise never produce a bet. At T+12 the
    # PAT and kickoff have not happened yet, so the real post-kickoff row cannot
    # be used without leaking. Build the state instead, from constants measured
    # on this data rather than assumed.
    df["synth_wp_home"] = np.nan
    df["synth_score_diff_home"] = np.nan
    if not args.no_synthetic_score_state:
        eligible = np.zeros(len(df), dtype=bool)
        eligible[elig["_row"].to_numpy()] = True
        gid_arr = df["game_id"].to_numpy()
        nxt_e = next_eligible_index(gid_arr, eligible)

        scoring = as_mask(df["ev_type"].isin(["field_goal", "touchdown"]))
        measurable = scoring & (nxt_e >= 0)

        thome = pd.to_numeric(df["total_home_score"], errors="coerce").to_numpy(dtype=float)
        taway = pd.to_numeric(df["total_away_score"], errors="coerce").to_numpy(dtype=float)
        home_arr = df["home_team"].to_numpy().astype(object)
        scorer = df["ev_team"].to_numpy().astype(object)
        scorer_is_home = scorer == home_arr
        scorer_tot = np.where(scorer_is_home, thome, taway)

        gsr = pd.to_numeric(df["game_seconds_remaining"], errors="coerce").to_numpy(dtype=float)
        yl = pd.to_numeric(df["yardline_100"], errors="coerce").to_numpy(dtype=float)

        m = np.flatnonzero(measurable)
        nx = nxt_e[m]
        ko_yardline = float(np.nanmedian(yl[nx]))
        elapsed = float(np.nanmedian(gsr[m] - gsr[nx]))
        is_td = as_mask(df["ev_type"] == "touchdown")[m]
        # score the PAT against the SCORING team's running total, not the
        # intervening row's own team (ev_team is null on a PAT row)
        prev_nx = np.clip(nx - 1, 0, len(df) - 1)
        sh = scorer_is_home[m]
        pat_gain = (np.where(sh, thome[prev_nx], taway[prev_nx])
                    - np.where(sh, thome[m], taway[m]))
        pat_points = float(np.nanmean(pat_gain[is_td])) if is_td.any() else 0.0
        print(f"post-score state measured from {len(m):,} scoring plays: "
              f"yardline_100={ko_yardline:.0f}, clock elapsed={elapsed:.0f}s, "
              f"mean PAT points={pat_points:.3f}")

        sc = np.flatnonzero(scoring)
        recv = np.where(scorer_is_home[sc], df["away_team"].to_numpy()[sc], home_arr[sc])
        recv_is_home = (recv == home_arr[sc]).astype(float)

        add = np.where(as_mask(df["ev_type"] == "touchdown")[sc], pat_points, 0.0)
        home_sc = thome[sc] + np.where(scorer_is_home[sc], add, 0.0)
        away_sc = taway[sc] + np.where(scorer_is_home[sc], 0.0, add)
        diff_home = home_sc - away_sc

        pos_to = pd.to_numeric(df["posteam_timeouts_remaining"], errors="coerce").to_numpy(dtype=float)[sc]
        def_to = pd.to_numeric(df["defteam_timeouts_remaining"], errors="coerce").to_numpy(dtype=float)[sc]
        pos_was_scorer = (scorer[sc] == df["posteam"].to_numpy().astype(object)[sc])
        new_pos_to = np.where(pos_was_scorer, def_to, pos_to)
        new_def_to = np.where(pos_was_scorer, pos_to, def_to)

        half_arr = pd.to_numeric(df["half"], errors="coerce").to_numpy(dtype=float)[sc]
        q3 = (df[as_mask(df["qtr"] == 3) & df["posteam"].notna()]
              .groupby("game_id", sort=False)["posteam"].first())
        q3_team = df["game_id"].map(q3).to_numpy().astype(object)[sc]

        spread = pd.to_numeric(df["spread_line"], errors="coerce").to_numpy(dtype=float)[sc]
        syn = pd.DataFrame({
            "down": 1.0,
            "ydstogo": 10.0,
            "yardline_100": ko_yardline,
            "goal_to_go": 0.0,
            "game_seconds_remaining": np.clip(gsr[sc] - elapsed, 0.0, None),
            "posteam_is_home": recv_is_home,
            "posteam_timeouts_remaining": new_pos_to,
            "defteam_timeouts_remaining": new_def_to,
            "score_differential": np.where(recv_is_home == 1.0, diff_home, -diff_home),
            "posteam_spread": np.where(recv_is_home == 1.0, spread, -spread),
            "receive_2h_ko": ((half_arr == 1) & (recv == q3_team)).astype(float),
            "is_second_half": (half_arr == 2).astype(float),
        })
        hsr = np.where(half_arr == 1,
                       np.clip(syn["game_seconds_remaining"].to_numpy() - 1800.0, 0.0, None),
                       syn["game_seconds_remaining"].to_numpy())
        syn["half_seconds_remaining"] = hsr
        if "skill_diff" in df.columns:
            sd = pd.to_numeric(df["skill_diff"], errors="coerce").to_numpy(dtype=float)[sc]
            syn["posteam_skill_diff"] = np.where(recv_is_home == 1.0, sd, -sd)

        # Carry every remaining model feature onto the synthetic row. Anything
        # tied to a team swaps when the scoring team was the one with the ball,
        # since possession flips on the kickoff. Anything belonging to the game
        # as a whole copies straight across. Add new feature families here.
        SWAP_PAIRS = [
            ("posteam_off_ypp", "defteam_off_ypp"),
            ("posteam_def_ypp", "defteam_def_ypp"),
            ("posteam_epa_pp", "defteam_epa_pp"),
            ("posteam_qb_changed", "defteam_qb_changed"),
        ]
        PER_GAME = ["pregame_total", "wind_mph", "temp_f", "is_indoors",
                    "plays_so_far", "sab_plays"]
        # every sab_*_diff is possession-team-minus-opponent, so it negates when
        # possession flips on the kickoff
        NEGATE_ON_FLIP = [c for c in feats
                          if c.startswith("sab_") and c.endswith("_diff")]
        DERIVED = {
            "off_ypp_matchup": ("posteam_off_ypp", "defteam_def_ypp"),
            "def_ypp_matchup": ("defteam_off_ypp", "posteam_def_ypp"),
            "epa_pp_diff": ("posteam_epa_pp", "defteam_epa_pp"),
            "qb_change_diff": ("defteam_qb_changed", "posteam_qb_changed"),
        }

        for a, b in SWAP_PAIRS:
            if a in df.columns and b in df.columns:
                va = pd.to_numeric(df[a], errors="coerce").to_numpy(dtype=float)[sc]
                vb = pd.to_numeric(df[b], errors="coerce").to_numpy(dtype=float)[sc]
                syn[a] = np.where(pos_was_scorer, vb, va)
                syn[b] = np.where(pos_was_scorer, va, vb)

        for c in NEGATE_ON_FLIP:
            if c in df.columns:
                v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)[sc]
                syn[c] = np.where(pos_was_scorer, -v, v)

        for c in PER_GAME:
            if c in df.columns and c not in syn.columns:
                syn[c] = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)[sc]

        for name, (lhs, rhs) in DERIVED.items():
            if name in feats and lhs in syn.columns and rhs in syn.columns:
                syn[name] = syn[lhs] - syn[rhs]

        dsyn = xgb.DMatrix(syn[EP_FEATURES].astype(float), feature_names=EP_FEATURES)
        ep_syn = (ep_booster.predict(dsyn, iteration_range=it) if it
                  else ep_booster.predict(dsyn)) @ NEXT_SCORE_VALUES
        syn = add_wp_features(syn, ep_syn, v2=use_v2)
        syn_missing = [f for f in feats if f not in syn.columns]
        if syn_missing:
            raise SystemExit(
                f"synthetic post-score state is missing {syn_missing}. Add each one "
                f"to SWAP_PAIRS (team-specific), PER_GAME (game-wide) or DERIVED "
                f"(computed) in nfl_wp_edge_backtest.py, or rerun with "
                f"--no-synthetic-score-state to drop scoring plays instead.")
        p_syn = wp_booster.predict(
            xgb.DMatrix(syn[feats].astype(float), feature_names=feats))
        if x_knots:
            p_syn = np.clip(np.interp(p_syn, x_knots, y_knots), 0.0, 1.0)
        wp_home_syn = np.where(recv_is_home == 1.0, p_syn, 1.0 - p_syn)

        df.loc[df.index[sc], "synth_wp_home"] = wp_home_syn
        df.loc[df.index[sc], "synth_score_diff_home"] = diff_home

    # ---- clean the snap sequence -----------------------------------------
    print("\nplay table:")
    n0 = len(df)
    if "time_of_day" not in df.columns:
        raise SystemExit("pbp export has no time_of_day; trades cannot be aligned to plays")
    df["wall"] = parse_time_of_day(
        df["time_of_day"], df["game_date"] if "game_date" in df.columns else None
    )
    df = df[df["wall"].notna()].copy()
    print(f"  {n0:,} -> {len(df):,} have time_of_day")

    n = len(df)
    df = df[df["play_type"].notna() & (df["play_type"].astype(str) != "")].copy()
    print(f"  {n:,} -> {len(df):,} are snaps (play_type present)")

    df = df.sort_values(["game_id", "play_id"], kind="mergesort").reset_index(drop=True)
    n = len(df)
    running_max = df.groupby("game_id", sort=False)["wall"].cummax()
    df = df[as_mask(df["wall"] >= running_max)].copy()
    print(f"  {n:,} -> {len(df):,} wall clock in order")

    n = len(df)
    df = df[~df.duplicated(subset=["game_id", "wall"], keep="first")].copy()
    print(f"  {n:,} -> {len(df):,} distinct timestamps")

    n = len(df)
    df = df[as_mask(df["qtr"] <= 4)].copy()
    print(f"  {n:,} -> {len(df):,} regulation")

    # ---- pair each snap with the next snap's state ------------------------
    df = df.sort_values(["game_id", "wall"], kind="mergesort").reset_index(drop=True)
    g = df.groupby("game_id", sort=False)
    df["next_wall"] = g["wall"].shift(-1)
    df["state_wp_home"] = g["wp_home"].shift(-1)
    df["state_qtr"] = g["qtr"].shift(-1)
    df["state_down"] = pd.to_numeric(g["down"].shift(-1), errors="coerce")
    df["state_ydstogo"] = pd.to_numeric(g["ydstogo"].shift(-1), errors="coerce")
    df["state_posteam"] = g["posteam"].shift(-1)
    df["state_score_diff_home"] = g["score_differential"].shift(-1) * np.where(
        as_mask(g["posteam_is_home"].shift(-1) == 1), 1.0, -1.0)

    n = len(df)
    df = df[df["next_wall"].notna()].copy()
    print(f"  {n:,} -> {len(df):,} have a following snap")

    use_syn = df["state_wp_home"].isna() & df["synth_wp_home"].notna()
    if use_syn.any():
        df.loc[use_syn, "state_wp_home"] = df.loc[use_syn, "synth_wp_home"]
        df.loc[use_syn, "state_score_diff_home"] = df.loc[use_syn, "synth_score_diff_home"]
        print(f"  {int(use_syn.sum()):,} scoring plays given a modelled post-score state")

    n = len(df)
    df = df[df["state_wp_home"].notna()].copy()
    print(f"  {n:,} -> {len(df):,} have a usable entry state")

    df["entry"] = df["wall"] + pd.to_timedelta(args.offset, unit="s")
    n = len(df)
    df = df[as_mask(df["entry"] < df["next_wall"])].copy()
    print(f"  {n:,} -> {len(df):,} entry lands before the next snap")

    if df.empty:
        raise SystemExit("no usable plays after cleaning")

    # ---- trades -----------------------------------------------------------
    print(f"\nreading {trades_path.name} ...")
    tr = pd.read_csv(trades_path, usecols=TRADE_COLS)
    print(f"  {len(tr):,} trade rows")

    tr["ts"] = to_utc(tr["timestamp"], label="timestamp ")
    n = len(tr)
    tr = tr[tr["ts"].notna()].copy()
    if len(tr) < n:
        print(f"  {n:,} -> {len(tr):,} with a parseable timestamp")

    # Resolve game_id on the unique (home, away, date) triples only - the tape
    # is millions of rows and per-row string work on it is the bottleneck.
    key_cols = ["home_team", "away_team", "game_date"]
    keys = tr[key_cols].drop_duplicates().reset_index(drop=True)
    keys["home_n"] = norm_team(keys["home_team"])
    keys["away_n"] = norm_team(keys["away_team"])
    keys["gd"] = pd.to_datetime(keys["game_date"], errors="coerce")

    lookup = sched.set_index(["home_n", "away_n"])
    gid, matched = [], 0
    for _, k in keys.iterrows():
        try:
            cand = lookup.loc[[(k["home_n"], k["away_n"])]]
        except KeyError:
            gid.append(None)
            continue
        if pd.isna(k["gd"]):
            cand2 = cand
        else:
            cand2 = cand[(cand["gameday"] - k["gd"]).abs() <= pd.Timedelta(days=1)]
        if len(cand2) == 1:
            gid.append(cand2["game_id"].iloc[0])
            matched += 1
        else:
            gid.append(None)
    keys["game_id"] = gid
    print(f"  game keys: {len(keys):,}, matched to a game_id: {matched:,}")
    if matched == 0:
        raise SystemExit("no trade game keys matched schedules.parquet; check TEAM_ALIASES")
    if matched < len(keys):
        miss = keys[keys["game_id"].isna()][key_cols].head(10)
        print("  unmatched game keys (first 10):")
        print(miss.to_string(index=False))

    tr = tr.merge(keys[key_cols + ["game_id", "home_n"]], on=key_cols, how="left")
    n = len(tr)
    tr = tr[tr["game_id"].notna()].copy()
    print(f"  {n:,} -> {len(tr):,} trades on matched games")

    tmap = {t: TEAM_ALIASES.get(str(t).strip().upper(), str(t).strip().upper())
            for t in tr["team_yes"].dropna().unique()}
    tr["team_yes_n"] = tr["team_yes"].map(tmap)

    tr = tr[as_mask(tr["game_id"].isin(set(df["game_id"])))].copy()
    print(f"  {len(tr):,} trades on games that survived play cleaning")

    tr["yes_price_cents"] = pd.to_numeric(tr["yes_price_cents"], errors="coerce")
    tr = tr[tr["yes_price_cents"].notna()].copy()
    tr["print_is_yes"] = as_mask(tr["taker_side"].astype(str).str.lower() == "yes")
    tr["yes_is_home"] = as_mask(tr["team_yes_n"] == tr["home_n"])

    # price of whichever side the print is on, and whether that side is home
    tr["print_price"] = np.where(tr["print_is_yes"],
                                 tr["yes_price_cents"],
                                 100.0 - tr["yes_price_cents"])
    tr["print_on_home"] = np.where(tr["print_is_yes"], tr["yes_is_home"], ~tr["yes_is_home"])

    tr = tr.sort_values(["game_id", "ts"], kind="mergesort").reset_index(drop=True)

    # ---- fill: next print at or after entry -------------------------------
    df["game_id"] = df["game_id"].astype(str)
    tr["game_id"] = tr["game_id"].astype(str)
    kickoff = df.groupby("game_id", sort=False)["wall"].min().rename("kickoff")
    df = df.sort_values(["entry"], kind="mergesort").reset_index(drop=True)
    tr_s = tr.sort_values(["ts"], kind="mergesort").reset_index(drop=True)

    fills = pd.merge_asof(
        df[["game_id", "entry", "wall", "qtr", "state_wp_home",
            "state_score_diff_home", "play_id", "ev_type", "ev_team",
            "home_team", "away_team", "state_down", "state_ydstogo",
            "state_posteam"]],
        tr_s[["game_id", "ts", "print_price", "print_on_home"]],
        left_on="entry", right_on="ts", by="game_id",
        direction="forward",
        tolerance=pd.Timedelta(seconds=args.max_stale),
    )
    n = len(fills)
    fills = fills[fills["ts"].notna()].copy()
    print(f"\nfills: {n:,} entries -> {len(fills):,} with a print within "
          f"{args.max_stale:.0f}s")
    fills["fill_lag"] = (fills["ts"] - fills["entry"]).dt.total_seconds()

    # ---- model vs market, on every filled entry ---------------------------
    # This is the question underneath the whole strategy: on the states we
    # actually trade, is the model a better probability than the print? If the
    # print wins here, no threshold or sizing rule can produce an edge.
    res_all = sched.set_index("game_id")["result"]
    marg = pd.to_numeric(fills["game_id"].map(res_all), errors="coerce")
    decisive = marg.notna() & (marg != 0)
    if decisive.any():
        pp_all = fills.loc[decisive, "print_price"].to_numpy(dtype=float)
        on_home_all = as_mask(fills.loc[decisive, "print_on_home"])
        p_mkt = np.where(on_home_all, pp_all, 100.0 - pp_all) / 100.0
        p_mod = fills.loc[decisive, "state_wp_home"].to_numpy(dtype=float)
        y_home = (marg[decisive].to_numpy() > 0).astype(float)

        eps = 1e-6
        def _ll(p):
            p = np.clip(p, eps, 1 - eps)
            return float(-np.mean(y_home * np.log(p) + (1 - y_home) * np.log(1 - p)))

        ll_m, ll_k = _ll(p_mod), _ll(p_mkt)
        br_m = float(np.mean((np.clip(p_mod, 0, 1) - y_home) ** 2))
        br_k = float(np.mean((np.clip(p_mkt, 0, 1) - y_home) ** 2))
        closer = float(np.mean(np.abs(p_mod - y_home) < np.abs(p_mkt - y_home)))

        print(f"\nMODEL vs MARKET on {int(decisive.sum()):,} filled entries")
        print(f"  model  logloss {ll_m:.5f}   brier {br_m:.5f}")
        print(f"  print  logloss {ll_k:.5f}   brier {br_k:.5f}")
        print(f"  model beats print by {ll_k - ll_m:+.5f} logloss "
              f"({'model' if ll_m < ll_k else 'PRINT'} is better)")
        print(f"  model closer to the outcome on {closer:.1%} of entries")
        print("  NOTE: the print is one side's traded price, so it carries about "
              "half the spread against itself")

        rows = []
        for lo, hi in ((11, 30), (31, 50), (51, 70), (71, 90)):
            hm = np.where(on_home_all, pp_all, 100.0 - pp_all)
            mk = (hm >= lo) & (hm <= hi)
            if mk.sum() == 0:
                continue
            pm, pk, yy = p_mod[mk], p_mkt[mk], y_home[mk]
            pmc, pkc = np.clip(pm, eps, 1 - eps), np.clip(pk, eps, 1 - eps)
            rows.append({
                "home price": f"{lo}-{hi}c",
                "n": f"{int(mk.sum()):,}",
                "model ll": f"{-np.mean(yy * np.log(pmc) + (1 - yy) * np.log(1 - pmc)):.5f}",
                "print ll": f"{-np.mean(yy * np.log(pkc) + (1 - yy) * np.log(1 - pkc)):.5f}",
            })
        if rows:
            print(pd.DataFrame(rows).to_string(index=False))


    # ---- cost of each side ------------------------------------------------
    pp = fills["print_price"].to_numpy(dtype=float)
    on_home = as_mask(fills["print_on_home"])
    cost_home = np.where(on_home, pp, 101.0 - pp)
    cost_away = np.where(on_home, 101.0 - pp, pp)

    p_home = fills["state_wp_home"].to_numpy(dtype=float)
    edge_home = p_home - cost_home / 100.0
    edge_away = (1.0 - p_home) - cost_away / 100.0

    take_home = edge_home >= args.min_edge
    take_away = edge_away >= args.min_edge

    fills["bet_home"] = take_home
    fills["bet_side"] = np.where(take_home, "home", np.where(take_away, "away", ""))
    fills["cost_cents"] = np.where(take_home, cost_home, cost_away)
    fills["edge"] = np.where(take_home, edge_home, edge_away)
    fills["model_p"] = np.where(take_home, p_home, 1.0 - p_home)

    bets = fills[fills["bet_side"] != ""].copy()
    print(f"edge >= {args.min_edge:.1%}: {len(bets):,} bets "
          f"({len(bets) / max(len(fills), 1):.1%} of entries)")

    n = len(bets)
    bets = bets[as_mask((bets["cost_cents"] >= args.min_cost)
                        & (bets["cost_cents"] <= args.max_cost))].copy()
    print(f"  {n:,} -> {len(bets):,} inside the {args.min_cost}-{args.max_cost}c cost band")

    if bets.empty:
        raise SystemExit("no bets qualified")

    # ---- settle -----------------------------------------------------------
    res = sched.set_index("game_id")["result"]
    margin = pd.to_numeric(bets["game_id"].map(res), errors="coerce")
    n = len(bets)
    bets = bets[margin.notna() & (margin != 0)].copy()
    margin = margin.loc[bets.index]
    print(f"  {n:,} -> {len(bets):,} on games with a decisive result")

    home_won = (margin > 0).to_numpy()
    bets["won"] = np.where(as_mask(bets["bet_home"]), home_won, ~home_won).astype(int)

    contracts, zscore = edge_scaled_size(
        bets["edge"].to_numpy(dtype=float), float(args.contracts),
        args.size_slope, args.size_min, args.size_max)
    bets["edge_z"] = zscore
    bets["contracts"] = contracts

    c = bets["contracts"].to_numpy(dtype=float)
    bets["staked"] = bets["cost_cents"] * c / 100.0
    bets["fee"] = kalshi_fee(c, bets["cost_cents"].to_numpy(), args.fee_coeff)
    bets["gross"] = np.where(bets["won"] == 1,
                             (100.0 - bets["cost_cents"]) * c / 100.0,
                             -bets["cost_cents"] * c / 100.0)
    bets["net"] = bets["gross"] - bets["fee"]

    print(f"\nsizing    slope {args.size_slope:g}, multiplier clipped to "
          f"[{args.size_min:.2f}, {args.size_max:.2f}]")
    print(f"          avg {c.mean():.1f} contracts  "
          f"min {c.min():.0f}  median {np.median(c):.0f}  max {c.max():.0f}")

    # ---- game metadata: week, month, pregame favorite ---------------------
    gmeta = sched.set_index("game_id")
    bets["week"] = pd.to_numeric(bets["game_id"].map(gmeta["week"]), errors="coerce")
    gday = pd.to_datetime(bets["game_id"].map(gmeta["gameday"]), errors="coerce")
    bets["month"] = gday.dt.strftime("%Y-%m")

    spread_home = pd.to_numeric(gmeta["spread_line"], errors="coerce")
    home_fav_spread = pd.Series(np.where(spread_home > 0, 1.0,
                                         np.where(spread_home < 0, 0.0, np.nan)),
                                index=gmeta.index)

    home_fav = home_fav_spread
    role_note = "spread_line from schedules.parquet"
    if args.pregame_role == "kalshi":
        pre = tr[["game_id", "ts", "print_price", "print_on_home"]].copy()
        pre["home_price"] = np.where(as_mask(pre["print_on_home"]),
                                     pre["print_price"], 100.0 - pre["print_price"])
        k = kickoff.reset_index().sort_values("kickoff", kind="mergesort")
        pre = pre.sort_values("ts", kind="mergesort")
        last_pre = pd.merge_asof(
            k, pre[["game_id", "ts", "home_price"]],
            left_on="kickoff", right_on="ts", by="game_id", direction="backward",
        ).set_index("game_id")
        hp = last_pre["home_price"]
        home_fav_kalshi = pd.Series(
            np.where(hp > 50, 1.0, np.where(hp < 50, 0.0, np.nan)), index=hp.index)
        home_fav = home_fav_kalshi.reindex(gmeta.index)
        n_kalshi = int(home_fav.notna().sum())
        home_fav = home_fav.fillna(home_fav_spread)
        role_note = (f"last Kalshi print before the first snap for {n_kalshi} games, "
                     f"spread_line for the rest")
    print(f"\npregame favorite defined by: {role_note}")

    hf = bets["game_id"].map(home_fav)
    bets["backed_is_fav"] = np.where(
        hf.isna(), np.nan,
        np.where(as_mask(bets["bet_home"]), hf, 1.0 - hf))
    bets["backed_score_diff"] = np.where(as_mask(bets["bet_home"]),
                                         bets["state_score_diff_home"],
                                         -bets["state_score_diff_home"])
    bets["score_bucket"] = score_bucket(bets["backed_score_diff"].to_numpy())

    at_home_arr = as_mask(bets["bet_home"])
    backed_team = np.where(at_home_arr, bets["home_team"], bets["away_team"])
    opp_team = np.where(at_home_arr, bets["away_team"], bets["home_team"])
    et = bets["ev_type"].to_numpy().astype(object)
    ew = bets["ev_team"].to_numpy().astype(object)
    on_backed = ew == backed_team
    on_opp = ew == opp_team

    backed_event = np.full(len(bets), "other", dtype=object)
    backed_event[(et == "first_down") & on_backed] = "backed converted 1st down"
    backed_event[(et == "field_goal") & on_backed] = "backed made FG"
    backed_event[(et == "touchdown") & on_backed] = "backed scored TD"
    backed_event[(et == "punt") & on_backed] = "backed punted"
    backed_event[(et == "turnover") & on_backed] = "backed turnover"
    backed_event[(et == "first_down") & on_opp] = "opponent converted 1st down"
    backed_event[(et == "field_goal") & on_opp] = "opponent made FG"
    backed_event[(et == "touchdown") & on_opp] = "opponent scored TD"
    backed_event[(et == "punt") & on_opp] = "opponent punted"
    backed_event[(et == "turnover") & on_opp] = "opponent turnover"

    # Everything still unlabelled is a play that did not convert, score, punt
    # or turn the ball over, so the offense faces 2nd, 3rd or 4th down. Bucket
    # those by the down and distance at entry, i.e. the next snap's state.
    still_other = backed_event == "other"
    sd = pd.to_numeric(bets["state_down"], errors="coerce").to_numpy(dtype=float)
    sy = pd.to_numeric(bets["state_ydstogo"], errors="coerce").to_numpy(dtype=float)
    sp = bets["state_posteam"].to_numpy().astype(object)
    has_dd = np.isfinite(sd) & np.isfinite(sy)
    off_backed = still_other & has_dd & (sp == backed_team)
    off_opp = still_other & has_dd & (sp == opp_team)

    d2_long = (sd == 2) & (sy >= 8)
    d2_short = (sd == 2) & (sy <= 4)
    d3_long = (sd == 3) & (sy >= 5)
    d3_short = (sd == 3) & (sy <= 2)

    for mask, backed_label, opp_label in (
        (d2_long, "backed 2nd and 8+", "opponent 2nd and 8+"),
        (d2_short, "backed 2nd and 4 or less", "opponent 2nd and 4 or less"),
        (d3_long, "backed 3rd and 5+", "opponent 3rd and 5+"),
        (d3_short, "backed 3rd and 2 or less", "opponent 3rd and 2 or less"),
    ):
        backed_event[off_backed & mask] = backed_label
        backed_event[off_opp & mask] = opp_label

    bets["backed_event"] = backed_event

    # ---- report -----------------------------------------------------------
    staked = float(bets["staked"].sum())
    fees = float(bets["fee"].sum())
    gross = float(bets["gross"].sum())
    net = float(bets["net"].sum())
    print("\n" + "=" * 62)
    print(f"BETS      {len(bets):,} across {bets['game_id'].nunique():,} games")
    print(f"avg cost  {bets['cost_cents'].mean():.1f}c   "
          f"avg edge {bets['edge'].mean():+.3f}   "
          f"avg model p {bets['model_p'].mean():.3f}")
    print(f"staked    ${staked:,.0f}")
    print(f"fees      ${fees:,.0f}  ({fees / staked:.2%} of stake)")
    print(f"gross     ${gross:+,.0f}  ({gross / staked:+.2%})")
    print(f"net       ${net:+,.0f}  ({net / staked:+.2%})")
    print(f"win rate  {bets['won'].mean():.1%}   "
          f"break-even {bets['cost_cents'].mean() / 100:.1%}")
    print(f"fill lag  median {bets['fill_lag'].median():.0f}s  "
          f"p90 {bets['fill_lag'].quantile(0.9):.0f}s")
    print("=" * 62)

    per_game = bets.groupby("game_id")["net"].sum()
    print(f"\nper game: mean ${per_game.mean():+,.0f}  "
          f"win rate {(per_game > 0).mean():.1%}  "
          f"bets/game {len(bets) / per_game.size:.0f}  games {per_game.size}")

    print_table([summarize(bets[bets["qtr"] == q], f"Q{int(q)}") for q in (1, 2, 3, 4)],
                "BY QUARTER (quarter of the snap)")

    edges = [(0.02, 0.03), (0.03, 0.05), (0.05, 0.08), (0.08, 0.12),
             (0.12, 0.20), (0.20, 1.01)]
    print_table(
        [summarize(bets[as_mask((bets["edge"] >= lo) & (bets["edge"] < hi))],
                   (f"{lo:.2f}+" if hi > 1 else f"{lo:.2f}-{hi:.2f}"))
         for lo, hi in edges],
        "BY MODEL EDGE",
    )

    bands = [(1, 5), (6, 10), (11, 20), (21, 30), (31, 40), (41, 50),
             (51, 60), (61, 70), (71, 80), (81, 90), (91, 94), (95, 99)]
    print_table(
        [summarize(bets[as_mask((bets["cost_cents"] >= lo) & (bets["cost_cents"] <= hi))],
                   f"{lo}-{hi}c") for lo, hi in bands],
        "BY ENTRY COST",
    )

    print_table([summarize(bets[as_mask(bets["bet_home"])], "home"),
                 summarize(bets[as_mask(~bets["bet_home"].astype(bool))], "away")],
                "BY SIDE BACKED")

    is_fav = as_mask(bets["backed_is_fav"] == 1.0)
    is_dog = as_mask(bets["backed_is_fav"] == 0.0)
    unknown = bets["backed_is_fav"].isna().to_numpy()
    print_table([summarize(bets[is_fav], "pregame favorite"),
                 summarize(bets[is_dog], "pregame underdog"),
                 summarize(bets[unknown], "pick'em / unknown")],
                "BY PREGAME ROLE OF THE BACKED TEAM")

    at_home = as_mask(bets["bet_home"])
    print_table([summarize(bets[at_home & is_fav], "home favorite"),
                 summarize(bets[at_home & is_dog], "home underdog"),
                 summarize(bets[~at_home & is_fav], "away favorite"),
                 summarize(bets[~at_home & is_dog], "away underdog")],
                "BY VENUE x PREGAME ROLE")

    weeks = sorted(w for w in bets["week"].dropna().unique())
    print_table([summarize(bets[as_mask(bets["week"] == w)], f"week {int(w)}")
                 for w in weeks],
                "BY WEEK")

    months = sorted(m for m in bets["month"].dropna().unique())
    print_table([summarize(bets[as_mask(bets["month"] == m)], m) for m in months],
                "BY MONTH")

    print_table([summarize(bets[as_mask(bets["score_bucket"] == b)], b)
                 for b in SCORE_BUCKETS],
                "BY SCORE DIFFERENTIAL OF THE BACKED TEAM AT ENTRY")

    print_table([summarize(bets[as_mask(bets["backed_event"] == e)], e)
                 for e in BACKED_EVENTS],
                "BY EVENT ON THE PLAY JUST BEFORE ENTRY")

    if args.per_bet_out:
        cols = ["game_id", "play_id", "week", "month", "qtr", "wall", "entry", "ts",
                "fill_lag", "bet_side", "backed_is_fav", "model_p", "cost_cents",
                "edge", "edge_z", "state_wp_home", "state_score_diff_home",
                "backed_score_diff", "score_bucket", "state_down", "state_ydstogo",
                "ev_type", "ev_team",
                "backed_event", "contracts", "staked", "fee",
                "gross", "net", "won"]
        bets[cols].to_csv(args.per_bet_out, index=False)
        print(f"\nwrote {args.per_bet_out}")


if __name__ == "__main__":
    main()
