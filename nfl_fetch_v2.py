"""
nfl_fetch_history.py
Pulls nflverse play-by-play and schedules into the cache so the model can train
on far more than four seasons.

Two things this buys beyond volume:

1. spread_line is populated 100% of the way back to 1999, so --skill-source
   spread can train on 20+ seasons.
2. The full nflverse export carries columns the trimmed cache never had:
   success, qb_dropback, qb_epa, cpoe, air_yards, xpass, pass_oe, first_down,
   series_success, drive, and the event flags (interception, fumble_lost,
   fourth_down_failed, penalty, sp, touchdown, field_goal_result). Those are
   the underlying rate stats, and they also close the gaps in the event table.

Existing pbp_YYYY.parquet files are NOT overwritten unless --overwrite is
given. Refetching every season with --overwrite is worth it once, so the whole
cache shares one schema.

Usage:
    python nfl_fetch_history.py --seasons 2015-2024
    python nfl_fetch_history.py --seasons 2015-2025 --schedules
    python nfl_fetch_history.py --seasons 2021-2025 --overwrite
    python nfl_fetch_history.py --list-columns 2015
"""

from __future__ import annotations

import argparse
import io
import ssl
import urllib.request
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from nfl_common import cache_dir

PBP_URL = ("https://github.com/nflverse/nflverse-data/releases/download/pbp/"
           "play_by_play_{season}.parquet")
GAMES_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"

# Must be present or the season is unusable for this pipeline.
REQUIRED = [
    "game_id", "play_id", "season", "week", "season_type", "game_date",
    "home_team", "away_team", "posteam", "defteam", "qtr", "down", "ydstogo",
    "yardline_100", "goal_to_go", "game_seconds_remaining",
    "half_seconds_remaining", "quarter_seconds_remaining", "posteam_score",
    "defteam_score", "score_differential", "total_home_score",
    "total_away_score", "posteam_timeouts_remaining",
    "defteam_timeouts_remaining", "play_type", "result", "yards_gained", "epa",
]

# Taken when present. Older seasons are missing some of these; that is fine,
# every consumer treats them as optional.
OPTIONAL = [
    # matches the existing 45-column cache
    "home_timeouts_remaining", "away_timeouts_remaining", "home_score",
    "away_score", "start_time", "time_of_day", "wp", "id",
    "passer_player_id", "passer_player_name", "passing_yards", "rushing_yards",
    "pass_attempt", "rush_attempt", "complete_pass", "sack",
    # underlying rate stats
    "success", "qb_dropback", "qb_epa", "cpoe", "air_yards",
    "yards_after_catch", "xpass", "pass_oe", "first_down", "series_success",
    "drive", "wpa",
    # event flags the trimmed cache never had
    "sp", "touchdown", "td_team", "return_touchdown", "field_goal_result",
    "extra_point_result", "two_point_conv_result", "safety", "interception",
    "fumble", "fumble_lost", "fourth_down_failed", "fourth_down_converted",
    "third_down_converted", "third_down_failed", "penalty", "penalty_team",
    "penalty_yards", "kick_distance", "rusher_player_id", "receiver_player_id",
    # game-level, saves a join
    "spread_line", "total_line", "div_game", "roof", "surface", "temp", "wind",
    "receive_2h_ko",
]

SCHEDULE_COLS = [
    "game_id", "season", "game_type", "week", "gameday", "weekday", "gametime",
    "home_team", "away_team", "home_score", "away_score", "result", "total",
    "overtime", "home_moneyline", "away_moneyline", "spread_line",
    "home_spread_odds", "away_spread_odds", "total_line", "under_odds",
    "over_odds", "div_game", "roof", "surface", "temp", "wind", "home_rest",
    "away_rest", "stadium",
]


def parse_seasons(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.replace(",", " ").split():
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=str, default="2015-2024",
                    help="e.g. 2015-2024 or '2015 2016 2020-2024'")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="defaults to the NFL_CACHE_DIR cache")
    ap.add_argument("--overwrite", action="store_true",
                    help="refetch seasons whose pbp file already exists")
    ap.add_argument("--schedules", action="store_true",
                    help="also rebuild schedules.parquet for every season 1999+")
    ap.add_argument("--insecure", action="store_true",
                    help="last resort: skip TLS verification if neither requests "
                         "nor certifi is installed and the Windows cert store is broken")
    ap.add_argument("--list-columns", type=int, default=None,
                    help="download one season, print its column list, write nothing")
    return ap.parse_args()


_SESSION = {"mode": None}


def _pick_downloader(insecure: bool) -> str:
    """
    Python on Windows loads CA certs from the Windows store, and a malformed
    entry there raises SSLError [ASN1: NOT_ENOUGH_DATA] before any request is
    made. Prefer requests (which ships certifi), then urllib pointed at
    certifi explicitly, then unverified only if asked.
    """
    if insecure:
        return "insecure"
    try:
        import requests  # noqa: F401
        return "requests"
    except ImportError:
        pass
    try:
        import certifi  # noqa: F401
        return "certifi"
    except ImportError:
        pass
    return "default"


def download(url: str, insecure: bool = False) -> bytes:
    if _SESSION["mode"] is None:
        _SESSION["mode"] = _pick_downloader(insecure)
        print(f"  download mode: {_SESSION['mode']}")
    mode = _SESSION["mode"]

    if mode == "requests":
        import requests
        r = requests.get(url, timeout=300, headers={"User-Agent": "nfl_fetch"})
        r.raise_for_status()
        return r.content

    ctx = None
    if mode == "certifi":
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    elif mode == "insecure":
        ctx = ssl._create_unverified_context()

    req = urllib.request.Request(url, headers={"User-Agent": "nfl_fetch"})
    with urllib.request.urlopen(req, timeout=300, context=ctx) as r:
        return r.read()


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir).expanduser() if args.out_dir else cache_dir()
    out.mkdir(parents=True, exist_ok=True)
    print(f"cache dir: {out}")

    if args.list_columns:
        raw = download(PBP_URL.format(season=args.list_columns), args.insecure)
        names = list(pq.ParquetFile(io.BytesIO(raw)).schema_arrow.names)
        print(f"{args.list_columns}: {len(names)} columns")
        print(names)
        return

    seasons = parse_seasons(args.seasons)
    print(f"seasons: {seasons}")

    wrote, skipped, failed = [], [], []
    for s in seasons:
        dest = out / f"pbp_{s}.parquet"
        if dest.exists() and not args.overwrite:
            print(f"  {s}: exists, skipping (use --overwrite to refetch)")
            skipped.append(s)
            continue
        try:
            raw = download(PBP_URL.format(season=s), args.insecure)
        except Exception as e:                      # noqa: BLE001
            print(f"  {s}: DOWNLOAD FAILED - {type(e).__name__}: {e}")
            failed.append(s)
            continue

        buf = io.BytesIO(raw)
        names = set(pq.ParquetFile(buf).schema_arrow.names)
        missing = [c for c in REQUIRED if c not in names]
        if missing:
            print(f"  {s}: SKIPPED, missing required column(s) {missing}")
            failed.append(s)
            continue

        have_opt = [c for c in OPTIONAL if c in names]
        buf.seek(0)
        df = pd.read_parquet(buf, columns=REQUIRED + have_opt)
        df.to_parquet(dest, index=False)
        gone = [c for c in OPTIONAL if c not in names]
        print(f"  {s}: {len(df):,} plays, {len(df.columns)} cols -> {dest.name}"
              + (f"   (absent: {gone})" if gone else ""))
        wrote.append(s)

    if args.schedules:
        print("\nschedules ...")
        try:
            g = pd.read_csv(io.BytesIO(download(GAMES_URL, args.insecure)))
        except Exception as e:                      # noqa: BLE001
            print(f"  schedules DOWNLOAD FAILED - {type(e).__name__}: {e}")
            g = None
    if args.schedules and g is not None:
        keep = [c for c in SCHEDULE_COLS if c in g.columns]
        miss = [c for c in SCHEDULE_COLS if c not in g.columns]
        g = g[keep]
        dest = out / "schedules.parquet"
        g.to_parquet(dest, index=False)
        cov = g.groupby("season")["spread_line"].apply(lambda x: x.notna().mean())
        first_full = cov[cov > 0.99].index.min() if (cov > 0.99).any() else None
        print(f"  {len(g):,} games, seasons {int(g.season.min())}-{int(g.season.max())}"
              f" -> {dest.name}")
        print(f"  spread_line is complete from {first_full} onward")
        if miss:
            print(f"  columns not in the source: {miss}")

    print(f"\nwrote {len(wrote)}, skipped {len(skipped)}, failed {len(failed)}")
    if failed:
        print(f"failed seasons: {failed}")


if __name__ == "__main__":
    main()
