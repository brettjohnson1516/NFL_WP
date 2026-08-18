"""
nfl_fetch_kalshi.py
Rebuild the Kalshi NFL trade tape from the public API — no auth, no copying.

Produces the same columns nfl_backtest_v5.py expects:
    home_team, away_team, game_date, timestamp, team_yes,
    yes_price_cents, count, taker_side, ticker, trade_id

home_team / away_team / game_date are resolved by joining the ticker's team
pair and date against schedules.parquet, so they are authoritative rather than
guessed from the ticker's ordering. Run nfl_fetch_v2.py --schedules first.

Resumable: each market's trades are cached under <out-dir>/raw/ as JSONL, so a
killed run re-uses what it already pulled. --force refetches.

    python nfl_fetch_kalshi.py --season 2025
    python nfl_fetch_kalshi.py --season 2025 --include-postseason
    python nfl_fetch_kalshi.py --probe
"""

from __future__ import annotations

import argparse
import json
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from nfl_common import cache_dir

API = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXNFLGAME"

MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}

_SESSION: dict = {"mode": None}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025,
                    help="NFL season year (the Sep-Jan season starting that year)")
    ap.add_argument("--series", default=SERIES)
    ap.add_argument("--out", default=None,
                    help="output CSV; defaults to kalshi_nfl_<season>_regular.csv "
                         "beside the cache")
    ap.add_argument("--out-dir", default=None,
                    help="working dir for the per-market cache; defaults to "
                         "<cache>/kalshi_raw")
    ap.add_argument("--include-postseason", action="store_true",
                    help="keep playoff games too; default is REG only")
    ap.add_argument("--limit", type=int, default=1000, help="trades per page")
    ap.add_argument("--sleep", type=float, default=0.15,
                    help="seconds between API calls")
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--force", action="store_true",
                    help="refetch markets already cached")
    ap.add_argument("--probe", action="store_true",
                    help="fetch a few markets, print tickers and one trade, write nothing")
    ap.add_argument("--insecure", action="store_true",
                    help="last resort if the Windows cert store is broken and "
                         "neither requests nor certifi is installed")
    return ap.parse_args()


def _pick_mode(insecure: bool) -> str:
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


def get(path: str, params: dict, insecure: bool, retries: int, sleep: float) -> dict:
    """GET with the same cert fallback as nfl_fetch_v2, plus backoff."""
    if _SESSION["mode"] is None:
        _SESSION["mode"] = _pick_mode(insecure)
        print(f"  download mode: {_SESSION['mode']}")
    url = f"{API}{path}?{urllib.parse.urlencode(params)}"
    mode = _SESSION["mode"]

    last = None
    for attempt in range(retries):
        try:
            if mode == "requests":
                import requests
                r = requests.get(url, timeout=60,
                                 headers={"User-Agent": "nfl_fetch_kalshi"})
                if r.status_code == 429:
                    raise RuntimeError("rate limited")
                r.raise_for_status()
                return r.json()
            ctx = None
            if mode == "certifi":
                import certifi
                ctx = ssl.create_default_context(cafile=certifi.where())
            elif mode == "insecure":
                ctx = ssl._create_unverified_context()
            req = urllib.request.Request(
                url, headers={"User-Agent": "nfl_fetch_kalshi"})
            with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
                return json.loads(resp.read())
        except Exception as e:                       # noqa: BLE001
            last = e
            wait = sleep * (2 ** attempt) + 0.5
            time.sleep(wait)
    raise RuntimeError(f"GET {path} failed after {retries} tries: {last}")


def list_markets(series: str, insecure: bool, retries: int, sleep: float) -> list[dict]:
    """Every market in the series, across all statuses, paginated."""
    out, cursor = [], None
    while True:
        p = {"series_ticker": series, "limit": 1000}
        if cursor:
            p["cursor"] = cursor
        d = get("/markets", p, insecure, retries, sleep)
        batch = d.get("markets", [])
        out += batch
        cursor = d.get("cursor")
        print(f"  {len(out):,} markets listed", flush=True)
        if not cursor or not batch:
            break
        time.sleep(sleep)
    return out


def parse_event(ticker: str, teams: set[str]) -> tuple[pd.Timestamp, str, str] | None:
    """
    KXNFLGAME-25SEP07BALKC-KC -> (2025-09-07, {BAL, KC}) using the team codes
    that actually appear in schedules, so no guessing about 2 vs 3 letters.
    """
    parts = ticker.split("-")
    if len(parts) < 2:
        return None
    ev = parts[1]
    if len(ev) < 8:
        return None
    yy, mon, dd, rest = ev[:2], ev[2:5], ev[5:7], ev[7:]
    if mon not in MONTHS or not (yy.isdigit() and dd.isdigit()):
        return None
    try:
        date = pd.Timestamp(2000 + int(yy), MONTHS[mon], int(dd))
    except ValueError:
        return None
    for cut in (2, 3, 4):
        a, b = rest[:cut], rest[cut:]
        if a in teams and b in teams:
            return date, a, b
    return None


def fetch_trades(ticker: str, cache: Path, args: argparse.Namespace) -> list[dict]:
    f = cache / f"{ticker}.jsonl"
    if f.exists() and not args.force:
        return [json.loads(line) for line in f.read_text().splitlines() if line]
    rows, cursor = [], None
    while True:
        p = {"ticker": ticker, "limit": args.limit}
        if cursor:
            p["cursor"] = cursor
        d = get("/markets/trades", p, args.insecure, args.retries, args.sleep)
        batch = d.get("trades", [])
        rows += batch
        cursor = d.get("cursor")
        if not cursor or not batch:
            break
        time.sleep(args.sleep)
    f.write_text("\n".join(json.dumps(r) for r in rows))
    return rows


def main() -> None:
    args = parse_args()
    cd = cache_dir()
    sched_path = cd / "schedules.parquet"
    if not sched_path.exists():
        raise SystemExit(
            f"{sched_path} not found. Run this first:\n"
            f"  python nfl_fetch_v2.py --seasons {args.season} --schedules")

    sched = pd.read_parquet(sched_path)
    sched = sched[sched["season"] == args.season]
    if not args.include_postseason and "game_type" in sched.columns:
        sched = sched[sched["game_type"] == "REG"]
    if sched.empty:
        raise SystemExit(f"no {args.season} games in schedules.parquet")
    sched["gameday"] = pd.to_datetime(sched["gameday"])
    teams = set(sched["home_team"]) | set(sched["away_team"])
    print(f"schedules: {len(sched)} games, {len(teams)} team codes, "
          f"{sched.gameday.min().date()} to {sched.gameday.max().date()}")

    out_dir = Path(args.out_dir) if args.out_dir else cd / "kalshi_raw"
    raw = out_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    print(f"listing {args.series} markets ...")
    markets = list_markets(args.series, args.insecure, args.retries, args.sleep)

    # a game can be played the day after the ticker date in UTC terms, so allow +/-1
    by_key: dict[tuple, pd.Series] = {}
    for _, g in sched.iterrows():
        for off in (-1, 0, 1):
            key = (g["gameday"] + pd.Timedelta(days=off),
                   frozenset({g["home_team"], g["away_team"]}))
            by_key.setdefault(key, g)

    wanted: list[tuple[str, pd.Series]] = []
    unmatched = 0
    for m in markets:
        t = m.get("ticker", "")
        p = parse_event(t, teams)
        if not p:
            continue
        date, a, b = p
        g = by_key.get((date, frozenset({a, b})))
        if g is None:
            unmatched += 1
            continue
        wanted.append((t, g))

    print(f"{len(markets):,} markets -> {len(wanted):,} match a {args.season} "
          f"scheduled game ({unmatched:,} parsed but off-schedule)")

    if args.probe:
        for t, g in wanted[:5]:
            print(f"  {t}  ->  {g['away_team']} @ {g['home_team']} "
                  f"{g['gameday'].date()}")
        if wanted:
            tr = fetch_trades(wanted[0][0], raw, args)
            print(f"\nfirst market has {len(tr):,} trades; one row:")
            print(json.dumps(tr[0], indent=2) if tr else "  (none)")
        return

    if not wanted:
        raise SystemExit("no markets matched; run --probe to see the tickers")

    frames, done = [], 0
    t0 = time.time()
    for t, g in wanted:
        trades = fetch_trades(t, raw, args)
        done += 1
        if trades:
            d = pd.DataFrame(trades)
            d["ticker"] = t
            d["home_team"] = g["home_team"]
            d["away_team"] = g["away_team"]
            d["game_date"] = g["gameday"].date().isoformat()
            d["team_yes"] = t.rsplit("-", 1)[-1]
            frames.append(d)
        if done % 25 == 0 or done == len(wanted):
            n = sum(len(f) for f in frames)
            print(f"  {done}/{len(wanted)} markets, {n:,} trades, "
                  f"{(time.time() - t0) / 60:.0f}m", flush=True)

    if not frames:
        raise SystemExit("no trades returned")

    df = pd.concat(frames, ignore_index=True)

    ren = {"created_time": "timestamp", "yes_price": "yes_price_cents",
           "taker_side": "taker_side", "count": "count", "trade_id": "trade_id"}
    for src, dst in ren.items():
        if src in df.columns and src != dst:
            df = df.rename(columns={src: dst})

    need = ["home_team", "away_team", "game_date", "timestamp", "team_yes",
            "yes_price_cents", "count", "taker_side", "ticker", "trade_id"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(
            f"API response is missing {missing}. Columns returned: "
            f"{sorted(df.columns)}. Run --probe to inspect one trade.")

    df = df[need].drop_duplicates(subset="trade_id")
    df = df.sort_values(["game_date", "ticker", "timestamp"])

    out = Path(args.out) if args.out else cd.parent / f"kalshi_nfl_{args.season}_regular.csv"
    df.to_csv(out, index=False)
    print(f"\n{len(df):,} trades across {df.ticker.nunique():,} markets "
          f"and {df.groupby(['home_team', 'away_team', 'game_date']).ngroups} games")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
