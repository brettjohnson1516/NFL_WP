"""
nfl_kalshi_v6.py
Rebuild the Kalshi NFL trade tape from the public API — no auth, no copying.

Output columns match what the backtest reads:
    home_team, away_team, game_date, timestamp, team_yes,
    yes_price_cents, count, taker_side, ticker, trade_id

What the live API actually does, measured rather than assumed:
  * /markets returns only ~162 currently-open markets for this series and does
    not paginate, so it cannot reach a finished season.
  * min_close_ts / max_close_ts on /markets returns 0 items.
  * /events pages with a cursor, but without a status it returns only the
    current season, and nested markets come back only for OPEN events.
So: page /events once per status, filter to the season on the EVENT ticker,
then pull each event's markets by event_ticker when nesting is empty.

Resumable: per-market trades cached as JSONL under <cache>/kalshi_raw/raw/.

    python nfl_kalshi_v6.py --season 2025 --probe
    python nfl_kalshi_v6.py --season 2025
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

# Kalshi ticker code -> nflverse schedule code. ONLY codes actually observed
# in KXNFLGAME tickers. Do not add speculative entries: every alias widens the
# set the splitter accepts, and a wrong one silently steals a valid split
# (CLV as an alias for CLE turned LACLV into LA/CLE instead of LAC/LV).
TEAM_ALIAS = {
    "JAC": "JAX",
    "LAR": "LA",
}

_SESSION: dict = {"mode": None}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--series", default=SERIES)
    ap.add_argument("--status", default="settled,closed,open",
                    help="event statuses to page; the API returns only the "
                         "current season if you omit this")
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--include-postseason", action="store_true")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--sleep", type=float, default=0.15)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--probe", action="store_true",
                    help="show what matched and one raw trade, write nothing")
    ap.add_argument("--diag-match", action="store_true",
                    help="print the scheduled games with no matching event "
                         "and the leftover event tickers, then stop")
    ap.add_argument("--diag-trades", action="store_true",
                    help="probe /trades endpoint shapes, write nothing")
    ap.add_argument("--insecure", action="store_true")
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
                                 headers={"User-Agent": "nfl_kalshi"})
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
            req = urllib.request.Request(url, headers={"User-Agent": "nfl_kalshi"})
            with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
                return json.loads(resp.read())
        except Exception as e:                       # noqa: BLE001
            last = e
            time.sleep(sleep * (2 ** attempt) + 0.5)
    raise RuntimeError(f"GET {path} failed after {retries} tries: {last}")


_CUTOFF: dict = {}


def load_cutoff(insecure: bool, retries: int, sleep: float) -> dict:
    """
    GET /historical/cutoff gives the live/historical boundary timestamps.
    Anything older than trades_created_ts is only on /historical/trades;
    anything settled before market_settled_ts is only on /historical/markets.
    """
    if _CUTOFF:
        return _CUTOFF
    try:
        d = get("/historical/cutoff", {}, insecure, retries, sleep)
    except Exception as e:                           # noqa: BLE001
        print(f"  /historical/cutoff failed ({str(e)[:60]}); "
              "will try live first and fall back to historical")
        _CUTOFF["_failed"] = True
        return _CUTOFF
    if isinstance(d.get("cutoff"), dict):
        d = d["cutoff"]
    _CUTOFF.update(d)
    return _CUTOFF


def _cutoff_ts(field: str) -> int | None:
    v = _CUTOFF.get(field)
    if v is None:
        return None
    if isinstance(v, str):
        try:
            return int(pd.Timestamp(v).timestamp())
        except Exception:                            # noqa: BLE001
            return None
    return int(v)


def ev_ticker(ev: dict) -> str:
    return ev.get("event_ticker") or ev.get("ticker") or ""


def list_events(series: str, statuses: list[str], insecure: bool,
                retries: int, sleep: float) -> list[dict]:
    seen: dict[str, dict] = {}
    for status in statuses:
        cursor, pages, n0 = None, 0, len(seen)
        while True:
            p = {"series_ticker": series, "limit": 200,
                 "with_nested_markets": "true", "status": status}
            if cursor:
                p["cursor"] = cursor
            try:
                d = get("/events", p, insecure, retries, sleep)
            except Exception as e:                   # noqa: BLE001
                print(f"  status={status}: FAILED {str(e)[:80]}")
                break
            events = d.get("events", [])
            for ev in events:
                t = ev_ticker(ev)
                if not t:
                    continue
                # keep whichever copy actually carries markets
                if t not in seen or (not seen[t].get("markets")
                                     and ev.get("markets")):
                    seen[t] = ev
            pages += 1
            cursor = d.get("cursor")
            if not cursor or not events:
                break
            time.sleep(sleep)
        print(f"  status={status}: {pages} pages, {len(seen) - n0:,} new "
              f"({len(seen):,} events total)", flush=True)
    return list(seen.values())


def _page_markets(path: str, event_ticker: str, insecure: bool, retries: int,
                  sleep: float) -> list[dict]:
    out, cursor = [], None
    while True:
        p = {"event_ticker": event_ticker, "limit": 1000}
        if cursor:
            p["cursor"] = cursor
        try:
            d = get(path, p, insecure, retries, sleep)
        except Exception:                            # noqa: BLE001
            return out
        batch = d.get("markets", [])
        out += batch
        cursor = d.get("cursor")
        if not cursor or not batch:
            break
        time.sleep(sleep)
    return out


def markets_for_event(event_ticker: str, insecure: bool, retries: int,
                      sleep: float) -> tuple[list[dict], str]:
    """
    Settled markets older than market_settled_ts are dropped from /markets and
    only exist on /historical/markets, so try both.
    """
    m = _page_markets("/markets", event_ticker, insecure, retries, sleep)
    if m:
        return m, "/markets"
    m = _page_markets("/historical/markets", event_ticker, insecure, retries,
                      sleep)
    return m, "/historical/markets"


def markets_from_event_detail(event_ticker: str, insecure: bool, retries: int,
                              sleep: float) -> list[dict]:
    """
    GET /events/{ticker} returns the event's markets nested even when the event
    is settled, which the /events LIST and /markets?event_ticker= calls do not.
    Response shape varies, so read markets from either the top level or the
    nested event object.
    """
    try:
        d = get(f"/events/{event_ticker}", {"with_nested_markets": "true"},
                insecure, retries, sleep)
    except Exception:                                # noqa: BLE001
        return []
    if isinstance(d.get("markets"), list) and d["markets"]:
        return d["markets"]
    ev = d.get("event") or {}
    if isinstance(ev.get("markets"), list):
        return ev["markets"]
    return []


def parse_event(ticker: str, teams: set[str]):
    """
    KXNFLGAME-25SEP07BALKC        -> (2025-09-07, BAL, KC, BAL, KC)
    KXNFLGAME-25SEP07BALKC-KC     -> same; the side segment is ignored here.

    The team blob is split by trying 2/3/4-char cuts against the codes that
    appear in schedules.parquet PLUS the Kalshi-only spellings in TEAM_ALIAS,
    so JAC/LAR split correctly. Returns the schedule codes first (for matching)
    and the raw ticker codes second (for building market tickers).
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
    known = set(teams) | set(TEAM_ALIAS)
    cands = []
    for cut in (2, 3, 4):
        a, b = rest[:cut], rest[cut:]
        if a in known and b in known:
            # a split using real schedule codes beats one that needs an alias,
            # so LACLV resolves to LAC/LV rather than LA/<alias>
            score = (a in teams) + (b in teams)
            cands.append((score, cut, a, b))
    if not cands:
        return None
    _, _, a, b = max(cands, key=lambda c: (c[0], -c[1]))
    return (date, TEAM_ALIAS.get(a, a), TEAM_ALIAS.get(b, b), a, b)


def _page_trades(path: str, ticker: str, args: argparse.Namespace) -> list[dict]:
    rows, cursor = [], None
    while True:
        p = {"ticker": ticker, "limit": args.limit}
        if cursor:
            p["cursor"] = cursor
        try:
            d = get(path, p, args.insecure, args.retries, args.sleep)
        except Exception:                            # noqa: BLE001
            return rows
        batch = d.get("trades", [])
        rows += batch
        cursor = d.get("cursor")
        if not cursor or not batch:
            break
        time.sleep(args.sleep)
    return rows


def fetch_trades(ticker: str, cache: Path, args: argparse.Namespace,
                 game_ts: int | None = None) -> tuple[list[dict], str]:
    """
    Trades older than the trades_created_ts cutoff are dropped from
    /markets/trades and live only on /historical/trades. Route on the game's
    own timestamp so a mixed-age pull works, and try the other endpoint if the
    routed one comes back empty.
    """
    f = cache / f"{ticker}.jsonl"
    if f.exists() and not args.force:
        return [json.loads(x) for x in f.read_text().splitlines() if x], "cache"

    cut = _cutoff_ts("trades_created_ts")
    hist_first = (cut is not None and game_ts is not None and game_ts < cut)
    order = (["/historical/trades", "/markets/trades"] if hist_first
             else ["/markets/trades", "/historical/trades"])

    rows, used = [], order[0]
    for path in order:
        rows = _page_trades(path, ticker, args)
        used = path
        if rows:
            break
    if rows:
        f.write_text("\n".join(json.dumps(r) for r in rows))
    return rows, used


def raw_get(path: str, params: dict, args) -> dict | str:
    try:
        return get(path, params, args.insecure, 2, args.sleep)
    except Exception as e:                           # noqa: BLE001
        return f"ERROR {type(e).__name__}: {str(e)[:70]}"


def diag_trades(settled_ticker: str, open_ticker: str, args) -> None:
    """
    Find out why a correctly-formed ticker returns nothing.

    Prints the response's TOP-LEVEL KEYS as well: if the payload nests trades
    under a different name, d.get("trades") silently yields zero and everything
    looks empty for no visible reason.
    """
    for label, tk in (("settled 2025", settled_ticker),
                      ("open 2026", open_ticker)):
        if not tk:
            continue
        print(f"\n--- {label}: {tk}")
        trials = [
            ("ticker only", "/markets/trades", {"ticker": tk}),
            ("ticker+limit", "/markets/trades", {"ticker": tk, "limit": 100}),
            ("market_ticker", "/markets/trades", {"market_ticker": tk,
                                                  "limit": 100}),
            ("path form", f"/markets/{tk}/trades", {"limit": 100}),
            ("bare /trades", "/trades", {"ticker": tk, "limit": 100}),
            ("historical", "/historical/trades", {"ticker": tk, "limit": 100}),
            ("hist market", "/historical/markets", {"tickers": tk}),
        ]
        for name, path, params in trials:
            d = raw_get(path, params, args)
            if isinstance(d, str):
                print(f"  {name:16} {d}")
                continue
            keys = list(d.keys())
            lens = {k: len(v) for k, v in d.items() if isinstance(v, list)}
            print(f"  {name:16} keys={keys} list_lens={lens}")
            for v in d.values():
                if isinstance(v, list) and v:
                    print(f"      sample: {json.dumps(v[0])[:170]}")
                    break
            time.sleep(args.sleep)


def main() -> None:
    args = parse_args()
    cd = cache_dir()
    sp = cd / "schedules.parquet"
    if not sp.exists():
        raise SystemExit(f"{sp} not found. Run nfl_fetch_v2.py --schedules first.")

    sched = pd.read_parquet(sp)
    sched = sched[sched["season"] == args.season]
    if not args.include_postseason and "game_type" in sched.columns:
        sched = sched[sched["game_type"] == "REG"]
    if sched.empty:
        raise SystemExit(f"no {args.season} games in schedules.parquet")
    sched["gameday"] = pd.to_datetime(sched["gameday"])
    teams = set(sched["home_team"]) | set(sched["away_team"])
    print(f"schedules: {len(sched)} games, {len(teams)} team codes, "
          f"{sched.gameday.min().date()} to {sched.gameday.max().date()}")

    cut = load_cutoff(args.insecure, args.retries, args.sleep)
    if cut and not cut.get("_failed"):
        print("  historical cutoffs: " + ", ".join(
            f"{k}={v}" for k, v in sorted(cut.items())))

    out_dir = Path(args.out_dir) if args.out_dir else cd / "kalshi_raw"
    raw = out_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    # Earlier versions cached a file even when the fetch returned nothing, so a
    # zero-byte file means "not fetched", not "no trades". Drop those or every
    # later run reads the empty cache and reports 0.
    pruned = 0
    for f in raw.glob("*.jsonl"):
        if f.stat().st_size == 0:
            f.unlink()
            pruned += 1
    print(f"  trade cache: {raw}")
    if pruned:
        print(f"  removed {pruned:,} empty cache files")

    statuses = [x.strip() for x in args.status.split(",") if x.strip()]
    print(f"listing {args.series} events, statuses {statuses} ...")
    events = list_events(args.series, statuses, args.insecure, args.retries,
                         args.sleep)

    # kickoff can roll into the next UTC day, so allow +/-1
    by_key: dict[tuple, pd.Series] = {}
    for _, g in sched.iterrows():
        for off in (-1, 0, 1):
            by_key.setdefault(
                (g["gameday"] + pd.Timedelta(days=off),
                 frozenset({g["home_team"], g["away_team"]})), g)

    by_year: dict[int, int] = {}
    matched: list[tuple[dict, pd.Series, tuple[str, str]]] = []
    unparsed: list[str] = []
    leftover: list[tuple[str, object, str, str]] = []
    for ev in events:
        t = ev_ticker(ev)
        p = parse_event(t, teams)
        if not p:
            unparsed.append(t)
            continue
        date, a, b, ra, rb = p
        by_year[date.year] = by_year.get(date.year, 0) + 1
        g = by_key.get((date, frozenset({a, b})))
        if g is not None:
            matched.append((ev, g, (ra, rb)))
        else:
            leftover.append((t, date.date(), a, b))

    print(f"  {len(events):,} events; parsed dates by year "
          f"{dict(sorted(by_year.items()))}")
    print(f"  {len(matched):,} events match a {args.season} scheduled game")
    if not matched:
        raise SystemExit(
            "no events matched. If the year breakdown above shows no "
            f"{args.season}, the API is not returning that season's events for "
            "these statuses — paste the output and we adjust the status list.")

    if args.diag_match:
        hit = {(g["home_team"], g["away_team"],
                g["gameday"].date()) for _, g, _ in matched}
        miss = [g for _, g in sched.iterrows()
                if (g["home_team"], g["away_team"],
                    g["gameday"].date()) not in hit]
        print(f"\n{len(miss)} scheduled games with no matching event:")
        for g in miss:
            wk = g["week"] if "week" in g.index else "?"
            print(f"  wk{wk:>3}  {g['gameday'].date()}  "
                  f"{g['away_team']} @ {g['home_team']}")

        # only the leftovers near the season are informative
        near = [x for x in leftover
                if sched.gameday.min().date() - pd.Timedelta(days=7).to_pytimedelta()
                <= x[1]
                <= sched.gameday.max().date() + pd.Timedelta(days=7).to_pytimedelta()]
        print(f"\n{len(near)} event tickers in the season window that matched "
              f"nothing ({len(leftover)} leftovers overall):")
        for t, d, a, b in sorted(near, key=lambda x: x[1]):
            print(f"  {d}  {a}/{b}   {t}")

        if unparsed:
            print(f"\n{len(unparsed)} event tickers the parser could not "
                  f"split at all:")
            for t in unparsed[:40]:
                print(f"  {t}")
        return

    # a probe only needs a couple of events to show the shape of things
    work = matched[:3] if args.probe else matched

    # /markets?event_ticker= returns nothing for settled events, so do not rely
    # on it. A market ticker is just "<event ticker>-<team code>", and both team
    # codes are already known from the schedule row, so build them directly and
    # let the trades call confirm they exist.
    wanted: list[tuple[str, pd.Series]] = []
    src_count = {"list nesting": 0, "event detail": 0, "constructed": 0}
    for i, (ev, g, raw_codes) in enumerate(work, 1):
        t_ev = ev_ticker(ev)

        mkts = ev.get("markets") or []
        src = "list nesting"
        if not mkts:
            mkts = markets_from_event_detail(t_ev, args.insecure, args.retries,
                                             args.sleep)
            src = "event detail"
        if not mkts:
            mkts, endpoint = markets_for_event(t_ev, args.insecure,
                                               args.retries, args.sleep)
            src = f"{endpoint}?event_ticker"
            if src not in src_count:
                src_count[src] = 0

        tickers = [m.get("ticker") for m in mkts if m.get("ticker")]
        if not tickers:
            tickers = [f"{t_ev}-{c}" for c in raw_codes]
            src = "constructed"

        for t in tickers:
            wanted.append((t, g))

        src_count[src] += len(tickers)

        if not args.probe and (i % 25 == 0 or i == len(work)):
            print(f"  resolving markets {i}/{len(work)}", flush=True)
        time.sleep(args.sleep)

    shown = ", ".join(f"{k}: {v:,}" for k, v in src_count.items() if v)
    print(f"  {len(wanted):,} market tickers ({shown})")

    if args.diag_trades:
        settled = wanted[0][0] if wanted else ""
        open_t = ""
        for ev in events:
            t = ev_ticker(ev)
            p2 = parse_event(t, teams)
            if p2 and p2[0].year == 2026:
                open_t = f"{t}-{p2[3]}"
                break
        diag_trades(settled, open_t, args)
        return

    if args.probe:
        t_ev0 = ev_ticker(work[0][0])
        print(f"\nraw /events/{t_ev0}:")
        d0 = raw_get(f"/events/{t_ev0}", {"with_nested_markets": "true"}, args)
        if isinstance(d0, str):
            print(f"  {d0}")
        else:
            print(f"  top-level keys: {list(d0.keys())}")
            ev0 = d0.get("event") or {}
            if isinstance(ev0, dict):
                print(f"  event keys: {list(ev0.keys())}")
            m0 = d0.get("markets") or ev0.get("markets") or []
            print(f"  markets returned: {len(m0)}")
            if m0:
                print(f"  market tickers: {[m.get('ticker') for m in m0]}")
                print(f"  one market: {json.dumps(m0[0])[:400]}")

        for t, g in wanted[:6]:
            print(f"  {t}  ->  {g['away_team']} @ {g['home_team']} "
                  f"{g['gameday'].date()}")
        ok, first = 0, None
        for t, g in wanted[:6]:
            gts = int(pd.Timestamp(g["gameday"]).timestamp())
            tr, used = fetch_trades(t, raw, args, gts)
            print(f"  {t:34} {len(tr):>7,} trades  via {used}")
            ok += bool(tr)
            if tr and first is None:
                first = tr[0]
        if not ok:
            print("\nNo trades on any probed ticker. Run --diag-trades and "
                  "paste the output.")
        else:
            print("\none raw trade:")
            print(json.dumps(first, indent=2))
        return

    frames, done = [], 0
    t0 = time.time()
    for t, g in wanted:
        gts = int(pd.Timestamp(g["gameday"]).timestamp())
        trades, _used = fetch_trades(t, raw, args, gts)
        done += 1
        if trades:
            d = pd.DataFrame(trades)
            d["ticker"] = t
            d["home_team"] = g["home_team"]
            d["away_team"] = g["away_team"]
            d["game_date"] = g["gameday"].date().isoformat()
            code = t.rsplit("-", 1)[-1]
            d["team_yes"] = TEAM_ALIAS.get(code, code)
            frames.append(d)
        if done % 25 == 0 or done == len(wanted):
            n = sum(len(f) for f in frames)
            print(f"  {done}/{len(wanted)} markets, {n:,} trades, "
                  f"{(time.time() - t0) / 60:.0f}m", flush=True)

    if not frames:
        raise SystemExit(
            "no trades returned for any ticker. Run --probe to check the "
            "constructed market tickers against the API.")

    df = pd.concat(frames, ignore_index=True)

    # The trades payload now returns fixed-point STRINGS: count_fp "10.00" and
    # yes_price_dollars "0.5600". The old integer count / yes_price fields are
    # gone, so build the cent and count columns from whichever form came back.
    if "timestamp" not in df.columns and "created_time" in df.columns:
        df = df.rename(columns={"created_time": "timestamp"})

    if "yes_price_cents" not in df.columns:
        if "yes_price_dollars" in df.columns:
            df["yes_price_cents"] = (
                pd.to_numeric(df["yes_price_dollars"], errors="coerce") * 100
            ).round().astype("Int64")
        elif "yes_price" in df.columns:
            df["yes_price_cents"] = pd.to_numeric(df["yes_price"],
                                                  errors="coerce")

    if "count" not in df.columns and "count_fp" in df.columns:
        df["count"] = pd.to_numeric(df["count_fp"], errors="coerce")

    # taker_side is deprecated in favour of taker_outcome_side; both mean the
    # same thing, so accept either.
    if "taker_side" not in df.columns and "taker_outcome_side" in df.columns:
        df["taker_side"] = df["taker_outcome_side"]

    need = ["home_team", "away_team", "game_date", "timestamp", "team_yes",
            "yes_price_cents", "count", "taker_side", "ticker", "trade_id"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"missing {missing}; columns returned: {sorted(df.columns)}")

    df = df[need].drop_duplicates(subset="trade_id")
    df = df.sort_values(["game_date", "ticker", "timestamp"])

    out = (Path(args.out) if args.out
           else cd.parent / f"kalshi_nfl_{args.season}_regular.csv")
    df.to_csv(out, index=False)
    print(f"\n{len(df):,} trades, {df.ticker.nunique():,} markets, "
          f"{df.groupby(['home_team', 'away_team', 'game_date']).ngroups} games")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
