"""
nfl_common.py
Shared cache discovery, play-by-play loading, next-score labelling and feature
construction for the two-stage NFL model:

    Stage 1 (EP): field position + down & distance  ->  expected points
    Stage 2 (WP): score, clock, EP, pregame line    ->  live win probability

The pbp_YYYY.parquet exports do NOT carry spread_line, so it is joined on
game_id from schedules.parquet. The odds_api_lines_nfl_YYYY.parquet files are
joined optionally to add skill_diff.

Environment overrides (both optional):
    NFL_CACHE_DIR   directory holding the parquet files
                    (default: %USERPROFILE%\\OneDrive\\Documents\\nfl_cache)
    NFL_MODEL_DIR   where model artifacts are written (default: <cache>\\models)
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

COMMON_VERSION = "2026-08-09f"


def cache_dir() -> Path:
    env = os.environ.get("NFL_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / "OneDrive" / "Documents" / "nfl_cache"


def model_dir(create: bool = True) -> Path:
    env = os.environ.get("NFL_MODEL_DIR")
    p = Path(env).expanduser() if env else cache_dir() / "models"
    p = p.resolve()
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def available_seasons(directory: Path | None = None) -> list[int]:
    d = directory or cache_dir()
    out = []
    for f in sorted(d.glob("pbp_*.parquet")):
        m = re.fullmatch(r"pbp_(\d{4})", f.stem)
        if m:
            out.append(int(m.group(1)))
    return out


# --------------------------------------------------------------------------
# columns
# --------------------------------------------------------------------------

# Hard requirements inside pbp_YYYY.parquet. Note spread_line is NOT here: it
# comes from schedules.parquet.
REQUIRED_COLS = [
    "game_id",
    "home_team",
    "away_team",
    "posteam",
    "qtr",
    "down",
    "ydstogo",
    "yardline_100",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "total_home_score",
    "total_away_score",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "result",
]

OPTIONAL_COLS = [
    "play_id",
    "season",
    "week",
    "season_type",
    "defteam",
    "score_differential",
    "goal_to_go",
    "receive_2h_ko",
    "game_date",
    # not model features; carried through for the trade-tape backtest
    "play_type",
    "time_of_day",
    "quarter_seconds_remaining",
    "yards_gained",
    "epa",
    "passer_player_id",
    # sabermetric inputs, all present and ~97-99% populated back to 2011
    "success", "qb_dropback", "qb_hit", "qb_scramble", "cpoe", "pass_oe",
    "pass_attempt", "rush_attempt", "series_success", "first_down",
    "interception", "fumble_lost", "penalty_team", "penalty_yards",
    "third_down_converted", "third_down_failed",
]

# schedules.parquet
SCHEDULE_COLS = ["game_id", "season", "gameday", "home_team", "away_team", "spread_line"]
SCHEDULE_CONTEXT = ["temp", "wind", "roof"]

# odds_api_lines_nfl_YYYY.parquet
LINES_COLS = ["home_abbr", "away_abbr", "date_str", "skill_diff"]
LINES_OPTIONAL = ["pregame_total"]

# --------------------------------------------------------------------------
# next-score classes
# --------------------------------------------------------------------------

# Order defines the label encoding used by the EP booster. Do not reorder
# without retraining.
NEXT_SCORE_CLASSES = [
    "No_Score",
    "TD",
    "FG",
    "Safety",
    "Opp_TD",
    "Opp_FG",
    "Opp_Safety",
]
NEXT_SCORE_VALUES = np.array([0.0, 7.0, 3.0, 2.0, -7.0, -3.0, -2.0], dtype=float)

EP_FEATURES = [
    "down",
    "ydstogo",
    "yardline_100",
    "half_seconds_remaining",
    "goal_to_go",
    "posteam_is_home",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
]

# Everything in the WP model that is not a pregame team-strength term.
WP_FEATURES_BASE = [
    "score_differential",
    "ep",
    "exp_score_diff",
    "diff_time_ratio",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "down",
    "ydstogo",
    "yardline_100",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "posteam_is_home",
    "receive_2h_ko",
    "is_second_half",
]

SKILL_SOURCES = ("spread", "skill_diff", "both")

# Season-to-date efficiency, counting only games BEFORE the current one.
YTD_FEATURES = [
    "posteam_off_ypp",
    "posteam_def_ypp",
    "defteam_off_ypp",
    "defteam_def_ypp",
    "off_ypp_matchup",
    "def_ypp_matchup",
]


# Structural features. These add arithmetic the trees cannot easily build for
# themselves (division, square roots, rule-based clock), not new opinions about
# team strength — which is what the YTD experiment showed the closing line has
# already priced.
V2_FEATURES = [
    "pregame_total",        # expected combined points, from the closing total
    "remaining_points",     # expected points still to be scored by both teams
    "std_lead",             # lead measured in units of remaining scoring
    "std_spread",           # pregame edge measured the same way
    "runoff_seconds",       # clock the offense can bleed given defensive timeouts
    "clock_after_runoff",   # time that survives a kneel-out attempt
]

PLAY_CLOCK_SECONDS = 40.0   # NFL rule, not a fitted constant

# How each side has actually played SO FAR IN THIS GAME. The pregame line
# cannot contain this: it is fixed at kickoff, while these update every snap.
# This is the one kind of team-strength signal the closing line has not already
# priced, which is what separates it from the season-to-date YPP experiment.
INGAME_FEATURES = [
    "posteam_epa_pp",   # possession team's EPA per play so far this game
    "defteam_epa_pp",   # opponent's EPA per play so far this game
    "epa_pp_diff",      # the difference
    "plays_so_far",     # how much evidence the two averages rest on
]

# Directions the model should not be allowed to violate: more expected points,
# a bigger lead, or a stronger pregame line can only raise win probability.
# Conditions the pregame line prices for the game as a whole but which the WP
# model has never seen at all. Wind in particular changes how easily a trailing
# team can move the ball and kick.
CONTEXT_FEATURES = ["wind_mph", "temp_f", "is_indoors"]

# A mid-game quarterback change is the sharpest in-game news there is, and the
# closing line cannot contain it. NOTE: an earlier version of this model had
# QB-change features soak up disproportionate capacity, so this is deliberately
# only three columns.
QB_FEATURES = ["posteam_qb_changed", "defteam_qb_changed", "qb_change_diff"]

# Rate stats measured over the plays already run IN THIS GAME. Every one is the
# possession team's OFFENSE minus the opponent's OFFENSE, so a positive value
# always means the team with the ball has been the better unit so far. Defensive
# quality shows up as the opponent's offensive numbers being poor.
SABER_FEATURES = [
    "sab_success_diff",      # share of plays with positive EPA
    "sab_epa_dropback_diff", # EPA per dropback
    "sab_epa_rush_diff",     # EPA per rush
    "sab_early_epa_diff",    # EPA per play on 1st and 2nd down
    "sab_explosive_diff",    # share of plays gaining 15+
    "sab_cpoe_diff",         # completion percentage over expected
    "sab_series_diff",       # series conversion rate
    "sab_pass_oe_diff",      # pass rate over expected
    "sab_disruption_diff",   # sacks + QB hits allowed per dropback (sign flipped)
    "sab_third_conv_diff",   # third down conversion rate
    "sab_turnover_diff",     # interceptions + lost fumbles per play (sign flipped)
    "sab_penalty_diff",      # penalty yards per play (sign flipped)
    "sab_plays",             # plays run so far, so the model can discount early noise
]

# Short name -> feature, so subsets can be selected on the command line.
SABER_SHORT = {c[len("sab_"):].replace("_diff", ""): c for c in SABER_FEATURES}

# Coherent families, for testing a handful of runs instead of thirteen.
SABER_GROUPS = {
    "efficiency": ["success", "epa_dropback", "epa_rush", "early_epa"],
    "explosive":  ["explosive", "cpoe", "series"],
    "tendency":   ["pass_oe"],
    "disruption": ["disruption", "turnover", "penalty"],
    "situational": ["third_conv"],
    "volume":     ["plays"],
}


def resolve_saber(spec: str | None) -> list[str]:
    """Turn a comma list of short names or group names into feature columns."""
    if not spec:
        return list(SABER_FEATURES)
    picked: list[str] = []
    unknown: list[str] = []
    for raw in spec.replace(" ", "").split(","):
        if not raw:
            continue
        if raw in SABER_GROUPS:
            picked += SABER_GROUPS[raw]
        elif raw in SABER_SHORT:
            picked.append(raw)
        elif raw in SABER_FEATURES:
            picked.append(raw[len("sab_"):].replace("_diff", ""))
        else:
            unknown.append(raw)
    if unknown:
        raise ValueError(
            f"unknown saber name(s) {unknown}. "
            f"features: {sorted(SABER_SHORT)}  groups: {sorted(SABER_GROUPS)}"
        )
    seen, out = set(), []
    for name in picked:
        col = SABER_SHORT[name]
        if col not in seen:
            seen.add(col)
            out.append(col)
    return out


MONOTONE_UP = [
    "score_differential", "ep", "exp_score_diff", "diff_time_ratio",
    "posteam_spread", "spread_time", "posteam_skill_diff", "skill_diff_time",
    "std_lead", "std_spread",
]


def monotone_constraints(feats: list[str]) -> str:
    """XGBoost monotone_constraints string aligned to the feature list."""
    return "(" + ",".join("1" if f in MONOTONE_UP else "0" for f in feats) + ")"


def wp_features(skill_source: str = "both", ytd: bool = False,
                v2: bool = False, ingame: bool = False,
                context: bool = False, qb: bool = False,
                saber: bool | list[str] = False) -> list[str]:
    """Feature list for the WP booster, given which pregame strength terms are used."""
    if skill_source not in SKILL_SOURCES:
        raise ValueError(f"skill_source must be one of {SKILL_SOURCES}")
    feats = list(WP_FEATURES_BASE)
    if skill_source in ("spread", "both"):
        feats += ["posteam_spread", "spread_time"]
    if skill_source in ("skill_diff", "both"):
        feats += ["posteam_skill_diff", "skill_diff_time"]
    if ytd:
        feats += list(YTD_FEATURES)
    if v2:
        feats += list(V2_FEATURES)
    if ingame:
        feats += list(INGAME_FEATURES)
    if context:
        feats += list(CONTEXT_FEATURES)
    if qb:
        feats += list(QB_FEATURES)
    if saber:
        feats += list(SABER_FEATURES) if saber is True else list(saber)
    return feats


def skill_columns(skill_source: str = "both") -> list[str]:
    """The raw per-game strength columns needed for the given skill_source."""
    cols = []
    if skill_source in ("spread", "both"):
        cols.append("posteam_spread")
    if skill_source in ("skill_diff", "both"):
        cols.append("posteam_skill_diff")
    return cols


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def as_mask(x) -> np.ndarray:
    """Comparison result -> plain numpy bool array, NA -> False."""
    if isinstance(x, pd.Series):
        return x.fillna(False).to_numpy(dtype=bool)
    return np.asarray(x).astype(bool)


def load_schedules(directory: Path | None = None) -> pd.DataFrame:
    d = directory or cache_dir()
    f = d / "schedules.parquet"
    if not f.exists():
        raise FileNotFoundError(
            f"missing {f}. spread_line is not present in the pbp exports and is "
            f"read from schedules.parquet."
        )
    names = set(pq.ParquetFile(f).schema_arrow.names)
    missing = [c for c in SCHEDULE_COLS if c not in names]
    if missing:
        raise ValueError(f"schedules.parquet is missing column(s): {missing}")
    have = [c for c in SCHEDULE_CONTEXT if c in names]
    sched = pd.read_parquet(f, columns=SCHEDULE_COLS + have)
    for c in SCHEDULE_CONTEXT:
        if c not in sched.columns:
            sched[c] = np.nan
    sched["gameday"] = pd.to_datetime(sched["gameday"], errors="coerce").dt.strftime("%Y-%m-%d")
    sched["spread_line"] = pd.to_numeric(sched["spread_line"], errors="coerce")
    sched["season"] = pd.to_numeric(sched["season"], errors="coerce").astype("Int64")
    return sched


def load_lines(seasons: list[int], directory: Path | None = None) -> pd.DataFrame:
    """
    Read odds_api_lines_nfl_YYYY.parquet and attach game_id by matching
    (home team, away team, date) against schedules.parquet.

    Returns game_id + skill_diff. skill_diff is oriented HOME-positive, matching
    the spread_line convention, and this is verified against outcomes in
    check_skill_signs() before training.
    """
    d = directory or cache_dir()
    sched = load_schedules(d)

    frames = []
    for s in seasons:
        f = d / f"odds_api_lines_nfl_{s}.parquet"
        if not f.exists():
            print(f"  no lines file for {s}: {f.name}")
            continue
        names = set(pq.ParquetFile(f).schema_arrow.names)
        missing = [c for c in LINES_COLS if c not in names]
        if missing:
            raise ValueError(f"{f.name} is missing column(s): {missing}")
        use = LINES_COLS + [c for c in LINES_OPTIONAL if c in names]
        df = pd.read_parquet(f, columns=use)
        for c in LINES_OPTIONAL:
            if c not in df.columns:
                df[c] = np.nan
        df["season_file"] = s
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["game_id", "skill_diff"] + LINES_OPTIONAL)

    lines = pd.concat(frames, ignore_index=True)
    lines["date_str"] = pd.to_datetime(lines["date_str"], errors="coerce").dt.strftime("%Y-%m-%d")
    lines["skill_diff"] = pd.to_numeric(lines["skill_diff"], errors="coerce")

    # primary match: home + away + exact date
    merged = lines.merge(
        sched[["game_id", "season", "gameday", "home_team", "away_team"]],
        left_on=["home_abbr", "away_abbr", "date_str"],
        right_on=["home_team", "away_team", "gameday"],
        how="left",
    )

    # fallback: home + away + season, where that is unique
    unmatched = merged["game_id"].isna()
    if unmatched.any():
        pair_counts = sched.groupby(["season", "home_team", "away_team"]).size()
        unique_pairs = (
            sched.set_index(["season", "home_team", "away_team"])
            .loc[pair_counts[pair_counts == 1].index, "game_id"]
            .reset_index()
        )
        fb = merged.loc[unmatched, ["season_file", "home_abbr", "away_abbr"]].merge(
            unique_pairs,
            left_on=["season_file", "home_abbr", "away_abbr"],
            right_on=["season", "home_team", "away_team"],
            how="left",
        )
        merged.loc[unmatched, "game_id"] = fb["game_id"].to_numpy()

    n_total = len(merged)
    n_matched = int(merged["game_id"].notna().sum())
    print(f"  lines rows: {n_total:,}  matched to a game_id: {n_matched:,} "
          f"({n_matched / max(n_total, 1):.1%})")
    if n_matched == 0:
        raise RuntimeError(
            "no odds rows matched schedules.parquet. Team abbreviations or dates "
            "in the odds files do not line up with nflverse; inspect both before "
            "training with --skill-source skill_diff."
        )
    if n_matched / max(n_total, 1) < 0.95:
        bad = merged.loc[merged["game_id"].isna(),
                         ["season_file", "home_abbr", "away_abbr", "date_str"]].head(10)
        print("  unmatched sample:")
        print(bad.to_string(index=False))

    out = merged.loc[merged["game_id"].notna(),
                     ["game_id", "skill_diff"] + LINES_OPTIONAL]
    out = out.drop_duplicates(subset=["game_id"], keep="first").reset_index(drop=True)
    return out


def load_pbp(
    seasons: list[int],
    directory: Path | None = None,
    with_lines: bool = False,
) -> pd.DataFrame:
    """Read pbp_YYYY.parquet for the requested seasons, join spread_line from
    schedules.parquet, and optionally join skill_diff from the odds files."""
    d = directory or cache_dir()
    if not d.exists():
        raise FileNotFoundError(f"cache directory not found: {d}")

    frames = []
    for s in seasons:
        f = d / f"pbp_{s}.parquet"
        if not f.exists():
            raise FileNotFoundError(f"missing play-by-play file: {f}")

        names = set(pq.ParquetFile(f).schema_arrow.names)
        missing = [c for c in REQUIRED_COLS if c not in names]
        if missing:
            raise ValueError(
                f"{f.name} is missing required column(s): {missing}\n"
                f"Run nfl_inspect.py to see what the file actually contains."
            )

        use = [c for c in REQUIRED_COLS + OPTIONAL_COLS if c in names]
        df = pd.read_parquet(f, columns=use)
        if "season" not in df.columns:
            df["season"] = s
        else:
            df["season"] = pd.to_numeric(df["season"], errors="coerce").fillna(s).astype(int)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    sched = load_schedules(d)
    before = len(out)
    out = out.merge(sched[["game_id", "spread_line"] + SCHEDULE_CONTEXT],
                    on="game_id", how="left")
    assert len(out) == before, "schedules join duplicated pbp rows"
    n_missing_spread = int(out["spread_line"].isna().sum())
    print(f"  spread_line joined from schedules.parquet; "
          f"{n_missing_spread:,} of {len(out):,} rows without a line")

    if with_lines:
        lines = load_lines(seasons, d)
        out = out.merge(lines, on="game_id", how="left")
        assert len(out) == before, "lines join duplicated pbp rows"
        n_missing_skill = int(out["skill_diff"].isna().sum())
        print(f"  skill_diff joined; {n_missing_skill:,} of {len(out):,} rows without one")

    return out


# --------------------------------------------------------------------------
# scoring events -> next-score labels
# --------------------------------------------------------------------------

def _tag_scoring_plays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive the scoring event on each play from the running score alone, so the
    labelling works regardless of which descriptive columns the pbp export kept.

    Point deltas are read as:
        >= 6  -> touchdown (valued at 7, i.e. TD + expected PAT)
        == 3  -> field goal
        == 2  -> safety, UNLESS it closely follows a touchdown by either team,
                 in which case it is a two-point conversion or a defensive PAT
                 return and is folded into the preceding TD
        == 1  -> extra point, folded into the preceding TD
    """
    g = df.groupby("game_id", sort=False)
    dh = g["total_home_score"].diff()
    da = g["total_away_score"].diff()

    first = ~df["game_id"].duplicated()
    dh[first] = df.loc[first, "total_home_score"]
    da[first] = df.loc[first, "total_away_score"]

    dh = dh.fillna(0.0).to_numpy()
    da = da.fillna(0.0).to_numpy()

    gid = df["game_id"].to_numpy()
    home = df["home_team"].to_numpy()
    away = df["away_team"].to_numpy()

    n = len(df)
    ev_type = np.full(n, None, dtype=object)
    ev_team = np.full(n, None, dtype=object)

    last_td_row: dict[object, int] = {}

    for i in np.flatnonzero((dh != 0) | (da != 0)):
        if dh[i] > 0:
            pts, team = dh[i], home[i]
        elif da[i] > 0:
            pts, team = da[i], away[i]
        else:
            continue  # negative delta = stat correction, ignore

        pts = int(round(float(pts)))
        prev_td = last_td_row.get(gid[i])
        if pts in (1, 2) and prev_td is not None and (i - prev_td) <= 3:
            continue  # PAT kick / two-point try / defensive PAT return

        if pts >= 6:
            ev_type[i] = "TD"
            ev_team[i] = team
            last_td_row[gid[i]] = i
        elif pts == 3:
            ev_type[i] = "FG"
            ev_team[i] = team
        elif pts == 2:
            ev_type[i] = "Safety"
            ev_team[i] = team
        # anything else (1, 4, 5) is ignored

    df = df.copy()
    df["_ev_type"] = ev_type
    df["_ev_team"] = ev_team
    return df


def add_next_score_label(df: pd.DataFrame) -> pd.DataFrame:
    """Attach next_score_label: the next scoring event in the same half, from
    the possession team's point of view."""
    df = _tag_scoring_plays(df)

    grp = df.groupby(["game_id", "half"], sort=False)
    nxt_type = grp["_ev_type"].bfill()
    nxt_team = grp["_ev_team"].bfill()

    scored = nxt_type.notna()
    same_team = scored & (nxt_team == df["posteam"])

    label = np.where(
        ~scored,
        "No_Score",
        np.where(same_team, nxt_type.fillna("TD"), "Opp_" + nxt_type.fillna("TD")),
    )

    df["next_score_label"] = label
    df = df.drop(columns=["_ev_type", "_ev_team"])
    return df


# --------------------------------------------------------------------------
# feature construction
# --------------------------------------------------------------------------

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, derive the shared columns, and attach the next-score label."""
    sort_cols = ["game_id"] + (["play_id"] if "play_id" in df.columns else [])
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    df["qtr"] = pd.to_numeric(df["qtr"], errors="coerce")

    # running score must be gap-free for the score-delta labelling to work
    for c in ("total_home_score", "total_away_score"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df.groupby("game_id", sort=False)[c].ffill().fillna(0.0)

    df["half"] = np.where(df["qtr"] <= 2, 1, np.where(df["qtr"] <= 4, 2, 3))
    df["is_second_half"] = (df["half"] == 2).astype(int)
    df["posteam_is_home"] = (df["posteam"] == df["home_team"]).astype(int)

    if "goal_to_go" in df.columns:
        df["goal_to_go"] = pd.to_numeric(df["goal_to_go"], errors="coerce").fillna(0).astype(int)
    else:
        df["goal_to_go"] = (df["yardline_100"] <= df["ydstogo"]).astype(int)

    # score differential from the possession team's view, at the snap
    if "score_differential" in df.columns and df["score_differential"].notna().mean() > 0.9:
        df["score_differential"] = pd.to_numeric(df["score_differential"], errors="coerce")
    else:
        g = df.groupby("game_id", sort=False)
        pre_h = g["total_home_score"].shift(1).fillna(0.0)
        pre_a = g["total_away_score"].shift(1).fillna(0.0)
        df["score_differential"] = np.where(
            df["posteam_is_home"] == 1, pre_h - pre_a, pre_a - pre_h
        )

    # pregame strength terms, flipped to the possession team's view
    if "spread_line" in df.columns:
        df["spread_line"] = pd.to_numeric(df["spread_line"], errors="coerce")
        df["posteam_spread"] = np.where(
            df["posteam_is_home"] == 1, df["spread_line"], -df["spread_line"]
        )
    if "skill_diff" in df.columns:
        df["skill_diff"] = pd.to_numeric(df["skill_diff"], errors="coerce")
        df["posteam_skill_diff"] = np.where(
            df["posteam_is_home"] == 1, df["skill_diff"], -df["skill_diff"]
        )

    # who receives the second-half kickoff
    if "receive_2h_ko" in df.columns and df["receive_2h_ko"].notna().mean() > 0.9:
        df["receive_2h_ko"] = pd.to_numeric(df["receive_2h_ko"], errors="coerce").fillna(0).astype(int)
    else:
        q3 = (
            df[(df["qtr"] == 3) & df["posteam"].notna()]
            .groupby("game_id", sort=False)["posteam"]
            .first()
        )
        q3_team = df["game_id"].map(q3)
        df["receive_2h_ko"] = (
            (df["half"] == 1) & df["posteam"].notna() & (df["posteam"] == q3_team)
        ).astype(int)

    df = add_ytd_efficiency(df)
    df = add_ingame_form(df)
    df = add_context(df)
    df = add_qb_change(df)
    df = add_saber_form(df)
    df = add_next_score_label(df)
    return df


def _prior_rate(gid: pd.Series, home_off: np.ndarray, away_off: np.ndarray,
                num: np.ndarray, den: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-team rate over PRIOR plays in the same game.

    num and den are already zeroed on plays that should not count. Each team's
    running totals have the current play subtracted back out, so a play can
    never contribute to its own rate.
    """
    out = {}
    for tag, side in (("home", home_off), ("away", away_off)):
        n_side = np.where(side, num, 0.0)
        d_side = np.where(side, den, 0.0)
        n = pd.Series(n_side).groupby(gid.to_numpy(), sort=False).cumsum().to_numpy() - n_side
        d = pd.Series(d_side).groupby(gid.to_numpy(), sort=False).cumsum().to_numpy() - d_side
        with np.errstate(invalid="ignore", divide="ignore"):
            out[tag] = np.where(d > 0, n / d, np.nan)
    return out["home"], out["away"]


def add_saber_form(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the in-game sabermetric rate differentials.

    Every metric is accumulated for whichever team has the ball, over plays
    strictly before the current one, then expressed as possession team minus
    opponent. Metrics where low is good (disruption, turnovers, penalties) are
    sign-flipped so that positive is always better for the team with the ball.
    """
    for c in SABER_FEATURES:
        df[c] = np.nan

    need = ["epa", "success", "qb_dropback"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        print(f"  no {missing} in the pbp export; sabermetric features left empty")
        return df

    def col(name, default=0.0):
        if name not in df.columns:
            return np.full(len(df), default, dtype=float)
        return pd.to_numeric(df[name], errors="coerce").fillna(default).to_numpy(dtype=float)

    gid = df["game_id"]
    home_off = as_mask(df["posteam"] == df["home_team"])
    away_off = as_mask(df["posteam"].notna()) & ~home_off

    pt = df["play_type"].astype(str) if "play_type" in df.columns else pd.Series("", index=df.index)
    scrim = pt.isin(["run", "pass"]).to_numpy().astype(float)

    epa = col("epa")
    dropback = col("qb_dropback")
    rush = col("rush_attempt")
    down = col("down", np.nan)
    early = np.where(np.isfinite(down) & (down <= 2), 1.0, 0.0) * scrim
    yards = col("yards_gained")
    passatt = col("pass_attempt")
    cpoe_raw = df["cpoe"] if "cpoe" in df.columns else pd.Series(np.nan, index=df.index)
    cpoe_ok = pd.to_numeric(cpoe_raw, errors="coerce").notna().to_numpy().astype(float)
    cpoe = col("cpoe")
    poe_raw = df["pass_oe"] if "pass_oe" in df.columns else pd.Series(np.nan, index=df.index)
    poe_ok = pd.to_numeric(poe_raw, errors="coerce").notna().to_numpy().astype(float)
    poe = col("pass_oe")
    series_ok = (pd.to_numeric(df["series_success"], errors="coerce").notna().to_numpy().astype(float)
                 if "series_success" in df.columns else np.zeros(len(df)))
    series = col("series_success")
    disrupt = np.clip(col("sack") + col("qb_hit"), 0.0, 1.0)
    third_c, third_f = col("third_down_converted"), col("third_down_failed")
    turnover = np.clip(col("interception") + col("fumble_lost"), 0.0, 1.0)
    pen_team = df["penalty_team"].astype("string") if "penalty_team" in df.columns else None
    pen_on_off = (as_mask(pen_team == df["posteam"]).astype(float)
                  if pen_team is not None else np.zeros(len(df)))
    pen_yards = col("penalty_yards") * pen_on_off

    # (feature, numerator, denominator, sign)
    specs = [
        ("sab_success_diff",      col("success") * scrim, scrim,               +1.0),
        ("sab_epa_dropback_diff", epa * dropback,         dropback,            +1.0),
        ("sab_epa_rush_diff",     epa * rush,             rush,                +1.0),
        ("sab_early_epa_diff",    epa * early,            early,               +1.0),
        ("sab_explosive_diff",    (yards >= 15).astype(float) * scrim, scrim,  +1.0),
        ("sab_cpoe_diff",         cpoe * cpoe_ok * passatt, cpoe_ok * passatt, +1.0),
        ("sab_series_diff",       series * series_ok,     series_ok,           +1.0),
        ("sab_pass_oe_diff",      poe * poe_ok,           poe_ok,              +1.0),
        ("sab_disruption_diff",   disrupt * dropback,     dropback,            -1.0),
        ("sab_third_conv_diff",   third_c,                third_c + third_f,   +1.0),
        ("sab_turnover_diff",     turnover * scrim,       scrim,               -1.0),
        ("sab_penalty_diff",      pen_yards * scrim,      scrim,               -1.0),
    ]

    for name, num, den, sign in specs:
        h, a = _prior_rate(gid, home_off, away_off, num, den)
        df[name] = sign * np.where(home_off, h - a, a - h)

    n_side = np.where(home_off | away_off, scrim, 0.0)
    total = pd.Series(n_side).groupby(gid.to_numpy(), sort=False).cumsum().to_numpy() - n_side
    df["sab_plays"] = total

    cov = float(pd.Series(df["sab_success_diff"]).notna().mean())
    print(f"  sabermetric form attached; {cov:.1%} of rows have prior plays for both sides")
    return df


def add_context(df: pd.DataFrame) -> pd.DataFrame:
    """Wind, temperature and whether the roof is shut, from schedules.parquet."""
    roof = df["roof"].astype(str).str.lower() if "roof" in df.columns else None
    indoors = (roof.isin(["dome", "closed"]) if roof is not None
               else pd.Series(False, index=df.index))
    df["is_indoors"] = indoors.astype(int)

    wind = pd.to_numeric(df.get("wind"), errors="coerce")
    temp = pd.to_numeric(df.get("temp"), errors="coerce")
    # indoor games have no weather rather than missing weather
    df["wind_mph"] = np.where(indoors, 0.0, wind)
    df["temp_f"] = np.where(indoors, 70.0, temp)

    cov = float(pd.Series(df["wind_mph"]).notna().mean())
    print(f"  game context attached; {cov:.1%} of rows have a wind reading, "
          f"{float(indoors.mean()):.1%} indoors")
    return df


def add_qb_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag each side once its passer differs from the one who threw first for that
    team in this game. Uses only plays before the current one.
    """
    for c in QB_FEATURES:
        df[c] = 0
    if "passer_player_id" not in df.columns:
        print("  no passer_player_id in the pbp export; QB-change left at zero")
        return df

    pid = df["passer_player_id"].astype("string")
    has = pid.notna() & df["posteam"].notna()

    home_off = as_mask(df["posteam"] == df["home_team"])
    out = {}
    for tag, side in (("home", home_off), ("away", as_mask(df["posteam"].notna()) & ~home_off)):
        m = has.to_numpy() & side
        pid_side = pid.where(pd.Series(m, index=df.index))
        grp = pid_side.groupby(df["game_id"], sort=False)
        first = grp.transform("first").ffill()          # that team's first passer
        last = pid_side.groupby(df["game_id"], sort=False).ffill()  # most recent passer
        # shift so the current play cannot flag itself
        last_prior = last.groupby(df["game_id"], sort=False).shift(1)
        first_prior = first.groupby(df["game_id"], sort=False).ffill()
        changed = (last_prior.notna() & first_prior.notna()
                   & (last_prior != first_prior))
        out[tag] = changed.groupby(df["game_id"], sort=False).cummax().fillna(False).to_numpy()

    df["posteam_qb_changed"] = np.where(home_off, out["home"], out["away"]).astype(int)
    df["defteam_qb_changed"] = np.where(home_off, out["away"], out["home"]).astype(int)
    df["qb_change_diff"] = df["defteam_qb_changed"] - df["posteam_qb_changed"]

    n = int((df["posteam_qb_changed"] | df["defteam_qb_changed"]).sum())
    print(f"  QB-change attached; {n:,} rows are after a change by either side")
    return df


def add_ingame_form(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach each side's EPA per play SO FAR IN THIS GAME, counting only plays
    strictly before the current one.

    Sorted by game_id then play_id, so "before" is the real play order. A team
    with no prior plays gets NaN and XGBoost routes the missing branch itself.
    """
    for c in INGAME_FEATURES:
        df[c] = np.nan
    if "epa" not in df.columns:
        print("  no epa column in the pbp export; in-game form left empty")
        return df

    epa = pd.to_numeric(df["epa"], errors="coerce")
    home_off = as_mask(df["posteam"] == df["home_team"])
    away_off = as_mask(df["posteam"].notna()) & ~home_off
    counts = epa.notna().to_numpy()

    e = epa.fillna(0.0).to_numpy()
    g = df.groupby("game_id", sort=False)

    out = {}
    for tag, mask in (("home", home_off & counts), ("away", away_off & counts)):
        cum_e = pd.Series(np.where(mask, e, 0.0), index=df.index).groupby(
            df["game_id"], sort=False).cumsum().to_numpy()
        cum_n = pd.Series(mask.astype(float), index=df.index).groupby(
            df["game_id"], sort=False).cumsum().to_numpy()
        # drop the current play out of its own average
        out[tag + "_e"] = cum_e - np.where(mask, e, 0.0)
        out[tag + "_n"] = cum_n - mask.astype(float)

    with np.errstate(invalid="ignore", divide="ignore"):
        home_pp = np.where(out["home_n"] > 0, out["home_e"] / out["home_n"], np.nan)
        away_pp = np.where(out["away_n"] > 0, out["away_e"] / out["away_n"], np.nan)

    df["posteam_epa_pp"] = np.where(home_off, home_pp, away_pp)
    df["defteam_epa_pp"] = np.where(home_off, away_pp, home_pp)
    df["epa_pp_diff"] = df["posteam_epa_pp"] - df["defteam_epa_pp"]
    df["plays_so_far"] = out["home_n"] + out["away_n"]

    cov = float(df["epa_pp_diff"].notna().mean())
    print(f"  in-game form attached; {cov:.1%} of rows have prior plays for both sides")
    return df


def add_ytd_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach season-to-date yards per play, offense and defense, for both teams.

    Only games EARLIER in the same season count, so a game never sees its own
    plays. Week 1 has no prior games and stays NaN; XGBoost routes missing
    values on its own rather than being handed a made-up league average.
    """
    for c in YTD_FEATURES:
        df[c] = np.nan

    need = {"play_type", "yards_gained"}
    if not need.issubset(df.columns):
        print(f"  no {sorted(need - set(df.columns))} in the pbp export; "
              f"season-to-date efficiency left empty")
        return df

    if "defteam" in df.columns and df["defteam"].notna().mean() > 0.5:
        defteam = df["defteam"]
    else:
        defteam = pd.Series(
            np.where(as_mask(df["posteam"] == df["home_team"]),
                     df["away_team"], df["home_team"]),
            index=df.index,
        ).where(df["posteam"].notna())

    scrim = (df["play_type"].isin(["run", "pass"])
             & pd.to_numeric(df["yards_gained"], errors="coerce").notna()
             & df["posteam"].notna())
    if not scrim.any():
        print("  no run/pass plays with yards_gained; "
              "season-to-date efficiency left empty")
        return df

    core = pd.DataFrame({
        "season": df.loc[scrim, "season"].to_numpy(),
        "game_id": df.loc[scrim, "game_id"].to_numpy(),
        "off": df.loc[scrim, "posteam"].to_numpy(),
        "deff": defteam[scrim].to_numpy(),
        "yards": pd.to_numeric(df.loc[scrim, "yards_gained"], errors="coerce").to_numpy(),
    })

    off = (core.groupby(["season", "game_id", "off"], sort=False)["yards"]
           .agg(["sum", "size"]).reset_index()
           .rename(columns={"off": "team", "sum": "off_yards", "size": "off_plays"}))
    dfn = (core.groupby(["season", "game_id", "deff"], sort=False)["yards"]
           .agg(["sum", "size"]).reset_index()
           .rename(columns={"deff": "team", "sum": "def_yards", "size": "def_plays"}))
    tg = off.merge(dfn, on=["season", "game_id", "team"], how="outer").fillna(
        {"off_yards": 0.0, "off_plays": 0.0, "def_yards": 0.0, "def_plays": 0.0})

    # order games within a season
    if "week" in df.columns:
        order = df.groupby("game_id", sort=False)["week"].first()
    elif "game_date" in df.columns:
        order = df.groupby("game_id", sort=False)["game_date"].first()
    else:
        order = pd.Series(tg["game_id"].unique(), index=tg["game_id"].unique())
    tg["_order"] = tg["game_id"].map(order)
    tg = tg.sort_values(["season", "team", "_order", "game_id"], kind="mergesort")

    grp = tg.groupby(["season", "team"], sort=False)
    for c in ("off_yards", "off_plays", "def_yards", "def_plays"):
        tg["prior_" + c] = grp[c].cumsum() - tg[c]

    with np.errstate(invalid="ignore", divide="ignore"):
        tg["prior_off_ypp"] = np.where(tg["prior_off_plays"] > 0,
                                       tg["prior_off_yards"] / tg["prior_off_plays"],
                                       np.nan)
        tg["prior_def_ypp"] = np.where(tg["prior_def_plays"] > 0,
                                       tg["prior_def_yards"] / tg["prior_def_plays"],
                                       np.nan)

    key = tg.set_index(["game_id", "team"])[["prior_off_ypp", "prior_def_ypp"]]
    gid = df["game_id"]
    for side, team_col in (("posteam", df["posteam"]), ("defteam", defteam)):
        idx = pd.MultiIndex.from_arrays([gid, team_col])
        vals = key.reindex(idx)
        df[f"{side}_off_ypp"] = vals["prior_off_ypp"].to_numpy()
        df[f"{side}_def_ypp"] = vals["prior_def_ypp"].to_numpy()

    df["off_ypp_matchup"] = df["posteam_off_ypp"] - df["defteam_def_ypp"]
    df["def_ypp_matchup"] = df["defteam_off_ypp"] - df["posteam_def_ypp"]

    own = float(df["posteam_off_ypp"].notna().mean())
    both = float(df["off_ypp_matchup"].notna().mean())
    print(f"  season-to-date efficiency attached; {own:.1%} of rows have prior games "
          f"for the possession team, {both:.1%} for both teams")
    return df


def model_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Plays usable for training: real scrimmage snaps in regulation."""
    m = (
        df["posteam"].notna()
        & df["down"].between(1, 4)
        & df["ydstogo"].notna()
        & df["yardline_100"].between(1, 99)
        & df["half_seconds_remaining"].notna()
        & df["game_seconds_remaining"].notna()
        & (df["qtr"] <= 4)
        & df["posteam_timeouts_remaining"].notna()
        & df["defteam_timeouts_remaining"].notna()
    )
    return df.loc[m].reset_index(drop=True)


def add_wp_features(df: pd.DataFrame, ep: np.ndarray, v2: bool = False) -> pd.DataFrame:
    """Attach EP-derived and clock-decayed features. `ep` is aligned to df."""
    df = df.copy()
    df["ep"] = ep
    df["exp_score_diff"] = df["score_differential"] + df["ep"]

    gsr = df["game_seconds_remaining"].clip(lower=0)
    elapsed_share = (3600.0 - gsr) / 3600.0
    decay = np.exp(-4.0 * elapsed_share)

    df["diff_time_ratio"] = df["exp_score_diff"] / decay
    if "posteam_spread" in df.columns:
        df["spread_time"] = df["posteam_spread"] * decay
    if "posteam_skill_diff" in df.columns:
        df["skill_diff_time"] = df["posteam_skill_diff"] * decay

    if v2:
        total = pd.to_numeric(df.get("pregame_total"), errors="coerce")
        if total is None or total.isna().all():
            raise ValueError(
                "v2 features need pregame_total from odds_api_lines_nfl_YYYY.parquet; "
                "load the play-by-play with with_lines=True"
            )
        df["pregame_total"] = total

        # Expected points still to be scored by BOTH teams. A lead is worth more
        # when less scoring remains, which is what the model cannot infer from
        # clock alone: 3 points up with 5 minutes left is safer in a 38-point
        # game than a 52-point game.
        df["remaining_points"] = total * gsr / 3600.0

        # Margins scale with the square root of remaining scoring, so dividing
        # by it puts every game state on one comparable axis.
        denom = np.sqrt(df["remaining_points"].clip(lower=1.0))
        df["std_lead"] = df["exp_score_diff"] / denom
        if "posteam_spread" in df.columns:
            df["std_spread"] = df["posteam_spread"] / np.sqrt(total.clip(lower=1.0))
        else:
            df["std_spread"] = np.nan

        # With the ball and a lead, the offense can kneel out roughly one play
        # clock per timeout the defense does not have.
        to_left = pd.to_numeric(df["defteam_timeouts_remaining"], errors="coerce")
        df["runoff_seconds"] = (PLAY_CLOCK_SECONDS * (3.0 - to_left)).clip(lower=0.0)
        df["clock_after_runoff"] = gsr - df["runoff_seconds"]

    return df


def posteam_win_target(df: pd.DataFrame) -> np.ndarray:
    """1 if the possession team won, 0 if it lost, 0.5 for a tie."""
    result = pd.to_numeric(df["result"], errors="coerce")  # home margin
    home_win = np.where(result > 0, 1.0, np.where(result < 0, 0.0, 0.5))
    return np.where(df["posteam_is_home"] == 1, home_win, 1.0 - home_win)


def check_skill_signs(df: pd.DataFrame, y: np.ndarray, cols: list[str]) -> dict[str, float]:
    """
    Sanity check on the orientation of each pregame strength column: it should
    correlate positively with the possession team winning. Raises if not.
    """
    y = np.asarray(y, dtype=float)
    out = {}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").to_numpy()
        ok = np.isfinite(s) & np.isfinite(y)
        corr = float(np.corrcoef(s[ok], y[ok])[0, 1]) if ok.sum() > 1 else float("nan")
        out[c] = corr
        if not np.isfinite(corr) or corr <= 0:
            raise RuntimeError(
                f"{c} does not correlate positively with the possession team winning "
                f"(corr={corr}). The sign convention for that source is not "
                f"'positive = home favored'; fix it in nfl_common.prepare() before training."
            )
    return out
