"""
nfl_wp_predict.py
Scores a live game state with the trained two-stage model.

The WP booster's raw output is passed through the isotonic map in
nfl_wp_calibration.json (fitted on leave-one-season-out predictions during
training). Use --no-calibration to see the raw booster output instead; both are
returned in batch mode as wp_raw and wp.

The required inputs depend on which --skill-source the WP model was trained
with; the feature list is read back from nfl_wp_meta.json, so this stays in
sync automatically.

    skill_source = spread      -> needs spread_line
    skill_source = skill_diff  -> needs skill_diff
    skill_source = both        -> needs both

Both are given from the HOME team's point of view, matching training:
positive = home team favored. score_differential is always from the possession
team's point of view.

Importable:
    from nfl_wp_predict import NflWinProb
    m = NflWinProb()
    m.score_state(qtr=3, clock="07:32", score_differential=-4, down=2, ydstogo=8,
                  yardline_100=61, posteam_is_home=1, spread_line=-2.5,
                  skill_diff=-0.44, posteam_timeouts=3, defteam_timeouts=2)

Command line:
    python nfl_wp_predict.py --qtr 3 --clock 07:32 --score-diff -4 --down 2 `
        --ydstogo 8 --yardline-100 61 --posteam-is-home 1 --spread-line -2.5 `
        --skill-diff -0.44 --posteam-timeouts 3 --defteam-timeouts 2

Batch:
    python nfl_wp_predict.py --csv states.csv --out scored.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from nfl_common import EP_FEATURES, NEXT_SCORE_VALUES, model_dir

BASE_STATE_COLUMNS = [
    "qtr",
    "clock_seconds",
    "score_differential",
    "down",
    "ydstogo",
    "yardline_100",
    "posteam_is_home",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
]


def clock_to_seconds(clock) -> float:
    """Accepts 'MM:SS', 'M:SS' or a number of seconds remaining in the quarter."""
    if isinstance(clock, (int, float, np.integer, np.floating)):
        return float(clock)
    s = str(clock).strip()
    if ":" in s:
        mm, ss = s.split(":")
        return float(int(mm) * 60 + float(ss))
    return float(s)


class NflWinProb:
    def __init__(self, ep_model=None, wp_model=None, model_directory=None,
                 use_calibration: bool = True):
        md = Path(model_directory) if model_directory else model_dir(create=False)
        self.model_directory = md
        self.ep_path = Path(ep_model) if ep_model else md / "nfl_ep_model.json"
        self.wp_path = Path(wp_model) if wp_model else md / "nfl_wp_model.json"

        for p in (self.ep_path, self.wp_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} not found. Train first, or point at the right directory "
                    f"with NFL_MODEL_DIR / --model-dir."
                )

        self.ep = xgb.Booster()
        self.ep.load_model(str(self.ep_path))
        self.wp = xgb.Booster()
        self.wp.load_model(str(self.wp_path))

        self.ep_meta = self._read_json(md / "nfl_ep_meta.json")
        self.wp_meta = self._read_json(md / "nfl_wp_meta.json")
        self.ep_best = self.ep_meta.get("best_iteration")

        self.features = self.wp_meta.get("features")
        if not self.features:
            raise RuntimeError(
                f"{md / 'nfl_wp_meta.json'} has no feature list; retrain the WP model."
            )
        self.needs_spread = "posteam_spread" in self.features
        self.needs_skill = "posteam_skill_diff" in self.features
        self.ytd_inputs = [c for c in ("posteam_off_ypp", "posteam_def_ypp",
                                       "defteam_off_ypp", "defteam_def_ypp")
                           if c in self.features
                           or "off_ypp_matchup" in self.features]
        self.needs_total = "pregame_total" in self.features
        self.context_inputs = [c for c in ("wind_mph", "temp_f", "is_indoors")
                               if c in self.features]
        self.qb_inputs = [c for c in ("posteam_qb_changed", "defteam_qb_changed")
                          if c in self.features or "qb_change_diff" in self.features]
        self.ingame_inputs = [c for c in ("posteam_epa_pp", "defteam_epa_pp",
                                          "plays_so_far")
                              if c in self.features or "epa_pp_diff" in self.features]

        calib = self._read_json(md / "nfl_wp_calibration.json")
        self.x_knots = calib.get("x_knots") or []
        self.y_knots = calib.get("y_knots") or []
        self.use_calibration = bool(use_calibration and self.x_knots)
        self.calibration_available = bool(self.x_knots)

    @staticmethod
    def _read_json(path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except (ValueError, OSError):
            return {}

    def required_state_columns(self) -> list[str]:
        cols = list(BASE_STATE_COLUMNS)
        if self.needs_spread:
            cols.append("spread_line")
        if self.needs_skill:
            cols.append("skill_diff")
        cols += self.ytd_inputs + self.context_inputs + self.qb_inputs
        if self.needs_total:
            cols.append("pregame_total")
        cols += self.ingame_inputs
        return cols

    def calibrate(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if not self.use_calibration:
            return p
        return np.clip(np.interp(p, self.x_knots, self.y_knots), 0.0, 1.0)

    # ------------------------------------------------------------------
    def score_frame(self, states: pd.DataFrame) -> pd.DataFrame:
        """states: one row per game state. Returns the frame with ep, wp_raw,
        wp (calibrated when available) and home_wp."""
        df = states.copy()

        missing = [c for c in self.required_state_columns() if c not in df.columns]
        if missing:
            raise ValueError(f"missing state column(s): {missing}")
        if "receive_2h_ko" not in df.columns:
            df["receive_2h_ko"] = 0

        df["clock_seconds"] = df["clock_seconds"].map(clock_to_seconds).astype(float)
        df["qtr"] = df["qtr"].astype(float)

        # regulation clock derivation
        df["game_seconds_remaining"] = np.where(
            df["qtr"] <= 4, df["clock_seconds"] + 900.0 * (4.0 - df["qtr"]), 0.0
        ).clip(min=0.0)
        df["half_seconds_remaining"] = np.where(
            df["qtr"] <= 2,
            df["clock_seconds"] + 900.0 * (2.0 - df["qtr"]),
            np.where(df["qtr"] <= 4, df["clock_seconds"] + 900.0 * (4.0 - df["qtr"]),
                     df["clock_seconds"]),
        ).clip(min=0.0)

        df["goal_to_go"] = (df["yardline_100"] <= df["ydstogo"]).astype(int)
        df["is_second_half"] = (df["qtr"] >= 3).astype(int)

        # ---- stage 1
        dep = xgb.DMatrix(df[EP_FEATURES].astype(float), feature_names=EP_FEATURES)
        it = (0, self.ep_best + 1) if self.ep_best is not None else None
        probs = self.ep.predict(dep, iteration_range=it) if it else self.ep.predict(dep)
        df["ep"] = probs @ NEXT_SCORE_VALUES

        # ---- stage 2
        df["exp_score_diff"] = df["score_differential"] + df["ep"]
        elapsed_share = (3600.0 - df["game_seconds_remaining"].clip(lower=0)) / 3600.0
        decay = np.exp(-4.0 * elapsed_share)
        df["diff_time_ratio"] = df["exp_score_diff"] / decay

        if self.needs_spread:
            df["posteam_spread"] = np.where(
                df["posteam_is_home"] == 1, df["spread_line"], -df["spread_line"]
            )
            df["spread_time"] = df["posteam_spread"] * decay
        if self.needs_skill:
            df["posteam_skill_diff"] = np.where(
                df["posteam_is_home"] == 1, df["skill_diff"], -df["skill_diff"]
            )
            df["skill_diff_time"] = df["posteam_skill_diff"] * decay

        if self.needs_total:
            gsr = df["game_seconds_remaining"].clip(lower=0)
            total = pd.to_numeric(df["pregame_total"], errors="coerce")
            df["remaining_points"] = total * gsr / 3600.0
            df["std_lead"] = df["exp_score_diff"] / np.sqrt(
                df["remaining_points"].clip(lower=1.0))
            df["std_spread"] = (df["posteam_spread"] / np.sqrt(total.clip(lower=1.0))
                                if "posteam_spread" in df.columns else np.nan)
            to_left = pd.to_numeric(df["defteam_timeouts_remaining"], errors="coerce")
            df["runoff_seconds"] = (40.0 * (3.0 - to_left)).clip(lower=0.0)
            df["clock_after_runoff"] = gsr - df["runoff_seconds"]

        if "qb_change_diff" in self.features:
            df["qb_change_diff"] = df["defteam_qb_changed"] - df["posteam_qb_changed"]

        if "epa_pp_diff" in self.features:
            df["epa_pp_diff"] = df["posteam_epa_pp"] - df["defteam_epa_pp"]

        if "off_ypp_matchup" in self.features:
            df["off_ypp_matchup"] = df["posteam_off_ypp"] - df["defteam_def_ypp"]
            df["def_ypp_matchup"] = df["defteam_off_ypp"] - df["posteam_def_ypp"]

        dwp = xgb.DMatrix(df[self.features].astype(float), feature_names=self.features)
        df["wp_raw"] = self.wp.predict(dwp)
        df["wp"] = self.calibrate(df["wp_raw"].to_numpy())

        df["home_wp"] = np.where(df["posteam_is_home"] == 1, df["wp"], 1.0 - df["wp"])
        return df

    def score_state(
        self,
        qtr,
        clock,
        score_differential,
        down,
        ydstogo,
        yardline_100,
        posteam_is_home,
        spread_line=None,
        skill_diff=None,
        pregame_total=None,
        posteam_timeouts=3,
        defteam_timeouts=3,
        receive_2h_ko=0,
    ) -> dict:
        row = {
            "qtr": qtr,
            "clock_seconds": clock,
            "score_differential": score_differential,
            "down": down,
            "ydstogo": ydstogo,
            "yardline_100": yardline_100,
            "posteam_is_home": posteam_is_home,
            "posteam_timeouts_remaining": posteam_timeouts,
            "defteam_timeouts_remaining": defteam_timeouts,
            "receive_2h_ko": receive_2h_ko,
        }
        if self.needs_spread:
            if spread_line is None:
                raise ValueError("this model was trained with spread_line; supply it")
            row["spread_line"] = spread_line
        if self.needs_skill:
            if skill_diff is None:
                raise ValueError("this model was trained with skill_diff; supply it")
            row["skill_diff"] = skill_diff
        if self.needs_total:
            if pregame_total is None:
                raise ValueError("this model was trained with pregame_total; supply it")
            row["pregame_total"] = pregame_total

        out = self.score_frame(pd.DataFrame([row])).iloc[0]
        return {
            "ep": float(out["ep"]),
            "posteam_wp": float(out["wp"]),
            "posteam_wp_raw": float(out["wp_raw"]),
            "home_wp": float(out["home_wp"]),
            "game_seconds_remaining": float(out["game_seconds_remaining"]),
            "half_seconds_remaining": float(out["half_seconds_remaining"]),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--qtr", type=int)
    ap.add_argument("--clock", type=str, help="MM:SS remaining in the quarter")
    ap.add_argument("--score-diff", type=float, help="possession team minus opponent")
    ap.add_argument("--down", type=int)
    ap.add_argument("--ydstogo", type=float)
    ap.add_argument("--yardline-100", type=float, help="yards from the opponent goal line")
    ap.add_argument("--posteam-is-home", type=int, choices=[0, 1])
    ap.add_argument("--spread-line", type=float, help="closing spread, positive = home favored")
    ap.add_argument("--skill-diff", type=float, help="skill_diff, positive = home favored")
    ap.add_argument("--pregame-total", type=float, help="closing total for the game")
    ap.add_argument("--posteam-timeouts", type=int, default=3)
    ap.add_argument("--defteam-timeouts", type=int, default=3)
    ap.add_argument("--receive-2h-ko", type=int, default=0)
    ap.add_argument("--no-calibration", action="store_true",
                    help="return the raw booster output instead of the calibrated one")
    ap.add_argument("--model-dir", type=str, default=None)
    ap.add_argument("--ep-model", type=str, default=None)
    ap.add_argument("--wp-model", type=str, default=None)
    args = ap.parse_args()

    m = NflWinProb(ep_model=args.ep_model, wp_model=args.wp_model,
                   model_directory=args.model_dir,
                   use_calibration=not args.no_calibration)
    print(f"model dir: {m.model_directory}")
    print(f"skill source: {m.wp_meta.get('skill_source')}")
    if not m.calibration_available:
        print("calibration: none found (nfl_wp_calibration.json missing)")
    else:
        print(f"calibration: {'on' if m.use_calibration else 'off'}")

    if args.csv:
        scored = m.score_frame(pd.read_csv(args.csv))
        if args.out:
            scored.to_csv(args.out, index=False)
            print(f"wrote {args.out}")
        else:
            print(scored.to_string(index=False))
        return

    required = [args.qtr, args.clock, args.score_diff, args.down, args.ydstogo,
                args.yardline_100, args.posteam_is_home]
    if any(v is None for v in required):
        ap.error("provide --csv, or all of --qtr --clock --score-diff --down "
                 "--ydstogo --yardline-100 --posteam-is-home")

    res = m.score_state(
        qtr=args.qtr,
        clock=args.clock,
        score_differential=args.score_diff,
        down=args.down,
        ydstogo=args.ydstogo,
        yardline_100=args.yardline_100,
        posteam_is_home=args.posteam_is_home,
        spread_line=args.spread_line,
        skill_diff=args.skill_diff,
        pregame_total=args.pregame_total,
        posteam_timeouts=args.posteam_timeouts,
        defteam_timeouts=args.defteam_timeouts,
        receive_2h_ko=args.receive_2h_ko,
    )
    print(f"EP  (possession team) : {res['ep']:+.3f}")
    print(f"WP  (possession team) : {res['posteam_wp']:.4f}  (raw {res['posteam_wp_raw']:.4f})")
    print(f"WP  (home team)       : {res['home_wp']:.4f}")


if __name__ == "__main__":
    main()
