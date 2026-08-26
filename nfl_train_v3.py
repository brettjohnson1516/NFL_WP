"""
nfl_wp_train.py
Stage 2 of the two-stage model.

Takes the stage-1 expected-points output as a feature and trains a binary
win-probability model on score differential, EP, expected score differential,
clock, down & distance, field position, timeouts, second-half kickoff
possession, and the pregame team-strength term(s) selected by --skill-source.

Fitting procedure
-----------------
1. Leave-one-season-out over the training seasons. For each season s, fit on
   the other training seasons and early-stop on s. This measures generalisation
   to an unseen SEASON, which a random slice of games from all seasons cannot.
2. The out-of-fold predictions from step 1 (every training play predicted by a
   model that never saw its season) are pooled and used to fit an isotonic
   calibration map.
3. The final booster is refit on all training seasons for the median of the
   per-fold best iterations, scaled by K/(K-1) for the extra data.
4. The eval season is scored with the final booster, raw and calibrated.
   With --calib-bins 0 steps 2 and the calibration map are skipped entirely and
   every reported figure is the raw booster output.

--skill-source
    spread      spread_line from schedules.parquet, flipped to the possession
                team's view (positive = possession team favored)
    skill_diff  skill_diff from odds_api_lines_nfl_YYYY.parquet, same flip
    both        (default) supply both to the booster

Usage:
    python nfl_wp_train.py
    python nfl_wp_train.py --seasons 2021 2022 2023 2024 --eval-season 2025
    python nfl_wp_train.py --skill-source spread

Writes:
    <model_dir>/nfl_wp_model.json
    <model_dir>/nfl_wp_calibration.json
    <model_dir>/nfl_wp_meta.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from nfl_common import (
    COMMON_VERSION,
    EP_FEATURES,
    NEXT_SCORE_VALUES,
    SKILL_SOURCES,
    add_wp_features,
    available_seasons,
    cache_dir,
    check_skill_signs,
    load_pbp,
    model_dir,
    model_rows,
    monotone_constraints,
    resolve_saber,
    posteam_win_target,
    prepare,
    skill_columns,
    wp_features,
)


SCRIPT_VERSION = "2026-08-09f"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=None,
                    help="training seasons (default: 2021-2024 if present)")
    ap.add_argument("--eval-season", type=int, nargs="+", default=None,
                    help="pure holdout season(s), reported only (default: 2025 if present)")
    ap.add_argument("--skill-source", choices=SKILL_SOURCES, default="both",
                    help="which pregame team-strength term(s) to use")
    ap.add_argument("--saber", action="store_true",
                    help="add the in-game sabermetric rate differentials: success "
                         "rate, EPA by dropback/rush/early down, explosive rate, "
                         "CPOE, series conversion, pass rate over expected, "
                         "disruption allowed, third down rate, turnovers, penalties")
    ap.add_argument("--saber-only", type=str, default=None,
                    help="restrict --saber to a subset: comma list of feature short "
                         "names (success, epa_dropback, epa_rush, early_epa, "
                         "explosive, cpoe, series, pass_oe, disruption, third_conv, "
                         "turnover, penalty, plays) or group names (efficiency, "
                         "explosive, tendency, disruption, situational, volume)")
    ap.add_argument("--context", action="store_true",
                    help="add wind, temperature and indoor/outdoor from schedules")
    ap.add_argument("--qb", action="store_true",
                    help="add in-game quarterback-change flags for both sides")
    ap.add_argument("--round-select", choices=["median", "curve"], default="median",
                    help="how the final round count is chosen: median of the fold "
                         "best iterations, or the minimum of the averaged fold "
                         "validation curves (slower, no early stopping)")
    ap.add_argument("--curve-rounds", type=int, default=800,
                    help="rounds to run per fold when --round-select curve")
    ap.add_argument("--ingame", action="store_true",
                    help="add each side's EPA per play so far in the current game, "
                         "counting only plays before the current one")
    ap.add_argument("--monotone", action="store_true",
                    help="force win probability to rise with expected score "
                         "differential and with the pregame line")
    ap.add_argument("--v2", action="store_true",
                    help="add the structural feature group: pregame total, remaining "
                         "expected points, variance-scaled lead and spread, and "
                         "rule-based clock runoff")
    ap.add_argument("--ytd", action="store_true",
                    help="add season-to-date yards per play (offense, defense, and "
                         "the two matchup differentials) for both teams, counting "
                         "only games earlier in the same season")
    ap.add_argument("--ep-model", type=str, default=None,
                    help="path to nfl_ep_model.json (default: <model_dir>/nfl_ep_model.json)")
    ap.add_argument("--calib-season", type=int, default=None,
                    help="hold one TRAINING season out of the final fit and use it "
                         "only to fit the calibration map, so the map is measured on "
                         "predictions from the final model rather than from the folds")
    ap.add_argument("--calib-bins", type=int, default=1000,
                    help="quantile bins used to fit the isotonic map (0 disables calibration)")
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--eta", type=float, default=0.03)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--min-child-weight", type=float, default=300.0)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample", type=float, default=0.7)
    ap.add_argument("--reg-lambda", type=float, default=5.0)
    ap.add_argument("--reg-alpha", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=5000)
    ap.add_argument("--early-stopping", type=int, default=100)
    return ap.parse_args()


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    eps = 1e-9
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y, dtype=float)
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    brier = float(np.mean((p - y) ** 2))
    rate = float(np.mean(y))
    base = min(max(rate, eps), 1 - eps)
    base_ll = float(-np.mean(y * np.log(base) + (1 - y) * np.log(1 - base)))
    return {"n": int(len(y)), "logloss": logloss, "brier": brier,
            "base_rate": rate, "base_rate_logloss": base_ll}


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        rows.append({
            "bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}",
            "n": int(m.sum()),
            "pred": float(p[m].mean()),
            "actual": float(y[m].mean()),
            "gap": float(p[m].mean() - y[m].mean()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# isotonic calibration (pool adjacent violators, no sklearn dependency)
# --------------------------------------------------------------------------

def _pava(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    vals: list[float] = []
    wts: list[float] = []
    cnt: list[int] = []
    for v, w in zip(values, weights):
        vals.append(float(v))
        wts.append(float(w))
        cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnt.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            wts.append(w1 + w2)
            cnt.append(c1 + c2)
    return np.concatenate([np.full(c, v) for v, c in zip(vals, cnt)])


def fit_isotonic(p: np.ndarray, y: np.ndarray, n_bins: int = 1000) -> tuple[list, list]:
    """Returns (x_knots, y_knots) of a monotone step map from raw to calibrated."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(p, kind="mergesort")
    p, y = p[order], y[order]

    n = len(p)
    k = int(min(max(n_bins, 2), n))
    edges = np.linspace(0, n, k + 1).astype(int)

    xs, ys, ws = [], [], []
    for i in range(k):
        a, b = edges[i], edges[i + 1]
        if b <= a:
            continue
        xs.append(float(p[a:b].mean()))
        ys.append(float(y[a:b].mean()))
        ws.append(float(b - a))

    xs = np.asarray(xs)
    ys = _pava(np.asarray(ys), np.asarray(ws))

    # np.interp needs strictly increasing x; keep the last entry of any tie
    keep = np.ones(len(xs), dtype=bool)
    keep[:-1] = xs[1:] > xs[:-1]
    xs, ys = xs[keep], ys[keep]
    return xs.tolist(), ys.tolist()


def apply_isotonic(p: np.ndarray, x_knots, y_knots) -> np.ndarray:
    if not x_knots:
        return np.asarray(p, dtype=float)
    return np.clip(np.interp(np.asarray(p, dtype=float), x_knots, y_knots), 0.0, 1.0)


# --------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    md = model_dir()
    print(f"script version: {SCRIPT_VERSION}")
    print(f"nfl_common version: {COMMON_VERSION}")
    print(f"cache dir: {cache_dir()}")
    print(f"model dir: {md}")

    have = available_seasons()

    train_seasons = args.seasons
    if train_seasons is None:
        train_seasons = [s for s in (2021, 2022, 2023, 2024) if s in have]
    train_seasons = sorted(set(train_seasons))
    if not train_seasons:
        raise SystemExit(f"no usable training pbp files in {cache_dir()}")

    if args.eval_season is None:
        eval_seasons = [s for s in (2025,) if s in have]
    else:
        eval_seasons = sorted(set(args.eval_season))
        overlap = set(eval_seasons) & set(train_seasons)
        if overlap:
            raise SystemExit(
                f"season(s) {sorted(overlap)} appear in both --seasons and --eval-season"
            )

    feats = wp_features(args.skill_source, ytd=args.ytd, v2=args.v2,
                        ingame=args.ingame, context=args.context, qb=args.qb,
                        saber=(resolve_saber(args.saber_only) if args.saber else False))
    skill_cols = skill_columns(args.skill_source)

    print(f"train seasons: {train_seasons}")
    print(f"eval  seasons: {eval_seasons if eval_seasons else '(none)'}")
    print(f"skill source : {args.skill_source} -> {skill_cols}")
    print(f"season-to-date efficiency features: {'on' if args.ytd else 'off'}")
    print(f"structural (v2) features: {'on' if args.v2 else 'off'}")
    print(f"in-game form features: {'on' if args.ingame else 'off'}")
    print(f"monotone constraints: {'on' if args.monotone else 'off'}")
    print(f"game context features: {'on' if args.context else 'off'}")
    print(f"QB-change features: {'on' if args.qb else 'off'}")
    print(f"sabermetric features: {'on' if args.saber else 'off'}"
          + (f" -> {args.saber_only}" if args.saber and args.saber_only else ""))
    print(f"round selection: {args.round_select}")

    ep_path = args.ep_model or (md / "nfl_ep_model.json")
    ep_booster = xgb.Booster()
    ep_booster.load_model(str(ep_path))
    print(f"loaded EP model: {ep_path}")

    need_lines = args.skill_source in ("skill_diff", "both") or args.v2
    raw = load_pbp(sorted(set(train_seasons) | set(eval_seasons)), with_lines=need_lines)
    df = prepare(raw)
    df = model_rows(df)
    print(f"model rows: {len(df):,}")

    usable = pd.to_numeric(df["result"], errors="coerce").notna()
    for c in skill_cols:
        if c not in df.columns:
            raise SystemExit(
                f"{c} was not built. Check the load step above for a missing "
                f"schedules.parquet or odds_api_lines file."
            )
        usable &= df[c].notna()
    dropped = int((~usable).sum())
    if dropped:
        print(f"dropping {dropped:,} rows with no final result or no pregame line")
    df = df.loc[usable].reset_index(drop=True)

    # ---- stage 1 -> ep -----------------------------------------------------
    dep = xgb.DMatrix(df[EP_FEATURES].astype(float), feature_names=EP_FEATURES)
    ep = ep_booster.predict(dep) @ NEXT_SCORE_VALUES
    df = add_wp_features(df, ep, v2=args.v2)
    print(f"EP: mean={ep.mean():.3f}  min={ep.min():.3f}  max={ep.max():.3f}")

    y = posteam_win_target(df)

    corrs = check_skill_signs(df, y, skill_cols)
    for c, v in corrs.items():
        print(f"corr({c}, posteam_win) = {v:+.4f}")

    X = df[feats].astype(float)
    season = pd.to_numeric(df["season"], errors="coerce").to_numpy()
    is_eval = np.isin(season, eval_seasons)
    in_pool = np.isin(season, train_seasons)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": args.eta,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample,
        "lambda": args.reg_lambda,
        "alpha": args.reg_alpha,
        "tree_method": "hist",
        "seed": args.seed,
    }
    if args.monotone:
        params["monotone_constraints"] = monotone_constraints(feats)

    # ---- leave-one-season-out ---------------------------------------------
    K = len(train_seasons)
    oof_pred = np.full(len(df), np.nan)
    best_iters: list[int] = []
    fold_curves: list[np.ndarray] = []
    fold_metrics: dict[str, dict] = {}

    if K < 2:
        raise SystemExit(
            "leave-one-season-out needs at least 2 training seasons; "
            f"got {train_seasons}"
        )

    print(f"\n=== leave-one-season-out over {train_seasons} ===")
    for s in train_seasons:
        hold = in_pool & (season == s)
        fit = in_pool & (season != s)

        dfit = xgb.DMatrix(X[fit], label=y[fit], feature_names=feats)
        dhold = xgb.DMatrix(X[hold], label=y[hold], feature_names=feats)

        fold_hist: dict = {}
        if args.round_select == "curve":
            bst = xgb.train(
                params, dfit, num_boost_round=args.curve_rounds,
                evals=[(dhold, "hold")], evals_result=fold_hist, verbose_eval=False,
            )
            curve = np.asarray(fold_hist["hold"]["logloss"], dtype=float)
            fold_curves.append(curve)
            bi = int(np.argmin(curve))
        else:
            bst = xgb.train(
                params, dfit, num_boost_round=args.rounds,
                evals=[(dhold, f"hold{s}")],
                early_stopping_rounds=args.early_stopping, verbose_eval=False,
            )
            bi = int(getattr(bst, "best_iteration", 0) or 0)
        best_iters.append(bi)

        p = bst.predict(dhold, iteration_range=(0, bi + 1))
        oof_pred[hold] = p
        m = metrics(y[hold], p)
        fold_metrics[str(s)] = {**m, "best_iteration": bi}
        print(f"  fold {s}: best_iter={bi:>5}  n={m['n']:>7,}  "
              f"logloss={m['logloss']:.5f}  brier={m['brier']:.5f}")

    oof_mask = in_pool & np.isfinite(oof_pred)
    m_oof = metrics(y[oof_mask], oof_pred[oof_mask])
    print(f"\npooled out-of-fold: logloss={m_oof['logloss']:.5f}  "
          f"brier={m_oof['brier']:.5f}  n={m_oof['n']:,}")

    if args.calib_season is not None and args.calib_season not in train_seasons:
        raise SystemExit(f"--calib-season {args.calib_season} is not in {train_seasons}")

    # ---- calibration map from the out-of-fold predictions -------------------
    if args.calib_bins > 0 and args.calib_season is None:
        x_knots, y_knots = fit_isotonic(
            oof_pred[oof_mask], y[oof_mask], n_bins=args.calib_bins
        )
        m_oof_cal = metrics(
            y[oof_mask], apply_isotonic(oof_pred[oof_mask], x_knots, y_knots)
        )
        print(f"out-of-fold calibrated: logloss={m_oof_cal['logloss']:.5f}  "
              f"brier={m_oof_cal['brier']:.5f}  ({len(x_knots)} knots)")
    else:
        x_knots, y_knots = [], []
        m_oof_cal = None
        if args.calib_bins <= 0:
            print("calibration disabled (--calib-bins 0)")

    # ---- final refit -------------------------------------------------------

    if args.round_select == "curve" and fold_curves:
        L = min(len(c) for c in fold_curves)
        mean_curve = np.mean([c[:L] for c in fold_curves], axis=0)
        best_round = int(np.argmin(mean_curve))
        n_rounds = max(1, int(round((best_round + 1) * K / (K - 1))))
        print(f"\naveraged fold curve minimum at round {best_round} "
              f"(mean logloss {mean_curve[best_round]:.5f}); "
              f"per-fold minima {best_iters}")
    else:
        n_rounds = int(round(float(np.median(best_iters) + 1) * K / (K - 1)))
    n_rounds = max(n_rounds, 1)

    fit_seasons = [s for s in train_seasons if s != args.calib_season]
    fit_mask = np.isin(season, fit_seasons)
    if args.calib_season is not None:
        n_rounds = max(1, int(round(n_rounds * len(fit_seasons) / K)))
        print(f"\nfinal refit on {fit_seasons} for {n_rounds} rounds; "
              f"{args.calib_season} reserved for calibration only")
    else:
        print(f"\nfinal refit on {train_seasons} for {n_rounds} rounds "
              f"(median fold best_iter {int(np.median(best_iters))}, scaled by {K}/{K - 1})")

    dall = xgb.DMatrix(X[fit_mask], label=y[fit_mask], feature_names=feats)
    booster = xgb.train(params, dall, num_boost_round=n_rounds, verbose_eval=False)

    # ---- calibration fitted on a season the FINAL model never saw ----------
    if args.calib_season is not None and args.calib_bins > 0:
        cal_mask = season == args.calib_season
        dcal = xgb.DMatrix(X[cal_mask], label=y[cal_mask], feature_names=feats)
        p_cal_raw = booster.predict(dcal)
        y_cal = y[cal_mask]
        x_knots, y_knots = fit_isotonic(p_cal_raw, y_cal, n_bins=args.calib_bins)
        before = metrics(y_cal, p_cal_raw)
        after = metrics(y_cal, apply_isotonic(p_cal_raw, x_knots, y_knots))
        print(f"calibration fitted on {args.calib_season} "
              f"({int(cal_mask.sum()):,} rows, {len(x_knots)} knots): "
              f"logloss {before['logloss']:.5f} -> {after['logloss']:.5f} "
              f"(in-sample for the map)")

    # ---- holdout report ----------------------------------------------------
    has_calib = bool(x_knots)
    label = "calibrated" if has_calib else "raw"

    m_eval = m_eval_cal = None
    if is_eval.sum() > 0:
        deval = xgb.DMatrix(X[is_eval], label=y[is_eval], feature_names=feats)
        p_raw = booster.predict(deval)
        p_cal = apply_isotonic(p_raw, x_knots, y_knots)
        y_eval = y[is_eval]

        m_eval = metrics(y_eval, p_raw)
        m_eval_cal = metrics(y_eval, p_cal) if has_calib else None
        print(f"\nHOLDOUT {eval_seasons}  n={m_eval['n']:,}  "
              f"base logloss={m_eval['base_rate_logloss']:.5f}")
        print(f"  raw       : logloss={m_eval['logloss']:.5f}  brier={m_eval['brier']:.5f}")
        if has_calib:
            print(f"  calibrated: logloss={m_eval_cal['logloss']:.5f}  "
                  f"brier={m_eval_cal['brier']:.5f}")

        print(f"\ncalibration (holdout, {label} predictions):")
        print(calibration_table(y_eval, p_cal).to_string(index=False))

        print(f"\nholdout by quarter ({label}):")
        q = pd.to_numeric(df.loc[is_eval, "qtr"], errors="coerce").to_numpy()
        for qq in (1, 2, 3, 4):
            mq = q == qq
            if mq.sum() == 0:
                continue
            mm = metrics(y_eval[mq], p_cal[mq])
            print(f"  Q{qq}  n={mm['n']:>7,}  logloss={mm['logloss']:.5f}  "
                  f"brier={mm['brier']:.5f}")
    else:
        print("\nWARNING: no eval-season rows found - holdout report skipped")

    print("\nfeature gain:")
    gain = booster.get_score(importance_type="gain")
    for k, v in sorted(gain.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<28s} {v:10.2f}")

    # ---- save --------------------------------------------------------------
    model_path = md / "nfl_wp_model.json"
    calib_path = md / "nfl_wp_calibration.json"
    meta_path = md / "nfl_wp_meta.json"

    booster.save_model(str(model_path))
    calib_path.write_text(json.dumps(
        {"method": "isotonic", "x_knots": x_knots, "y_knots": y_knots,
         "fitted_on": "pooled leave-one-season-out predictions",
         "train_seasons": train_seasons},
        indent=2,
    ))

    meta = {
        "features": feats,
        "skill_source": args.skill_source,
        "ytd": bool(args.ytd),
        "v2": bool(args.v2),
        "ingame": bool(args.ingame),
        "saber": bool(args.saber),
        "saber_only": args.saber_only,
        "context": bool(args.context),
        "qb": bool(args.qb),
        "monotone": bool(args.monotone),
        "ep_features": EP_FEATURES,
        "ep_model": str(ep_path),
        "train_seasons": train_seasons,
        "eval_seasons": eval_seasons,
        "fit_procedure": "leave-one-season-out early stopping; final refit on the "
                         "training seasons, minus --calib-season when given, whose "
                         "predictions then fit the isotonic map",
        "fit_seasons": fit_seasons,
        "final_rounds": n_rounds,
        "fold_best_iterations": best_iters,
        "fold_metrics": fold_metrics,
        "oof_metrics": m_oof,
        "oof_metrics_calibrated": m_oof_cal,
        "holdout_metrics_raw": m_eval,
        "holdout_metrics_calibrated": m_eval_cal,
        "calibration_file": str(calib_path),
        "skill_sign_corr": corrs,
        "sign_convention": "posteam_spread / posteam_skill_diff positive = possession "
                           "team favored; source spread_line and skill_diff positive = "
                           "home team favored",
        "params": params,
        "n_train_rows": int(in_pool.sum()),
        "n_eval_rows": int(is_eval.sum()),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {model_path}")
    print(f"wrote {calib_path}")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
