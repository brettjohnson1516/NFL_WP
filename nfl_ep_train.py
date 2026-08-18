"""
nfl_ep_train.py
Stage 1 of the two-stage model.

Trains a 7-class next-score model on field position and down & distance:
    No_Score, TD, FG, Safety, Opp_TD, Opp_FG, Opp_Safety
Expected points is then the probability-weighted sum of the class values
(+/-7, +/-3, +/-2, 0) from the possession team's point of view.

Labels come from the next scoring event in the same half, derived from the
running score in the play-by-play.

Usage:
    python nfl_ep_train.py
    python nfl_ep_train.py --seasons 2021 2022 2023 2024

Writes:
    <model_dir>/nfl_ep_model.json
    <model_dir>/nfl_ep_meta.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from nfl_common import (
    EP_FEATURES,
    NEXT_SCORE_CLASSES,
    NEXT_SCORE_VALUES,
    available_seasons,
    cache_dir,
    load_pbp,
    model_dir,
    model_rows,
    prepare,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=None,
                    help="seasons to train on (default: 2021-2024 if present)")
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="fraction of GAMES held out for early stopping")
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--min-child-weight", type=float, default=50.0)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample", type=float, default=0.8)
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--early-stopping", type=int, default=60)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    md = model_dir()
    print(f"cache dir: {cache_dir()}")
    print(f"model dir: {md}")

    seasons = args.seasons
    if seasons is None:
        have = available_seasons()
        seasons = [s for s in (2021, 2022, 2023, 2024) if s in have]
        if not seasons:
            raise SystemExit(f"no usable pbp files in {cache_dir()}")
    print(f"seasons: {seasons}")

    raw = load_pbp(seasons)
    print(f"raw plays loaded: {len(raw):,}")

    df = prepare(raw)
    df = model_rows(df)
    print(f"model rows (regulation scrimmage snaps): {len(df):,}")

    class_index = {c: i for i, c in enumerate(NEXT_SCORE_CLASSES)}
    y = df["next_score_label"].map(class_index)
    if y.isna().any():
        bad = df.loc[y.isna(), "next_score_label"].unique()
        raise RuntimeError(f"unmapped next-score labels: {bad}")
    y = y.astype(int).to_numpy()

    dist = (
        pd.Series(np.asarray(NEXT_SCORE_CLASSES)[y])
        .value_counts(normalize=True)
        .reindex(NEXT_SCORE_CLASSES)
        .fillna(0.0)
    )
    print("\nnext-score label distribution:")
    for c in NEXT_SCORE_CLASSES:
        print(f"  {c:<12s} {dist[c]:6.3%}")

    X = df[EP_FEATURES].astype(float)

    # split by game so plays from one game never straddle the split
    games = np.asarray(df["game_id"].unique(), dtype=object)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(games)
    n_val = max(1, int(len(games) * args.val_frac))
    val_games = set(games[:n_val])
    is_val = df["game_id"].isin(val_games).to_numpy()
    print(f"\ngames: {len(games):,}  (validation games: {n_val:,})")
    print(f"train rows: {(~is_val).sum():,}   val rows: {is_val.sum():,}")

    dtrain = xgb.DMatrix(X[~is_val], label=y[~is_val], feature_names=EP_FEATURES)
    dval = xgb.DMatrix(X[is_val], label=y[is_val], feature_names=EP_FEATURES)

    params = {
        "objective": "multi:softprob",
        "num_class": len(NEXT_SCORE_CLASSES),
        "eval_metric": "mlogloss",
        "eta": args.eta,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample,
        "tree_method": "hist",
        "seed": args.seed,
    }

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=args.rounds,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=args.early_stopping,
        verbose_eval=50,
    )

    best_it = getattr(booster, "best_iteration", None)
    best_score = getattr(booster, "best_score", None)
    print(f"\nbest iteration: {best_it}   val mlogloss: {best_score}")

    # ---- sanity: EP curve for 1st & 10 by field position -------------------
    grid = pd.DataFrame({
        "down": 1.0,
        "ydstogo": 10.0,
        "yardline_100": np.arange(95, 4, -10, dtype=float),
        "half_seconds_remaining": 1800.0,
        "goal_to_go": 0.0,
        "posteam_is_home": 1.0,
        "posteam_timeouts_remaining": 3.0,
        "defteam_timeouts_remaining": 3.0,
    })
    dgrid = xgb.DMatrix(grid[EP_FEATURES], feature_names=EP_FEATURES)
    probs = booster.predict(dgrid, iteration_range=(0, (best_it or 0) + 1))
    ep_grid = probs @ NEXT_SCORE_VALUES
    print("\nEP sanity check, 1st & 10, 1800s left in half, 3 timeouts each:")
    for yl, e in zip(grid["yardline_100"], ep_grid):
        print(f"  yardline_100={yl:5.0f}   EP={e:6.3f}")

    model_path = md / "nfl_ep_model.json"
    meta_path = md / "nfl_ep_meta.json"
    booster.save_model(str(model_path))
    meta = {
        "features": EP_FEATURES,
        "classes": NEXT_SCORE_CLASSES,
        "class_values": NEXT_SCORE_VALUES.tolist(),
        "seasons": seasons,
        "best_iteration": int(best_it) if best_it is not None else None,
        "val_mlogloss": float(best_score) if best_score is not None else None,
        "params": params,
        "n_train_rows": int((~is_val).sum()),
        "n_val_rows": int(is_val.sum()),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {model_path}")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
