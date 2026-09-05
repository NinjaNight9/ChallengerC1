#!/usr/bin/env python
"""Parallelized exact Rev5 fundamental execution.

Mode=horizon evaluates exactly one frozen history horizon on development data.
Mode=finalize compares the untouched horizon summaries, selects the lower frozen
selection score once, then opens the Jun-01..Aug-30 holdout for that winner.
Parallelism changes wall-clock only; model code, folds, seeds and scoring do not.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Import applies the verified Rev5 same-timestamp batching patch.
import rev5_exact_feature_patch  # noqa: F401
from rev5_final_fundamental import (
    apply_rev5_target_definition,
    base_model_metrics,
    calibration_table,
    grouped_metrics,
    metric_dict,
    validate_complete_tournament_weeks,
)
from challenger_c1.data import load_tennismylife_directory
from challenger_c1.features import ChallengerFeatureBuilder
from challenger_c1.model import (
    _metric_dict,
    fit_final_fundamental,
    save_json,
    walk_forward_fixed_architecture,
    walk_forward_fundamental,
)


def load_features(data_dir: str, years: list[int], config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    matches = load_tennismylife_directory(data_dir, years)
    matches = apply_rev5_target_definition(matches)
    builder = ChallengerFeatureBuilder(config)
    features = builder.build(matches)
    features["date"] = pd.to_datetime(features["date"])
    return matches, features, {"builder": builder}


def horizon_mode(args, config: dict) -> None:
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    matches, features, _ = load_features(args.data_dir, args.years, config)
    h0 = pd.Timestamp(args.holdout_start).normalize()
    dev = features[features["date"] < h0].copy()
    horizon = int(args.horizon)
    oof, folds = walk_forward_fundamental(dev, config, horizon)
    metrics = _metric_dict(oof["target"].to_numpy(int), oof["predicted_probability"].to_numpy(float))
    fold_ll_std = float(folds["log_loss"].std(ddof=0)) if len(folds) > 1 else 0.0
    score = float(metrics["log_loss"] + 0.12 * fold_ll_std + 0.20 * metrics["brier"])
    summary = {
        "horizon_days": horizon,
        **metrics,
        "fold_logloss_std": fold_ll_std,
        "selection_score": score,
        "development_rows": int(len(dev)),
        "oof_rows": int(len(oof)),
        "rev5_qualifying_targets_removed": int(matches.attrs.get("rev5_qualifying_targets_removed", 0)),
    }
    oof.to_csv(out / f"02_dev_oof_{horizon}d.csv", index=False)
    folds.to_csv(out / f"03_dev_folds_{horizon}d.csv", index=False)
    save_json(summary, out / f"horizon_{horizon}_summary.json")
    print(json.dumps(summary, indent=2))


def finalize_mode(args, config: dict) -> None:
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    incoming = Path(args.horizon_dir)
    summaries = []
    for horizon in (365, 730):
        p = incoming / f"horizon_{horizon}_summary.json"
        if not p.exists():
            raise FileNotFoundError(p)
        summaries.append(json.loads(p.read_text()))
    summaries.sort(key=lambda x: (x["selection_score"], x["log_loss"], x["brier"]))
    best = int(summaries[0]["horizon_days"])
    horizon_table = pd.DataFrame(summaries)
    horizon_table.to_csv(out / "01_horizon_comparison_development.csv", index=False)
    for horizon in (365, 730):
        for prefix in ("02_dev_oof", "03_dev_folds"):
            src = incoming / f"{prefix}_{horizon}d.csv"
            if src.exists():
                (out / src.name).write_bytes(src.read_bytes())

    matches = load_tennismylife_directory(args.data_dir, args.years)
    raw_rows = int(len(matches))
    matches = apply_rev5_target_definition(matches)
    provenance = validate_complete_tournament_weeks(matches, args.holdout_start, args.holdout_end)
    source_counts = matches.groupby(["source_kind", "level_role"], dropna=False).size().rename("matches").reset_index()
    source_counts.to_csv(out / "00_cross_level_source_counts.csv", index=False)

    builder = ChallengerFeatureBuilder(config)
    features = builder.build(matches)
    features["date"] = pd.to_datetime(features["date"])
    h0 = pd.Timestamp(args.holdout_start).normalize()
    h1 = pd.Timestamp(args.holdout_end).normalize()
    dev = features[features["date"] < h0].copy()
    holdout_available = features[(features["date"] >= h0) & (features["date"] <= h1)].copy()
    dev_oof = pd.read_csv(incoming / f"02_dev_oof_{best}d.csv", low_memory=False)

    dev_art = fit_final_fundamental(dev, config, best, dev_oof, cutoff=h0)
    holdout = walk_forward_fixed_architecture(
        features, config, best, h0, dev_art.weights, dev_art.calibrator, end_date=h1
    )
    if holdout.empty:
        raise RuntimeError("No holdout predictions produced")
    holdout.to_csv(out / "04_UNTOUCHED_holdout_predictions.csv", index=False)
    base_model_metrics(holdout).to_csv(out / "05_holdout_base_model_metrics.csv", index=False)
    grouped = holdout.copy()
    grouped["month"] = pd.to_datetime(grouped["date"]).dt.to_period("M").astype(str)
    grouped_metrics(grouped, "month").to_csv(out / "06_holdout_metrics_by_month.csv", index=False)
    grouped_metrics(grouped, "surface").to_csv(out / "07_holdout_metrics_by_surface.csv", index=False)
    if "round" in grouped:
        grouped_metrics(grouped, "round").to_csv(out / "08_holdout_metrics_by_round.csv", index=False)
    calibration_table(grouped).to_csv(out / "09_holdout_calibration.csv", index=False)

    holdout_metrics = metric_dict(holdout)
    dev_metrics = metric_dict(dev_oof)
    latest_cutoff = features["date"].max() + pd.Timedelta(days=1)
    meta_all = pd.concat([dev_oof, holdout], ignore_index=True)
    final_fund = fit_final_fundamental(features, config, best, meta_all, cutoff=latest_cutoff)
    final_fund.metadata.update({
        "revision": "Rev5",
        "parallel_execution_only": True,
        "architecture_selected_on_development_only": True,
        "holdout_graded_after_horizon_freeze": True,
        "selected_history_days": best,
        "holdout_start": str(h0.date()),
        "holdout_end": str(h1.date()),
        "results_provenance": provenance,
        "betting_policy_approved": False,
    })
    final_fund.save(out / "challenger_c1_fundamental_FROZEN.joblib")
    builder.h2h = defaultdict(partial(list, [0, 0]), dict(builder.h2h))
    joblib.dump(builder, out / "challenger_c1_feature_state_FROZEN.joblib")

    audit = {
        "model": "ChallengerC1-FUND",
        "revision": "Rev5",
        "execution": "parallel_horizons_exact_equivalent",
        "status": "C1-FUND frozen; no historical-odds profitability claim",
        "years_loaded": args.years,
        "cross_level_matches_loaded": raw_rows,
        "source_counts": source_counts.to_dict(orient="records"),
        "rev5_qualifying_target_rows_removed": int(matches.attrs.get("rev5_qualifying_targets_removed", 0)),
        "challenger_feature_rows": int(len(features)),
        "development_rows": int(len(dev)),
        "holdout_target_rows_available": int(len(holdout_available)),
        "holdout_predictions": int(len(holdout)),
        "holdout_start": str(h0.date()),
        "holdout_end": str(h1.date()),
        "selected_history_days": best,
        "horizon_comparison": summaries,
        "ensemble_weights_frozen_from_development": {
            name: float(w) for name, w in zip(dev_art.base_model_names, dev_art.weights)
        },
        "development_oof_metrics": dev_metrics,
        "untouched_holdout_metrics": holdout_metrics,
        "results_provenance": provenance,
        "betting_policy_approved": False,
    }
    save_json(audit, out / "FUNDAMENTAL_FREEZE_AUDIT.json")
    print(json.dumps(audit, indent=2, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["horizon", "finalize"], required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025, 2026])
    ap.add_argument("--config", default=str(ROOT / "config" / "default.json"))
    ap.add_argument("--holdout-start", default="2026-06-01")
    ap.add_argument("--holdout-end", default="2026-08-30")
    ap.add_argument("--horizon", type=int)
    ap.add_argument("--horizon-dir")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.mode == "horizon":
        if args.horizon not in (365, 730):
            raise ValueError("--horizon must be 365 or 730")
        horizon_mode(args, config)
    else:
        if not args.horizon_dir:
            raise ValueError("--horizon-dir is required for finalize")
        finalize_mode(args, config)


if __name__ == "__main__":
    main()
