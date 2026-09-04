#!/usr/bin/env python
"""Leakage-safe fundamental benchmark for ChallengerC1.

This deliberately excludes betting odds. It chooses the 365d vs 730d history
window on development data only, freezes ensemble weights/calibration, then
scores the untouched 2026-06-01..2026-08-31 holdout with blockwise base-model
refits that can only use information available before each block.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from challenger_c1.data import load_tennismylife_directory
from challenger_c1.features import ChallengerFeatureBuilder
from challenger_c1.model import (
    _metric_dict,
    compare_horizons,
    fit_final_fundamental,
    save_json,
    walk_forward_fixed_architecture,
)


def metric_dict(frame: pd.DataFrame, pred_col: str = "predicted_probability") -> dict:
    if frame.empty:
        return {"n": 0, "accuracy": None, "brier": None, "log_loss": None, "auc": None}
    return _metric_dict(frame["target"].to_numpy(int), frame[pred_col].to_numpy(float))


def grouped_metrics(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    rows = []
    for value, g in frame.groupby(key, dropna=False):
        m = metric_dict(g)
        m[key] = str(value)
        rows.append(m)
    cols = [key, "n", "accuracy", "brier", "log_loss", "auc"]
    return pd.DataFrame(rows)[cols].sort_values("n", ascending=False) if rows else pd.DataFrame(columns=cols)


def calibration_table(frame: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    x = frame[["target", "predicted_probability"]].dropna().copy()
    if x.empty:
        return pd.DataFrame()
    try:
        x["bucket"] = pd.qcut(x["predicted_probability"], q=bins, duplicates="drop")
    except ValueError:
        x["bucket"] = pd.cut(x["predicted_probability"], bins=bins, duplicates="drop")
    out = (
        x.groupby("bucket", observed=True)
        .agg(n=("target", "size"), mean_prediction=("predicted_probability", "mean"), actual_win_rate=("target", "mean"))
        .reset_index()
    )
    out["bucket"] = out["bucket"].astype(str)
    out["calibration_error"] = out["actual_win_rate"] - out["mean_prediction"]
    return out


def base_model_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in sorted(c for c in frame.columns if c.startswith("p_")):
        m = _metric_dict(frame["target"].to_numpy(int), frame[col].to_numpy(float))
        m["model"] = col[2:]
        rows.append(m)
    if "predicted_probability" in frame:
        m = metric_dict(frame)
        m["model"] = "C1_ensemble_calibrated"
        rows.append(m)
    cols = ["model", "n", "accuracy", "brier", "log_loss", "auc"]
    return pd.DataFrame(rows)[cols].sort_values("log_loss") if rows else pd.DataFrame(columns=cols)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025, 2026])
    ap.add_argument("--config", default=str(ROOT / "config" / "default.json"))
    ap.add_argument("--holdout-start", default="2026-06-01")
    ap.add_argument("--holdout-end", default="2026-08-31")
    ap.add_argument("--out-dir", default=str(ROOT / "reports" / "fundamental_benchmark"))
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text())

    matches = load_tennismylife_directory(args.data_dir, args.years)
    source_counts = (
        matches.groupby(["source_kind", "level_role"], dropna=False)
        .size().rename("matches").reset_index()
    )
    source_counts.to_csv(out / "00_cross_level_source_counts.csv", index=False)

    builder = ChallengerFeatureBuilder(config)
    features = builder.build(matches)
    features["date"] = pd.to_datetime(features["date"])

    holdout_start = pd.Timestamp(args.holdout_start)
    holdout_end = pd.Timestamp(args.holdout_end)
    dev = features[features["date"] < holdout_start].copy()
    holdout_rows = features[(features["date"] >= holdout_start) & (features["date"] <= holdout_end)].copy()
    if len(dev) < 1200:
        raise RuntimeError(f"Development set too small: {len(dev)}")
    if len(holdout_rows) < 200:
        raise RuntimeError(f"Untouched holdout too small: {len(holdout_rows)}")

    best_horizon, horizon_table, oof_by_h, folds_by_h = compare_horizons(dev, config)
    horizon_table.to_csv(out / "01_horizon_comparison_development.csv", index=False)
    for h in sorted(oof_by_h):
        oof_by_h[h].to_csv(out / f"02_dev_oof_{h}d.csv", index=False)
        folds_by_h[h].to_csv(out / f"03_dev_folds_{h}d.csv", index=False)

    dev_oof = oof_by_h[best_horizon]
    dev_art = fit_final_fundamental(dev, config, best_horizon, dev_oof, cutoff=holdout_start)

    holdout = walk_forward_fixed_architecture(
        features,
        config,
        best_horizon,
        holdout_start,
        dev_art.weights,
        dev_art.calibrator,
        end_date=holdout_end,
    )
    if holdout.empty:
        raise RuntimeError("No holdout predictions were produced")
    holdout.to_csv(out / "04_UNTOUCHED_holdout_predictions.csv", index=False)

    dev_metrics = metric_dict(dev_oof)
    holdout_metrics = metric_dict(holdout)
    base_metrics = base_model_metrics(holdout)
    base_metrics.to_csv(out / "05_holdout_base_model_metrics.csv", index=False)

    holdout = holdout.copy()
    holdout["month"] = pd.to_datetime(holdout["date"]).dt.to_period("M").astype(str)
    grouped_metrics(holdout, "month").to_csv(out / "06_holdout_metrics_by_month.csv", index=False)
    grouped_metrics(holdout, "surface").to_csv(out / "07_holdout_metrics_by_surface.csv", index=False)
    if "round" in holdout:
        grouped_metrics(holdout, "round").to_csv(out / "08_holdout_metrics_by_round.csv", index=False)
    calibration_table(holdout).to_csv(out / "09_holdout_calibration.csv", index=False)

    weights = {name: float(w) for name, w in zip(dev_art.base_model_names, dev_art.weights)}
    audit = {
        "model": "ChallengerC1-FUND",
        "status": "fundamental benchmark only; no historical-odds profitability claim",
        "years_loaded": args.years,
        "all_cross_level_matches_loaded": int(len(matches)),
        "source_counts": source_counts.to_dict(orient="records"),
        "challenger_feature_rows": int(len(features)),
        "development_rows": int(len(dev)),
        "holdout_target_rows_available": int(len(holdout_rows)),
        "holdout_predictions": int(len(holdout)),
        "holdout_start": str(holdout_start.date()),
        "holdout_end": str(holdout_end.date()),
        "selected_history_days": int(best_horizon),
        "ensemble_weights_frozen_from_development": weights,
        "development_oof_metrics": dev_metrics,
        "untouched_holdout_metrics": holdout_metrics,
        "note": "History horizon, ensemble weights and calibration are selected without using holdout outcomes. Base models may refit prospectively by block using prior data only.",
    }
    save_json(audit, out / "FUNDAMENTAL_BENCHMARK_AUDIT.json")

    print("=== ChallengerC1 fundamental benchmark ===")
    print(json.dumps(audit, indent=2, default=str))
    print("\n=== Holdout base-model comparison ===")
    print(base_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
