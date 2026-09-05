#!/usr/bin/env python
"""Final ChallengerC1 Rev5 fundamental execution.

Runs the frozen architecture on fresh TennisMyLife data with the Rev5 target
and complete-tournament-week rules.  No odds are used and no betting policy is
approved by this script.
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

from challenger_c1.data import load_tennismylife_directory
from challenger_c1.features import ChallengerFeatureBuilder
from challenger_c1.model import (
    _metric_dict,
    compare_horizons,
    fit_final_fundamental,
    save_json,
    walk_forward_fixed_architecture,
)

QUALIFYING_ROUNDS = {"Q1", "Q2", "Q3", "Q4"}


def metric_dict(frame: pd.DataFrame, pred_col: str = "predicted_probability") -> dict:
    if frame.empty:
        return {"n": 0, "accuracy": None, "brier": None, "log_loss": None, "auc": None}
    return _metric_dict(frame["target"].to_numpy(int), frame[pred_col].to_numpy(float))


def grouped_metrics(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    rows = []
    for value, group in frame.groupby(key, dropna=False):
        row = metric_dict(group)
        row[key] = str(value)
        rows.append(row)
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
        .agg(
            n=("target", "size"),
            mean_prediction=("predicted_probability", "mean"),
            actual_win_rate=("target", "mean"),
        )
        .reset_index()
    )
    out["bucket"] = out["bucket"].astype(str)
    out["calibration_error"] = out["actual_win_rate"] - out["mean_prediction"]
    return out


def base_model_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in sorted(c for c in frame.columns if c.startswith("p_")):
        row = _metric_dict(frame["target"].to_numpy(int), frame[col].to_numpy(float))
        row["model"] = col[2:]
        rows.append(row)
    row = metric_dict(frame)
    row["model"] = "C1_ensemble_calibrated"
    rows.append(row)
    cols = ["model", "n", "accuracy", "brier", "log_loss", "auc"]
    return pd.DataFrame(rows)[cols].sort_values("log_loss")


def apply_rev5_target_definition(matches: pd.DataFrame) -> pd.DataFrame:
    """Keep qualifying matches as state history, but never as target labels."""
    out = matches.copy()
    rounds = out["round"].fillna("").astype(str).str.upper().str.strip()
    before = int(out["is_target_challenger"].astype(bool).sum())
    qual_target = out["is_target_challenger"].astype(bool) & rounds.isin(QUALIFYING_ROUNDS)
    out.loc[qual_target, "is_target_challenger"] = False
    after = int(out["is_target_challenger"].astype(bool).sum())
    out.attrs["rev5_target_rows_before"] = before
    out.attrs["rev5_qualifying_targets_removed"] = int(qual_target.sum())
    out.attrs["rev5_target_rows_after"] = after
    return out


def validate_complete_tournament_weeks(
    matches: pd.DataFrame,
    holdout_start: str | pd.Timestamp,
    holdout_end: str | pd.Timestamp,
) -> dict:
    """Exact Rev5 event-week coverage rule for TennisMyLife/Sackmann data."""
    h0 = pd.Timestamp(holdout_start).normalize()
    h1 = pd.Timestamp(holdout_end).normalize()
    if h0.weekday() != 0 or h1.weekday() != 6:
        raise ValueError(
            "complete_tournament_weeks requires Monday/Sunday boundaries; "
            f"got {h0.date()} through {h1.date()}"
        )
    terminal_week = h1 - pd.Timedelta(days=6)
    if h0 > terminal_week:
        raise ValueError("Holdout contains no complete tournament week")

    target = matches[matches["is_target_challenger"].astype(bool)].copy()
    if target.empty:
        raise ValueError("No clean Challenger main-draw target rows were loaded")
    proxy = pd.to_datetime(target["proxy_date"], errors="coerce")
    event = pd.to_datetime(target["tourney_date"], errors="coerce").dt.normalize()
    min_proxy = proxy.min().normalize()
    max_proxy = proxy.max().normalize()
    min_event = event.min().normalize()
    max_event = event.max().normalize()

    if max_event < terminal_week:
        raise ValueError(
            "Result source is incomplete for the declared complete-week holdout: latest Challenger "
            f"tournament week is {max_event.date()}, before terminal week {terminal_week.date()}."
        )
    if min_event > h0:
        raise ValueError(
            f"Result source begins after holdout start week: earliest Challenger tournament week is {min_event.date()}"
        )

    # Structural completion check uses all Challenger rows, including retirements.
    all_ch = matches[matches["tourney_level"].astype(str).str.upper().eq("C")].copy()
    all_event = pd.to_datetime(all_ch["tourney_date"], errors="coerce").dt.normalize()
    terminal = all_ch[all_event.eq(terminal_week)].copy()
    if terminal.empty:
        raise ValueError(f"No Challenger events found for terminal tournament week {terminal_week.date()}")
    event_key = "tourney_id" if "tourney_id" in terminal.columns else "tourney_name"
    terminal[event_key] = terminal[event_key].fillna("").astype(str)
    terminal["_round"] = terminal["round"].fillna("").astype(str).str.upper().str.strip()
    grouped = terminal.groupby(event_key, dropna=False)["_round"].agg(lambda x: sorted(set(x)))
    incomplete = [str(key) for key, rounds in grouped.items() if "F" not in rounds]
    if incomplete:
        raise ValueError(
            f"Terminal Challenger week {terminal_week.date()} contains {len(incomplete)} event(s) without a final row: "
            + ", ".join(incomplete[:8])
        )

    in_hold = target[(event >= h0) & (event <= terminal_week)]
    if in_hold.empty:
        raise ValueError("No Challenger target rows fall inside the declared complete-week holdout")

    return {
        "source": "tennismylife",
        "rows": int(len(matches)),
        "target_rows": int(len(target)),
        "holdout_target_rows_by_event_week": int(len(in_hold)),
        "holdout_start": str(h0.date()),
        "holdout_end": str(h1.date()),
        "terminal_tournament_week": str(terminal_week.date()),
        "terminal_week_events": int(len(grouped)),
        "terminal_week_events_without_final": 0,
        "min_target_proxy_date": str(min_proxy.date()),
        "max_target_proxy_date": str(max_proxy.date()),
        "max_target_event_start": str(max_event.date()),
        "exact_match_timestamps": False,
        "approval_eligible_from_results": True,
        "date_basis": "complete_tournament_week",
        "boundary_mode": "complete_tournament_weeks",
        "boundary_warning": None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025, 2026])
    ap.add_argument("--config", default=str(ROOT / "config" / "default.json"))
    ap.add_argument("--holdout-start", default="2026-06-01")
    ap.add_argument("--holdout-end", default="2026-08-30")
    ap.add_argument("--out-dir", default=str(ROOT / "reports" / "rev5_final_fundamental"))
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text())

    matches = load_tennismylife_directory(args.data_dir, args.years)
    raw_rows = int(len(matches))
    matches = apply_rev5_target_definition(matches)
    target_before = int(matches.attrs["rev5_target_rows_before"])
    qual_removed = int(matches.attrs["rev5_qualifying_targets_removed"])
    target_after = int(matches.attrs["rev5_target_rows_after"])
    provenance = validate_complete_tournament_weeks(matches, args.holdout_start, args.holdout_end)

    source_counts = (
        matches.groupby(["source_kind", "level_role"], dropna=False)
        .size().rename("matches").reset_index()
    )
    source_counts.to_csv(out / "00_cross_level_source_counts.csv", index=False)

    builder = ChallengerFeatureBuilder(config)
    features = builder.build(matches)
    features["date"] = pd.to_datetime(features["date"])

    h0 = pd.Timestamp(args.holdout_start).normalize()
    h1 = pd.Timestamp(args.holdout_end).normalize()
    dev = features[features["date"] < h0].copy()
    holdout_available = features[(features["date"] >= h0) & (features["date"] <= h1)].copy()
    if len(dev) < 1200:
        raise RuntimeError(f"Development set too small: {len(dev)}")
    if len(holdout_available) < 200:
        raise RuntimeError(f"Untouched holdout too small: {len(holdout_available)}")

    best_horizon, horizon_table, oof_by_h, folds_by_h = compare_horizons(dev, config)
    horizon_table.to_csv(out / "01_horizon_comparison_development.csv", index=False)
    for horizon in sorted(oof_by_h):
        oof_by_h[horizon].to_csv(out / f"02_dev_oof_{horizon}d.csv", index=False)
        folds_by_h[horizon].to_csv(out / f"03_dev_folds_{horizon}d.csv", index=False)

    dev_oof = oof_by_h[best_horizon]
    dev_art = fit_final_fundamental(dev, config, best_horizon, dev_oof, cutoff=h0)
    holdout = walk_forward_fixed_architecture(
        features,
        config,
        best_horizon,
        h0,
        dev_art.weights,
        dev_art.calibrator,
        end_date=h1,
    )
    if holdout.empty:
        raise RuntimeError("No holdout predictions were produced")
    dates = pd.to_datetime(holdout["date"])
    if dates.min().normalize() < h0 or dates.max().normalize() > h1:
        raise RuntimeError("Holdout prediction dates escaped the frozen Rev5 boundary")
    holdout.to_csv(out / "04_UNTOUCHED_holdout_predictions.csv", index=False)

    dev_metrics = metric_dict(dev_oof)
    holdout_metrics = metric_dict(holdout)
    base_metrics = base_model_metrics(holdout)
    base_metrics.to_csv(out / "05_holdout_base_model_metrics.csv", index=False)

    holdout_grouped = holdout.copy()
    holdout_grouped["month"] = pd.to_datetime(holdout_grouped["date"]).dt.to_period("M").astype(str)
    grouped_metrics(holdout_grouped, "month").to_csv(out / "06_holdout_metrics_by_month.csv", index=False)
    grouped_metrics(holdout_grouped, "surface").to_csv(out / "07_holdout_metrics_by_surface.csv", index=False)
    if "round" in holdout_grouped.columns:
        grouped_metrics(holdout_grouped, "round").to_csv(out / "08_holdout_metrics_by_round.csv", index=False)
    calibration_table(holdout_grouped).to_csv(out / "09_holdout_calibration.csv", index=False)

    # Holdout has now been graded exactly once. Refit the frozen fundamental
    # predictor on all completed history while keeping architecture/horizon fixed.
    latest_cutoff = features["date"].max() + pd.Timedelta(days=1)
    meta_all = pd.concat([dev_oof, holdout], ignore_index=True)
    final_fund = fit_final_fundamental(features, config, best_horizon, meta_all, cutoff=latest_cutoff)
    final_fund.metadata.update({
        "revision": "Rev5",
        "architecture_selected_on_development_only": True,
        "holdout_graded_before_final_refit": True,
        "selected_history_days": int(best_horizon),
        "holdout_start": str(h0.date()),
        "holdout_end": str(h1.date()),
        "results_approval_eligible": True,
        "results_provenance": provenance,
        "betting_policy_approved": False,
    })
    final_fund.save(out / "challenger_c1_fundamental_FROZEN.joblib")

    # Existing repo builder uses a lambda-backed defaultdict. Convert only the
    # container factory to a pickle-safe equivalent; state values are unchanged.
    builder.h2h = defaultdict(partial(list, [0, 0]), dict(builder.h2h))
    joblib.dump(builder, out / "challenger_c1_feature_state_FROZEN.joblib")

    weights = {name: float(weight) for name, weight in zip(dev_art.base_model_names, dev_art.weights)}
    audit = {
        "model": "ChallengerC1-FUND",
        "revision": "Rev5",
        "mode": "fundamental_only",
        "status": "C1-FUND frozen; no historical-odds profitability claim",
        "years_loaded": args.years,
        "cross_level_matches_loaded": raw_rows,
        "source_counts": source_counts.to_dict(orient="records"),
        "rev5_target_rows_before_qualifying_exclusion": target_before,
        "rev5_qualifying_target_rows_removed": qual_removed,
        "rev5_target_rows_after_exclusion": target_after,
        "challenger_feature_rows": int(len(features)),
        "development_rows": int(len(dev)),
        "holdout_target_rows_available": int(len(holdout_available)),
        "holdout_predictions": int(len(holdout)),
        "holdout_start": str(h0.date()),
        "holdout_end": str(h1.date()),
        "selected_history_days": int(best_horizon),
        "horizon_comparison": horizon_table.to_dict(orient="records"),
        "ensemble_weights_frozen_from_development": weights,
        "development_oof_metrics": dev_metrics,
        "untouched_holdout_metrics": holdout_metrics,
        "results_provenance": provenance,
        "approval_eligible_from_results": True,
        "betting_policy_approved": False,
        "note": "History horizon, ensemble weights and calibration were selected without holdout outcomes. Q1-Q4 Challenger qualifying rounds update player state but are excluded from target labels. Holdout uses complete Monday-Sunday tournament weeks through the 2026-08-24 terminal week.",
    }
    save_json(audit, out / "FUNDAMENTAL_FREEZE_AUDIT.json")

    print("=== ChallengerC1 Rev5 final fundamental ===")
    print(json.dumps(audit, indent=2, default=str))
    print("\n=== Holdout base-model comparison ===")
    print(base_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
