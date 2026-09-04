#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from challenger_c1.market import fit_final_market, walk_forward_market
from challenger_c1.model import _metric_dict
from challenger_c1.policy import (
    FrozenPolicy,
    bootstrap_roi,
    candidate_sides,
    diagnostic_tables,
    evaluate_policy_grid,
    summarize_bets,
)
from tennisexplorer_matcher import attach_tennisexplorer_odds, no_vig_probability


def metric(frame: pd.DataFrame, col: str) -> dict:
    if frame.empty:
        return {"n": 0, "accuracy": None, "brier": None, "log_loss": None, "auc": None}
    return _metric_dict(frame["target"].to_numpy(int), frame[col].to_numpy(float))


def choose_policy(grid: pd.DataFrame, fraction: float, name: str) -> FrozenPolicy:
    sub = grid[np.isclose(grid["fraction_cap"], fraction)].copy()
    credible = sub[sub["robust_score"] > -900]
    if not credible.empty:
        best = credible.iloc[0]
    else:
        fallback = sub[(sub["roi_haircut_5pct"] >= 0) & (sub["bets"] >= 30)].copy()
        if fallback.empty:
            fallback = sub.sort_values(["bets", "min_reliability"], ascending=[False, False])
        best = fallback.iloc[0]
    return FrozenPolicy(
        name=name,
        fraction_cap=float(best["fraction_cap"]),
        edge_floor=float(best["edge_floor"]),
        ev_floor=float(best["ev_floor"]),
        min_reliability=float(best["min_reliability"]),
        max_decimal_odds=float(best["max_decimal_odds"]),
        min_history=4.0 if fraction <= 0.10 else 2.0,
        min_surface_history=2.0 if fraction <= 0.10 else 0.0,
    )


def policy_dict(p: FrozenPolicy) -> dict:
    return {
        "name": p.name,
        "fraction_cap": p.fraction_cap,
        "edge_floor": p.edge_floor,
        "ev_floor": p.ev_floor,
        "min_reliability": p.min_reliability,
        "max_decimal_odds": p.max_decimal_odds,
        "min_history": p.min_history,
        "min_surface_history": p.min_surface_history,
    }


def forced_fraction(candidates: pd.DataFrame, fraction: float) -> pd.DataFrame:
    selected = []
    for slate, group in candidates.groupby("slate_date", sort=True):
        cap = max(1, int(math.ceil(fraction * len(group))))
        selected.append(group.sort_values(["selection_score", "edge_pp"], ascending=False).head(cap))
    return pd.concat(selected, ignore_index=True) if selected else candidates.iloc[:0].copy()


def audit_counts(audit: pd.DataFrame) -> dict:
    if audit.empty or "status" not in audit:
        return {}
    return {str(k): int(v) for k, v in audit["status"].value_counts().items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fund-dir", required=True)
    ap.add_argument("--odds", required=True)
    ap.add_argument("--config", default=str(ROOT / "config" / "default.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "reports" / "value_evaluation"))
    args = ap.parse_args()

    fund = Path(args.fund_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text())

    horizon_table = pd.read_csv(fund / "01_horizon_comparison_development.csv")
    best_horizon = int(horizon_table.sort_values("selection_score").iloc[0]["horizon_days"])
    dev = pd.read_csv(fund / f"02_dev_oof_{best_horizon}d.csv", low_memory=False)
    hold = pd.read_csv(fund / "04_UNTOUCHED_holdout_predictions.csv", low_memory=False)
    odds = pd.read_csv(args.odds, low_memory=False)
    if "valid_price_pair" in odds:
        odds = odds[pd.to_numeric(odds["valid_price_pair"], errors="coerce").fillna(0).astype(int).eq(1)].copy()

    dev_odds, dev_audit = attach_tennisexplorer_odds(dev, odds)
    hold_odds, hold_audit = attach_tennisexplorer_odds(hold, odds)
    dev_audit.to_csv(out / "01_odds_match_audit_development.csv", index=False)
    hold_audit.to_csv(out / "02_odds_match_audit_holdout.csv", index=False)
    dev_odds.to_csv(out / "03_development_matched_odds.csv", index=False)
    hold_odds.to_csv(out / "04_holdout_matched_odds.csv", index=False)

    if len(dev_odds) < 800:
        raise RuntimeError(f"Only {len(dev_odds)} development rows matched; refusing value fit")
    if len(hold_odds) < 150:
        raise RuntimeError(f"Only {len(hold_odds)} holdout rows matched; refusing holdout grade")

    # Strictly walk-forward value predictions on development OOF rows.
    value_dev = walk_forward_market(dev_odds)
    value_dev.to_csv(out / "05_value_oof_development.csv", index=False)

    # The holdout gets one fixed market-residual model fitted only to development.
    market_art = fit_final_market(dev_odds)
    hold_value = hold_odds.copy()
    hold_value["value_probability_a"] = market_art.model.predict(hold_value)
    hold_value.to_csv(out / "06_value_UNTOUCHED_holdout_predictions.csv", index=False)

    # Predictive benchmark: market vs fundamental vs C1-VALUE.
    predictive = {
        "development": {
            "market": metric(value_dev, "market_probability_a"),
            "fundamental": metric(value_dev, "predicted_probability"),
            "value": metric(value_dev, "value_probability_a"),
        },
        "holdout": {
            "market": metric(hold_value, "market_probability_a"),
            "fundamental": metric(hold_value, "predicted_probability"),
            "value": metric(hold_value, "value_probability_a"),
        },
    }

    # All grid choices are made on development only. Holdout is never used to
    # choose a threshold, fraction cap, reliability gate or max price.
    dev_candidates, grid = evaluate_policy_grid(value_dev, config)
    grid.to_csv(out / "07_policy_grid_DEVELOPMENT_ONLY.csv", index=False)
    dev_candidates.to_csv(out / "08_candidates_development.csv", index=False)

    fractions = [0.10, 0.15, 0.20, 0.25, 0.30]
    policies = {f: choose_policy(grid, f, f"C1-TOP{int(f*100)}") for f in fractions}
    (out / "09_frozen_fraction_policies.json").write_text(
        json.dumps({str(f): policy_dict(p) for f, p in policies.items()}, indent=2)
    )

    hold_candidates = candidate_sides(hold_value, "value_probability_a")
    hold_candidates["slate_date"] = pd.to_datetime(hold_candidates["odds_date"]).dt.normalize()
    hold_candidates.to_csv(out / "10_holdout_candidates.csv", index=False)

    holdout_policy_results = {}
    forced_results = {}
    for f, policy in policies.items():
        selected = policy.apply(hold_candidates)
        selected.to_csv(out / f"11_holdout_gated_top{int(f*100)}.csv", index=False)
        base = summarize_bets(selected)
        stress = summarize_bets(selected, 0.05)
        boot = bootstrap_roi(selected) if len(selected) >= 20 else {}
        holdout_policy_results[str(f)] = {
            "policy": policy_dict(policy),
            "base": base,
            "price_haircut_5pct": stress,
            "bootstrap": boot,
        }
        forced = forced_fraction(hold_candidates, f)
        forced.to_csv(out / f"12_holdout_FORCED_top{int(f*100)}.csv", index=False)
        forced_results[str(f)] = {
            "base": summarize_bets(forced),
            "price_haircut_5pct": summarize_bets(forced, 0.05),
        }
        for dname, table in diagnostic_tables(hold_candidates, selected).items():
            table.to_csv(out / f"13_top{int(f*100)}_{dname}.csv", index=False)

    volume = holdout_policy_results["0.2"]["base"]
    volume_stress = holdout_policy_results["0.2"]["price_haircut_5pct"]
    core = holdout_policy_results["0.1"]["base"]
    volume_approved = bool(
        volume["bets"] >= 60
        and volume["profit_units"] > 0
        and volume_stress["profit_units"] > 0
        and volume["profit_without_best_win"] > 0
        and volume["profitable_months"] >= max(1, math.ceil(0.5 * volume["months"]))
    )
    core_approved = bool(
        core["bets"] >= 30
        and core["profit_units"] > 0
        and core["profit_without_best_win"] > -1.0
    )

    audit = {
        "model": "ChallengerC1 value evaluation",
        "selected_history_days": best_horizon,
        "odds_source": "TennisExplorer historical displayed-average moneyline",
        "odds_rows_valid": int(len(odds)),
        "development_rows": int(len(dev)),
        "development_odds_matched": int(len(dev_odds)),
        "development_match_fraction": float(len(dev_odds) / len(dev)),
        "development_match_audit": audit_counts(dev_audit),
        "holdout_rows": int(len(hold)),
        "holdout_odds_matched": int(len(hold_odds)),
        "holdout_match_fraction": float(len(hold_odds) / len(hold)),
        "holdout_match_audit": audit_counts(hold_audit),
        "predictive_metrics": predictive,
        "development_policy_selection": "robustness-first grid; holdout not used",
        "gated_fraction_holdout_results": holdout_policy_results,
        "forced_fraction_holdout_diagnostics": forced_results,
        "core10_approved_for_prospective_tracking": core_approved,
        "volume20_approved_for_prospective_tracking": volume_approved,
        "note": "Holdout is Jun-Aug 2026 and was not used for policy/model selection. Historical ROI is not a guarantee of future profit.",
    }
    (out / "FINAL_VALUE_EVALUATION_AUDIT.json").write_text(json.dumps(audit, indent=2, default=str))
    print(json.dumps(audit, indent=2, default=str))


if __name__ == "__main__":
    main()
