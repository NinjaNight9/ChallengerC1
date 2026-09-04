#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from challenger_c1.market import fit_final_market, walk_forward_market
from challenger_c1.policy import (
    FrozenPolicy,
    bootstrap_roi,
    candidate_sides,
    diagnostic_tables,
    evaluate_policy_grid,
    summarize_bets,
)
from challenger_c1.tennisexplorer_matcher import attach_tennisexplorer_odds


def choose_policy(grid: pd.DataFrame, fraction: float, name: str) -> tuple[FrozenPolicy, dict]:
    sub = grid[np.isclose(grid["fraction_cap"], fraction)].copy()
    credible = sub[sub["robust_score"] > -900]
    if not credible.empty:
        best = credible.iloc[0]
    else:
        fallback = sub[(sub["roi_haircut_5pct"] >= 0) & (sub["bets"] >= 30)].copy()
        if fallback.empty:
            fallback = sub.sort_values(["bets", "min_reliability"], ascending=[False, False])
        best = fallback.iloc[0]
    policy = FrozenPolicy(
        name=name,
        fraction_cap=float(best["fraction_cap"]),
        edge_floor=float(best["edge_floor"]),
        ev_floor=float(best["ev_floor"]),
        min_reliability=float(best["min_reliability"]),
        max_decimal_odds=float(best["max_decimal_odds"]),
        min_history=4.0 if fraction <= 0.10 else 2.0,
        min_surface_history=2.0 if fraction <= 0.10 else 0.0,
    )
    return policy, best.to_dict()


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


def quality_summary(matched: pd.DataFrame, audit: pd.DataFrame) -> dict:
    out = {
        "matched": int(len(matched)),
        "audit_status": audit["status"].value_counts().to_dict() if not audit.empty else {},
    }
    if not matched.empty:
        out.update({
            "median_name_pair_score": float(matched["odds_name_pair_score"].median()),
            "p05_name_pair_score": float(matched["odds_name_pair_score"].quantile(0.05)),
            "median_tournament_similarity": float(matched["odds_tournament_similarity"].median()),
            "p05_match_margin": float(matched["odds_match_margin"].quantile(0.05)),
            "same_proxy_day_fraction": float((matched["odds_date_gap"] == 0).mean()),
            "within_2_days_fraction": float((matched["odds_date_gap"] <= 2).mean()),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fund-dir", required=True)
    ap.add_argument("--odds", required=True)
    ap.add_argument("--config", default=str(ROOT / "config" / "default.json"))
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    fund = Path(args.fund_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text())

    horizon_table = pd.read_csv(fund / "01_horizon_comparison_development.csv")
    horizon = int(horizon_table.iloc[0]["horizon_days"])
    dev = pd.read_csv(fund / f"02_dev_oof_{horizon}d.csv", low_memory=False)
    hold = pd.read_csv(fund / "04_UNTOUCHED_holdout_predictions.csv", low_memory=False)
    odds = pd.read_csv(args.odds, low_memory=False)

    dev_odds, dev_audit = attach_tennisexplorer_odds(dev, odds)
    hold_odds, hold_audit = attach_tennisexplorer_odds(hold, odds)
    dev_audit.to_csv(out / "01_dev_odds_match_audit.csv", index=False)
    hold_audit.to_csv(out / "02_holdout_odds_match_audit.csv", index=False)
    dev_odds.to_csv(out / "03_dev_oof_with_odds.csv", index=False)
    hold_odds.to_csv(out / "04_holdout_fund_with_odds.csv", index=False)

    # Matching is part of validation. Abort rather than silently evaluate a tiny or
    # questionable sample.
    if len(dev_odds) < 10000:
        raise RuntimeError(f"Insufficient development odds matches: {len(dev_odds)} / {len(dev)}")
    if len(hold_odds) < 1200:
        raise RuntimeError(f"Insufficient holdout odds matches: {len(hold_odds)} / {len(hold)}")
    if dev_odds["odds_name_pair_score"].quantile(0.05) < 0.80:
        raise RuntimeError("Development matcher p05 name score below 0.80")
    if hold_odds["odds_name_pair_score"].quantile(0.05) < 0.80:
        raise RuntimeError("Holdout matcher p05 name score below 0.80")

    value_dev = walk_forward_market(dev_odds)
    value_dev.to_csv(out / "05_value_oof_development.csv", index=False)
    market_art = fit_final_market(dev_odds)
    hold_odds = hold_odds.copy()
    hold_odds["value_probability_a"] = market_art.model.predict(hold_odds)
    hold_odds.to_csv(out / "06_value_holdout_predictions.csv", index=False)

    candidates_dev, grid = evaluate_policy_grid(value_dev, config)
    candidates_dev.to_csv(out / "07_candidates_development.csv", index=False)
    grid.to_csv(out / "08_policy_grid_development.csv", index=False)
    core, core_dev = choose_policy(grid, 0.10, "C1-CORE10")
    volume, volume_dev = choose_policy(grid, 0.20, "C1-VOLUME20")

    hold_candidates = candidate_sides(hold_odds, "value_probability_a")
    hold_candidates["slate_date"] = pd.to_datetime(hold_candidates["odds_date"]).dt.normalize()
    hold_core = core.apply(hold_candidates)
    hold_volume = volume.apply(hold_candidates)
    hold_candidates.to_csv(out / "09_holdout_candidates.csv", index=False)
    hold_core.to_csv(out / "10_holdout_core10.csv", index=False)
    hold_volume.to_csv(out / "11_holdout_volume20.csv", index=False)

    core_summary = summarize_bets(hold_core)
    core_5 = summarize_bets(hold_core, 0.05)
    volume_summary = summarize_bets(hold_volume)
    volume_5 = summarize_bets(hold_volume, 0.05)
    core_boot = bootstrap_roi(hold_core) if len(hold_core) >= 20 else {}
    volume_boot = bootstrap_roi(hold_volume) if len(hold_volume) >= 20 else {}

    for label, selected in (("core10", hold_core), ("volume20", hold_volume)):
        for table_name, table in diagnostic_tables(hold_candidates, selected).items():
            table.to_csv(out / f"12_{label}_{table_name}.csv", index=False)

    core_approved = bool(
        core_summary["bets"] >= 30
        and core_summary["profit_units"] > 0
        and core_summary["profit_without_best_win"] > -1.0
    )
    volume_approved = bool(
        volume_summary["bets"] >= 60
        and volume_summary["profit_units"] > 0
        and volume_5["profit_units"] > 0
        and volume_summary["profit_without_best_win"] > 0
        and volume_summary["profitable_months"] >= max(1, int(np.ceil(0.5 * volume_summary["months"])))
    )

    report = {
        "model": "ChallengerC1 value-stage evaluation",
        "selected_history_days": horizon,
        "development_fundamental_rows": int(len(dev)),
        "holdout_fundamental_rows": int(len(hold)),
        "development_matching": quality_summary(dev_odds, dev_audit),
        "holdout_matching": quality_summary(hold_odds, hold_audit),
        "core_policy": policy_dict(core),
        "volume20_policy": policy_dict(volume),
        "core_development_policy_row": core_dev,
        "volume20_development_policy_row": volume_dev,
        "core_holdout": core_summary,
        "core_holdout_5pct_price_haircut": core_5,
        "core_bootstrap": core_boot,
        "volume20_holdout": volume_summary,
        "volume20_holdout_5pct_price_haircut": volume_5,
        "volume20_bootstrap": volume_boot,
        "core_approved_for_prospective_tracking": core_approved,
        "volume20_approved_for_prospective_tracking": volume_approved,
        "note": "The holdout was not used to select the history window, market-model architecture, or policy thresholds.",
    }
    (out / "FINAL_VALUE_EVALUATION.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
