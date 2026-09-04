#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from challenger_c1.data import (
    canonicalize_generic_odds,
    load_sackmann_directory,
    load_tennismylife_directory,
)
from challenger_c1.features import ChallengerFeatureBuilder
from challenger_c1.market import attach_odds_to_features, fit_final_market, walk_forward_market
from challenger_c1.model import (
    _metric_dict,
    compare_horizons,
    fit_final_fundamental,
    save_json,
    walk_forward_fixed_architecture,
)
from challenger_c1.policy import (
    FrozenPolicy,
    bootstrap_roi,
    candidate_sides,
    diagnostic_tables,
    evaluate_policy_grid,
    summarize_bets,
)


def choose_policy(grid: pd.DataFrame, fraction: float, name: str) -> FrozenPolicy:
    sub = grid[np.isclose(grid["fraction_cap"], fraction)].copy()
    credible = sub[sub["robust_score"] > -900]
    if not credible.empty:
        best = credible.iloc[0]
    else:
        # If development history is short, choose the most conservative policy
        # with nonnegative 5% haircut ROI rather than maximizing a tiny sample.
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


def policy_to_dict(p: FrozenPolicy) -> dict:
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Full ChallengerC1 research -> untouched holdout -> freeze workflow")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--source", choices=["sackmann", "tennismylife"], default="tennismylife")
    ap.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025, 2026])
    ap.add_argument("--odds", required=True, help="Historical odds CSV with both sides")
    ap.add_argument("--odds-map", help="JSON mapping canonical field -> source column")
    ap.add_argument("--config", default=str(ROOT / "config" / "default.json"))
    ap.add_argument("--holdout-start", default="2026-06-01")
    ap.add_argument("--holdout-end", default="2026-08-31")
    ap.add_argument("--out-dir", default=str(ROOT / "reports" / "freeze_run"))
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text())
    if args.source == "tennismylife":
        matches = load_tennismylife_directory(args.data_dir, args.years)
    else:
        matches = load_sackmann_directory(args.data_dir, args.years)

    builder = ChallengerFeatureBuilder(config)
    features = builder.build(matches)
    features.to_csv(out / "challenger_features_all.csv", index=False)
    joblib.dump(builder, out / "feature_state_through_latest.joblib")

    holdout_start = pd.Timestamp(args.holdout_start)
    holdout_end = pd.Timestamp(args.holdout_end)
    dev = features[pd.to_datetime(features["date"]) < holdout_start].copy()
    if len(dev) < 1200:
        raise ValueError(f"Development set too small: {len(dev)} Challenger matches")

    best_horizon, horizon_table, oof_by_h, folds_by_h = compare_horizons(dev, config)
    horizon_table.to_csv(out / "01_horizon_comparison_development.csv", index=False)
    for h in oof_by_h:
        oof_by_h[h].to_csv(out / f"02_fundamental_oof_dev_{h}d.csv", index=False)
        folds_by_h[h].to_csv(out / f"03_fundamental_folds_dev_{h}d.csv", index=False)
    dev_oof = oof_by_h[best_horizon]

    # Fit meta weights/calibration from development OOF only.  The returned base
    # models are not used for holdout; holdout refits bases by block while keeping
    # these architecture/meta decisions fixed.
    dev_art = fit_final_fundamental(dev, config, best_horizon, dev_oof, cutoff=holdout_start)
    holdout_fund = walk_forward_fixed_architecture(
        features, config, best_horizon, holdout_start, dev_art.weights, dev_art.calibrator, end_date=holdout_end
    )
    holdout_fund.to_csv(out / "04_fundamental_holdout_predictions.csv", index=False)
    fund_dev_metrics = _metric_dict(dev_oof["target"].to_numpy(int), dev_oof["predicted_probability"].to_numpy(float))
    fund_hold_metrics = _metric_dict(holdout_fund["target"].to_numpy(int), holdout_fund["predicted_probability"].to_numpy(float))

    mapping = json.loads(Path(args.odds_map).read_text()) if args.odds_map else None
    odds = canonicalize_generic_odds(args.odds, mapping)
    dev_odds, dev_audit = attach_odds_to_features(dev_oof, odds)
    hold_odds, hold_audit = attach_odds_to_features(holdout_fund, odds)
    dev_audit.to_csv(out / "05_odds_match_audit_dev.csv", index=False)
    hold_audit.to_csv(out / "06_odds_match_audit_holdout.csv", index=False)
    if len(dev_odds) < 800:
        raise ValueError(f"Only {len(dev_odds)} development OOF rows matched to odds; value model not credible")
    if len(hold_odds) < 150:
        raise ValueError(f"Only {len(hold_odds)} holdout rows matched to odds; holdout not credible")

    value_dev = walk_forward_market(dev_odds)
    value_dev.to_csv(out / "07_value_oof_development.csv", index=False)
    market_art_dev = fit_final_market(dev_odds)
    hold_odds = hold_odds.copy()
    hold_odds["value_probability_a"] = market_art_dev.model.predict(hold_odds)
    hold_odds.to_csv(out / "08_value_holdout_predictions.csv", index=False)

    candidates_dev, grid = evaluate_policy_grid(value_dev, config)
    grid.to_csv(out / "09_policy_grid_development.csv", index=False)
    candidates_dev.to_csv(out / "10_candidates_development.csv", index=False)
    core = choose_policy(grid, 0.10, "C1-CORE10")
    volume = choose_policy(grid, 0.20, "C1-VOLUME20")
    save_json({"core": policy_to_dict(core), "volume": policy_to_dict(volume)}, out / "11_frozen_policies.json")

    hold_candidates = candidate_sides(hold_odds, "value_probability_a")
    hold_candidates["slate_date"] = pd.to_datetime(hold_candidates["odds_date"]).dt.normalize()
    hold_core = core.apply(hold_candidates)
    hold_volume = volume.apply(hold_candidates)
    hold_candidates.to_csv(out / "12_holdout_candidates.csv", index=False)
    hold_core.to_csv(out / "13_holdout_core_bets.csv", index=False)
    hold_volume.to_csv(out / "14_holdout_volume20_bets.csv", index=False)

    core_summary = summarize_bets(hold_core)
    volume_summary = summarize_bets(hold_volume)
    core_stress = summarize_bets(hold_core, 0.05)
    volume_stress = summarize_bets(hold_volume, 0.05)
    core_boot = bootstrap_roi(hold_core) if len(hold_core) >= 20 else {}
    volume_boot = bootstrap_roi(hold_volume) if len(hold_volume) >= 20 else {}
    for label, selected in [("core", hold_core), ("volume20", hold_volume)]:
        for dname, table in diagnostic_tables(hold_candidates, selected).items():
            table.to_csv(out / f"15_{label}_{dname}.csv", index=False)

    # Conservative deployment gates. Passing does not prove future profit; it
    # only permits a prospective freeze instead of immediate redesign.
    volume_approved = bool(
        volume_summary["bets"] >= 60
        and volume_summary["profit_units"] > 0
        and volume_stress["profit_units"] > 0
        and volume_summary["profit_without_best_win"] > 0
        and volume_summary["profitable_months"] >= max(1, math.ceil(0.5 * volume_summary["months"]))
    )
    core_approved = bool(
        core_summary["bets"] >= 30
        and core_summary["profit_units"] > 0
        and core_summary["profit_without_best_win"] > -1.0
    )

    # After the holdout has been graded exactly once, production parameters may
    # be refit on all *out-of-sample* historical predictions while keeping the
    # architecture and betting policy fixed. This does not change the already
    # reported holdout score; it simply gives the prospective freeze fresher
    # calibration/weights.
    latest_cutoff = pd.to_datetime(features["date"]).max() + pd.Timedelta(days=1)
    meta_all = pd.concat([dev_oof, holdout_fund], ignore_index=True)
    final_base = fit_final_fundamental(features, config, best_horizon, meta_all, cutoff=latest_cutoff)
    final_base.metadata["architecture_selected_on_development_only"] = True
    final_base.metadata["policy_selected_on_development_only"] = True
    final_base.metadata["holdout_graded_before_final_refit"] = True
    final_base.metadata["holdout_start"] = str(holdout_start.date())
    final_base.metadata["holdout_end"] = str(holdout_end.date())
    final_base.save(out / "challenger_c1_fundamental_FROZEN.joblib")

    market_training_all = pd.concat([dev_odds, hold_odds], ignore_index=True)
    final_market = fit_final_market(market_training_all)
    final_market.metadata["architecture_selected_on_development_only"] = True
    final_market.metadata["holdout_graded_before_final_refit"] = True
    final_market.save(out / "challenger_c1_value_FROZEN.joblib")
    joblib.dump(builder, out / "challenger_c1_feature_state_FROZEN.joblib")

    report = {
        "model": "ChallengerC1",
        "source": args.source,
        "years_loaded": args.years,
        "cross_level_matches_loaded": len(matches),
        "challenger_target_rows": len(features),
        "development_rows": len(dev),
        "selected_history_days": best_horizon,
        "fundamental_development": fund_dev_metrics,
        "fundamental_holdout": fund_hold_metrics,
        "development_odds_matched": len(dev_odds),
        "holdout_odds_matched": len(hold_odds),
        "core_policy": policy_to_dict(core),
        "volume20_policy": policy_to_dict(volume),
        "core_holdout": core_summary,
        "core_holdout_5pct_price_haircut": core_stress,
        "core_bootstrap": core_boot,
        "volume20_holdout": volume_summary,
        "volume20_holdout_5pct_price_haircut": volume_stress,
        "volume20_bootstrap": volume_boot,
        "core_approved_for_prospective_tracking": core_approved,
        "volume20_approved_for_prospective_tracking": volume_approved,
        "note": "Approval is a research gate, not evidence or guarantee of future profitability.",
    }
    save_json(report, out / "FINAL_RESEARCH_AUDIT.json")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    import math
    main()
