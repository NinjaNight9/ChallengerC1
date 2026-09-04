#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from challenger_c1.data import load_tennismylife_directory, normalize_name
from challenger_c1.features import ChallengerFeatureBuilder
from challenger_c1.market import fit_final_market, no_vig_probability, walk_forward_market
from challenger_c1.model import _metric_dict, fit_final_fundamental, save_json
from challenger_c1.policy import (
    FrozenPolicy,
    bootstrap_roi,
    candidate_sides,
    diagnostic_tables,
    evaluate_policy_grid,
    summarize_bets,
)


def short_signature(name: object) -> tuple[str, str]:
    """TennisExplorer uses surname(s) + initial, e.g. 'Ugo Carabelli C.'."""
    toks = normalize_name(name).split()
    if len(toks) < 2:
        return (normalize_name(name), "")
    return (" ".join(toks[:-1]), toks[-1][:1])


def full_signatures(name: object) -> set[tuple[str, str]]:
    """Generate suffix surname candidates from a full player name."""
    toks = normalize_name(name).split()
    if not toks:
        return set()
    initial = toks[0][:1]
    max_suffix = min(4, len(toks))
    return {(" ".join(toks[-k:]), initial) for k in range(1, max_suffix + 1)}


def pair_key(a: tuple[str, str], b: tuple[str, str]) -> str:
    parts = sorted([f"{a[0]}::{a[1]}", f"{b[0]}::{b[1]}"])
    return "||".join(parts)


def full_matches_short(full: object, short: object) -> bool:
    return short_signature(short) in full_signatures(full)


def tournament_similarity(a: object, b: object) -> float:
    aa, bb = normalize_name(a), normalize_name(b)
    if not aa or not bb:
        return 0.0
    return SequenceMatcher(None, aa, bb).ratio()


def attach_tennisexplorer_odds(
    features: pd.DataFrame,
    odds: pd.DataFrame,
    max_date_gap_days: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach abbreviated TennisExplorer prices to full-name benchmark rows.

    Matching never uses result labels. It uses only player-name signatures,
    date proximity and tournament-name similarity. This fixes the mismatch
    between full names in TennisMyLife and 'Surname I.' names in TennisExplorer.
    """
    o = odds.copy().reset_index(drop=True)
    o["_date"] = pd.to_datetime(o["date"], errors="coerce").dt.normalize()
    o = o.dropna(subset=["_date", "odd1", "odd2"]).copy()
    o["odd1"] = pd.to_numeric(o["odd1"], errors="coerce")
    o["odd2"] = pd.to_numeric(o["odd2"], errors="coerce")
    o = o[(o["odd1"] > 1.0) & (o["odd2"] > 1.0)].copy()

    by_pair: dict[str, list[int]] = defaultdict(list)
    for idx, row in o.iterrows():
        by_pair[pair_key(short_signature(row["player1"]), short_signature(row["player2"]))].append(idx)

    selected: list[dict] = []
    audit: list[dict] = []
    used_odds: dict[int, int] = defaultdict(int)

    for feature_index, row in features.reset_index(drop=True).iterrows():
        keys = set()
        for sa in full_signatures(row["player_a"]):
            for sb in full_signatures(row["player_b"]):
                keys.add(pair_key(sa, sb))
        ids: set[int] = set()
        for key in keys:
            ids.update(by_pair.get(key, []))
        if not ids:
            audit.append({"feature_index": feature_index, "status": "no_pair"})
            continue

        fdate = pd.Timestamp(row["date"]).normalize()
        ranked: list[tuple[int, float, int]] = []
        for oid in ids:
            orow = o.loc[oid]
            gap = abs((orow["_date"] - fdate).days)
            if gap > max_date_gap_days:
                continue
            orientation_ok = (
                full_matches_short(row["player_a"], orow["player1"])
                and full_matches_short(row["player_b"], orow["player2"])
            ) or (
                full_matches_short(row["player_a"], orow["player2"])
                and full_matches_short(row["player_b"], orow["player1"])
            )
            if not orientation_ok:
                continue
            sim = tournament_similarity(row.get("tourney_name", ""), orow.get("tournament", ""))
            ranked.append((gap, -sim, oid))
        if not ranked:
            audit.append({"feature_index": feature_index, "status": "date_or_orientation_fail"})
            continue

        ranked.sort()
        gap, neg_sim, oid = ranked[0]
        best = o.loc[oid]
        sim = -neg_sim
        if full_matches_short(row["player_a"], best["player1"]):
            odd_a, odd_b = float(best["odd1"]), float(best["odd2"])
        else:
            odd_a, odd_b = float(best["odd2"]), float(best["odd1"])

        rec = row.to_dict()
        rec.update({
            "odd_a": odd_a,
            "odd_b": odd_b,
            "odds_date": best["_date"],
            "odds_source": best.get("source", "TennisExplorer"),
            "odds_date_gap": int(gap),
            "odds_tournament_similarity": float(sim),
            "odds_index": int(oid),
        })
        rec["market_probability_a"] = float(no_vig_probability(np.array([odd_a]), np.array([odd_b]))[0])
        rec["market_logit_a"] = float(np.log(rec["market_probability_a"] / (1.0 - rec["market_probability_a"])))
        selected.append(rec)
        used_odds[int(oid)] += 1
        audit.append({
            "feature_index": feature_index,
            "status": "matched",
            "odds_index": int(oid),
            "date_gap": int(gap),
            "tourney_similarity": float(sim),
        })

    matched = pd.DataFrame(selected)
    audit_df = pd.DataFrame(audit)
    if not audit_df.empty:
        audit_df["odds_reuse_count"] = audit_df.get("odds_index", pd.Series(index=audit_df.index, dtype=float)).map(used_odds)
    return matched, audit_df


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-dir", required=True)
    ap.add_argument("--odds", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025, 2026])
    ap.add_argument("--config", default=str(ROOT / "config" / "default.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "reports" / "finalize"))
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bdir = Path(args.benchmark_dir)
    config = json.loads(Path(args.config).read_text())
    audit0 = json.loads((bdir / "FUNDAMENTAL_BENCHMARK_AUDIT.json").read_text())
    horizon = int(audit0["selected_history_days"])

    dev_oof = pd.read_csv(bdir / f"02_dev_oof_{horizon}d.csv", low_memory=False)
    hold_fund = pd.read_csv(bdir / "04_UNTOUCHED_holdout_predictions.csv", low_memory=False)
    odds = pd.read_csv(args.odds, low_memory=False)

    dev_odds, dev_match_audit = attach_tennisexplorer_odds(dev_oof, odds)
    hold_odds, hold_match_audit = attach_tennisexplorer_odds(hold_fund, odds)
    dev_match_audit.to_csv(out / "01_odds_match_audit_development.csv", index=False)
    hold_match_audit.to_csv(out / "02_odds_match_audit_holdout.csv", index=False)
    dev_odds.to_csv(out / "03_development_with_odds.csv", index=False)
    hold_odds.to_csv(out / "04_holdout_with_odds.csv", index=False)

    if len(dev_odds) < 8000:
        raise RuntimeError(f"Development odds coverage too low: {len(dev_odds)}/{len(dev_oof)}")
    if len(hold_odds) < 900:
        raise RuntimeError(f"Holdout odds coverage too low: {len(hold_odds)}/{len(hold_fund)}")

    value_dev = walk_forward_market(dev_odds)
    value_dev.to_csv(out / "05_value_oof_development.csv", index=False)
    market_dev = fit_final_market(dev_odds)
    hold_odds = hold_odds.copy()
    hold_odds["value_probability_a"] = market_dev.model.predict(hold_odds)
    hold_odds.to_csv(out / "06_value_holdout_predictions.csv", index=False)

    market_hold_metrics = _metric_dict(hold_odds["target"].to_numpy(int), hold_odds["market_probability_a"].to_numpy(float))
    value_hold_metrics = _metric_dict(hold_odds["target"].to_numpy(int), hold_odds["value_probability_a"].to_numpy(float))
    fund_hold_metrics = _metric_dict(hold_odds["target"].to_numpy(int), hold_odds["predicted_probability"].to_numpy(float))

    candidates_dev, grid = evaluate_policy_grid(value_dev, config)
    grid.to_csv(out / "07_policy_grid_development.csv", index=False)
    candidates_dev.to_csv(out / "08_candidates_development.csv", index=False)
    core = choose_policy(grid, 0.10, "C1-CORE10")
    volume = choose_policy(grid, 0.20, "C1-VOLUME20")
    save_json({"core": policy_dict(core), "volume": policy_dict(volume)}, out / "09_FROZEN_POLICIES.json")

    hold_candidates = candidate_sides(hold_odds, "value_probability_a")
    hold_candidates["slate_date"] = pd.to_datetime(hold_candidates["odds_date"]).dt.normalize()
    core_bets = core.apply(hold_candidates)
    volume_bets = volume.apply(hold_candidates)
    hold_candidates.to_csv(out / "10_holdout_candidates.csv", index=False)
    core_bets.to_csv(out / "11_holdout_core10_bets.csv", index=False)
    volume_bets.to_csv(out / "12_holdout_volume20_bets.csv", index=False)

    core_summary = summarize_bets(core_bets)
    volume_summary = summarize_bets(volume_bets)
    core_stress = summarize_bets(core_bets, 0.05)
    volume_stress = summarize_bets(volume_bets, 0.05)
    core_boot = bootstrap_roi(core_bets) if len(core_bets) >= 20 else {}
    volume_boot = bootstrap_roi(volume_bets) if len(volume_bets) >= 20 else {}

    for label, selected in [("core10", core_bets), ("volume20", volume_bets)]:
        for dname, table in diagnostic_tables(hold_candidates, selected).items():
            table.to_csv(out / f"13_{label}_{dname}.csv", index=False)

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

    # Build fresh deployment artifacts only after the untouched holdout is graded.
    matches = load_tennismylife_directory(args.data_dir, args.years)
    builder = ChallengerFeatureBuilder(config)
    features = builder.build(matches)
    latest_cutoff = pd.to_datetime(features["date"]).max() + pd.Timedelta(days=1)
    meta_all = pd.concat([dev_oof, hold_fund], ignore_index=True)
    final_fund = fit_final_fundamental(features, config, horizon, meta_all, cutoff=latest_cutoff)
    final_fund.metadata.update({
        "holdout_graded_before_final_refit": True,
        "selected_history_days": horizon,
        "holdout_start": audit0["holdout_start"],
        "holdout_end": audit0["holdout_end"],
    })
    final_fund.save(out / "challenger_c1_fundamental_FROZEN.joblib")

    final_market = fit_final_market(pd.concat([dev_odds, hold_odds], ignore_index=True))
    final_market.metadata["holdout_graded_before_final_refit"] = True
    final_market.save(out / "challenger_c1_value_FROZEN.joblib")
    joblib.dump(builder, out / "challenger_c1_feature_state_FROZEN.joblib")

    report = {
        "model": "ChallengerC1",
        "selected_history_days": horizon,
        "fundamental_original_holdout": audit0["untouched_holdout_metrics"],
        "development_odds_matched": int(len(dev_odds)),
        "development_odds_total": int(len(dev_oof)),
        "holdout_odds_matched": int(len(hold_odds)),
        "holdout_odds_total": int(len(hold_fund)),
        "holdout_market_metrics": market_hold_metrics,
        "holdout_fundamental_metrics_on_odds_subset": fund_hold_metrics,
        "holdout_value_metrics": value_hold_metrics,
        "core_policy": policy_dict(core),
        "volume20_policy": policy_dict(volume),
        "core_holdout": core_summary,
        "core_holdout_5pct_price_haircut": core_stress,
        "core_bootstrap": core_boot,
        "volume20_holdout": volume_summary,
        "volume20_holdout_5pct_price_haircut": volume_stress,
        "volume20_bootstrap": volume_boot,
        "core_approved_for_prospective_tracking": core_approved,
        "volume20_approved_for_prospective_tracking": volume_approved,
        "latest_feature_date": str(pd.to_datetime(features["date"]).max().date()),
        "note": "Approval permits prospective tracking only; it is not a guarantee of future profitability.",
    }
    save_json(report, out / "FINAL_RESEARCH_AUDIT.json")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
