#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from challenger_c1.policy import (
    FrozenPolicy,
    bootstrap_roi,
    candidate_sides,
    diagnostic_tables,
    select_fraction_policy,
    summarize_bets,
)


def robust_row(bets: pd.DataFrame) -> dict:
    base = summarize_bets(bets, 0.0)
    stress5 = summarize_bets(bets, 0.05)
    row = {
        **base,
        "roi_haircut_5pct": stress5["roi"],
        "profit_haircut_5pct": stress5["profit_units"],
    }
    if base["bets"] >= 80 and base["months"] >= 3:
        positive_month_share = base["profitable_months"] / base["months"]
        row["robust_score"] = (
            0.35 * base["roi"]
            + 0.30 * stress5["roi"]
            + 0.20 * base["roi_without_best_win"]
            + 0.15 * base["median_month_roi"]
            - 0.025 * max(0.0, 0.60 - positive_month_share)
        )
    else:
        row["robust_score"] = -999.0
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--value-dir", required=True)
    ap.add_argument("--config", default=str(ROOT / "config" / "default.json"))
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    value_dir = Path(args.value_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text())

    old_report = json.loads((value_dir / "FINAL_VALUE_EVALUATION.json").read_text())
    if old_report.get("selected_history_days") != 730:
        raise RuntimeError("Expected exact Rev5 730-day FUND artifact")
    if old_report.get("evaluation_grade") != "NON_APPROVAL_GRADE_SENSITIVITY":
        raise RuntimeError("Expected non-approval-grade sensitivity input")

    dev = pd.read_csv(value_dir / "05_value_oof_development.csv", low_memory=False)
    hold = pd.read_csv(value_dir / "06_value_holdout_predictions.csv", low_memory=False)

    dev_candidates = candidate_sides(dev, "value_probability_a")
    dev_candidates["slate_date"] = pd.to_datetime(dev_candidates["odds_date"]).dt.normalize()
    hold_candidates = candidate_sides(hold, "value_probability_a")
    hold_candidates["slate_date"] = pd.to_datetime(hold_candidates["odds_date"]).dt.normalize()

    pcfg = config["policy"]
    dcfg = config["deployment"]
    fraction = float(dcfg["core_fraction_cap"])
    min_history = float(dcfg["minimum_history_matches"])
    min_surface = float(dcfg["minimum_surface_matches"])

    rows = []
    for edge, ev, rel, max_odds in itertools.product(
        pcfg["edge_floors"], pcfg["ev_floors"], pcfg["min_reliability"], pcfg["max_decimal_odds"]
    ):
        bets = select_fraction_policy(
            dev_candidates,
            fraction,
            float(edge),
            float(ev),
            float(rel),
            float(max_odds),
            min_history,
            min_surface,
        )
        row = {
            "fraction_cap": fraction,
            "edge_floor": float(edge),
            "ev_floor": float(ev),
            "min_reliability": float(rel),
            "max_decimal_odds": float(max_odds),
            "min_history": min_history,
            "min_surface_history": min_surface,
            **robust_row(bets),
        }
        rows.append(row)

    grid = pd.DataFrame(rows).sort_values(["robust_score", "bets"], ascending=[False, False]).reset_index(drop=True)
    credible = grid[grid["robust_score"] > -900]
    if not credible.empty:
        best = credible.iloc[0]
    else:
        fallback = grid[(grid["roi_haircut_5pct"] >= 0) & (grid["bets"] >= 30)].copy()
        if fallback.empty:
            fallback = grid.sort_values(["bets", "min_reliability"], ascending=[False, False])
        best = fallback.iloc[0]

    core = FrozenPolicy(
        name="C1-CORE10",
        fraction_cap=fraction,
        edge_floor=float(best["edge_floor"]),
        ev_floor=float(best["ev_floor"]),
        min_reliability=float(best["min_reliability"]),
        max_decimal_odds=float(best["max_decimal_odds"]),
        min_history=min_history,
        min_surface_history=min_surface,
    )

    hold_bets = core.apply(hold_candidates)
    core_summary = summarize_bets(hold_bets)
    core_5 = summarize_bets(hold_bets, 0.05)
    core_boot = bootstrap_roi(hold_bets) if len(hold_bets) >= 20 else {}

    # Regression checks: development selection and holdout application must use
    # the same frozen 4/2 eligibility rule.
    for name, frame in (("development candidates", dev_candidates), ("holdout bets", hold_bets)):
        if name == "holdout bets" and not frame.empty:
            if float(frame["min_experience"].min()) < min_history:
                raise RuntimeError("CORE10 holdout history eligibility mismatch")
            if float(frame["min_surface_experience"].min()) < min_surface:
                raise RuntimeError("CORE10 holdout surface eligibility mismatch")

    grid.to_csv(out / "CORE10_DEVELOPMENT_GRID_CORRECTED.csv", index=False)
    hold_bets.to_csv(out / "CORE10_HOLDOUT_BETS_CORRECTED.csv", index=False)
    for table_name, table in diagnostic_tables(hold_candidates, hold_bets).items():
        table.to_csv(out / f"CORE10_{table_name.upper()}_CORRECTED.csv", index=False)

    policy = {
        "name": core.name,
        "fraction_cap": core.fraction_cap,
        "edge_floor": core.edge_floor,
        "ev_floor": core.ev_floor,
        "min_reliability": core.min_reliability,
        "max_decimal_odds": core.max_decimal_odds,
        "min_history": core.min_history,
        "min_surface_history": core.min_surface_history,
    }
    report = {
        "model": "ChallengerC1",
        "revision": "Rev5",
        "correction": "CORE10 development eligibility mismatch fix",
        "selected_history_days": 730,
        "core_policy_corrected": policy,
        "core_development_corrected": best.to_dict(),
        "core_holdout_corrected": core_summary,
        "core_holdout_5pct_price_haircut_corrected": core_5,
        "core_bootstrap": core_boot,
        "volume20_unchanged": old_report.get("volume20_holdout"),
        "evaluation_grade": "NON_APPROVAL_GRADE_SENSITIVITY",
        "odds_source": "TennisExplorer",
        "odds_price_type": "historical_displayed_average",
        "immutable_prestart_quote_provenance": False,
        "betting_policy_approved": False,
        "bug_explanation": (
            "The original development grid used select_fraction_policy defaults min_history=2 and "
            "min_surface_history=0, while the frozen CORE10 holdout policy used the predeclared "
            "deployment thresholds 4 and 2. This correction applies the same predeclared 4/2 "
            "eligibility to development selection and holdout application. No threshold was chosen "
            "from holdout outcomes."
        ),
    }
    (out / "CORE10_CORRECTED_EVALUATION.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
