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

from challenger_c1.market import no_vig_probability
from challenger_c1.policy import candidate_sides, select_fraction_policy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slate", required=True, help="CSV with player1/player2/date/surface and optional odds/meta")
    ap.add_argument("--config", default=str(ROOT / "config" / "default.json"))
    ap.add_argument("--feature-state", default=str(ROOT / "artifacts" / "feature_state.joblib"))
    ap.add_argument("--fundamental", default=str(ROOT / "artifacts" / "challenger_c1_fundamental.joblib"))
    ap.add_argument("--value", default=str(ROOT / "artifacts" / "challenger_c1_value.joblib"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs"))
    args = ap.parse_args()

    config = json.loads(Path(args.config).read_text())
    builder = joblib.load(args.feature_state)
    fund = joblib.load(args.fundamental)
    slate = pd.read_csv(args.slate)
    live_features = builder.build_live(slate)
    pred = fund.predict_frame(live_features)

    if {"odd_a", "odd_b"}.issubset(pred.columns):
        pred["market_probability_a"] = no_vig_probability(pred["odd_a"], pred["odd_b"])
        if Path(args.value).exists():
            value = joblib.load(args.value)
            pred = value.predict_frame(pred, fundamental_col="fundamental_probability_a")
            prob_col = "value_probability_a"
        else:
            # Without a trained historical market layer, fundamental-vs-market
            # disagreement is visible but not promoted to an official bet card.
            pred["value_probability_a"] = pred["fundamental_probability_a"]
            prob_col = "value_probability_a"
        candidates = candidate_sides(pred, prob_col)
        candidates["slate_date"] = pd.to_datetime(candidates["date"]).dt.normalize()
        dcfg = config["deployment"]
        volume = select_fraction_policy(
            candidates,
            float(dcfg["volume_fraction_cap"]),
            0.04, 0.02, float(dcfg["minimum_reliability"]), float(dcfg["maximum_decimal_odds"]),
            float(dcfg["minimum_history_matches"]), float(dcfg["minimum_surface_matches"]),
        )
        core = select_fraction_policy(
            candidates,
            float(dcfg["core_fraction_cap"]),
            0.06, 0.04, max(0.60, float(dcfg["minimum_reliability"])), min(4.0, float(dcfg["maximum_decimal_odds"])),
            max(6.0, float(dcfg["minimum_history_matches"])), max(3.0, float(dcfg["minimum_surface_matches"])),
        )
    else:
        candidates = pred.copy(); volume = pred.iloc[:0].copy(); core = pred.iloc[:0].copy()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out / "all_predictions.csv", index=False)
    candidates.to_csv(out / "all_candidates.csv", index=False)
    volume.to_csv(out / "volume20_bets.csv", index=False)
    core.to_csv(out / "core_bets.csv", index=False)
    print(f"All matches: {len(pred)} | volume: {len(volume)} | core: {len(core)}")


if __name__ == "__main__":
    main()
