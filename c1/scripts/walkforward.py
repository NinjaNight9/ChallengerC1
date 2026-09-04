#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from challenger_c1.data import canonicalize_generic_odds
from challenger_c1.market import attach_odds_to_features, fit_final_market, walk_forward_market
from challenger_c1.model import compare_horizons, fit_final_fundamental, save_json
from challenger_c1.policy import evaluate_policy_grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(ROOT / "data" / "processed" / "challenger_features.csv"))
    ap.add_argument("--config", default=str(ROOT / "config" / "default.json"))
    ap.add_argument("--odds", help="Optional generic historical odds CSV")
    ap.add_argument("--odds-map", help="Optional JSON mapping canonical field -> odds CSV column")
    ap.add_argument("--out-dir", default=str(ROOT / "reports"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text())
    frame = pd.read_csv(args.features, low_memory=False)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["proxy_date"] = pd.to_datetime(frame["proxy_date"])

    best, horizon_summary, oof_by_h, folds_by_h = compare_horizons(frame, config)
    horizon_summary.to_csv(out_dir / "horizon_comparison.csv", index=False)
    for h, oof in oof_by_h.items():
        oof.to_csv(out_dir / f"fundamental_oof_{h}d.csv", index=False)
        folds_by_h[h].to_csv(out_dir / f"fundamental_folds_{h}d.csv", index=False)

    best_oof = oof_by_h[best]
    artifact = fit_final_fundamental(frame, config, best, best_oof)
    artifact.save(ROOT / "artifacts" / "challenger_c1_fundamental.joblib")
    save_json(artifact.metadata, out_dir / "fundamental_final_metadata.json")
    print("Best fundamental history horizon:", best)
    print(horizon_summary.to_string(index=False))

    if args.odds:
        mapping = json.loads(Path(args.odds_map).read_text()) if args.odds_map else None
        odds = canonicalize_generic_odds(args.odds, mapping)
        matched, audit = attach_odds_to_features(best_oof, odds)
        matched.to_csv(out_dir / "oof_with_odds_matched.csv", index=False)
        audit.to_csv(out_dir / "odds_match_audit.csv", index=False)
        print(f"Historical odds matched: {len(matched):,}/{len(best_oof):,}")
        if len(matched) >= 500:
            value_oof = walk_forward_market(matched)
            value_oof.to_csv(out_dir / "value_oof.csv", index=False)
            market_art = fit_final_market(matched)
            market_art.save(ROOT / "artifacts" / "challenger_c1_value.joblib")
            candidates, grid = evaluate_policy_grid(value_oof, config)
            candidates.to_csv(out_dir / "value_candidates_oof.csv", index=False)
            grid.to_csv(out_dir / "policy_grid.csv", index=False)
            print("Top robust policy candidates:")
            print(grid.head(15).to_string(index=False))
        else:
            print("Too few matched OOF odds rows for a credible value layer; not fitting it.")


if __name__ == "__main__":
    main()
