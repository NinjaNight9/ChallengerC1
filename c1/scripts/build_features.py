#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from challenger_c1.data import load_sackmann_directory, load_tennismylife_directory
from challenger_c1.features import ChallengerFeatureBuilder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Directory containing Sackmann yearly CSV files")
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--source", choices=["sackmann", "tennismylife"], default="tennismylife")
    ap.add_argument("--config", default=str(ROOT / "config" / "default.json"))
    ap.add_argument("--output", default=str(ROOT / "data" / "processed" / "challenger_features.csv"))
    ap.add_argument("--state-output", default=str(ROOT / "artifacts" / "feature_state.joblib"))
    args = ap.parse_args()

    config = json.loads(Path(args.config).read_text())
    matches = (
        load_tennismylife_directory(args.data_dir, args.years)
        if args.source == "tennismylife"
        else load_sackmann_directory(args.data_dir, args.years)
    )
    builder = ChallengerFeatureBuilder(config)
    features = builder.build(matches)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out, index=False)
    state = Path(args.state_output)
    state.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(builder, state)
    print(f"Loaded {len(matches):,} cross-level matches; built {len(features):,} Challenger target rows")
    print(f"Features: {out}")
    print(f"Live feature state: {state}")


if __name__ == "__main__":
    main()
