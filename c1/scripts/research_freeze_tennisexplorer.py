#!/usr/bin/env python
"""Run the canonical research_freeze workflow with TennisExplorer-safe matching.

This wrapper deliberately does not alter the research architecture. It replaces
only the generic exact-name odds join with the audited abbreviated-name matcher
needed for TennisExplorer (`Surname I.` vs full player names).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import challenger_c1.market as market
from tennisexplorer_matcher import attach_tennisexplorer_odds

# Patch before loading research_freeze so its from-import binds the robust join.
market.attach_odds_to_features = attach_tennisexplorer_odds

spec = importlib.util.spec_from_file_location("challenger_c1_research_freeze", ROOT / "scripts" / "research_freeze.py")
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load research_freeze.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.attach_odds_to_features = attach_tennisexplorer_odds

if __name__ == "__main__":
    module.main()
