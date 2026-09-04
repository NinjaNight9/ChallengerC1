from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from challenger_c1.data import load_sackmann_file
from challenger_c1.features import ChallengerFeatureBuilder, FEATURE_COLUMNS
from challenger_c1.model import walk_forward_fundamental
from challenger_c1.market import add_market_meta_features, walk_forward_market
from challenger_c1.policy import evaluate_policy_grid


def raw_feature_smoke(tmp: Path, config: dict) -> None:
    names = ["Alpha One", "Beta Two", "Gamma Three", "Delta Four", "Epsilon Five", "Zeta Six"]
    rows = []
    for i in range(36):
        w = names[i % len(names)]
        l = names[(i * 2 + 1) % len(names)]
        if w == l:
            l = names[(i + 3) % len(names)]
        rows.append({
            "tourney_id": f"2024-{i//6:03d}", "tourney_name": "Test Challenger",
            "surface": ["Hard", "Clay", "Grass"][i % 3], "draw_size": 32,
            "tourney_level": "C", "tourney_date": 20240101 + (i // 6) * 100,
            "match_num": i, "winner_id": 100 + names.index(w), "winner_seed": np.nan,
            "winner_entry": "Q" if i % 7 == 0 else "", "winner_name": w,
            "winner_hand": "R", "winner_ht": 183, "winner_ioc": "USA", "winner_age": 24,
            "loser_id": 100 + names.index(l), "loser_seed": np.nan, "loser_entry": "",
            "loser_name": l, "loser_hand": "R", "loser_ht": 181, "loser_ioc": "GBR", "loser_age": 25,
            "score": "6-4 6-3", "best_of": 3, "round": ["R32", "R16", "QF", "SF", "F", "R32"][i % 6],
            "minutes": 90, "w_ace": 7, "w_df": 2, "w_svpt": 60, "w_1stIn": 38,
            "w_1stWon": 28, "w_2ndWon": 12, "w_SvGms": 10, "w_bpSaved": 3, "w_bpFaced": 4,
            "l_ace": 4, "l_df": 3, "l_svpt": 64, "l_1stIn": 40, "l_1stWon": 24,
            "l_2ndWon": 10, "l_SvGms": 10, "l_bpSaved": 4, "l_bpFaced": 7,
            "winner_rank": 150 + i, "winner_rank_points": 400, "loser_rank": 240 + i,
            "loser_rank_points": 220,
        })
    path = tmp / "tiny.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    loaded = load_sackmann_file(path, "qual_chall")
    features = ChallengerFeatureBuilder(config).build(loaded)
    assert len(features) == len(loaded)
    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert features["target"].isin([0, 1]).all()


def model_smoke(config: dict) -> None:
    rng = np.random.default_rng(7)
    n = 1800
    dates = pd.date_range("2023-01-01", "2026-08-31", periods=n)
    x = pd.DataFrame({col: rng.normal(size=n) for col in FEATURE_COLUMNS})
    # Repair count / binary feature domains.
    for c in ["min_experience", "min_surface_experience", "min_stat_matches", "h2h_matches"]:
        x[c] = rng.integers(0, 30, size=n)
    x["stat_coverage"] = rng.uniform(0, 1, size=n)
    for c in ["surface_hard", "surface_clay", "surface_grass", "surface_carpet", "surface_unknown", "best_of_5"]:
        x[c] = 0.0
    surfaces = rng.choice(["Hard", "Clay", "Grass"], size=n, p=[0.5, 0.45, 0.05])
    x["surface_hard"] = (surfaces == "Hard").astype(float)
    x["surface_clay"] = (surfaces == "Clay").astype(float)
    x["surface_grass"] = (surfaces == "Grass").astype(float)
    latent = 0.8*x["elo_surface_blend_diff_100"] + 0.5*x["recent_residual_90_diff"] + 0.25*x["spw_90_diff"]
    p = 1/(1+np.exp(-latent))
    y = rng.binomial(1, p)
    frame = pd.DataFrame({
        "date": dates, "proxy_date": dates, "tourney_id": [f"T{i//20}" for i in range(n)],
        "tourney_name": "Synthetic Challenger", "surface": surfaces, "round": "R32",
        "player_a": [f"A{i}" for i in range(n)], "player_b": [f"B{i}" for i in range(n)],
        "player_a_key": [f"a{i}" for i in range(n)], "player_b_key": [f"b{i}" for i in range(n)],
        "target": y,
    })
    frame = pd.concat([frame, x], axis=1)
    cfg = dict(config)
    cfg["minimum_train_matches"] = 250
    cfg["minimum_test_matches"] = 30
    cfg["validation_block_days"] = 120
    oof, folds = walk_forward_fundamental(frame, cfg, 365)
    assert len(oof) > 100
    assert oof["predicted_probability"].between(0,1).all()
    assert len(folds) >= 2

    # Market/policy smoke using synthetic two-sided decimal odds.
    oof = oof.copy()
    market = np.clip(0.5 + 0.75*(oof["predicted_probability"]-0.5) + rng.normal(0,0.05,len(oof)), .08, .92)
    margin = 1.05
    oof["market_probability_a"] = market
    oof["market_logit_a"] = np.log(market/(1-market))
    oof["odd_a"] = 1/(market*margin)
    oof["odd_b"] = 1/((1-market)*margin)
    oof["odds_date"] = oof["date"]
    value = walk_forward_market(oof, min_meta_train=120)
    cfg["policy"] = {
        "top_fractions": [0.10,0.20], "edge_floors": [0.02,0.04], "ev_floors": [0.0],
        "min_reliability": [0.2,0.5], "max_decimal_odds": [3.5,6.0], "price_haircuts": [0.05]
    }
    candidates, grid = evaluate_policy_grid(value, cfg)
    assert len(candidates) == len(value)
    assert len(grid) > 0


if __name__ == "__main__":
    config = json.loads((ROOT / "config" / "default.json").read_text())
    tmp = ROOT / "tests" / "_tmp"; tmp.mkdir(exist_ok=True)
    raw_feature_smoke(tmp, config)
    model_smoke(config)
    print("ChallengerC1 smoke tests passed")
