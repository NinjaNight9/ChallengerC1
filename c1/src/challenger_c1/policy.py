from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def american_odds(decimal: float) -> str:
    if not np.isfinite(decimal) or decimal <= 1:
        return ""
    if decimal >= 2.0:
        return f"+{int(round((decimal - 1.0) * 100))}"
    return str(int(round(-100.0 / (decimal - 1.0))))


def candidate_sides(frame: pd.DataFrame, probability_col: str = "value_probability_a") -> pd.DataFrame:
    out = frame.copy()
    p_a = np.clip(out[probability_col].to_numpy(float), 1e-5, 1 - 1e-5)
    p_mkt_a = np.clip(out["market_probability_a"].to_numpy(float), 1e-5, 1 - 1e-5)
    odd_a = out["odd_a"].to_numpy(float)
    odd_b = out["odd_b"].to_numpy(float)
    ev_a = p_a * odd_a - 1.0
    ev_b = (1.0 - p_a) * odd_b - 1.0
    choose_a = ev_a >= ev_b
    out["bet_side"] = np.where(choose_a, "A", "B")
    out["bet_player"] = np.where(choose_a, out["player_a"], out["player_b"])
    out["opponent"] = np.where(choose_a, out["player_b"], out["player_a"])
    out["bet_probability"] = np.where(choose_a, p_a, 1.0 - p_a)
    out["market_probability"] = np.where(choose_a, p_mkt_a, 1.0 - p_mkt_a)
    out["decimal_odds"] = np.where(choose_a, odd_a, odd_b)
    out["edge_pp"] = out["bet_probability"] - out["market_probability"]
    out["expected_roi"] = out["bet_probability"] * out["decimal_odds"] - 1.0
    out["bet_american"] = out["decimal_odds"].map(american_odds)

    # Consensus quality: a large edge matters, but reliable player history and
    # low ensemble dispersion determine how much of that disagreement we trust.
    reliability = out["reliability"].fillna(0.0).to_numpy(float)
    dispersion = out["ensemble_dispersion"].fillna(0.12).to_numpy(float)
    agreement = np.clip(1.0 - dispersion / 0.18, 0.0, 1.0)
    edge = np.maximum(out["edge_pp"].to_numpy(float), 0.0)
    ev = np.maximum(out["expected_roi"].to_numpy(float), 0.0)
    out["selection_score"] = (
        (edge + 0.22 * ev)
        * (0.45 + 0.55 * reliability)
        * (0.65 + 0.35 * agreement)
    )
    if "target" in out:
        won = np.where(choose_a, out["target"].to_numpy(int) == 1, out["target"].to_numpy(int) == 0)
        out["won"] = won
        out["profit_units"] = np.where(won, out["decimal_odds"] - 1.0, -1.0)
    return out


def _worse_price(decimal: np.ndarray, haircut: float) -> np.ndarray:
    # Haircut only the profit portion.  2.50 at 5% -> 2.425.
    return 1.0 + (np.asarray(decimal, dtype=float) - 1.0) * (1.0 - haircut)


def _max_drawdown(profits: np.ndarray) -> float:
    if len(profits) == 0:
        return 0.0
    equity = np.cumsum(profits)
    running_peak = np.maximum.accumulate(np.maximum(equity, 0.0))
    return float(np.max(running_peak - equity))


def summarize_bets(bets: pd.DataFrame, haircut: float = 0.0) -> dict[str, float]:
    if bets.empty:
        return {
            "bets": 0, "wins": 0, "hit_rate": np.nan, "profit_units": 0.0,
            "roi": np.nan, "max_drawdown": 0.0, "profitable_months": 0,
            "months": 0, "median_month_roi": np.nan, "worst_month_roi": np.nan,
            "profit_without_best_win": 0.0, "roi_without_best_win": np.nan,
        }
    odds = _worse_price(bets["decimal_odds"].to_numpy(float), haircut)
    won = bets["won"].to_numpy(bool)
    profits = np.where(won, odds - 1.0, -1.0)
    tmp = bets[["slate_date"]].copy()
    tmp["profit"] = profits
    tmp["month"] = pd.to_datetime(tmp["slate_date"]).dt.to_period("M").astype(str)
    monthly = tmp.groupby("month")["profit"].agg(["sum", "count"])
    monthly["roi"] = monthly["sum"] / monthly["count"]
    best = float(np.max(profits))
    without = float(np.sum(profits) - best)
    denom_without = max(1, len(profits) - 1)
    return {
        "bets": int(len(bets)),
        "wins": int(won.sum()),
        "hit_rate": float(won.mean()),
        "profit_units": float(profits.sum()),
        "roi": float(profits.mean()),
        "max_drawdown": _max_drawdown(profits),
        "profitable_months": int((monthly["sum"] > 0).sum()),
        "months": int(len(monthly)),
        "median_month_roi": float(monthly["roi"].median()),
        "worst_month_roi": float(monthly["roi"].min()),
        "profit_without_best_win": without,
        "roi_without_best_win": without / denom_without,
    }


def select_fraction_policy(
    candidates: pd.DataFrame,
    fraction: float,
    edge_floor: float,
    ev_floor: float,
    min_reliability: float,
    max_decimal_odds: float,
    min_history: float = 2.0,
    min_surface_history: float = 0.0,
) -> pd.DataFrame:
    x = candidates.copy()
    x = x[
        (x["edge_pp"] >= edge_floor)
        & (x["expected_roi"] >= ev_floor)
        & (x["reliability"] >= min_reliability)
        & (x["decimal_odds"] <= max_decimal_odds)
        & (x["min_experience"] >= min_history)
        & (x["min_surface_experience"] >= min_surface_history)
    ].copy()
    if x.empty:
        return x

    # The cap is based on the complete matched slate, not just qualifiers. This
    # implements "up to top 20%" rather than forcing 20% of whatever survives.
    slate_sizes = candidates.groupby("slate_date").size().to_dict()
    selected = []
    for slate, group in x.groupby("slate_date", sort=True):
        cap = max(1, int(math.ceil(float(fraction) * slate_sizes.get(slate, len(group)))))
        selected.append(group.sort_values(["selection_score", "edge_pp"], ascending=False).head(cap))
    return pd.concat(selected, ignore_index=True) if selected else x.iloc[:0].copy()


def evaluate_policy_grid(
    frame: pd.DataFrame,
    config: dict[str, Any],
    probability_col: str = "value_probability_a",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = candidate_sides(frame, probability_col=probability_col)
    if "odds_date" in candidates:
        candidates["slate_date"] = pd.to_datetime(candidates["odds_date"]).dt.normalize()
    else:
        candidates["slate_date"] = pd.to_datetime(candidates["date"]).dt.normalize()

    pcfg = config["policy"]
    rows = []
    for fraction, edge, ev, rel, max_odds in itertools.product(
        pcfg["top_fractions"], pcfg["edge_floors"], pcfg["ev_floors"],
        pcfg["min_reliability"], pcfg["max_decimal_odds"],
    ):
        bets = select_fraction_policy(
            candidates, float(fraction), float(edge), float(ev), float(rel), float(max_odds)
        )
        base = summarize_bets(bets, 0.0)
        stress5 = summarize_bets(bets, 0.05)
        row = {
            "fraction_cap": float(fraction),
            "edge_floor": float(edge),
            "ev_floor": float(ev),
            "min_reliability": float(rel),
            "max_decimal_odds": float(max_odds),
            **base,
            "roi_haircut_5pct": stress5["roi"],
            "profit_haircut_5pct": stress5["profit_units"],
        }
        # Robustness-first policy score. It does not reward a policy whose raw
        # ROI is carried by one long shot or collapses under a modest price move.
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
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["robust_score", "bets"], ascending=[False, False]).reset_index(drop=True)
    return candidates, table


def bootstrap_roi(bets: pd.DataFrame, n_boot: int = 3000, seed: int = 260904) -> dict[str, float]:
    if bets.empty:
        return {"bootstrap_roi_p05": np.nan, "bootstrap_roi_p50": np.nan, "bootstrap_roi_p95": np.nan}
    profits = bets["profit_units"].to_numpy(float)
    rng = np.random.default_rng(seed)
    chunk = 500
    means = []
    remaining = n_boot
    while remaining > 0:
        n = min(chunk, remaining)
        idx = rng.integers(0, len(profits), size=(n, len(profits)))
        means.extend(profits[idx].mean(axis=1).tolist())
        remaining -= n
    q = np.quantile(means, [0.05, 0.50, 0.95])
    return {"bootstrap_roi_p05": float(q[0]), "bootstrap_roi_p50": float(q[1]), "bootstrap_roi_p95": float(q[2])}


def diagnostic_tables(candidates: pd.DataFrame, selected: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if selected.empty:
        return out
    tmp = selected.copy()
    tmp["month"] = pd.to_datetime(tmp["slate_date"]).dt.to_period("M").astype(str)
    out["by_month"] = tmp.groupby("month").agg(
        bets=("won", "size"), wins=("won", "sum"), profit=("profit_units", "sum")
    ).assign(roi=lambda x: x["profit"] / x["bets"]).reset_index()
    out["by_surface"] = tmp.groupby("surface").agg(
        bets=("won", "size"), wins=("won", "sum"), profit=("profit_units", "sum")
    ).assign(roi=lambda x: x["profit"] / x["bets"]).reset_index()
    bins = [1.0, 1.35, 1.65, 2.0, 2.75, 4.0, 6.0, 100.0]
    tmp["odds_band"] = pd.cut(tmp["decimal_odds"], bins=bins, right=False)
    out["by_odds_band"] = tmp.groupby("odds_band", observed=True).agg(
        bets=("won", "size"), wins=("won", "sum"), profit=("profit_units", "sum")
    ).assign(roi=lambda x: x["profit"] / x["bets"]).reset_index()
    edge_bins = [-1, 0.02, 0.04, 0.06, 0.08, 0.12, 1]
    tmp["edge_band"] = pd.cut(tmp["edge_pp"], bins=edge_bins, right=False)
    out["by_edge_band"] = tmp.groupby("edge_band", observed=True).agg(
        bets=("won", "size"), wins=("won", "sum"), profit=("profit_units", "sum")
    ).assign(roi=lambda x: x["profit"] / x["bets"]).reset_index()
    return out


@dataclass
class FrozenPolicy:
    name: str
    fraction_cap: float
    edge_floor: float
    ev_floor: float
    min_reliability: float
    max_decimal_odds: float
    min_history: float
    min_surface_history: float

    def apply(self, candidates: pd.DataFrame) -> pd.DataFrame:
        return select_fraction_policy(
            candidates,
            self.fraction_cap,
            self.edge_floor,
            self.ev_floor,
            self.min_reliability,
            self.max_decimal_odds,
            self.min_history,
            self.min_surface_history,
        )
