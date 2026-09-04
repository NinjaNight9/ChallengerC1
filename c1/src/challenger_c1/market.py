from __future__ import annotations

import math
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from .data import normalize_name

MARKET_RESIDUAL_FEATURES = [
    "fund_market_logit_gap",
    "fund_market_prob_gap",
    "reliability_centered",
    "ensemble_dispersion",
    "elo_surface_blend_diff_100",
    "elo_dynamic_surface_blend_diff_100",
    "elo_speed_gap_diff_100",
    "recent_residual_90_diff",
    "surface_residual_180_diff",
    "challenger_residual_180_diff",
    "spw_90_diff",
    "rpw_90_diff",
    "rank_log_diff",
    "points_log_diff",
    "min_experience_log",
    "min_surface_experience_log",
    "stat_coverage",
    "surface_clay",
    "surface_hard",
    "surface_grass",
]


def no_vig_probability(odd_a: np.ndarray, odd_b: np.ndarray) -> np.ndarray:
    ia = 1.0 / np.asarray(odd_a, dtype=float)
    ib = 1.0 / np.asarray(odd_b, dtype=float)
    return ia / (ia + ib)


def _pair_key(a: object, b: object) -> str:
    x, y = sorted((normalize_name(a), normalize_name(b)))
    return f"{x}||{y}"


def _tournament_similarity(a: object, b: object) -> float:
    aa, bb = normalize_name(a), normalize_name(b)
    if not aa or not bb:
        return 0.0
    return SequenceMatcher(None, aa, bb).ratio()


def attach_odds_to_features(
    features: pd.DataFrame,
    odds: pd.DataFrame,
    max_date_gap_days: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match a generic odds feed to Challenger feature rows without result use.

    Sackmann's tourney_date is an event-start date, while many odds feeds have
    the actual match date. We therefore pair by unordered normalized players and
    choose the closest date inside the event week, using tournament similarity
    only as a tie-breaker.

    Returns (matched_frame, audit_frame).
    """
    f = features.copy().reset_index(drop=False).rename(columns={"index": "_feature_index"})
    o = odds.copy().reset_index(drop=False).rename(columns={"index": "_odds_index"})
    f["_pair"] = [_pair_key(a, b) for a, b in zip(f["player_a"], f["player_b"])]
    o["_pair"] = [_pair_key(a, b) for a, b in zip(o["player1"], o["player2"])]
    f["_date"] = pd.to_datetime(f["date"]).dt.normalize()
    o["_date"] = pd.to_datetime(o["date"]).dt.normalize()

    odds_groups = {k: g for k, g in o.groupby("_pair", sort=False)}
    selected = []
    audit = []
    for _, row in f.iterrows():
        candidates = odds_groups.get(row["_pair"])
        if candidates is None or candidates.empty:
            audit.append({"feature_index": row["_feature_index"], "status": "no_pair", "pair": row["_pair"]})
            continue
        c = candidates.copy()
        c["date_gap"] = (c["_date"] - row["_date"]).dt.days.abs()
        c = c[c["date_gap"] <= max_date_gap_days]
        if c.empty:
            audit.append({"feature_index": row["_feature_index"], "status": "date_gap", "pair": row["_pair"]})
            continue
        if "tournament" in c.columns:
            c["tourney_sim"] = c["tournament"].map(lambda x: _tournament_similarity(row.get("tourney_name", ""), x))
        else:
            c["tourney_sim"] = 0.0
        c = c.sort_values(["date_gap", "tourney_sim"], ascending=[True, False], kind="mergesort")
        best = c.iloc[0]

        a_norm = normalize_name(row["player_a"])
        if normalize_name(best["player1"]) == a_norm:
            odd_a, odd_b = float(best["odd1"]), float(best["odd2"])
        elif normalize_name(best["player2"]) == a_norm:
            odd_a, odd_b = float(best["odd2"]), float(best["odd1"])
        else:
            audit.append({"feature_index": row["_feature_index"], "status": "orientation_fail", "pair": row["_pair"]})
            continue
        rec = row.to_dict()
        rec.update({
            "odd_a": odd_a,
            "odd_b": odd_b,
            "odds_date": best["_date"],
            "odds_source": best.get("source", "generic"),
            "odds_date_gap": int(best["date_gap"]),
            "odds_tournament_similarity": float(best["tourney_sim"]),
            "odds_index": int(best["_odds_index"]),
        })
        selected.append(rec)
        audit.append({
            "feature_index": row["_feature_index"], "status": "matched", "pair": row["_pair"],
            "odds_index": int(best["_odds_index"]), "date_gap": int(best["date_gap"]),
            "tourney_sim": float(best["tourney_sim"]),
        })

    matched = pd.DataFrame(selected)
    if not matched.empty:
        matched["market_probability_a"] = no_vig_probability(matched["odd_a"], matched["odd_b"])
        matched["market_logit_a"] = logit(np.clip(matched["market_probability_a"], 1e-5, 1 - 1e-5))
    return matched.drop(columns=[c for c in ["_pair", "_date"] if c in matched], errors="ignore"), pd.DataFrame(audit)


def add_market_meta_features(frame: pd.DataFrame, fundamental_col: str = "predicted_probability") -> pd.DataFrame:
    out = frame.copy()
    fund = np.clip(out[fundamental_col].to_numpy(float), 1e-5, 1 - 1e-5)
    market = np.clip(out["market_probability_a"].to_numpy(float), 1e-5, 1 - 1e-5)
    out["fund_market_logit_gap"] = logit(fund) - logit(market)
    out["fund_market_prob_gap"] = fund - market
    out["reliability_centered"] = out["reliability"].fillna(0.0) - 0.5
    out["min_experience_log"] = np.log1p(out["min_experience"].fillna(0.0))
    out["min_surface_experience_log"] = np.log1p(out["min_surface_experience"].fillna(0.0))
    return out


class MarketOffsetValueModel:
    """Bookmaker log-odds + regularized learned Challenger-specific correction."""

    def __init__(self, l2: float = 18.0, max_iter: int = 2500):
        self.l2 = float(l2)
        self.max_iter = int(max_iter)

    def fit(self, frame: pd.DataFrame, y: np.ndarray | pd.Series) -> "MarketOffsetValueModel":
        xframe = add_market_meta_features(frame)
        self.medians_ = xframe[MARKET_RESIDUAL_FEATURES].median().fillna(0.0)
        z = xframe[MARKET_RESIDUAL_FEATURES].fillna(self.medians_).to_numpy(float)
        self.means_ = z.mean(axis=0)
        self.scales_ = z.std(axis=0)
        self.scales_[self.scales_ < 1e-8] = 1.0
        z = (z - self.means_) / self.scales_
        z = np.column_stack([np.ones(len(z)), z])
        offset = xframe["market_logit_a"].to_numpy(float)
        target = np.asarray(y, dtype=float)

        def objective(beta: np.ndarray):
            lin = offset + z @ beta
            p = expit(lin)
            eps = 1e-12
            loss = -np.sum(target * np.log(p + eps) + (1 - target) * np.log(1 - p + eps))
            penalty = 0.5 * self.l2 * np.sum(beta[1:] ** 2)
            grad = z.T @ (p - target)
            grad[1:] += self.l2 * beta[1:]
            return loss + penalty, grad

        result = minimize(
            objective,
            np.zeros(z.shape[1]),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iter, "ftol": 1e-12},
        )
        if not result.success:
            raise RuntimeError(result.message)
        self.intercept_ = float(result.x[0])
        self.coef_ = result.x[1:]
        self.feature_names_ = list(MARKET_RESIDUAL_FEATURES)
        return self

    def predict(self, frame: pd.DataFrame, fundamental_col: str = "predicted_probability") -> np.ndarray:
        xframe = add_market_meta_features(frame, fundamental_col=fundamental_col)
        z = xframe[self.feature_names_].fillna(self.medians_).to_numpy(float)
        z = (z - self.means_) / self.scales_
        lin = xframe["market_logit_a"].to_numpy(float) + self.intercept_ + z @ self.coef_
        return np.clip(expit(lin), 1e-5, 1 - 1e-5)


@dataclass
class MarketArtifact:
    model_name: str
    model: MarketOffsetValueModel
    training_rows: int
    metadata: dict[str, Any]

    def predict_frame(self, frame: pd.DataFrame, fundamental_col: str = "fundamental_probability_a") -> pd.DataFrame:
        out = frame.copy()
        # Alias for the market model's historical OOF column convention.
        out["predicted_probability"] = out[fundamental_col]
        out["value_probability_a"] = self.model.predict(out, fundamental_col="predicted_probability")
        out["market_probability_a"] = no_vig_probability(out["odd_a"], out["odd_b"])
        return out

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)


def walk_forward_market(oof_with_odds: pd.DataFrame, min_meta_train: int = 350) -> pd.DataFrame:
    frame = oof_with_odds.sort_values(["fold_id", "date", "player_a_key"]).copy()
    parts = []
    for fold in sorted(frame["fold_id"].unique()):
        test = frame[frame["fold_id"] == fold].copy()
        train = frame[frame["fold_id"] < fold].copy()
        if len(train) >= min_meta_train and train["target"].nunique() == 2:
            mdl = MarketOffsetValueModel(l2=18.0).fit(train, train["target"].to_numpy(int))
            p = mdl.predict(test)
        else:
            # Before enough out-of-fold history exists, never manufacture a
            # Challenger edge: use the no-vig market as the conservative prior.
            p = test["market_probability_a"].to_numpy(float)
        test["value_probability_a"] = p
        parts.append(test)
    return pd.concat(parts, ignore_index=True)


def fit_final_market(oof_with_odds: pd.DataFrame) -> MarketArtifact:
    model = MarketOffsetValueModel(l2=18.0).fit(oof_with_odds, oof_with_odds["target"].to_numpy(int))
    return MarketArtifact(
        model_name="ChallengerC1-VALUE",
        model=model,
        training_rows=len(oof_with_odds),
        metadata={"feature_names": MARKET_RESIDUAL_FEATURES, "l2": 18.0},
    )
