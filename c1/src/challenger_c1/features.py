from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import derive_match_stats, normalize_name

DIFF_FEATURES = [
    "elo_slow_diff_100",
    "elo_dynamic_diff_100",
    "elo_fast_diff_100",
    "elo_surface_diff_100",
    "elo_surface_blend_diff_100",
    "elo_dynamic_surface_blend_diff_100",
    "elo_speed_gap_diff_100",
    "rank_log_diff",
    "points_log_diff",
    "age_diff_10",
    "height_diff_10",
    "experience_log_diff",
    "surface_experience_log_diff",
    "recent_residual_35_diff",
    "recent_residual_90_diff",
    "recent_residual_180_diff",
    "surface_residual_180_diff",
    "recent_win_35_diff",
    "recent_win_90_diff",
    "opp_elo_90_diff_100",
    "set_share_90_diff",
    "game_share_90_diff",
    "spw_90_diff",
    "rpw_90_diff",
    "hold_90_diff",
    "break_90_diff",
    "challenger_residual_180_diff",
    "main_residual_180_diff",
    "tour_qual_residual_180_diff",
    "futures_residual_180_diff",
    "rest_days_diff_14",
    "workload_7d_diff",
    "workload_14d_diff",
    "workload_28d_diff",
    "entry_q_diff",
    "entry_wc_diff",
    "entry_ll_diff",
    "entry_pr_diff",
    "h2h_advantage",
]

SHARED_FEATURES = [
    "min_experience",
    "min_surface_experience",
    "min_stat_matches",
    "stat_coverage",
    "h2h_matches",
    "surface_hard",
    "surface_clay",
    "surface_grass",
    "surface_carpet",
    "surface_unknown",
    "indoor_flag",
    "indoor_known",
    "round_order_scaled",
    "best_of_5",
]

FEATURE_COLUMNS = DIFF_FEATURES + SHARED_FEATURES


def _finite(x: Any, default: float = np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _logit_elo(diff: float, scale: float = 400.0) -> float:
    return 1.0 / (1.0 + 10.0 ** (-diff / scale))


def _rank_seed(rank: float, base: float, strength: float) -> float:
    if not np.isfinite(rank) or rank <= 0:
        return base
    raw = float(np.clip(2000.0 - 250.0 * math.log10(rank), 1150.0, 2050.0))
    return base + strength * (raw - base)


@dataclass
class Observation:
    date: pd.Timestamp
    win: float
    expected: float
    residual: float
    opponent_elo: float
    surface: str
    level_role: str
    spw: float = np.nan
    rpw: float = np.nan
    hold: float = np.nan
    break_rate: float = np.nan
    set_share: float = np.nan
    game_share: float = np.nan


@dataclass
class PlayerState:
    elo_slow: float
    elo_dynamic: float
    elo_fast: float
    matches: int = 0
    wins: int = 0
    last_date: pd.Timestamp | None = None
    surface_elo: dict[str, float] = field(default_factory=dict)
    surface_dynamic_elo: dict[str, float] = field(default_factory=dict)
    surface_matches: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    history: deque[Observation] = field(default_factory=lambda: deque(maxlen=140))
    latest_rank: float = np.nan
    latest_points: float = np.nan
    latest_age: float = np.nan
    latest_height: float = np.nan


class ChallengerFeatureBuilder:
    """Leakage-safe sequential feature engine across ATP/CH/qualifying/ITF levels.

    Every prediction row is created before the corresponding result updates any
    state. Challenger rows are retained as labels; other levels are useful for
    strength transfer and player-pool connectivity.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        rating = config["rating"]
        self.base = float(rating["base"])
        self.scale = float(rating["scale"])
        self.k_slow = float(rating["k_slow"])
        self.k_fast = float(rating["k_fast"])
        self.k_surface = float(rating["k_surface"])
        self.surface_prior = float(rating["surface_prior_matches"])
        self.rank_seed_strength = float(rating.get("rank_seed_strength", 0.45))
        self.states: dict[str, PlayerState] = {}
        self.h2h: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        self.name_to_key: dict[str, str] = {}

    @classmethod
    def from_config_file(cls, path: str | Path) -> "ChallengerFeatureBuilder":
        return cls(json.loads(Path(path).read_text()))

    def _state(self, key: str, rank: float = np.nan) -> PlayerState:
        if key not in self.states:
            seed = _rank_seed(rank, self.base, self.rank_seed_strength)
            self.states[key] = PlayerState(elo_slow=seed, elo_dynamic=seed, elo_fast=seed)
        return self.states[key]

    @staticmethod
    def _weighted_mean(
        history: deque[Observation],
        now: pd.Timestamp,
        field: str,
        half_life_days: float,
        default: float,
        surface: str | None = None,
        level_role: str | None = None,
    ) -> tuple[float, float]:
        vals: list[float] = []
        weights: list[float] = []
        ln2 = math.log(2.0)
        for obs in history:
            if surface is not None and obs.surface != surface:
                continue
            if level_role is not None and obs.level_role != level_role:
                continue
            value = getattr(obs, field)
            if not np.isfinite(value):
                continue
            age = max(0.0, float((now - obs.date).days))
            w = math.exp(-ln2 * age / half_life_days)
            vals.append(float(value))
            weights.append(w)
        if not vals:
            return default, 0.0
        return float(np.average(vals, weights=weights)), float(sum(weights))

    @staticmethod
    def _count_window(history: deque[Observation], now: pd.Timestamp, days: int) -> int:
        return sum(0 <= (now - obs.date).days <= days for obs in history)

    def _snapshot(
        self,
        state: PlayerState,
        now: pd.Timestamp,
        surface: str,
        rank: float,
        points: float,
        age: float,
        height: float,
    ) -> dict[str, float]:
        rank = rank if np.isfinite(rank) else state.latest_rank
        points = points if np.isfinite(points) else state.latest_points
        age = age if np.isfinite(age) else state.latest_age
        height = height if np.isfinite(height) else state.latest_height
        surface_rating = state.surface_elo.get(surface, state.elo_slow)
        surface_dynamic = state.surface_dynamic_elo.get(surface, state.elo_dynamic)
        n_surface = float(state.surface_matches.get(surface, 0))
        sw = n_surface / (n_surface + self.surface_prior)
        surface_blend = sw * surface_rating + (1.0 - sw) * state.elo_slow
        dynamic_surface_blend = sw * surface_dynamic + (1.0 - sw) * state.elo_dynamic

        resid35, _ = self._weighted_mean(state.history, now, "residual", 35, 0.0)
        resid90, _ = self._weighted_mean(state.history, now, "residual", 90, 0.0)
        resid180, _ = self._weighted_mean(state.history, now, "residual", 180, 0.0)
        sres180, _ = self._weighted_mean(state.history, now, "residual", 180, 0.0, surface=surface)
        win35, _ = self._weighted_mean(state.history, now, "win", 35, 0.5)
        win90, _ = self._weighted_mean(state.history, now, "win", 90, 0.5)
        opp90, _ = self._weighted_mean(state.history, now, "opponent_elo", 90, self.base)
        set90, _ = self._weighted_mean(state.history, now, "set_share", 90, 0.5)
        game90, _ = self._weighted_mean(state.history, now, "game_share", 90, 0.5)
        spw90, spw_eff = self._weighted_mean(state.history, now, "spw", 90, 0.5)
        rpw90, rpw_eff = self._weighted_mean(state.history, now, "rpw", 90, 0.5)
        hold90, hold_eff = self._weighted_mean(state.history, now, "hold", 90, 0.5)
        break90, break_eff = self._weighted_mean(state.history, now, "break_rate", 90, 0.5)

        role_resid = {}
        for role in ("challenger", "main", "tour_qual", "futures"):
            role_resid[role], _ = self._weighted_mean(
                state.history, now, "residual", 180, 0.0, level_role=role
            )

        rest_days = 21.0 if state.last_date is None else float(np.clip((now - state.last_date).days, 0, 60))
        stat_eff = min(spw_eff, rpw_eff, hold_eff, break_eff)
        return {
            "elo_slow": state.elo_slow,
            "elo_dynamic": state.elo_dynamic,
            "elo_fast": state.elo_fast,
            "elo_surface": surface_rating,
            "elo_surface_blend": surface_blend,
            "elo_dynamic_surface_blend": dynamic_surface_blend,
            "rank": rank if np.isfinite(rank) and rank > 0 else 750.0,
            "points": points if np.isfinite(points) and points >= 0 else 5.0,
            "age": age if np.isfinite(age) else 25.0,
            "height": height if np.isfinite(height) else 183.0,
            "experience": float(state.matches),
            "surface_experience": n_surface,
            "recent_residual_35": resid35,
            "recent_residual_90": resid90,
            "recent_residual_180": resid180,
            "surface_residual_180": sres180,
            "recent_win_35": win35,
            "recent_win_90": win90,
            "opp_elo_90": opp90,
            "set_share_90": set90,
            "game_share_90": game90,
            "spw_90": spw90,
            "rpw_90": rpw90,
            "hold_90": hold90,
            "break_90": break90,
            "challenger_residual_180": role_resid["challenger"],
            "main_residual_180": role_resid["main"],
            "tour_qual_residual_180": role_resid["tour_qual"],
            "futures_residual_180": role_resid["futures"],
            "rest_days": rest_days,
            "workload_7d": float(self._count_window(state.history, now, 7)),
            "workload_14d": float(self._count_window(state.history, now, 14)),
            "workload_28d": float(self._count_window(state.history, now, 28)),
            "stat_matches_eff": stat_eff,
        }

    @staticmethod
    def _entry_flags(entry: object) -> dict[str, float]:
        e = str(entry or "").upper()
        return {
            "entry_q": float(e in {"Q", "Q1", "Q2", "Q3"}),
            "entry_wc": float(e == "WC"),
            "entry_ll": float(e == "LL"),
            "entry_pr": float(e in {"PR", "P"}),
        }

    def _h2h_snapshot(self, a: str, b: str) -> tuple[float, float]:
        key = tuple(sorted((a, b)))
        wins_lo, wins_hi = self.h2h[key]
        if a == key[0]:
            aw, bw = wins_lo, wins_hi
        else:
            aw, bw = wins_hi, wins_lo
        n = aw + bw
        # Strong shrinkage: H2H is a weak contextual feature, not a narrative engine.
        return (aw + 1.5) / (n + 3.0) - 0.5, float(n)

    def _feature_row(
        self,
        a_key: str,
        b_key: str,
        now: pd.Timestamp,
        surface: str,
        a_meta: dict[str, float | str],
        b_meta: dict[str, float | str],
        round_order: float,
        best_of: float,
        indoor_flag: float = 0.0,
        indoor_known: float = 0.0,
    ) -> dict[str, float]:
        a_state = self._state(a_key, _finite(a_meta["rank"]))
        b_state = self._state(b_key, _finite(b_meta["rank"]))
        a = self._snapshot(
            a_state, now, surface,
            _finite(a_meta["rank"]), _finite(a_meta["points"]), _finite(a_meta["age"]), _finite(a_meta["height"])
        )
        b = self._snapshot(
            b_state, now, surface,
            _finite(b_meta["rank"]), _finite(b_meta["points"]), _finite(b_meta["age"]), _finite(b_meta["height"])
        )
        ae, be = self._entry_flags(a_meta.get("entry", "")), self._entry_flags(b_meta.get("entry", ""))
        h2h_adv, h2h_n = self._h2h_snapshot(a_key, b_key)
        stat_cov = min(a["stat_matches_eff"], b["stat_matches_eff"])
        # Smooth effective-count coverage. Around 5 effective recent matches is already useful;
        # 10+ is high confidence.
        stat_coverage = stat_cov / (stat_cov + 5.0) if stat_cov > 0 else 0.0
        features = {
            "elo_slow_diff_100": (a["elo_slow"] - b["elo_slow"]) / 100.0,
            "elo_dynamic_diff_100": (a["elo_dynamic"] - b["elo_dynamic"]) / 100.0,
            "elo_fast_diff_100": (a["elo_fast"] - b["elo_fast"]) / 100.0,
            "elo_surface_diff_100": (a["elo_surface"] - b["elo_surface"]) / 100.0,
            "elo_surface_blend_diff_100": (a["elo_surface_blend"] - b["elo_surface_blend"]) / 100.0,
            "elo_dynamic_surface_blend_diff_100": (a["elo_dynamic_surface_blend"] - b["elo_dynamic_surface_blend"]) / 100.0,
            "elo_speed_gap_diff_100": ((a["elo_fast"] - a["elo_slow"]) - (b["elo_fast"] - b["elo_slow"])) / 100.0,
            # Positive means A is stronger by ranking/points.
            "rank_log_diff": math.log(max(b["rank"], 1.0)) - math.log(max(a["rank"], 1.0)),
            "points_log_diff": math.log1p(max(a["points"], 0.0)) - math.log1p(max(b["points"], 0.0)),
            "age_diff_10": (a["age"] - b["age"]) / 10.0,
            "height_diff_10": (a["height"] - b["height"]) / 10.0,
            "experience_log_diff": math.log1p(a["experience"]) - math.log1p(b["experience"]),
            "surface_experience_log_diff": math.log1p(a["surface_experience"]) - math.log1p(b["surface_experience"]),
            "recent_residual_35_diff": a["recent_residual_35"] - b["recent_residual_35"],
            "recent_residual_90_diff": a["recent_residual_90"] - b["recent_residual_90"],
            "recent_residual_180_diff": a["recent_residual_180"] - b["recent_residual_180"],
            "surface_residual_180_diff": a["surface_residual_180"] - b["surface_residual_180"],
            "recent_win_35_diff": a["recent_win_35"] - b["recent_win_35"],
            "recent_win_90_diff": a["recent_win_90"] - b["recent_win_90"],
            "opp_elo_90_diff_100": (a["opp_elo_90"] - b["opp_elo_90"]) / 100.0,
            "set_share_90_diff": a["set_share_90"] - b["set_share_90"],
            "game_share_90_diff": a["game_share_90"] - b["game_share_90"],
            "spw_90_diff": a["spw_90"] - b["spw_90"],
            "rpw_90_diff": a["rpw_90"] - b["rpw_90"],
            "hold_90_diff": a["hold_90"] - b["hold_90"],
            "break_90_diff": a["break_90"] - b["break_90"],
            "challenger_residual_180_diff": a["challenger_residual_180"] - b["challenger_residual_180"],
            "main_residual_180_diff": a["main_residual_180"] - b["main_residual_180"],
            "tour_qual_residual_180_diff": a["tour_qual_residual_180"] - b["tour_qual_residual_180"],
            "futures_residual_180_diff": a["futures_residual_180"] - b["futures_residual_180"],
            "rest_days_diff_14": (a["rest_days"] - b["rest_days"]) / 14.0,
            "workload_7d_diff": a["workload_7d"] - b["workload_7d"],
            "workload_14d_diff": a["workload_14d"] - b["workload_14d"],
            "workload_28d_diff": a["workload_28d"] - b["workload_28d"],
            "entry_q_diff": ae["entry_q"] - be["entry_q"],
            "entry_wc_diff": ae["entry_wc"] - be["entry_wc"],
            "entry_ll_diff": ae["entry_ll"] - be["entry_ll"],
            "entry_pr_diff": ae["entry_pr"] - be["entry_pr"],
            "h2h_advantage": h2h_adv,
            "min_experience": min(a["experience"], b["experience"]),
            "min_surface_experience": min(a["surface_experience"], b["surface_experience"]),
            "min_stat_matches": stat_cov,
            "stat_coverage": stat_coverage,
            "h2h_matches": h2h_n,
            "surface_hard": float(surface == "Hard"),
            "surface_clay": float(surface == "Clay"),
            "surface_grass": float(surface == "Grass"),
            "surface_carpet": float(surface == "Carpet"),
            "surface_unknown": float(surface == "Unknown"),
            "indoor_flag": float(indoor_flag),
            "indoor_known": float(indoor_known),
            "round_order_scaled": float(round_order) / 10.0,
            "best_of_5": float(best_of == 5),
        }
        return features

    def _update_pair(
        self,
        winner_key: str,
        loser_key: str,
        now: pd.Timestamp,
        surface: str,
        level_role: str,
        winner_meta: dict[str, float],
        loser_meta: dict[str, float],
        match_stats: dict[str, float],
    ) -> None:
        w = self._state(winner_key, _finite(winner_meta.get("rank")))
        l = self._state(loser_key, _finite(loser_meta.get("rank")))
        pre_w_slow, pre_l_slow = w.elo_slow, l.elo_slow
        exp_w_slow = _logit_elo(pre_w_slow - pre_l_slow, self.scale)
        exp_w_dynamic = _logit_elo(w.elo_dynamic - l.elo_dynamic, self.scale)
        exp_w_fast = _logit_elo(w.elo_fast - l.elo_fast, self.scale)
        w_surf = w.surface_elo.get(surface, w.elo_slow)
        l_surf = l.surface_elo.get(surface, l.elo_slow)
        w_surf_dyn = w.surface_dynamic_elo.get(surface, w.elo_dynamic)
        l_surf_dyn = l.surface_dynamic_elo.get(surface, l.elo_dynamic)
        exp_w_surf = _logit_elo(w_surf - l_surf, self.scale)
        exp_w_surf_dyn = _logit_elo(w_surf_dyn - l_surf_dyn, self.scale)

        delta_slow = self.k_slow * (1.0 - exp_w_slow)
        # Tennis-Abstract-style experience decay, capped only for numerical safety.
        k_w = 250.0 / ((w.matches + 5.0) ** 0.4)
        k_l = 250.0 / ((l.matches + 5.0) ** 0.4)
        k_dyn = 0.5 * (k_w + k_l)
        delta_dynamic = k_dyn * (1.0 - exp_w_dynamic)
        delta_fast = self.k_fast * (1.0 - exp_w_fast)
        delta_surface = self.k_surface * (1.0 - exp_w_surf)
        swm = w.surface_matches.get(surface, 0); slm = l.surface_matches.get(surface, 0)
        k_surf_dyn = 0.5 * (250.0 / ((swm + 5.0) ** 0.4) + 250.0 / ((slm + 5.0) ** 0.4))
        delta_surf_dyn = k_surf_dyn * (1.0 - exp_w_surf_dyn)
        w.elo_slow += delta_slow
        l.elo_slow -= delta_slow
        w.elo_dynamic += delta_dynamic
        l.elo_dynamic -= delta_dynamic
        w.elo_fast += delta_fast
        l.elo_fast -= delta_fast
        w.surface_elo[surface] = w_surf + delta_surface
        l.surface_elo[surface] = l_surf - delta_surface
        w.surface_dynamic_elo[surface] = w_surf_dyn + delta_surf_dyn
        l.surface_dynamic_elo[surface] = l_surf_dyn - delta_surf_dyn

        for state, won, opp_elo, prefix in (
            (w, 1.0, pre_l_slow, "w"),
            (l, 0.0, pre_w_slow, "l"),
        ):
            expected = exp_w_slow if won == 1.0 else 1.0 - exp_w_slow
            state.history.append(
                Observation(
                    date=now,
                    win=won,
                    expected=expected,
                    residual=won - expected,
                    opponent_elo=opp_elo,
                    surface=surface,
                    level_role=level_role,
                    spw=_finite(match_stats.get(f"{prefix}_spw")),
                    rpw=_finite(match_stats.get(f"{prefix}_rpw")),
                    hold=_finite(match_stats.get(f"{prefix}_hold")),
                    break_rate=_finite(match_stats.get(f"{prefix}_break")),
                    set_share=_finite(match_stats.get(f"{prefix}_set_share")),
                    game_share=_finite(match_stats.get(f"{prefix}_game_share")),
                )
            )
            state.matches += 1
            state.wins += int(won)
            state.surface_matches[surface] = state.surface_matches.get(surface, 0) + 1
            state.last_date = now

        w.latest_rank = _finite(winner_meta.get("rank"), w.latest_rank)
        w.latest_points = _finite(winner_meta.get("points"), w.latest_points)
        w.latest_age = _finite(winner_meta.get("age"), w.latest_age)
        w.latest_height = _finite(winner_meta.get("height"), w.latest_height)
        l.latest_rank = _finite(loser_meta.get("rank"), l.latest_rank)
        l.latest_points = _finite(loser_meta.get("points"), l.latest_points)
        l.latest_age = _finite(loser_meta.get("age"), l.latest_age)
        l.latest_height = _finite(loser_meta.get("height"), l.latest_height)

        key = tuple(sorted((winner_key, loser_key)))
        if winner_key == key[0]:
            self.h2h[key][0] += 1
        else:
            self.h2h[key][1] += 1

    def build(self, matches: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for _, r in matches.iterrows():
            winner_key, loser_key = str(r["winner_key"]), str(r["loser_key"])
            self.name_to_key[normalize_name(r["winner_name"])] = winner_key
            self.name_to_key[normalize_name(r["loser_name"])] = loser_key
            # Deterministic orientation independent of result.
            a_is_winner = winner_key < loser_key
            a_key, b_key = (winner_key, loser_key) if a_is_winner else (loser_key, winner_key)
            a_side, b_side = ("winner", "loser") if a_is_winner else ("loser", "winner")
            now = pd.Timestamp(r["proxy_date"])
            surface = str(r["surface"])

            def meta(side: str) -> dict[str, float | str]:
                return {
                    "rank": _finite(r.get(f"{side}_rank")),
                    "points": _finite(r.get(f"{side}_rank_points")),
                    "age": _finite(r.get(f"{side}_age")),
                    "height": _finite(r.get(f"{side}_ht")),
                    "entry": str(r.get(f"{side}_entry", "")),
                }

            a_meta, b_meta = meta(a_side), meta(b_side)
            feats = self._feature_row(
                a_key, b_key, now, surface, a_meta, b_meta,
                _finite(r.get("round_order"), 6.0), _finite(r.get("best_of"), 3.0),
                _finite(r.get("indoor_flag"), 0.0), _finite(r.get("indoor_known"), 0.0)
            )
            target = int(a_is_winner)
            if bool(r.get("is_target_challenger", False)):
                record = {
                    "date": pd.Timestamp(r["tourney_date"]),
                    "proxy_date": now,
                    "tourney_id": r.get("tourney_id", ""),
                    "tourney_name": r.get("tourney_name", ""),
                    "surface": surface,
                    "round": r.get("round", ""),
                    "player_a": r[f"{a_side}_name"],
                    "player_b": r[f"{b_side}_name"],
                    "player_a_key": a_key,
                    "player_b_key": b_key,
                    "target": target,
                    "winner_name": r["winner_name"],
                    "loser_name": r["loser_name"],
                    "winner_rank": r.get("winner_rank", np.nan),
                    "loser_rank": r.get("loser_rank", np.nan),
                    **feats,
                }
                rows.append(record)

            # Retirements/walkovers are not treated as clean evidence of latent
            # playing strength. This also keeps historical betting labels aligned
            # with books whose retirement settlement rules differ.
            if bool(r.get("is_clean_completed", True)):
                stats = derive_match_stats(r)
                self._update_pair(
                    winner_key, loser_key, now, surface, str(r.get("level_role", "unknown")),
                    meta("winner"), meta("loser"), stats
                )
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out = out.sort_values(["date", "proxy_date", "tourney_id", "round", "player_a_key"], kind="mergesort").reset_index(drop=True)
        return out


    def _resolve_live_key(self, name: object, player_id: object = None) -> str:
        if player_id is not None:
            pid = str(player_id)
            if pid not in {"", "nan", "None"} and pid in self.states:
                return pid
        norm = normalize_name(name)
        return self.name_to_key.get(norm, f"name:{norm}")

    def build_live(self, slate: pd.DataFrame) -> pd.DataFrame:
        """Create pre-match features for an unplayed Challenger slate.

        Expected columns: player1, player2, date, surface. Optional columns:
        player1_id/player2_id, rank1/rank2, points1/points2, age1/age2,
        height1/height2, entry1/entry2, round_order, best_of, tournament,
        start_time, odd1, odd2. No state is updated by this method.
        """
        rows = []
        for _, r in slate.iterrows():
            p1, p2 = str(r["player1"]), str(r["player2"])
            k1 = self._resolve_live_key(p1, r.get("player1_id"))
            k2 = self._resolve_live_key(p2, r.get("player2_id"))
            a_is_p1 = k1 < k2
            a_key, b_key = (k1, k2) if a_is_p1 else (k2, k1)
            a_name, b_name = (p1, p2) if a_is_p1 else (p2, p1)
            def live_meta(prefix: str) -> dict[str, float | str]:
                return {
                    "rank": _finite(r.get(f"rank{prefix}")),
                    "points": _finite(r.get(f"points{prefix}")),
                    "age": _finite(r.get(f"age{prefix}")),
                    "height": _finite(r.get(f"height{prefix}")),
                    "entry": str(r.get(f"entry{prefix}", "")),
                }
            m1, m2 = live_meta("1"), live_meta("2")
            a_meta, b_meta = (m1, m2) if a_is_p1 else (m2, m1)
            now = pd.Timestamp(r["date"]).normalize()
            surface = str(r.get("surface", "Unknown")).title()
            feats = self._feature_row(
                a_key, b_key, now, surface, a_meta, b_meta,
                _finite(r.get("round_order"), 6.0), _finite(r.get("best_of"), 3.0),
                _finite(r.get("indoor_flag"), 0.0), _finite(r.get("indoor_known"), 0.0)
            )
            rec = {
                "date": now,
                "proxy_date": pd.Timestamp(r.get("start_time", now)),
                "tourney_id": str(r.get("tourney_id", "LIVE")),
                "tourney_name": str(r.get("tournament", r.get("tourney_name", ""))),
                "surface": surface,
                "round": str(r.get("round", "")),
                "player_a": a_name, "player_b": b_name,
                "player_a_key": a_key, "player_b_key": b_key,
                "source_player1": p1, "source_player2": p2,
                **feats,
            }
            if "odd1" in r and "odd2" in r:
                rec["odd_a"] = float(r["odd1"] if a_is_p1 else r["odd2"] )
                rec["odd_b"] = float(r["odd2"] if a_is_p1 else r["odd1"] )
            rows.append(rec)
        return pd.DataFrame(rows)


def mirror_features(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for col in DIFF_FEATURES:
        if col in out:
            out[col] = -out[col]
    return out


def augment_symmetric(X: pd.DataFrame, y: pd.Series | np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    yarr = np.asarray(y, dtype=int)
    mirrored = mirror_features(X)
    X2 = pd.concat([X, mirrored], ignore_index=True)
    y2 = np.concatenate([yarr, 1 - yarr])
    return X2, y2


def reliability_score(frame: pd.DataFrame, prediction_dispersion: np.ndarray | pd.Series) -> np.ndarray:
    """Data-quality + ensemble-agreement score in [0,1]."""
    exp = np.minimum(frame["min_experience"].to_numpy(float), 24.0) / 24.0
    surf = np.minimum(frame["min_surface_experience"].to_numpy(float), 12.0) / 12.0
    stats = frame["stat_coverage"].fillna(0.0).to_numpy(float)
    dispersion = np.asarray(prediction_dispersion, dtype=float)
    agreement = np.clip(1.0 - dispersion / 0.18, 0.0, 1.0)
    # Stats are deliberately the least-weighted component because coverage varies
    # by source and older Challenger events.
    return np.clip(0.32 * exp + 0.23 * surf + 0.15 * stats + 0.30 * agreement, 0.0, 1.0)
