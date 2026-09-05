#!/usr/bin/env python
"""Apply the verified Rev5 same-timestamp replay semantics, then run the frozen Rev5 fundamental script.

This is intentionally a narrow execution patch: model formulas, feature formulas,
configuration and policy are unchanged. Only ChallengerFeatureBuilder.build is
replaced with the Rev5 batching implementation that snapshots every row sharing
the same proxy timestamp before any result at that timestamp updates state.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from challenger_c1.data import derive_match_stats, normalize_name
from challenger_c1.features import ChallengerFeatureBuilder, _finite


def rev5_build(self: ChallengerFeatureBuilder, matches: pd.DataFrame) -> pd.DataFrame:
    """Verified Rev5 leakage-safe history replay."""
    rows: list[dict] = []
    if matches.empty:
        return pd.DataFrame(rows)

    work = matches.copy()
    work["proxy_date"] = pd.to_datetime(work["proxy_date"], errors="coerce")
    work = work.dropna(subset=["proxy_date"]).sort_values(
        ["proxy_date", "tourney_id", "round_order", "match_num", "winner_key", "loser_key"],
        kind="mergesort",
    )

    for proxy_time, group in work.groupby("proxy_date", sort=True, dropna=False):
        now = pd.Timestamp(proxy_time)
        pending_updates: list[tuple] = []

        # Phase 1: freeze every prediction at this information timestamp.
        for _, r in group.iterrows():
            winner_key, loser_key = str(r["winner_key"]), str(r["loser_key"])
            self.name_to_key[normalize_name(r["winner_name"])] = winner_key
            self.name_to_key[normalize_name(r["loser_name"])] = loser_key

            a_is_winner = winner_key < loser_key
            a_key, b_key = (winner_key, loser_key) if a_is_winner else (loser_key, winner_key)
            a_side, b_side = ("winner", "loser") if a_is_winner else ("loser", "winner")
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
                _finite(r.get("indoor_flag"), 0.0), _finite(r.get("indoor_known"), 0.0),
            )
            target = int(a_is_winner)
            if bool(r.get("is_target_challenger", False)):
                rows.append({
                    "date": now.normalize(),
                    "event_start_date": pd.Timestamp(r["tourney_date"]).normalize(),
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
                })

            if bool(r.get("is_clean_completed", True)):
                pending_updates.append((
                    winner_key,
                    loser_key,
                    now,
                    surface,
                    str(r.get("level_role", "unknown")),
                    meta("winner"),
                    meta("loser"),
                    derive_match_stats(r),
                ))

        # Phase 2: expose completed results only to later timestamps.
        for update in pending_updates:
            self._update_pair(*update)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["date", "proxy_date", "tourney_id", "round", "player_a_key"],
        kind="mergesort",
    ).reset_index(drop=True)


ChallengerFeatureBuilder.build = rev5_build

if __name__ == "__main__":
    runpy.run_path(str(ROOT / "scripts" / "rev5_final_fundamental.py"), run_name="__main__")
