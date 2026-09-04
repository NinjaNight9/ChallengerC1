from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.special import logit


def normalize_name(value: object) -> str:
    value = unicodedata.normalize("NFKD", str(value or ""))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.casefold().replace("’", "'")
    value = re.sub(r"\b(jr|sr)\.?$", "", value).strip()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def no_vig_probability(odd_a, odd_b):
    ia = 1.0 / np.asarray(odd_a, dtype=float)
    ib = 1.0 / np.asarray(odd_b, dtype=float)
    return ia / (ia + ib)


def _tokens(value: object) -> list[str]:
    return [t for t in normalize_name(value).split() if t]


def _split_te_name(value: object) -> tuple[list[str], list[str]]:
    """Split TennisExplorer's usual `Surname I.` representation."""
    toks = _tokens(value)
    if not toks:
        return [], []
    i = len(toks)
    while i > 0 and len(toks[i - 1]) <= 2:
        i -= 1
    surnames = toks[:i] if i > 0 else toks[:-1]
    initials = toks[i:] if i < len(toks) else []
    if not surnames and toks:
        surnames = [toks[0]]
    return surnames, initials


def _anchor_full(value: object) -> str:
    toks = _tokens(value)
    return toks[-1] if toks else ""


def _anchor_te(value: object) -> str:
    surnames, _ = _split_te_name(value)
    return surnames[-1] if surnames else ""


def _subseq_ratio(short_tokens: list[str], full_tokens: list[str]) -> float:
    if not short_tokens:
        return 0.0
    full_set = set(full_tokens)
    exact = sum(t in full_set for t in short_tokens) / len(short_tokens)
    fuzzy = [
        max([SequenceMatcher(None, s, f).ratio() for f in full_tokens] or [0.0])
        for s in short_tokens
    ]
    return max(exact, float(np.mean(fuzzy)))


@lru_cache(maxsize=250000)
def player_similarity(full_name: object, te_name: object) -> float:
    full_norm, te_norm = normalize_name(full_name), normalize_name(te_name)
    if not full_norm or not te_norm:
        return 0.0
    if full_norm == te_norm:
        return 1.0
    full_tokens = _tokens(full_name)
    surnames, initials = _split_te_name(te_name)
    if initials:
        surname_score = _subseq_ratio(surnames, full_tokens)
        full_initials = [x[0] for x in full_tokens if x]
        init_chars = [x[0] for x in initials if x]
        pos = hits = 0
        for ch in init_chars:
            while pos < len(full_initials) and full_initials[pos] != ch:
                pos += 1
            if pos < len(full_initials):
                hits += 1
                pos += 1
        init_score = hits / len(init_chars) if init_chars else 0.0
        return 0.82 * surname_score + 0.18 * init_score
    seq = SequenceMatcher(None, full_norm, te_norm).ratio()
    token_score = _subseq_ratio(_tokens(te_name), full_tokens)
    return 0.60 * seq + 0.40 * token_score


def _pair_orientation_score(a: object, b: object, p1: object, p2: object):
    direct = (player_similarity(a, p1) + player_similarity(b, p2)) / 2.0
    swapped = (player_similarity(a, p2) + player_similarity(b, p1)) / 2.0
    return (direct, "direct", swapped) if direct >= swapped else (swapped, "swapped", direct)


@lru_cache(maxsize=10000)
def _tournament_similarity(a: object, b: object) -> float:
    aa = normalize_name(a).replace(" challenger", "").strip()
    bb = normalize_name(b).replace(" challenger", "").strip()
    if not aa or not bb:
        return 0.0
    if aa == bb:
        return 1.0
    return SequenceMatcher(None, aa, bb).ratio()


def attach_tennisexplorer_odds(
    features: pd.DataFrame,
    odds: pd.DataFrame,
    max_date_gap_days: int = 8,
    min_pair_score: float = 0.78,
    ambiguity_margin: float = 0.035,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach historical TennisExplorer odds without using outcomes.

    Full-name fundamental rows are joined to TennisExplorer's abbreviated names
    by surname anchors and initials, with tournament/date used only as tie
    breakers. Ambiguous or low-confidence joins are dropped rather than guessed.
    """
    f = features.copy().reset_index(drop=False).rename(columns={"index": "_feature_index"})
    o = odds.copy().reset_index(drop=False).rename(columns={"index": "_odds_index"})

    if "valid_price_pair" in o.columns:
        valid = pd.to_numeric(o["valid_price_pair"], errors="coerce").fillna(0).astype(int).eq(1)
        o = o[valid].copy()
    o["odd1"] = pd.to_numeric(o["odd1"], errors="coerce")
    o["odd2"] = pd.to_numeric(o["odd2"], errors="coerce")
    o = o[o["odd1"].gt(1.0) & o["odd2"].gt(1.0)].copy()
    o["_date"] = pd.to_datetime(o["date"], errors="coerce").dt.normalize()
    o = o.dropna(subset=["_date"]).copy()

    fdate = f["proxy_date"] if "proxy_date" in f.columns else f["date"]
    f["_date"] = pd.to_datetime(fdate, errors="coerce").dt.normalize()

    o["_anchor1"] = [_anchor_te(a) for a in o["player1"]]
    o["_anchor2"] = [_anchor_te(b) for b in o["player2"]]
    o["_anchor"] = ["||".join(sorted((a, b))) for a, b in zip(o["_anchor1"], o["_anchor2"])]
    by_anchor = {k: g for k, g in o.groupby("_anchor", sort=False)}
    by_player: dict[str, list[int]] = defaultdict(list)
    by_date: dict[pd.Timestamp, list[int]] = defaultdict(list)
    for idx, day, a1, a2 in zip(o.index, o["_date"], o["_anchor1"], o["_anchor2"]):
        by_date[day].append(idx)
        if a1:
            by_player[a1].append(idx)
        if a2 and a2 != a1:
            by_player[a2].append(idx)

    selected: list[dict] = []
    audit: list[dict] = []
    for _, row in f.iterrows():
        a_anchor = _anchor_full(row["player_a"])
        b_anchor = _anchor_full(row["player_b"])
        pair_anchor = "||".join(sorted((a_anchor, b_anchor)))
        candidates = by_anchor.get(pair_anchor)
        candidate_source = "pair_anchor"
        if candidates is not None and not candidates.empty:
            c = candidates.copy()
            c["date_gap"] = (c["_date"] - row["_date"]).dt.days.abs()
            c = c[c["date_gap"] <= max_date_gap_days].copy()
        else:
            c = pd.DataFrame()

        # Compound-surname disagreement can break the exact pair anchor. Reuse
        # the agreeing player's surname before considering all matches by date.
        if c.empty:
            indices = set(by_player.get(a_anchor, [])) | set(by_player.get(b_anchor, []))
            c = o.loc[sorted(indices)].copy() if indices else pd.DataFrame()
            candidate_source = "single_anchor"
            if not c.empty:
                c["date_gap"] = (c["_date"] - row["_date"]).dt.days.abs()
                c = c[c["date_gap"] <= max_date_gap_days].copy()

        if c.empty:
            indices: list[int] = []
            if pd.notna(row["_date"]):
                for offset in range(-max_date_gap_days, max_date_gap_days + 1):
                    indices.extend(by_date.get(row["_date"] + pd.Timedelta(days=offset), []))
            c = o.loc[sorted(set(indices))].copy() if indices else pd.DataFrame()
            candidate_source = "date_fallback"
            if not c.empty:
                c["date_gap"] = (c["_date"] - row["_date"]).dt.days.abs()
                if "tournament" in c.columns:
                    ts = c["tournament"].map(
                        lambda x: _tournament_similarity(str(row.get("tourney_name", "")), str(x))
                    )
                    keep = ts >= 0.45
                    if keep.any():
                        c = c[keep].copy()

        if c.empty:
            audit.append({
                "feature_index": int(row["_feature_index"]), "status": "no_candidate",
                "player_a": row["player_a"], "player_b": row["player_b"],
            })
            continue

        scored = []
        for _, cand in c.iterrows():
            pair_score, orientation, opposite = _pair_orientation_score(
                row["player_a"], row["player_b"], cand["player1"], cand["player2"]
            )
            tourney_sim = _tournament_similarity(
                str(row.get("tourney_name", "")), str(cand.get("tournament", ""))
            )
            date_gap = float(cand["date_gap"])
            total = (
                0.82 * pair_score
                + 0.13 * tourney_sim
                + 0.05 * max(0.0, 1.0 - date_gap / (max_date_gap_days + 1.0))
            )
            scored.append((total, pair_score, tourney_sim, date_gap, orientation, opposite, cand))
        scored.sort(key=lambda x: (x[0], x[1], x[2], -x[3]), reverse=True)
        best = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else -1.0
        total, pair_score, tourney_sim, date_gap, orientation, opposite, best_row = best
        margin = total - second_score if second_score >= 0 else 1.0

        if pair_score < min_pair_score:
            audit.append({
                "feature_index": int(row["_feature_index"]), "status": "low_name_score",
                "pair_score": pair_score, "total_score": total, "candidate_count": len(scored),
            })
            continue
        if len(scored) > 1 and margin < ambiguity_margin and scored[1][1] >= min_pair_score:
            audit.append({
                "feature_index": int(row["_feature_index"]), "status": "ambiguous",
                "pair_score": pair_score, "total_score": total, "margin": margin,
                "candidate_count": len(scored),
            })
            continue

        if orientation == "direct":
            odd_a, odd_b = float(best_row["odd1"]), float(best_row["odd2"])
        else:
            odd_a, odd_b = float(best_row["odd2"]), float(best_row["odd1"])

        rec = row.to_dict()
        rec.update({
            "odd_a": odd_a, "odd_b": odd_b,
            "odds_date": best_row["_date"],
            "odds_source": best_row.get("source", "TennisExplorer"),
            "odds_date_gap": int(date_gap),
            "odds_tournament_similarity": float(tourney_sim),
            "odds_name_pair_score": float(pair_score),
            "odds_match_score": float(total),
            "odds_match_margin": float(margin),
            "odds_orientation": orientation,
            "odds_index": int(best_row["_odds_index"]),
        })
        selected.append(rec)
        audit.append({
            "feature_index": int(row["_feature_index"]), "status": "matched",
            "pair_score": pair_score, "total_score": total, "margin": margin,
            "orientation": orientation, "candidate_count": len(scored),
            "candidate_source": candidate_source, "odds_index": int(best_row["_odds_index"]),
            "date_gap": int(date_gap), "tourney_sim": tourney_sim,
        })

    matched = pd.DataFrame(selected)
    if not matched.empty:
        matched["market_probability_a"] = no_vig_probability(matched["odd_a"], matched["odd_b"])
        matched["market_logit_a"] = logit(np.clip(matched["market_probability_a"], 1e-5, 1 - 1e-5))
    return matched.drop(columns=[c for c in ["_date"] if c in matched], errors="ignore"), pd.DataFrame(audit)
