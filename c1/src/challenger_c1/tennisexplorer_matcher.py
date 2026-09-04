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


def _tokens(value: object) -> list[str]:
    return [x for x in normalize_name(value).split() if x]


def _full_anchor(value: object) -> str:
    t = _tokens(value)
    return t[-1] if t else ""


def _te_parts(value: object) -> tuple[list[str], list[str]]:
    t = _tokens(value)
    if not t:
        return [], []
    i = len(t)
    while i > 0 and len(t[i - 1]) <= 2:
        i -= 1
    surnames = t[:i] if i > 0 else t[:-1]
    initials = t[i:] if i < len(t) else []
    if not surnames and t:
        surnames = [t[0]]
    return surnames, initials


def _te_anchor(value: object) -> str:
    s, _ = _te_parts(value)
    return s[-1] if s else ""


def _subseq_ratio(short_tokens: list[str], full_tokens: list[str]) -> float:
    if not short_tokens:
        return 0.0
    exact = sum(t in set(full_tokens) for t in short_tokens) / len(short_tokens)
    fuzzy = [max([SequenceMatcher(None, s, f).ratio() for f in full_tokens] or [0.0]) for s in short_tokens]
    return max(exact, float(np.mean(fuzzy)))


@lru_cache(maxsize=300000)
def player_similarity(full_name: str, te_name: str) -> float:
    nf, nt = normalize_name(full_name), normalize_name(te_name)
    if not nf or not nt:
        return 0.0
    if nf == nt:
        return 1.0
    full = _tokens(full_name)
    surn, initials = _te_parts(te_name)
    if initials:
        surname_score = _subseq_ratio(surn, full)
        full_initials = [x[0] for x in full if x]
        init_chars = [x[0] for x in initials if x]
        hits = 0
        pos = 0
        for ch in init_chars:
            while pos < len(full_initials) and full_initials[pos] != ch:
                pos += 1
            if pos < len(full_initials):
                hits += 1
                pos += 1
        init_score = hits / len(init_chars) if init_chars else 0.0
        return 0.84 * surname_score + 0.16 * init_score
    return 0.6 * SequenceMatcher(None, nf, nt).ratio() + 0.4 * _subseq_ratio(_tokens(te_name), full)


@lru_cache(maxsize=20000)
def tournament_similarity(a: str, b: str) -> float:
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
    """Attach abbreviated TennisExplorer names to full-name C1 feature rows.

    Matching is result-blind. It uses player-name identity, event-week timing and
    tournament similarity only. Ambiguous/low-score rows are rejected.
    """
    f = features.copy().reset_index(drop=False).rename(columns={"index": "_feature_index"})
    o = odds.copy().reset_index(drop=False).rename(columns={"index": "_odds_index"})
    if "valid_price_pair" in o:
        o = o[pd.to_numeric(o["valid_price_pair"], errors="coerce").fillna(0).astype(int).eq(1)].copy()
    o["odd1"] = pd.to_numeric(o["odd1"], errors="coerce")
    o["odd2"] = pd.to_numeric(o["odd2"], errors="coerce")
    o = o[o["odd1"].gt(1) & o["odd2"].gt(1)].copy()
    o["_date"] = pd.to_datetime(o["date"], errors="coerce").dt.normalize()
    o = o.dropna(subset=["_date"]).copy()
    f["_date"] = pd.to_datetime(f["proxy_date"] if "proxy_date" in f else f["date"], errors="coerce").dt.normalize()
    o["_a1"] = o["player1"].map(_te_anchor)
    o["_a2"] = o["player2"].map(_te_anchor)
    o["_pair_anchor"] = ["||".join(sorted((a, b))) for a, b in zip(o["_a1"], o["_a2"])]
    by_pair = {k: g for k, g in o.groupby("_pair_anchor", sort=False)}
    by_player: dict[str, list[int]] = defaultdict(list)
    for idx, a1, a2 in zip(o.index, o["_a1"], o["_a2"]):
        if a1:
            by_player[a1].append(idx)
        if a2 and a2 != a1:
            by_player[a2].append(idx)

    selected: list[dict] = []
    audit: list[dict] = []
    for _, row in f.iterrows():
        aa, ab = _full_anchor(row["player_a"]), _full_anchor(row["player_b"])
        pair = "||".join(sorted((aa, ab)))
        c = by_pair.get(pair)
        source = "pair_anchor"
        if c is not None and not c.empty:
            c = c[(c["_date"] - row["_date"]).dt.days.abs().le(max_date_gap_days)].copy()
        else:
            c = pd.DataFrame()
        if c.empty:
            ids = set(by_player.get(aa, [])) | set(by_player.get(ab, []))
            c = o.loc[sorted(ids)].copy() if ids else pd.DataFrame()
            source = "single_anchor"
            if not c.empty:
                c = c[(c["_date"] - row["_date"]).dt.days.abs().le(max_date_gap_days)].copy()
        if c.empty:
            audit.append({"feature_index": int(row["_feature_index"]), "status": "no_candidate"})
            continue

        scored = []
        for _, cr in c.iterrows():
            direct = 0.5 * (player_similarity(str(row["player_a"]), str(cr["player1"])) + player_similarity(str(row["player_b"]), str(cr["player2"])))
            swapped = 0.5 * (player_similarity(str(row["player_a"]), str(cr["player2"])) + player_similarity(str(row["player_b"]), str(cr["player1"])))
            if direct >= swapped:
                ps, orient = direct, "direct"
            else:
                ps, orient = swapped, "swapped"
            ts = tournament_similarity(str(row.get("tourney_name", "")), str(cr.get("tournament", "")))
            dg = abs((cr["_date"] - row["_date"]).days)
            total = 0.84 * ps + 0.12 * ts + 0.04 * max(0.0, 1.0 - dg / (max_date_gap_days + 1.0))
            scored.append((total, ps, ts, dg, orient, cr))
        scored.sort(key=lambda x: (x[0], x[1], x[2], -x[3]), reverse=True)
        best = scored[0]
        second = scored[1][0] if len(scored) > 1 else -1.0
        total, ps, ts, dg, orient, br = best
        margin = total - second if second >= 0 else 1.0
        if ps < min_pair_score:
            audit.append({"feature_index": int(row["_feature_index"]), "status": "low_name_score", "pair_score": ps})
            continue
        if len(scored) > 1 and margin < ambiguity_margin and scored[1][1] >= min_pair_score:
            audit.append({"feature_index": int(row["_feature_index"]), "status": "ambiguous", "pair_score": ps, "margin": margin})
            continue

        odd_a, odd_b = (float(br["odd1"]), float(br["odd2"])) if orient == "direct" else (float(br["odd2"]), float(br["odd1"]))
        rec = row.to_dict()
        rec.update({
            "odd_a": odd_a, "odd_b": odd_b, "odds_date": br["_date"],
            "odds_source": br.get("source", "TennisExplorer"), "odds_date_gap": int(dg),
            "odds_tournament_similarity": float(ts), "odds_name_pair_score": float(ps),
            "odds_match_score": float(total), "odds_match_margin": float(margin),
            "odds_orientation": orient, "odds_index": int(br["_odds_index"]),
        })
        selected.append(rec)
        audit.append({"feature_index": int(row["_feature_index"]), "status": "matched", "pair_score": ps, "margin": margin, "date_gap": int(dg), "tourney_sim": ts, "source": source})

    matched = pd.DataFrame(selected)
    if not matched.empty:
        ia = 1.0 / matched["odd_a"].to_numpy(float)
        ib = 1.0 / matched["odd_b"].to_numpy(float)
        matched["market_probability_a"] = ia / (ia + ib)
        matched["market_logit_a"] = logit(np.clip(matched["market_probability_a"], 1e-5, 1 - 1e-5))
    return matched.drop(columns=[c for c in ["_date"] if c in matched], errors="ignore"), pd.DataFrame(audit)
