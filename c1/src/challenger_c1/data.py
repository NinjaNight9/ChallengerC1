from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

ROUND_ORDER = {
    "Q1": 0,
    "Q2": 1,
    "Q3": 2,
    "Q4": 3,
    "R128": 4,
    "R64": 5,
    "R32": 6,
    "R16": 7,
    "QF": 8,
    "SF": 9,
    "F": 10,
    "RR": 7,
    "BR": 11,
}


def normalize_name(value: object) -> str:
    value = unicodedata.normalize("NFKD", str(value or ""))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.casefold().replace("’", "'")
    value = re.sub(r"\b(jr|sr)\.?$", "", value).strip()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def _to_datetime_yyyymmdd(series: pd.Series) -> pd.Series:
    raw = pd.to_numeric(series, errors="coerce").astype("Int64").astype(str)
    return pd.to_datetime(raw, format="%Y%m%d", errors="coerce")


def _round_offset(round_code: object) -> int:
    # Sackmann supplies tournament start date, not exact match date.  The offset
    # is only an ordering proxy so later rounds can see prior-round results.
    code = str(round_code or "").upper().strip()
    order = ROUND_ORDER.get(code, 6)
    if code.startswith("Q") and code not in {"QF"}:
        return max(0, order)
    # Main-draw offsets are compressed to plausible tournament days.
    mapping = {"R128": 0, "R64": 1, "R32": 2, "R16": 3, "QF": 4, "SF": 5, "F": 6, "RR": 2, "BR": 6}
    return mapping.get(code, max(0, order - 4))


def _clean_surface(surface: object) -> str:
    s = str(surface or "Unknown").strip().title()
    aliases = {"Hard": "Hard", "Clay": "Clay", "Grass": "Grass", "Carpet": "Carpet"}
    return aliases.get(s, "Unknown")


def _source_level_role(source_kind: str, tourney_level: object) -> str:
    lvl = str(tourney_level or "").upper()
    if source_kind == "challenger":
        return "challenger"
    if source_kind in {"qual_chall", "tour_qual"}:
        return "challenger" if lvl == "C" else "tour_qual"
    if source_kind in {"futures", "itf"}:
        return "futures"
    if lvl == "C":
        return "challenger"
    return "main"


def _safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_sackmann_file(path: str | Path, source_kind: str) -> pd.DataFrame:
    """Load one Jeff Sackmann-format ATP results file into a canonical schema.

    source_kind is one of: main, qual_chall, futures.
    The canonical orientation is winner/loser here; it is converted to a
    deterministic A/B orientation during feature building, before the label is
    exposed to the model.
    """
    path = Path(path)
    raw = pd.read_csv(path, low_memory=False)
    required = {"tourney_date", "winner_name", "loser_name", "surface", "round"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{path.name}: missing required columns {sorted(missing)}")

    out = pd.DataFrame(index=raw.index)
    out["source_file"] = path.name
    out["source_kind"] = source_kind
    out["tourney_id"] = raw.get("tourney_id", "").astype(str)
    out["tourney_name"] = raw.get("tourney_name", "").astype(str)
    out["surface"] = raw["surface"].map(_clean_surface)
    indoor_raw = raw.get("indoor", pd.Series(index=raw.index, dtype=object))
    indoor_text = indoor_raw.fillna("").astype(str).str.casefold().str.strip()
    out["indoor_known"] = indoor_text.isin(["yes", "no", "true", "false", "1", "0", "indoor", "outdoor"]).astype(float)
    out["indoor_flag"] = indoor_text.isin(["yes", "true", "1", "indoor"]).astype(float)
    out["tourney_level"] = raw.get("tourney_level", "").astype(str).str.upper()
    out["level_role"] = [
        _source_level_role(source_kind, x) for x in out["tourney_level"]
    ]
    out["tourney_date"] = _to_datetime_yyyymmdd(raw["tourney_date"])
    out["round"] = raw["round"].astype(str).str.upper()
    out["round_order"] = out["round"].map(lambda x: ROUND_ORDER.get(x, 6)).astype(int)
    offsets = out["round"].map(_round_offset)
    out["proxy_date"] = out["tourney_date"] + pd.to_timedelta(offsets, unit="D")
    out["match_num"] = _safe_num(raw.get("match_num", pd.Series(index=raw.index, dtype=float))).fillna(0)
    out["best_of"] = _safe_num(raw.get("best_of", pd.Series(index=raw.index, dtype=float))).fillna(3)
    out["minutes"] = _safe_num(raw.get("minutes", pd.Series(index=raw.index, dtype=float)))

    for side in ("winner", "loser"):
        out[f"{side}_id"] = raw.get(f"{side}_id", pd.Series(index=raw.index, dtype=object)).astype(str)
        out[f"{side}_name"] = raw[f"{side}_name"].astype(str)
        out[f"{side}_key"] = out[f"{side}_id"].where(
            out[f"{side}_id"].notna() & ~out[f"{side}_id"].isin(["nan", "None", ""]),
            "name:" + out[f"{side}_name"].map(normalize_name),
        )
        for col in ("age", "ht", "rank", "rank_points"):
            source = f"{side}_{col}"
            out[source] = _safe_num(raw.get(source, pd.Series(index=raw.index, dtype=float)))
        out[f"{side}_entry"] = raw.get(f"{side}_entry", pd.Series(index=raw.index, dtype=object)).fillna("").astype(str).str.upper()

    # Canonical raw match statistics. Missing fields stay NaN and become an
    # uncertainty signal instead of being silently treated as average.
    stat_cols = [
        "ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced"
    ]
    for prefix in ("w", "l"):
        for c in stat_cols:
            out[f"{prefix}_{c}"] = _safe_num(raw.get(f"{prefix}_{c}", pd.Series(index=raw.index, dtype=float)))

    out["score"] = raw.get("score", pd.Series(index=raw.index, dtype=object)).fillna("").astype(str)
    score_upper = out["score"].str.upper()
    out["is_retirement"] = score_upper.str.contains(r"\bRET\b|\bDEF\b", regex=True)
    out["is_walkover"] = score_upper.str.contains(r"W/O|WALKOVER", regex=True)
    out["is_clean_completed"] = ~(out["is_retirement"] | out["is_walkover"])
    out["is_target_challenger"] = (out["tourney_level"].eq("C") | out["level_role"].eq("challenger")) & out["is_clean_completed"]
    out = out.dropna(subset=["tourney_date", "proxy_date", "winner_name", "loser_name"])
    out = out[out["winner_key"] != out["loser_key"]].copy()
    # Stable ordering: event week, approximate round, then Sackmann match_num.
    return out.sort_values(
        ["tourney_date", "round_order", "match_num", "winner_key", "loser_key"],
        kind="mergesort",
    ).reset_index(drop=True)


def load_sackmann_directory(
    root: str | Path,
    years: Iterable[int],
    include_main: bool = True,
    include_qual_chall: bool = True,
    include_futures: bool = True,
) -> pd.DataFrame:
    root = Path(root)
    frames: list[pd.DataFrame] = []
    patterns = []
    if include_main:
        patterns.append(("main", "atp_matches_{year}.csv"))
    if include_qual_chall:
        patterns.append(("qual_chall", "atp_matches_qual_chall_{year}.csv"))
    if include_futures:
        patterns.append(("futures", "atp_matches_futures_{year}.csv"))
    for year in years:
        for source_kind, pattern in patterns:
            path = root / pattern.format(year=year)
            if path.exists():
                frames.append(load_sackmann_file(path, source_kind))
    if not frames:
        expected = [p.format(year=list(years)[0] if list(years) else 2025) for _, p in patterns]
        raise FileNotFoundError(
            f"No Sackmann files found in {root}. Expected names like {expected}."
        )
    out = pd.concat(frames, ignore_index=True)
    # Deduplicate exact match identities when a source overlaps another source.
    identity = ["tourney_id", "tourney_date", "round", "winner_key", "loser_key"]
    out = out.drop_duplicates(identity, keep="first")
    return out.sort_values(
        ["tourney_date", "round_order", "match_num", "winner_key", "loser_key"],
        kind="mergesort",
    ).reset_index(drop=True)



def load_tennismylife_directory(root: str | Path, years: Iterable[int]) -> pd.DataFrame:
    """Load TennisMyLife-format ATP Tour, Challenger and qualifying files.

    Expected layout mirrors the site's download names:
      {year}.csv
      {year}_challenger.csv
      atp_quali/{year}_atp_quali.csv
    All three use a Sackmann-compatible column family, with an additional
    indoor field when available.
    """
    root = Path(root)
    frames: list[pd.DataFrame] = []
    for year in years:
        specs = [
            (root / f"{year}.csv", "main"),
            (root / f"{year}_challenger.csv", "challenger"),
            (root / "atp_quali" / f"{year}_atp_quali.csv", "tour_qual"),
        ]
        for path, kind in specs:
            if path.exists():
                frames.append(load_sackmann_file(path, kind))
    if not frames:
        raise FileNotFoundError(f"No TennisMyLife yearly files found under {root}")
    out = pd.concat(frames, ignore_index=True)
    identity = ["tourney_name", "tourney_date", "round", "winner_name", "loser_name"]
    out = out.drop_duplicates(identity, keep="first")
    return out.sort_values(
        ["tourney_date", "round_order", "match_num", "winner_key", "loser_key"],
        kind="mergesort",
    ).reset_index(drop=True)

def parse_score_share(score: object) -> tuple[float, float]:
    """Return winner set share and game share from a standard tennis score.

    Retired matches are retained if completed games exist, but match tie-break
    bracket tokens are ignored. Walkovers produce NaN.
    """
    text = str(score or "").upper().strip()
    if not text or "W/O" in text or "WALKOVER" in text:
        return np.nan, np.nan
    tokens = text.replace("RET", "").replace("DEF", "").split()
    sets_w = sets_l = games_w = games_l = 0
    for tok in tokens:
        if tok.startswith("["):
            continue
        # Remove tiebreak detail: 7-6(5) -> 7-6
        tok = re.sub(r"\([^)]*\)", "", tok)
        m = re.match(r"^(\d+)-(\d+)$", tok)
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        games_w += a
        games_l += b
        if a > b:
            sets_w += 1
        elif b > a:
            sets_l += 1
    set_total = sets_w + sets_l
    game_total = games_w + games_l
    return (
        sets_w / set_total if set_total else np.nan,
        games_w / game_total if game_total else np.nan,
    )


def derive_match_stats(row: Mapping[str, object]) -> dict[str, float]:
    """Derive per-player serve/return/hold/break rates for one completed match."""
    def f(key: str) -> float:
        try:
            x = float(row.get(key, np.nan))
            return x if np.isfinite(x) else np.nan
        except (TypeError, ValueError):
            return np.nan

    w_svpt, l_svpt = f("w_svpt"), f("l_svpt")
    w_spw = (f("w_1stWon") + f("w_2ndWon")) / w_svpt if w_svpt and np.isfinite(w_svpt) else np.nan
    l_spw = (f("l_1stWon") + f("l_2ndWon")) / l_svpt if l_svpt and np.isfinite(l_svpt) else np.nan
    w_rpw = 1.0 - l_spw if np.isfinite(l_spw) else np.nan
    l_rpw = 1.0 - w_spw if np.isfinite(w_spw) else np.nan

    w_svgms, l_svgms = f("w_SvGms"), f("l_SvGms")
    w_broken = f("w_bpFaced") - f("w_bpSaved")
    l_broken = f("l_bpFaced") - f("l_bpSaved")
    w_hold = 1.0 - w_broken / w_svgms if w_svgms and np.isfinite(w_broken) else np.nan
    l_hold = 1.0 - l_broken / l_svgms if l_svgms and np.isfinite(l_broken) else np.nan
    w_break = l_broken / l_svgms if l_svgms and np.isfinite(l_broken) else np.nan
    l_break = w_broken / w_svgms if w_svgms and np.isfinite(w_broken) else np.nan
    set_share, game_share = parse_score_share(row.get("score", ""))
    return {
        "w_spw": w_spw,
        "l_spw": l_spw,
        "w_rpw": w_rpw,
        "l_rpw": l_rpw,
        "w_hold": w_hold,
        "l_hold": l_hold,
        "w_break": w_break,
        "l_break": l_break,
        "w_set_share": set_share,
        "l_set_share": 1.0 - set_share if np.isfinite(set_share) else np.nan,
        "w_game_share": game_share,
        "l_game_share": 1.0 - game_share if np.isfinite(game_share) else np.nan,
    }


def canonicalize_generic_odds(
    path: str | Path,
    column_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Canonicalize a generic historical/live odds CSV.

    Required canonical fields after mapping:
      date, player1, player2, odd1, odd2
    Optional: tournament, surface, start_time, source.

    column_map maps canonical_name -> source_column_name.
    """
    raw = pd.read_csv(path, low_memory=False)
    column_map = dict(column_map or {})
    canonical = {}
    for col in ["date", "player1", "player2", "odd1", "odd2", "tournament", "surface", "start_time", "source"]:
        src = column_map.get(col, col)
        if src in raw.columns:
            canonical[col] = raw[src]
    missing = {"date", "player1", "player2", "odd1", "odd2"} - set(canonical)
    if missing:
        raise ValueError(f"Odds file missing canonical fields {sorted(missing)}. Supply column_map.")
    out = pd.DataFrame(canonical)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["player1_key_name"] = out["player1"].map(normalize_name)
    out["player2_key_name"] = out["player2"].map(normalize_name)
    out["odd1"] = pd.to_numeric(out["odd1"], errors="coerce")
    out["odd2"] = pd.to_numeric(out["odd2"], errors="coerce")
    out = out.dropna(subset=["date", "odd1", "odd2"])
    out = out[(out["odd1"] > 1.0) & (out["odd2"] > 1.0)].copy()
    if "surface" in out:
        out["surface"] = out["surface"].map(_clean_surface)
    return out.reset_index(drop=True)
