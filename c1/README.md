# ChallengerC1 — men's ATP Challenger forecasting/value system

ChallengerC1 is a leakage-safe, multi-surface men's Challenger research/deployment package. It is intentionally **not** a relaxed copy of Model71. It separates four jobs:

1. **C1-FUND** — estimate true match win probability from tennis information only.
2. **C1-VALUE** — start from the no-vig market and learn when Challenger-specific evidence justifies moving away from it.
3. **C1-CORE10** — high-quality low-volume policy capped at 10% of a slate.
4. **C1-VOLUME20** — the main volume policy, capped at 20% of a slate but never forced to fill the quota.

## Core design

- Targets only men's ATP Challenger matches.
- Strength state can be updated from ATP main draw, ATP qualifying, Challenger and Futures/ITF results.
- Separate slow Elo, fast Elo and surface Elo.
- Surface Elo is shrinkage-blended toward overall Elo when a player has little history on that surface.
- Recent form is opponent-adjusted via performance residuals against pre-match Elo expectations.
- Serve/return, hold/break, game share and set share are recency-weighted when statistics exist.
- Missing statistics reduce reliability rather than being interpreted as average evidence.
- Ranking and ranking points are features, not the truth.
- Deterministic player orientation plus mirrored training/prediction enforces approximate `P(A>B)=1-P(B>A)` symmetry.
- Every feature is snapshotted before the match outcome updates state.

## Historical horizon

The research workflow compares 365-day and 730-day training windows. Older matches can still initialize player state; recent form features use 35/90/180-day half-lives. The horizon is selected only by chronological development predictions.

## Untouched holdout

`research_freeze.py` defaults to:

- development: all Challenger target rows before 2026-06-01;
- untouched holdout: 2026-06-01 through 2026-08-31;
- architecture, history horizon, market correction and betting policies are selected without holdout results;
- holdout base estimators may refit at each 28-day block using only information already available before that block;
- ensemble weights, calibration, value model and bet policies remain frozen from development.

This is much cleaner than selecting a model on the same slates used to advertise its ROI.

## Model family

C1-FUND uses a conservative ensemble of:
- regularized logistic regression;
- histogram gradient boosting;
- Extra Trees;
- random forest.

Nonnegative ensemble weights are learned from earlier out-of-fold predictions with shrinkage toward equal weights. Calibration is a regularized logistic calibration on previous OOF predictions.

C1-VALUE uses bookmaker log-odds as a fixed offset and learns only a regularized correction from C1-FUND disagreement, surface/current-level signals and data reliability. This makes it deliberately difficult for a thin Challenger sample to invent fantasy probabilities far away from the market.

## 20% slate hypothesis

The research grid explicitly compares top 10%, 15%, 20%, 25% and 30% slate caps. C1-VOLUME20 is selected from **20%-cap policies only**. It is still subject to edge, expected-value, reliability and maximum-price gates, so a weak 60-match slate may produce only four bets rather than forcing twelve.

The diagnostic output also makes it easy to check whether bet quality declines monotonically from the very top of the ranked card toward 30% of the slate.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Build features from TennisMyLife-compatible files:

```bash
PYTHONPATH=src python scripts/build_features.py --data-dir data/raw/tennismylife --years 2023 2024 2025 2026
```

Full development / holdout / freeze research (requires a historical two-sided odds CSV):

```bash
PYTHONPATH=src python scripts/research_freeze.py \
  --source tennismylife \
  --data-dir data/raw/tennismylife \
  --years 2023 2024 2025 2026 \
  --odds data/raw/historical_odds.csv
```

If odds columns differ from the generic names, pass a JSON mapping with canonical fields `date`, `player1`, `player2`, `odd1`, `odd2`, and optionally `tournament`, `surface`, `source`.

Daily deployment after a successful freeze:

```bash
PYTHONPATH=src python scripts/deploy.py --slate data/live_slate.csv
```

## Important status distinction

A package that compiles and passes smoke tests is **not** a profitable trained model. C1 should only be labeled frozen/approved after the real multi-year Challenger + historical odds run completes and its untouched holdout passes the research gates. Even then, the next evidence is prospective performance on unseen future slates.
