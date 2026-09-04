# ChallengerC1 build report

Build date: 2026-09-04

## Status

**Engineering status: complete and smoke-tested.**

**Research/training status: NOT YET FROZEN.** The current chat runtime cannot directly materialize the multi-year public Challenger CSVs/odds files from the web into the local modeling container. Therefore this package does **not** invent a backtest or claim a trained profitable model. The next valid step is to import the actual historical files, run `scripts/research_freeze.py`, inspect the untouched 2026-06-01 through 2026-08-31 holdout once, and freeze only if it passes.

This distinction is intentional. A polished pipeline without real out-of-sample data is not evidence of edge.

## What was built

### 1. C1-FUND: fundamental win-probability model

Targets men's ATP Challenger main-draw matches while updating player strength from adjacent competition levels.

Player state contains:
- conservative slow Elo;
- high-reactivity fast Elo;
- Tennis-Abstract-style experience-decayed dynamic Elo;
- surface-specific Elo;
- dynamic surface Elo with shrinkage toward overall strength for sparse surfaces;
- opponent-adjusted recent performance residuals at 35/90/180-day half-lives;
- surface-specific current form;
- recent opponent quality;
- recency-weighted set/game share;
- serve-points-won, return-points-won, hold and break proxies when match stats exist;
- competition-level residuals for Challenger, ATP main draw, ATP qualifying and Futures/ITF;
- workload/rest, ranking, points, age, height, experience and entry-status context;
- heavily shrunk head-to-head context.

Retirements and walkovers are excluded from clean training labels and are not treated as normal Elo evidence.

### 2. Symmetry and leakage controls

The winner is never hard-coded as Player 1. Each historical matchup is oriented A/B deterministically from player identity before the label is created. Training rows are mirrored with all difference features sign-flipped and the target inverted. Inference also averages the direct and mirrored probabilities.

Thus the model is explicitly designed to satisfy approximately:

`P(A beats B) = 1 - P(B beats A)`

All rolling values and ratings are snapshotted before the result is applied.

### 3. C1-FUND ensemble

Base views:
- dynamic surface-blended Elo baseline;
- regularized logistic regression;
- histogram gradient boosting;
- Extra Trees;
- random forest.

Ensemble weights are nonnegative, regularized toward equal weighting, and learned only from earlier out-of-fold predictions. Probability calibration is a separate regularized logit calibrator trained only on previous OOF predictions in the walk-forward simulation.

The model explicitly compares 365-day and 730-day training horizons instead of assuming one is better. Recent performance remains much more responsive through 35/90/180-day state features even when the 730-day training window wins.

### 4. C1-VALUE: market-residual model

The value layer does not ask a generic tree model to rediscover the bookmaker market from scratch. It uses the no-vig bookmaker log-odds as a **fixed offset** and learns only a strongly regularized correction from:
- C1-FUND vs market disagreement;
- data reliability / ensemble dispersion;
- surface and dynamic Elo differences;
- recent opponent-adjusted form;
- serve/return evidence;
- ranking/points evidence;
- experience/stat coverage.

This architecture is deliberately skeptical because bookmaker probabilities are a very strong tennis baseline.

### 5. Reliability model

A 70% prediction based on 25 recent matches and strong serve/return coverage is not treated the same as a 70% prediction for a new wildcard with two observations.

Each candidate receives a reliability score based on:
- minimum player history;
- minimum surface history;
- recent statistic coverage;
- agreement/dispersion among the independent model views.

Missing stats reduce confidence; they do not silently become positive evidence.

### 6. Slate-ranking policies

The policy research grid compares caps of:
- top 10%;
- top 15%;
- **top 20%;**
- top 25%;
- top 30%.

It also tests multiple edge floors, modeled-EV floors, reliability floors and maximum prices.

Two named deployment policies are produced:
- **C1-CORE10**: maximum 10% of a slate, stronger history/reliability requirements;
- **C1-VOLUME20**: maximum 20% of a slate, intended as the main high-volume policy.

The 20% policy is a cap, not a quota. A 60-match slate can yield 12, 8, 4 or zero bets depending on whether candidates clear the frozen gates.

### 7. Robust policy selection

A historical policy is not rewarded merely for one giant underdog win. Development ranking incorporates:
- raw flat-stake ROI;
- ROI after a 5% price haircut;
- ROI after removing the single biggest winner;
- median monthly ROI;
- profitable-month share;
- minimum sample-size requirements.

Final diagnostics include month, surface, odds band, edge band, maximum drawdown and bootstrap ROI intervals.

## Clean validation plan

Default research protocol:

1. Build player state chronologically from 2023 onward across all supplied competition levels.
2. Use only Challenger matches as target labels.
3. Development period ends 2026-05-31.
4. Select 365 vs 730 days and all policy thresholds using development walk-forward predictions only.
5. Freeze ensemble architecture, calibration rule, value architecture and betting policies.
6. Grade 2026-06-01 through 2026-08-31 exactly once as the untouched holdout.
7. Holdout simulation can refit base estimators every 28 days using only results available before each block, but meta weights/calibration/value correction/policy remain fixed from development.
8. Only after the holdout is reported may production parameters be refit on all historical OOS predictions.
9. The resulting C1 artifacts are then frozen for prospective September+ tracking.

No random train/test split is used as headline evidence.

## Research gates before prospective deployment

The script uses intentionally conservative research gates. Among other requirements, C1-VOLUME20 must have meaningful holdout volume, positive flat-stake holdout profit, remain positive after a 5% price haircut, and remain positive after removing its single largest winner.

Passing those gates still does not establish future profitability. It only justifies freezing C1 and testing it prospectively rather than redesigning it immediately.

## Important external evidence incorporated into the design

Recent tennis forecasting research reinforces several choices made here:

- surface-adjusted Elo is a strong baseline for men's tennis;
- bookmaker probabilities are generally stronger than public-data models and must be treated as the benchmark rather than ignored;
- lower-level Challenger/ITF history materially improves player-strength ratings for ATP players who move between levels;
- sophisticated ML often adds only modest predictive improvement over a strong Elo baseline;
- claims of betting edge against sharp closing markets need unusually strict out-of-sample validation.

Because of that evidence, C1 is intentionally built to *earn* market disagreement rather than generate aggressive independent probabilities for every match.

## Files

- `src/challenger_c1/data.py` — source adapters, score/stat derivation, odds canonicalization
- `src/challenger_c1/features.py` — leakage-safe cross-level state and features
- `src/challenger_c1/model.py` — fundamental ensemble, nested walk-forward weighting/calibration, fixed-architecture holdout
- `src/challenger_c1/market.py` — odds matching and market-offset value model
- `src/challenger_c1/policy.py` — top-fraction policies, robustness scoring and diagnostics
- `scripts/research_freeze.py` — end-to-end development/holdout/freeze run
- `scripts/deploy.py` — future slate predictions/cards
- `tests/smoke_test.py` — feature/model/value/policy execution test
- `DATA_SOURCES.md` — preferred/fallback data feeds and limitations

## Verification performed in this build

The package was compiled with Python and the smoke test successfully executed:
- canonical results parsing;
- chronological feature state;
- symmetric model training and prediction;
- walk-forward fundamental ensemble;
- market-offset layer;
- candidate ranking and top-fraction policy grid.

Synthetic smoke-test performance is deliberately not reported as model performance because it has no betting meaning.
