# ChallengerC1 data plan

## Preferred results/stats source: TennisMyLife

Use 2023-2026 ATP Tour, ATP Challenger and ATP qualifying files. The site publishes yearly Challenger CSVs (`YYYY_challenger.csv`), ATP Tour files (`YYYY.csv`), ATP qualifying files, an ongoing Challenger file, ATP IDs, surface, indoor/outdoor, rankings, ages, heights, scores and serve/break-point totals. The site currently labels the database MIT-licensed and free to use.

Source page: https://stats.tennismylife.org/tennis-match-database

Why it is preferred for C1:
- explicit Challenger files through the current season;
- indoor/outdoor metadata;
- ATP identifiers and cross-level files;
- daily/current updates;
- permissive published license.

## Preferred odds source: TennisData.App

Use ATP season CSVs for 2023-2026 (or 2021-2026 if useful). The download documentation says each ATP season file combines main-tour and Challenger matches and includes match-winner odds when available, usually closing odds. Both sides are needed for no-vig probabilities and profit backtests.

Source page: https://tennisdata.app/downloads/

For reproducible model research, preserve the raw odds snapshot/files used for each run. Closing-line historical results are useful for conservative testing but do not guarantee the same price will be available live.

## Secondary/fallback source: Jeff Sackmann / Tennis Abstract

The `tennis_atp` dataset has separate annual main-tour, tour-qualifying + Challenger main-draw, and futures files. Challenger match statistics are broadly available from 2008 onward. This is particularly useful for connecting incoming ITF/Futures prospects to Challenger strength.

Source: https://github.com/JeffSackmann/tennis_atp

Important: the published license is CC BY-NC-SA 4.0 / non-commercial. Verify that the intended use is compatible before relying on it outside private research.

## Why more than one competition level matters

C1 predicts only ATP Challenger matches, but its player-strength state should be updated by ATP main draw, ATP qualifying, Challenger and (when a compatible source is available) ITF/Futures matches. That prevents a player moving between levels from appearing artificially unknown and automatically opponent-adjusts recent records.

## Date caveat

Several public tennis datasets use `tourney_date` as the tournament-week date rather than an exact match timestamp. C1 never pretends that is exact. Within an event it uses round order only to ensure later rounds can see prior-round results. Exact rest/travel calculations should use a source with real match timestamps when available.
