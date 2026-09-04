from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS, augment_symmetric, mirror_features, reliability_score


def _metric_dict(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=int)
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }
    out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan
    return out


def _prep_pipeline(model: Any, scale: bool) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    prep = ColumnTransformer([("num", Pipeline(steps), FEATURE_COLUMNS)], remainder="drop")
    return Pipeline([("prep", prep), ("model", model)])


class DynamicEloBaseline:
    """No-fit baseline using the experience-decayed, surface-shrunk Elo feature."""
    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series):
        self.classes_ = np.asarray([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        frame = pd.DataFrame(X)
        diff = frame["elo_dynamic_surface_blend_diff_100"].to_numpy(float) * 100.0
        p = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        return np.column_stack([1.0 - p, p])


def candidate_models(seed: int) -> dict[str, Any]:
    return {
        "dynamic_elo": DynamicEloBaseline(),
        "logistic_l2": _prep_pipeline(
            LogisticRegression(C=0.22, l1_ratio=0.0, max_iter=4000, random_state=seed), True
        ),
        "hist_gradient": _prep_pipeline(
            HistGradientBoostingClassifier(
                learning_rate=0.035,
                max_iter=240,
                max_leaf_nodes=15,
                min_samples_leaf=45,
                l2_regularization=5.0,
                random_state=seed,
            ),
            False,
        ),
        "extra_trees": _prep_pipeline(
            ExtraTreesClassifier(
                n_estimators=350,
                max_depth=8,
                min_samples_leaf=18,
                max_features=0.65,
                bootstrap=False,
                random_state=seed,
                n_jobs=-1,
            ),
            False,
        ),
        "random_forest": _prep_pipeline(
            RandomForestClassifier(
                n_estimators=350,
                max_depth=8,
                min_samples_leaf=18,
                max_features=0.70,
                random_state=seed,
                n_jobs=-1,
            ),
            False,
        ),
    }


def symmetric_predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    p = model.predict_proba(X[FEATURE_COLUMNS])[:, 1]
    pm = model.predict_proba(mirror_features(X[FEATURE_COLUMNS]))[:, 1]
    return np.clip(0.5 * (p + (1.0 - pm)), 1e-5, 1 - 1e-5)


def _fit_weights(preds: np.ndarray, y: np.ndarray, penalty: float = 0.08) -> np.ndarray:
    n_models = preds.shape[1]
    equal = np.full(n_models, 1.0 / n_models)
    if len(y) < 250:
        return equal

    def objective(w: np.ndarray) -> float:
        p = np.clip(preds @ w, 1e-6, 1 - 1e-6)
        ll = log_loss(y, p, labels=[0, 1])
        reg = penalty * float(np.sum((w - equal) ** 2))
        return ll + reg

    res = minimize(
        objective,
        equal,
        method="SLSQP",
        bounds=[(0.03, 0.85)] * n_models,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not res.success:
        return equal
    # Small equal-weight shrinkage makes final weights less fold-sensitive.
    w = 0.85 * res.x + 0.15 * equal
    return w / w.sum()


class LogitCalibrator:
    def __init__(self, C: float = 0.35):
        self.C = C
        self.model: LogisticRegression | None = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> "LogitCalibrator":
        p = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
        y = np.asarray(y, dtype=int)
        if len(y) < 250 or len(np.unique(y)) < 2:
            self.model = None
            return self
        x = logit(p).reshape(-1, 1)
        self.model = LogisticRegression(C=self.C, l1_ratio=0.0, max_iter=2000)
        self.model.fit(x, y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
        if self.model is None:
            return p
        return self.model.predict_proba(logit(p).reshape(-1, 1))[:, 1]


@dataclass
class FundamentalArtifact:
    model_name: str
    horizon_days: int
    feature_columns: list[str]
    base_models: dict[str, Any]
    base_model_names: list[str]
    weights: np.ndarray
    calibrator: LogitCalibrator
    training_cutoff: str
    training_rows: int
    metadata: dict[str, Any]

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        base = np.column_stack([symmetric_predict(self.base_models[name], frame) for name in self.base_model_names])
        raw = base @ self.weights
        calibrated = self.calibrator.predict(raw)
        out = frame.copy()
        for j, name in enumerate(self.base_model_names):
            out[f"p_{name}"] = base[:, j]
        out["fundamental_probability_a_raw"] = raw
        out["fundamental_probability_a"] = calibrated
        out["ensemble_dispersion"] = base.std(axis=1)
        out["reliability"] = reliability_score(out, out["ensemble_dispersion"].to_numpy())
        out["predicted_winner"] = np.where(
            calibrated >= 0.5, out["player_a"], out["player_b"]
        )
        out["predicted_winner_probability"] = np.where(calibrated >= 0.5, calibrated, 1.0 - calibrated)
        return out

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)


def _validation_blocks(
    frame: pd.DataFrame,
    horizon_days: int,
    block_days: int,
    min_train: int,
    min_test: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, np.ndarray, np.ndarray]]:
    dates = pd.to_datetime(frame["date"]).dt.normalize()
    start = dates.min() + pd.Timedelta(days=horizon_days)
    end = dates.max() + pd.Timedelta(days=1)
    blocks = []
    cursor = start
    while cursor < end:
        block_end = min(cursor + pd.Timedelta(days=block_days), end)
        train_start = cursor - pd.Timedelta(days=horizon_days)
        train_idx = np.flatnonzero((dates >= train_start) & (dates < cursor))
        test_idx = np.flatnonzero((dates >= cursor) & (dates < block_end))
        if len(train_idx) >= min_train and len(test_idx) >= min_test:
            blocks.append((cursor, block_end, train_idx, test_idx))
        cursor = block_end
    return blocks


def walk_forward_fundamental(
    frame: pd.DataFrame,
    config: dict[str, Any],
    horizon_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed = int(config["seed"])
    model_factories = candidate_models(seed)
    model_names = list(model_factories)
    blocks = _validation_blocks(
        frame,
        horizon_days=horizon_days,
        block_days=int(config.get("validation_block_days", 28)),
        min_train=int(config.get("minimum_train_matches", 600)),
        min_test=int(config.get("minimum_test_matches", 80)),
    )
    if not blocks:
        raise ValueError(f"No valid walk-forward blocks for {horizon_days}-day horizon")

    pred_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    past_base: list[np.ndarray] = []
    past_y: list[np.ndarray] = []
    past_meta_raw: list[np.ndarray] = []
    past_meta_y: list[np.ndarray] = []

    for fold_id, (start, end, train_idx, test_idx) in enumerate(blocks):
        train = frame.iloc[train_idx].copy()
        test = frame.iloc[test_idx].copy()
        X_train = train[FEATURE_COLUMNS]
        y_train = train["target"].to_numpy(int)
        X_aug, y_aug = augment_symmetric(X_train, y_train)
        base_test = []
        fitted_this_fold = {}
        for name, template in model_factories.items():
            # candidate_models creates independent unfitted objects; re-create each fold
            model = candidate_models(seed + fold_id)[name]
            model.fit(X_aug, y_aug)
            fitted_this_fold[name] = model
            base_test.append(symmetric_predict(model, test))
        base_test_arr = np.column_stack(base_test)

        if past_base:
            hist_base = np.vstack(past_base)
            hist_y = np.concatenate(past_y)
            weights = _fit_weights(hist_base, hist_y)
        else:
            weights = np.full(len(model_names), 1.0 / len(model_names))
        raw = base_test_arr @ weights

        # Calibration is itself walk-forward. It only sees ensemble predictions
        # that were genuinely produced for earlier validation blocks.
        calibrator = LogitCalibrator(C=0.35)
        if past_meta_raw:
            calibrator.fit(np.concatenate(past_meta_raw), np.concatenate(past_meta_y))
        calibrated = calibrator.predict(raw)

        meta_cols = [
            "date", "proxy_date", "tourney_id", "tourney_name", "surface", "round",
            "player_a", "player_b", "player_a_key", "player_b_key", "target",
        ]
        part = test[meta_cols + FEATURE_COLUMNS].copy()
        part["fold_id"] = fold_id
        part["fold_start"] = start
        part["fold_end"] = end
        for j, name in enumerate(model_names):
            part[f"p_{name}"] = base_test_arr[:, j]
        part["ensemble_raw"] = raw
        part["predicted_probability"] = calibrated
        part["ensemble_dispersion"] = base_test_arr.std(axis=1)
        part["reliability"] = reliability_score(test, part["ensemble_dispersion"].to_numpy())
        pred_parts.append(part)

        past_base.append(base_test_arr)
        past_y.append(test["target"].to_numpy(int))
        past_meta_raw.append(raw)
        past_meta_y.append(test["target"].to_numpy(int))

        metrics = _metric_dict(test["target"].to_numpy(int), calibrated)
        fold_rows.append({
            "horizon_days": horizon_days,
            "fold_id": fold_id,
            "start": start,
            "end": end,
            "train_n": len(train),
            "test_n": len(test),
            **metrics,
            **{f"w_{name}": float(weights[j]) for j, name in enumerate(model_names)},
        })

    oof = pd.concat(pred_parts, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    return oof, folds


def compare_horizons(
    frame: pd.DataFrame,
    config: dict[str, Any],
    horizons: list[int] | None = None,
) -> tuple[int, pd.DataFrame, dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    horizons = horizons or [int(x) for x in config.get("history_days_candidates", [365, 730])]
    rows = []
    oof_by_horizon: dict[int, pd.DataFrame] = {}
    folds_by_horizon: dict[int, pd.DataFrame] = {}
    for horizon in horizons:
        oof, folds = walk_forward_fundamental(frame, config, horizon)
        oof_by_horizon[horizon] = oof
        folds_by_horizon[horizon] = folds
        m = _metric_dict(oof["target"].to_numpy(int), oof["predicted_probability"].to_numpy(float))
        # Stability penalty: avoid choosing a horizon from a tiny aggregate edge.
        fold_ll_std = float(folds["log_loss"].std(ddof=0)) if len(folds) > 1 else 0.0
        score = m["log_loss"] + 0.12 * fold_ll_std + 0.20 * m["brier"]
        rows.append({"horizon_days": horizon, **m, "fold_logloss_std": fold_ll_std, "selection_score": score})
    summary = pd.DataFrame(rows).sort_values(["selection_score", "log_loss", "brier"]).reset_index(drop=True)
    best = int(summary.iloc[0]["horizon_days"])
    return best, summary, oof_by_horizon, folds_by_horizon


def fit_final_fundamental(
    frame: pd.DataFrame,
    config: dict[str, Any],
    horizon_days: int,
    oof: pd.DataFrame,
    cutoff: str | pd.Timestamp | None = None,
) -> FundamentalArtifact:
    cutoff_ts = pd.Timestamp(cutoff) if cutoff is not None else pd.to_datetime(frame["date"]).max() + pd.Timedelta(days=1)
    start = cutoff_ts - pd.Timedelta(days=horizon_days)
    train = frame[(pd.to_datetime(frame["date"]) >= start) & (pd.to_datetime(frame["date"]) < cutoff_ts)].copy()
    if len(train) < int(config.get("minimum_train_matches", 600)):
        raise ValueError("Not enough rows to fit final Challenger model")
    X_aug, y_aug = augment_symmetric(train[FEATURE_COLUMNS], train["target"].to_numpy(int))
    seed = int(config["seed"])
    models = {}
    names = list(candidate_models(seed))
    for name in names:
        model = candidate_models(seed)[name]
        model.fit(X_aug, y_aug)
        models[name] = model

    base_oof = np.column_stack([oof[f"p_{name}"].to_numpy(float) for name in names])
    weights = _fit_weights(base_oof, oof["target"].to_numpy(int))
    raw_oof = base_oof @ weights
    calibrator = LogitCalibrator(C=0.35).fit(raw_oof, oof["target"].to_numpy(int))
    metadata = {
        "oof_metrics_final_weighted": _metric_dict(oof["target"].to_numpy(int), calibrator.predict(raw_oof)),
        "weights": {name: float(weights[i]) for i, name in enumerate(names)},
        "train_start": str(start.date()),
        "train_end_exclusive": str(cutoff_ts.date()),
    }
    return FundamentalArtifact(
        model_name=str(config.get("model_name", "ChallengerC1")) + "-FUND",
        horizon_days=horizon_days,
        feature_columns=list(FEATURE_COLUMNS),
        base_models=models,
        base_model_names=names,
        weights=weights,
        calibrator=calibrator,
        training_cutoff=str(cutoff_ts),
        training_rows=len(train),
        metadata=metadata,
    )


def save_json(obj: Any, path: str | Path) -> None:
    def default(x: Any):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (pd.Timestamp,)):
            return x.isoformat()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(obj, indent=2, default=default))


def walk_forward_fixed_architecture(
    frame: pd.DataFrame,
    config: dict[str, Any],
    horizon_days: int,
    start_date: str | pd.Timestamp,
    weights: np.ndarray,
    calibrator: LogitCalibrator,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Prospective-style block simulation with architecture/meta-weights frozen.

    Base estimators are refit at each block using only the immediately preceding
    horizon, exactly as a live model could be refreshed. Ensemble weights and
    probability calibration remain frozen from development OOF data.
    """
    dates = pd.to_datetime(frame["date"]).dt.normalize()
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1) if end_date else dates.max() + pd.Timedelta(days=1)
    block_days = int(config.get("validation_block_days", 28))
    seed = int(config["seed"])
    names = list(candidate_models(seed))
    parts = []
    cursor = start_ts
    fold_id = 0
    while cursor < end_ts:
        block_end = min(cursor + pd.Timedelta(days=block_days), end_ts)
        train_start = cursor - pd.Timedelta(days=horizon_days)
        train = frame[(dates >= train_start) & (dates < cursor)].copy()
        test = frame[(dates >= cursor) & (dates < block_end)].copy()
        if len(test) == 0:
            cursor = block_end; fold_id += 1; continue
        if len(train) < int(config.get("minimum_train_matches", 600)):
            raise ValueError(f"Holdout block {cursor.date()} has only {len(train)} training rows")
        X_aug, y_aug = augment_symmetric(train[FEATURE_COLUMNS], train["target"].to_numpy(int))
        base = []
        for name in names:
            mdl = candidate_models(seed + fold_id)[name]
            mdl.fit(X_aug, y_aug)
            base.append(symmetric_predict(mdl, test))
        base_arr = np.column_stack(base)
        raw = base_arr @ np.asarray(weights, dtype=float)
        p = calibrator.predict(raw)
        meta_cols = [
            "date", "proxy_date", "tourney_id", "tourney_name", "surface", "round",
            "player_a", "player_b", "player_a_key", "player_b_key", "target",
        ]
        part = test[meta_cols + FEATURE_COLUMNS].copy()
        part["fold_id"] = fold_id
        part["fold_start"] = cursor
        part["fold_end"] = block_end
        for j, name in enumerate(names):
            part[f"p_{name}"] = base_arr[:, j]
        part["ensemble_raw"] = raw
        part["predicted_probability"] = p
        part["ensemble_dispersion"] = base_arr.std(axis=1)
        part["reliability"] = reliability_score(test, part["ensemble_dispersion"].to_numpy())
        parts.append(part)
        cursor = block_end
        fold_id += 1
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
