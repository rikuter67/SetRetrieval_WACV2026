"""Lazy-fit baseline inference server for the Kaggle evaluation API."""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:  # Optional dependency for faster CSV loading inside Kaggle
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover - fallback if Polars is unavailable
    pl = None  # type: ignore

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - fallback to manual ridge implementation
    Ridge = StandardScaler = None  # type: ignore
    _HAS_SKLEARN = False

try:
    from kaggle_evaluation.default_inference_server import DefaultInferenceServer
except Exception as exc:  # pragma: no cover - local usage without Kaggle package
    raise ImportError(
        "kaggle_evaluation is required to run this script. "
        "Install kaggle-evaluation inside a Kaggle Notebook or submission environment."
    ) from exc

warnings.filterwarnings("ignore")

_TARGET_COLUMN = os.getenv("HTMP_TARGET", "forward_returns")
_ID_CANDIDATES = {"row_id", "id", "ID"}
_DATE_CANDIDATES = {"date", "timestamp", "time", "Date"}
_DATA_ROOTS: tuple[str, ...] = (
    "/kaggle/input/hull-tactical-market-prediction",
    "/kaggle/input/hull-tactical-market-prediction/",
    "../input/hull-tactical-market-prediction",
    "../input/hull-tactical-market-prediction/",
    "data/raw",
    "./",
)


@dataclass(slots=True)
class _ModelArtifacts:
    feature_columns: list[str]
    medians: dict[str, float]
    scaler: Optional[StandardScaler]
    model: object


def _resolve_path(fname: str) -> Optional[str]:
    for root in _DATA_ROOTS:
        candidate = os.path.join(root, fname)
        if os.path.exists(candidate):
            return candidate
    if os.path.exists(fname):
        return fname
    return None


def _read_csv(path: str) -> pd.DataFrame:
    if pl is not None:
        return pl.read_csv(path).to_pandas()
    return pd.read_csv(path)


def _infer_numeric_columns(df: pd.DataFrame) -> list[str]:
    exclude: set[str] = set()
    exclude.update(col for col in df.columns if col in _ID_CANDIDATES)
    exclude.update(col for col in df.columns if col in _DATE_CANDIDATES)
    if _TARGET_COLUMN in df.columns:
        exclude.add(_TARGET_COLUMN)
    numeric_cols = [
        col
        for col in df.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
    ]
    return numeric_cols


def _median_impute(df: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    for col, value in medians.items():
        df[col] = df[col].fillna(value)
    return df


def _fit_manual_ridge(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    Xs = (X - mu) / std
    a = Xs.T @ Xs + lam * np.eye(Xs.shape[1])
    b = Xs.T @ y
    weights = np.linalg.solve(a, b)
    bias = float(y.mean() - (mu / std) @ weights)
    return weights, bias, mu, std


class LazyBaseline:
    """Fit-on-first-call baseline for streaming evaluation."""

    def __init__(self) -> None:
        self.artifacts: Optional[_ModelArtifacts] = None

    def _fit(self) -> None:
        train_path = _resolve_path("train.csv")
        if train_path is None:
            raise FileNotFoundError(
                "train.csv が見つかりませんでした。DATA_ROOTS または入力データの配置を確認してください。"
            )

        train_df = _read_csv(train_path)
        if _TARGET_COLUMN not in train_df.columns:
            raise KeyError(
                f"目的変数カラム '{_TARGET_COLUMN}' が見つかりません。"
                " 必要に応じて HTMP_TARGET 環境変数を設定してください。"
            )

        numeric_cols = _infer_numeric_columns(train_df)
        features = train_df[numeric_cols].copy()
        y = train_df[_TARGET_COLUMN].astype(float).values

        medians = {
            col: (features[col].median() if pd.api.types.is_numeric_dtype(features[col]) else 0.0)
            for col in numeric_cols
        }
        features = _median_impute(features, medians)

        if _HAS_SKLEARN:
            scaler = StandardScaler()
            Xs = scaler.fit_transform(features.values)
            model = Ridge(alpha=1.0, random_state=42)
            model.fit(Xs, y)
        else:  # pragma: no cover - fallback path
            scaler = None
            weights, bias, mu, std = _fit_manual_ridge(features.values, y)
            model = (weights, bias, mu, std)

        self.artifacts = _ModelArtifacts(
            feature_columns=numeric_cols,
            medians=medians,
            scaler=scaler,
            model=model,
        )

    def _ensure_fitted(self) -> _ModelArtifacts:
        if self.artifacts is None:
            self._fit()
        assert self.artifacts is not None
        return self.artifacts

    def _prepare_features(self, batch: pd.DataFrame, artifacts: _ModelArtifacts) -> pd.DataFrame:
        aligned = pd.DataFrame(index=batch.index)
        for col in artifacts.feature_columns:
            if col in batch.columns and pd.api.types.is_numeric_dtype(batch[col]):
                aligned[col] = batch[col]
            else:
                aligned[col] = np.nan
        aligned = _median_impute(aligned, artifacts.medians.copy())
        return aligned

    def predict(self, batch: pd.DataFrame | pl.DataFrame | Iterable[dict]) -> float:
        artifacts = self._ensure_fitted()

        if isinstance(batch, pl.DataFrame):
            frame = batch.to_pandas()
        elif isinstance(batch, pd.DataFrame):
            frame = batch
        else:
            frame = pd.DataFrame(batch)

        features = self._prepare_features(frame, artifacts)
        values = features.values

        if _HAS_SKLEARN and artifacts.scaler is not None:
            scaled = artifacts.scaler.transform(values)
            preds = artifacts.model.predict(scaled)  # type: ignore[call-arg]
        else:  # pragma: no cover - fallback path
            weights, bias, mu, std = artifacts.model  # type: ignore[misc]
            scaled = (values - mu) / std
            preds = scaled @ weights + bias

        return float(np.mean(preds))


_BASELINE = LazyBaseline()

def predict(batch: pl.DataFrame | pd.DataFrame | Iterable[dict]) -> float:
    """Entry point required by Kaggle's evaluation server."""
    return _BASELINE.predict(batch)


inference_server = DefaultInferenceServer(predict)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()
else:
    inference_server.run_local_gateway(("/kaggle/input/hull-tactical-market-prediction/",))
