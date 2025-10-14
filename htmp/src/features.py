"""Feature engineering for the Hull Tactical Market Prediction baseline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureConfig:
    drop_columns: Sequence[str]
    imputation_strategy: str = "median"
    scale: bool = True
    rolling_windows: Sequence[int] | None = None
    enable_interactions: bool = False
    time_column: str | None = None
    group_column: str | None = None


class SimpleFeatureExtractor:
    """Extracts simple statistical features suitable for the HTMP dataset."""

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.scaler: StandardScaler | None = None
        self.numeric_columns: List[str] = []

    def _prepare_columns(
        self, df: pd.DataFrame, target_column: str | None, fit: bool
    ) -> List[str]:
        drop_cols = set(self.config.drop_columns)
        if target_column is not None:
            drop_cols.add(target_column)

        if fit or not self.numeric_columns:
            numeric_columns = [
                col
                for col in df.columns
                if col not in drop_cols and pd.api.types.is_numeric_dtype(df[col])
            ]
            if not numeric_columns:
                raise ValueError("No numeric columns available for feature extraction.")
            self.numeric_columns = numeric_columns
        numeric_columns = [col for col in self.numeric_columns if col in df.columns]
        if not numeric_columns:
            raise ValueError("Configured numeric columns are missing from the provided dataframe.")
        return numeric_columns

    def fit_transform(self, df: pd.DataFrame, target_column: str | None = None) -> pd.DataFrame:
        return self._create_features(df.copy(), target_column=target_column, fit=True)

    def transform(self, df: pd.DataFrame, target_column: str | None = None) -> pd.DataFrame:
        return self._create_features(df.copy(), target_column=target_column, fit=False)

    def _create_features(
        self,
        df: pd.DataFrame,
        target_column: str | None = None,
        fit: bool = False,
    ) -> pd.DataFrame:
        numeric_columns = self._prepare_columns(df, target_column=target_column, fit=fit)
        df_numeric = df[numeric_columns].copy()

        df_numeric = self._apply_imputation(df_numeric)
        df_numeric = self._add_rolling_statistics(df_numeric, df)
        df_numeric = self._add_interactions(df_numeric)

        if self.config.scale:
            if fit:
                self.scaler = StandardScaler()
                scaled = self.scaler.fit_transform(df_numeric)
            else:
                if self.scaler is None:
                    raise RuntimeError("Scaler has not been fitted. Call fit_transform first.")
                scaled = self.scaler.transform(df_numeric)
            df_numeric = pd.DataFrame(scaled, columns=df_numeric.columns, index=df_numeric.index)

        return df_numeric

    def _apply_imputation(self, df_numeric: pd.DataFrame) -> pd.DataFrame:
        if self.config.imputation_strategy == "mean":
            return df_numeric.fillna(df_numeric.mean())
        if self.config.imputation_strategy == "median":
            return df_numeric.fillna(df_numeric.median())
        return df_numeric.fillna(0.0)

    def _add_rolling_statistics(self, df_numeric: pd.DataFrame, df_original: pd.DataFrame) -> pd.DataFrame:
        windows = self.config.rolling_windows or []
        time_col = self.config.time_column
        group_col = self.config.group_column
        if not windows or time_col is None or time_col not in df_original.columns:
            return df_numeric

        df_with_time = df_original[[time_col]].copy()
        if group_col and group_col in df_original.columns:
            df_with_time[group_col] = df_original[group_col]

        augmented = df_numeric.copy()
        for window in windows:
            suffix = f"_roll{window}"
            if group_col and group_col in df_with_time.columns:
                rolled_parts = []
                for col in df_numeric.columns:
                    rolled = (
                        df_numeric[col]
                        .groupby(df_with_time[group_col])
                        .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
                    )
                    rolled_parts.append(rolled.rename(f"{col}{suffix}"))
                augmented = pd.concat([augmented] + rolled_parts, axis=1)
            else:
                sorter = df_with_time[time_col].argsort()
                df_sorted = df_numeric.iloc[sorter]
                for col in df_numeric.columns:
                    rolled_values = df_sorted[col].rolling(window=window, min_periods=1).mean()
                    augmented.loc[:, f"{col}{suffix}"] = rolled_values.sort_index()
        return augmented

    def _add_interactions(self, df_numeric: pd.DataFrame) -> pd.DataFrame:
        if not self.config.enable_interactions:
            return df_numeric
        interaction_cols = {}
        for i, col_a in enumerate(self.numeric_columns):
            for col_b in self.numeric_columns[i + 1 :]:
                interaction_cols[f"{col_a}_x_{col_b}"] = df_numeric[col_a] * df_numeric[col_b]
        if interaction_cols:
            df_numeric = pd.concat([df_numeric, pd.DataFrame(interaction_cols, index=df_numeric.index)], axis=1)
        return df_numeric


__all__ = ["FeatureConfig", "SimpleFeatureExtractor"]
