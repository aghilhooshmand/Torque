"""
Load dataset from evolution_config.json: from a local file or from UCI ML Repository.

Config "dataset" section:
  - From file: "file" (path), "target_column", optional "delimiter", "header"
  - From UCI:  "uci_id" (int, e.g. 2 for Adult), optional "target_column" if targets have multiple columns
"""

import os
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd


def load_dataset_from_config(
    cfg: dict,
    root_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load X, y from config["dataset"]. Supports file path or UCI ML Repository.

    Returns:
        X: feature matrix (n_samples, n_features)
        y: target vector (n_samples,)
        info: dict with "target_name", "feature_names", "source" ("file" | "uci")
    """
    ds = cfg["dataset"]
    info = {"target_name": None, "feature_names": None, "source": None}

    # --- UCI ML Repository ---
    uci_id = ds.get("uci_id")
    if uci_id is not None:
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError:
            raise ImportError("UCI dataset requested but 'ucimlrepo' not installed. Run: pip install ucimlrepo")
        repo = fetch_ucirepo(id=int(uci_id))
        # data (as pandas DataFrames)
        X_df = repo.data.features
        y_df = repo.data.targets
        if X_df is None or y_df is None:
            raise ValueError(f"UCI id={uci_id}: missing features or targets in fetched data")
        # Target: if single column use it; else use target_column
        if y_df.shape[1] == 1:
            target_col = y_df.columns[0]
            y = y_df.iloc[:, 0].values
        else:
            target_col = ds.get("target_column")
            if not target_col or target_col not in y_df.columns:
                raise ValueError(
                    f"UCI id={uci_id}: targets have columns {list(y_df.columns)}; set dataset.target_column in config"
                )
            y = y_df[target_col].values
        # Convert to numpy; handle object/categorical for X
        X = _dataframe_to_numeric(X_df)
        y = _target_to_1d(y)
        info["target_name"] = target_col
        info["feature_names"] = list(X_df.columns) if hasattr(X_df, "columns") else None
        info["source"] = "uci"
        return X, y, info

    # --- Local file ---
    path = ds.get("file")
    if not path:
        raise ValueError('Dataset config must set either "file" or "uci_id"')
    if root_dir and not os.path.isabs(path):
        path = os.path.join(root_dir, path)
    df = pd.read_csv(
        path,
        sep=ds.get("delimiter", ","),
        header=0 if ds.get("header", True) else None,
    )
    target_col = ds.get("target_column")
    if not target_col or target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in {list(df.columns)}")
    y = df[target_col].values
    X = df.drop(columns=[target_col])
    feature_names = list(X.columns)
    X = _dataframe_to_numeric(X)
    y = _target_to_1d(y)
    info["target_name"] = target_col
    info["feature_names"] = feature_names
    info["source"] = "file"
    return X, y, info


def fetch_uci_data(uci_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch a dataset from UCI ML Repository by ID. For use in GUI: returns raw
    feature and target DataFrames so the user can choose target/feature columns.

    Returns:
        features_df: pandas DataFrame of features
        targets_df: pandas DataFrame of targets (may have multiple columns)
    """
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        raise ImportError("ucimlrepo is required. Run: pip install ucimlrepo")
    repo = fetch_ucirepo(id=int(uci_id))
    X_df = repo.data.features
    y_df = repo.data.targets
    if X_df is None or y_df is None:
        raise ValueError(f"UCI id={uci_id}: missing features or targets in fetched data")
    return X_df, y_df


def build_X_y_from_frames(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build X, y arrays from feature and target DataFrames (e.g. after UCI fetch).
    Converts object/category columns to numeric. For use in GUI.
    """
    if feature_columns is None:
        feature_columns = list(features_df.columns)
    X = _dataframe_to_numeric(features_df[feature_columns].copy())
    if target_column not in targets_df.columns:
        raise ValueError(f"Target column '{target_column}' not in {list(targets_df.columns)}")
    y = _target_to_1d(targets_df[target_column].values)
    return X, y


def sample_stratified_by_class(
    X: np.ndarray,
    y: np.ndarray,
    pct: float,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Keep a given percentage of each class (stratified sample). Use to reduce
    dataset size while preserving class balance. pct in [0, 100].

    Returns:
        X_sampled, y_sampled, indices_kept (so caller can subset a DataFrame).
    """
    n = len(y)
    rng = np.random.default_rng(random_state)
    if pct >= 100:
        return X, y, np.arange(n)
    indices = []
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        n_keep = max(1, int(round(len(idx_c) * pct / 100)))
        chosen = rng.choice(idx_c, size=min(n_keep, len(idx_c)), replace=False)
        indices.append(chosen)
    indices = np.concatenate(indices)
    rng.shuffle(indices)
    return X[indices], y[indices], indices


def _dataframe_to_numeric(df: pd.DataFrame) -> np.ndarray:
    """Convert DataFrame to numeric array; encode object/category columns if needed."""
    if df.empty:
        return np.zeros((0, 0))
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df = df.copy()
        df[col] = pd.Categorical(df[col]).codes
    return df.values.astype(np.float64, copy=False)


def _target_to_1d(y: Any) -> np.ndarray:
    """Ensure y is 1d and numeric (labels as int for classification)."""
    y = np.asarray(y)
    if y.ndim > 1:
        y = y.ravel()
    if y.dtype.kind in ("O", "U", "S") or pd.api.types.is_string_dtype(y):
        y = pd.Categorical(y).codes
    return np.asarray(y, dtype=np.intp)
