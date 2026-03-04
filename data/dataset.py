"""
Load and preprocess CIC IoMT 2024 dataset from Parquet (primary) or CSV.
Single source of truth for all quantum models (QSVM, VQC, QMLP, QNN).
"""
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA


# Default paths relative to project root
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "CIC_iomt_dataset"
DEFAULT_TRAIN_PARQUET = DEFAULT_DATA_DIR / "CIC_IoMT_2024_WiFi_MQTT_train.parquet"
DEFAULT_TEST_PARQUET = DEFAULT_DATA_DIR / "CIC_IoMT_2024_WiFi_MQTT_test.parquet"
DEFAULT_TRAIN_CSV = DEFAULT_DATA_DIR / "CIC_IoMT_2024_WiFi_MQTT_train.csv"
DEFAULT_TEST_CSV = DEFAULT_DATA_DIR / "CIC_IoMT_2024_WiFi_MQTT_test.csv"

LABEL_COLUMN = "label"


def _resolve_paths(
    data_dir: Optional[Path] = None,
    train_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Resolve train and test file paths; prefer Parquet over CSV."""
    base = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    train = Path(train_path) if train_path else None
    test = Path(test_path) if test_path else None

    if train and test:
        return train, test

    # Prefer Parquet
    pqt_train = train or base / "CIC_IoMT_2024_WiFi_MQTT_train.parquet"
    pqt_test = test or base / "CIC_IoMT_2024_WiFi_MQTT_test.parquet"
    if pqt_train.exists() and pqt_test.exists():
        return pqt_train, pqt_test

    # Fallback to CSV
    csv_train = train or base / "CIC_IoMT_2024_WiFi_MQTT_train.csv"
    csv_test = test or base / "CIC_IoMT_2024_WiFi_MQTT_test.csv"
    return csv_train, csv_test


def _load_file(path: Path, n_samples: Optional[int] = None) -> pd.DataFrame:
    """Load a single file as DataFrame (Parquet or CSV)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        # CSV: read in chunks if n_samples set to avoid huge memory
        df = pd.read_csv(path, nrows=n_samples, low_memory=False)

    if n_samples and len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
    return df


def load_cic_iomt_2024(
    data_dir: Optional[Path] = None,
    train_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
    label_column: str = LABEL_COLUMN,
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load CIC IoMT 2024 train and test (Parquet preferred, then CSV).
    Returns (X_train, y_train, X_test, y_test) as DataFrames/Series (not scaled).
    """
    train_path, test_path = _resolve_paths(data_dir, train_path, test_path)
    train_df = _load_file(train_path, n_train)
    test_df = _load_file(test_path, n_test)

    if label_column not in train_df.columns:
        raise ValueError(f"Label column '{label_column}' not in train columns: {list(train_df.columns)}")
    if label_column not in test_df.columns:
        raise ValueError(f"Label column '{label_column}' not in test columns")

    feature_cols = [c for c in train_df.columns if c != label_column]
    X_train = train_df[feature_cols].copy()
    y_train = train_df[label_column].astype(str)
    X_test = test_df[feature_cols].copy()
    y_test = test_df[label_column].astype(str)

    return X_train, y_train, X_test, y_test


def preprocess(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_components: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, Optional[PCA], LabelEncoder]:
    """
    Clean, scale, and optionally reduce dimension.
    Returns (X_train, y_train, X_test, y_test, scaler, pca, label_encoder).
    """
    # Align columns and fill NaN
    X_train = X_train.fillna(0).astype(np.float64)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0).astype(np.float64)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = None
    if n_components and n_components < X_train_scaled.shape[1]:
        pca = PCA(n_components=n_components, random_state=random_state)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_test_scaled = pca.transform(X_test_scaled)

    # Fit label encoder on union of train and test labels so that
    # labels that appear only in the test set are still encodable.
    le = LabelEncoder()
    all_labels = pd.concat([y_train, y_test], axis=0)
    le.fit(all_labels)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    return (
        np.asarray(X_train_scaled, dtype=np.float64),
        np.asarray(y_train_enc, dtype=np.int64),
        np.asarray(X_test_scaled, dtype=np.float64),
        np.asarray(y_test_enc, dtype=np.int64),
        scaler,
        pca,
        le,
    )


def get_dataset(
    data_dir: Optional[Path] = None,
    train_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
    label_column: str = LABEL_COLUMN,
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
    n_components: Optional[int] = None,
    random_state: int = 42,
):
    """
    Load and preprocess in one call.
    Returns (X_train, y_train, X_test, y_test, label_encoder).
    """
    X_train, y_train, X_test, y_test = load_cic_iomt_2024(
        data_dir=data_dir,
        train_path=train_path,
        test_path=test_path,
        label_column=label_column,
        n_train=n_train,
        n_test=n_test,
        random_state=random_state,
    )
    X_tr, y_tr, X_te, y_te, scaler, pca, le = preprocess(
        X_train, y_train, X_test, y_test,
        n_components=n_components,
        random_state=random_state,
    )
    return X_tr, y_tr, X_te, y_te, le
