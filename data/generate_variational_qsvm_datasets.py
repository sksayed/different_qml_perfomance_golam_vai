"""
One-off script to generate smaller, balanced datasets for
variational PennyLane models (VQC, QMLP, QNN) and QSVM.

Usage (from project root):
  python -m data.generate_variational_qsvm_datasets

This will:
  1) Load the full CIC_IoMT_2024_WiFi_MQTT_train.parquet file.
  2) Collapse labels into 4 meta-classes using _collapse_label:
       - DDoS, DoS, Benign, Other
  3) Apply per-class caps to build a ~O(80k) variational dataset:
       - If count < MIN_PER_CLASS: keep all rows
       - If MIN_PER_CLASS <= count <= MAX_PER_CLASS: keep all rows
       - If count > MAX_PER_CLASS: sample MAX_PER_CLASS rows
  4) Save this as CIC_IoMT_2024_Variational.parquet.
  5) Draw a stratified sample of N_QSVM rows from that file and
     save as CIC_IoMT_2024_QSVM.parquet for QSVM experiments.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from data.dataset import (
    DEFAULT_TRAIN_PARQUET,
    DEFAULT_DATA_DIR,
    LABEL_COLUMN,
    _collapse_label,
)


# Configuration for dataset sizes
MIN_PER_CLASS = 2_000   # minimum rows per meta-class (if available)
MAX_PER_CLASS = 15_000  # maximum rows per meta-class
N_QSVM = 5_000          # total rows for QSVM dataset
RANDOM_STATE = 42


def build_variational_dataset(
    train_parquet: Path,
    label_column: str = LABEL_COLUMN,
) -> pd.DataFrame:
    """Build the capped, balanced dataset for variational models."""
    print(f"Loading full train parquet from {train_parquet} ...", flush=True)
    df = pd.read_parquet(train_parquet)
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns.", flush=True)

    # Collapse to 4 meta-classes to match variational model outputs
    print("Collapsing labels to 4 meta-classes (DDoS, DoS, Benign, Other)...", flush=True)
    df = df.copy()
    # Overwrite the original label column so there is no target leakage
    # through an extra meta_label feature column.
    df[label_column] = df[label_column].astype(str).map(_collapse_label)

    print("Class counts before capping:")
    print(df[label_column].value_counts())

    groups = []
    rng = np.random.default_rng(RANDOM_STATE)

    # Group by the (overwritten) label column, which now contains the 4 meta-classes.
    for meta_label, g in df.groupby(label_column, sort=False):
        n = len(g)
        if n <= MIN_PER_CLASS:
            # Rare class: keep everything
            print(f"  {meta_label}: {n:,} rows (<= {MIN_PER_CLASS}), keeping all.")
            groups.append(g)
        elif n <= MAX_PER_CLASS:
            # Mid-sized class: keep everything
            print(f"  {meta_label}: {n:,} rows (between {MIN_PER_CLASS} and {MAX_PER_CLASS}), keeping all.")
            groups.append(g)
        else:
            # Very frequent class: cap at MAX_PER_CLASS via random sampling
            print(f"  {meta_label}: {n:,} rows (> {MAX_PER_CLASS}), sampling {MAX_PER_CLASS:,}.")
            groups.append(g.sample(n=MAX_PER_CLASS, random_state=RANDOM_STATE))

    df_var = pd.concat(groups, axis=0).reset_index(drop=True)
    df_var = df_var.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"\nVariational dataset size: {len(df_var):,} rows")
    print("Class counts after capping:")
    print(df_var[label_column].value_counts())

    return df_var


def build_qsvm_dataset(df_var: pd.DataFrame) -> pd.DataFrame:
    """Build a stratified QSVM dataset of N_QSVM rows from the variational dataset."""
    total_rows = len(df_var)
    if total_rows <= N_QSVM:
        print(
            f"Variational dataset has only {total_rows:,} rows "
            f"(<= N_QSVM={N_QSVM:,}); using all rows for QSVM.",
            flush=True,
        )
        return df_var.copy()

    print(f"\nBuilding QSVM dataset of {N_QSVM:,} rows (stratified by label)...", flush=True)
    counts = df_var[LABEL_COLUMN].value_counts()
    labels = counts.index.tolist()

    # Initial proportional allocation
    alloc = (counts / total_rows * N_QSVM).round().astype(int)
    # Ensure at least 1 row per class
    alloc[alloc < 1] = 1

    # Adjust allocations so that sum(alloc) == N_QSVM
    diff = N_QSVM - int(alloc.sum())
    while diff != 0:
        if diff > 0:
            # Need to add rows: distribute across classes with largest counts
            for label in alloc.sort_values(ascending=False).index:
                if diff == 0:
                    break
                alloc[label] += 1
                diff -= 1
        else:
            # Need to remove rows: reduce from classes with largest allocations
            for label in alloc.sort_values(ascending=False).index:
                if diff == 0:
                    break
                if alloc[label] > 1:
                    alloc[label] -= 1
                    diff += 1

    samples = []
    for label in labels:
        n = alloc[label]
        g = df_var[df_var[LABEL_COLUMN] == label]
        n = min(n, len(g))
        print(f"  QSVM: sampling {n} rows from class '{label}' (available {len(g):,}).")
        samples.append(g.sample(n=n, random_state=RANDOM_STATE))

    df_qsvm = pd.concat(samples, axis=0).reset_index(drop=True)
    df_qsvm = df_qsvm.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"QSVM dataset size: {len(df_qsvm):,} rows")
    print("QSVM class counts:")
    print(df_qsvm[LABEL_COLUMN].value_counts())

    return df_qsvm


def main() -> None:
    data_dir = DEFAULT_DATA_DIR
    train_parquet = DEFAULT_TRAIN_PARQUET

    if not train_parquet.exists():
        raise FileNotFoundError(f"Train parquet not found at {train_parquet}")

    print(f"Using data directory: {data_dir}")

    # 1) Build variational dataset
    df_var = build_variational_dataset(train_parquet, label_column=LABEL_COLUMN)

    # 2) Save variational dataset
    var_path = data_dir / "CIC_IoMT_2024_Variational.parquet"
    print(f"\nSaving variational dataset to {var_path} ...", flush=True)
    df_var.to_parquet(var_path, index=False)
    print("Saved variational dataset.\n")

    # 3) Build QSVM dataset from variational dataset
    df_qsvm = build_qsvm_dataset(df_var)

    # 4) Save QSVM dataset
    qsvm_path = data_dir / "CIC_IoMT_2024_QSVM.parquet"
    print(f"\nSaving QSVM dataset to {qsvm_path} ...", flush=True)
    df_qsvm.to_parquet(qsvm_path, index=False)
    print("Saved QSVM dataset.")


if __name__ == "__main__":
    main()

